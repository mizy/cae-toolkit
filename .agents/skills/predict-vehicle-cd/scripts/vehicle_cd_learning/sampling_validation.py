"""Validation-only preservation and camera-phase gates for frozen sampling arms."""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from . import sampling_experiment as experiment
from .delta_experiment import tensor_sha256
from .delta_metrics import score_anchors
from .phase_metrics import preservation_gate, sampling_scores
from .stable_delta import initialize_from_checkpoint

PATTERNS = ("common", "query_shifted", "reference_shifted")
sha256, write_json = experiment.sha256, experiment.write_json


def prepare(experiment_root):
    """@entry Freeze selection rules and validation IDs before scoring any model."""
    root = Path(experiment_root)
    p, cases = experiment.load(root)
    phase_plan_path = Path(p["phase_root"]) / "plan.json"
    phase_plan = json.loads(phase_plan_path.read_text())
    identities = phase_plan["selection"]["val"]
    by_id = {c["case_id"]: c for c in cases}
    if len(identities) != 12 or any(by_id[i]["split"] != "val" for i in identities):
        raise ValueError("Require exactly the 12 predeclared validation geometries")
    plan = {
        "validation_case_ids": identities,
        "phase_ids": p["phase_ids"],
        "anchors": p["anchors"],
        "preservation": {"tolerance": 0.05, "min_amplitude_fraction": 0.9, "direction_slack": 0.03},
        "stability": {"mae_over_delta_std": 0.05, "p95_over_delta_std": 0.10},
        "patterns": list(PATTERNS),
        "epochs": list(range(7)),
        "arms": list(p["arms"]),
        "score": "Maximum of the three pattern normalized_stability_score values; each score averages targets and supplied phases",
        "preservation_scope": "Each full nominal anchor plus pooled anchors and all66 nominal validation pairs versus original B; each aggregate phase pattern versus candidate nominal pairs",
        "fallback": "Prefer all gates; else nominal+phase preserving; else nominal preserving including original B fallback. Always mark failed stability/preservation explicitly.",
        "scope": "Engineering tolerances for representation sensitivity, not CFD noise. Validation geometries and correlated pair rows are not independent samples. No test or Radiant scoring.",
        "source_sha256": {str(path.resolve()): sha256(path) for path in (root / "protocol.json", phase_plan_path)},
        "code_sha256": {
            str(path.resolve()): sha256(path)
            for path in (
                Path(__file__),
                Path(__file__).with_name("phase_metrics.py"),
                Path(__file__).with_name("delta_metrics.py"),
            )
        },
    }
    path = root / "validation-plan.json"
    if path.exists() and json.loads(path.read_text()) != plan:
        raise ValueError("Frozen validation plan differs")
    if not path.exists():
        write_json(path, plan)
    return plan


def score_checkpoint(model, data, p, cases, phase_case_ids, phase_ids, split="val"):
    """Encode each requested image once, then score physical deltas from latents."""
    identities, phases = list(phase_case_ids), list(phase_ids)
    by_id = {c["case_id"]: c for c in cases}
    if len(identities) < 2 or len(set(identities)) != len(identities) or not phases or len(set(phases)) != len(phases):
        raise ValueError("Require unique phase cases and unique nonempty phases")
    if any(by_id[i]["split"] != split for i in identities):
        raise ValueError("Phase cases must belong to requested split")
    queries = sorted(c["case_id"] for c in cases if c["source"] == "drivaerrim" and c["split"] == split)
    nominal_ids = list(dict.fromkeys(p["anchors"] + queries))
    requests = [(data.index[i], None) for i in nominal_ids] + [
        (data.index[i], phase) for phase in phases for i in identities
    ]
    device, scale = next(model.parameters()).device, np.asarray(p["delta_std"])
    cache = {}
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(requests), 8):
            current = requests[start : start + 8]
            views, _, _ = experiment.batch_views(data, current, device)
            encoded = model.encode(views)
            cache.update({key: {n: value[k].cpu() for n, value in encoded.items()} for k, key in enumerate(current)})

        def predict(pairs):
            outputs = []
            for start in range(0, len(pairs), 64):
                current = pairs[start : start + 64]
                a, b = zip(*current)
                pa = {n: torch.stack([cache[key][n] for key in a]).to(device) for n in cache[a[0]]}
                pb = {n: torch.stack([cache[key][n] for key in b]).to(device) for n in cache[b[0]]}
                params = data.parameters[[key[0] for key in a]].to(device)
                outputs.append(model.predict_pair(pa, pb, params).cpu().numpy() * scale)
            return np.concatenate(outputs)

        anchor_pairs = [((data.index[a], None), (data.index[q], None)) for a in p["anchors"] for q in queries]
        anchor_values = predict(anchor_pairs).reshape(len(p["anchors"]), len(queries), 2)
        anchor_predictions = {
            a: {q: value.tolist() for q, value in zip(queries, values)}
            for a, values in zip(p["anchors"], anchor_values)
        }
        pairs = list(itertools.combinations(identities, 2))
        truth = np.asarray([np.asarray(by_id[b]["targets"]) - by_id[a]["targets"] for a, b in pairs])
        nominal = predict([((data.index[a], None), (data.index[b], None)) for a, b in pairs])
        self_phase = np.stack(
            [predict([((data.index[i], None), (data.index[i], phase)) for i in identities]) for phase in phases]
        )
        phase_predictions = {name: [] for name in PATTERNS}
        for phase in phases:
            for name in PATTERNS:
                left_phase = phase if name in ("common", "reference_shifted") else None
                right_phase = phase if name in ("common", "query_shifted") else None
                phase_predictions[name].append(
                    predict([((data.index[a], left_phase), (data.index[b], right_phase)) for a, b in pairs])
                )
    patterns = {
        name: sampling_scores(truth, nominal, values, self_phase, scale) for name, values in phase_predictions.items()
    }
    return {
        "nominal_anchors": score_anchors(cases, p["anchors"], anchor_predictions, split, p["delta_std"]),
        "nominal_anchor_predictions": anchor_predictions,
        "patterns": patterns,
        "phase_scoring": {
            "case_ids": identities,
            "phase_ids": phases,
            "pairs": [list(pair) for pair in pairs],
            "truth_pairs": truth.tolist(),
            "nominal_pred": nominal.tolist(),
            "self_phase_pred": self_phase.tolist(),
            **{name + "_pred": np.asarray(values).tolist() for name, values in phase_predictions.items()},
        },
        "encoded_image_count": len(cache),
        "split": split,
    }


def candidate_gates(reference, candidate, plan, delta_std):
    """Preserve aerodynamic response before considering stability improvements."""
    kwargs = plan["preservation"]
    full = {
        "pooled": preservation_gate(
            reference["nominal_anchors"]["pooled_correlated_pairs"],
            candidate["nominal_anchors"]["pooled_correlated_pairs"],
            **kwargs,
        )
    }
    full.update(
        {
            a: preservation_gate(
                reference["nominal_anchors"]["by_anchor"][a], candidate["nominal_anchors"]["by_anchor"][a], **kwargs
            )
            for a in plan["anchors"]
        }
    )
    nominal = preservation_gate(
        reference["patterns"]["common"]["nominal_score"], candidate["patterns"]["common"]["nominal_score"], **kwargs
    )
    phase = {
        name: preservation_gate(
            candidate["patterns"][name]["nominal_score"], candidate["patterns"][name]["phase_score"], **kwargs
        )
        for name in PATTERNS
    }
    reasons, stability = [], {}
    for name in PATTERNS:
        stability[name] = {}
        for target, scale in zip(("Cd", "Cl"), delta_std):
            values = candidate["patterns"][name]["targets"][target]
            normalized = {
                metric: values[metric] / scale
                for metric in (
                    "self_phase_mae",
                    "self_phase_p95_abs",
                    "common_phase_drift_mae",
                    "common_phase_drift_p95_abs",
                )
            }
            stability[name][target] = normalized
            for metric, value in normalized.items():
                limit = plan["stability"]["p95_over_delta_std" if "p95" in metric else "mae_over_delta_std"]
                if value > limit:
                    reasons.append(f"{name}/{target}/{metric} exceeds {limit} delta_std")
    nominal_passed = nominal["pass"] and all(g["pass"] for g in full.values())
    phase_preserved = all(g["pass"] for g in phase.values())
    preservation_reasons = [f"nominal/{name}: {reason}" for name, gate in full.items() for reason in gate["reasons"]]
    preservation_reasons += ["nominal/pairs: " + reason for reason in nominal["reasons"]]
    preservation_reasons += [f"phase/{name}: {reason}" for name, gate in phase.items() for reason in gate["reasons"]]
    return {
        "nominal_preserved": nominal_passed,
        "phase_preserved": phase_preserved,
        "stability_passed": not reasons,
        "validation_passed": nominal_passed and phase_preserved and not reasons,
        "nominal_anchor_gates": full,
        "nominal_pair_gate": nominal,
        "phase_gates": phase,
        "normalized_stability": stability,
        "score": max(candidate["patterns"][name]["normalized_stability_score"] for name in PATTERNS),
        "reasons": preservation_reasons + reasons,
    }


def choose_candidate(candidates, baseline):
    """Select only within the strongest available preservation/stability tier."""
    passing = [c for c in candidates if c["gates"]["validation_passed"]]
    preserving = [c for c in candidates if c["gates"]["nominal_preserved"] and c["gates"]["phase_preserved"]]
    nominal = [c for c in [baseline, *candidates] if c["gates"]["nominal_preserved"]]
    pool, tier = (
        (passing, "all_gates")
        if passing
        else (preserving, "preservation_only")
        if preserving
        else (nominal, "nominal_only_provisional")
    )
    if not pool:
        raise ValueError("Original B fallback must preserve its own nominal metrics")
    selected = min(pool, key=lambda c: (c["gates"]["score"], c["checkpoint"]))
    return {
        "selected_checkpoint": selected["checkpoint"],
        "selected_checkpoint_sha256": selected["checkpoint_sha256"],
        "selected_model_sha256": selected["checkpoint_sha256"],
        "chosen_arm": selected["arm"],
        "arm": selected["arm"],
        "epoch": selected["epoch"],
        "selection_tier": tier,
        "validation_passed": selected["gates"]["validation_passed"],
        "nominal_preserved": selected["gates"]["nominal_preserved"],
        "phase_preserved": selected["gates"]["phase_preserved"],
        "stability_passed": selected["gates"]["stability_passed"],
        "score": selected["gates"]["score"],
        "reasons": selected["gates"]["reasons"],
        "baseline_fallback": selected["arm"] == "original_B",
        "scope": "Selection uses validation only; failed gates remain failures even when a provisional checkpoint is retained",
    }


def completed_checkpoints(root, p):
    paths, exposure_reference = [], None
    for arm, config in p["arms"].items():
        directory = root / arm
        report = json.loads((directory / "training.json").read_text())
        if report["optimizer_steps"] != 540 or [row["epoch"] for row in report["history"]] != list(range(1, 7)):
            raise ValueError(f"All six epochs must finish before validation selection: {arm}")
        if (
            report["executed_pair_sha256"] != p["pair_trace_sha256"]
            or report["executed_aux_sha256"] != p["aux_trace_sha256"]
        ):
            raise ValueError(f"Actual training trace differs: {arm}")
        endpoints, phases = report["endpoint_exposures"], report["phase_geometry_exposures"]
        if sum(endpoints.values()) != 540 * 20 or sum(phases.values()) != (540 * 2 if config["phases"] else 0):
            raise ValueError(f"Image exposure budget differs: {arm}")
        if exposure_reference is not None and endpoints != exposure_reference:
            raise ValueError("Arms saw different physical geometry exposure counts")
        exposure_reference = endpoints
        for epoch in range(7):
            path = directory / f"epoch-{epoch:02d}.pt"
            if not path.is_file():
                raise ValueError(f"Missing retained epoch: {path}")
            paths.append((arm, epoch, path))
    return paths


def select(experiment_root):
    """@entry Score all retained epochs using frozen validation rules, then select."""
    root = Path(experiment_root)
    if (root / "selection.json").exists():
        raise FileExistsError("Validation selection already exists")
    p, cases = experiment.load(root)
    plan_path = root / "validation-plan.json"
    plan = json.loads(plan_path.read_text())
    for path, expected in {**plan["source_sha256"], **plan["code_sha256"]}.items():
        if sha256(path) != expected:
            raise ValueError(f"Frozen validation source changed: {path}")
    checkpoints = completed_checkpoints(root, p)
    data = experiment.SamplingCases(p, cases)
    torch.set_num_threads(2)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    cache, candidates = {}, []
    reference = None
    all_paths = [("original_B", None, Path(p["checkpoint"])), *checkpoints]
    for arm, epoch, path in all_paths:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        floor, filtered = checkpoint.get("coverage_floor", 0), checkpoint.get("filter_input", False)
        if arm != "original_B":
            config = p["arms"][arm]
            if (
                checkpoint["epoch"] != epoch
                or checkpoint["arm"] != arm
                or checkpoint["protocol_sha256"] != sha256(root / "protocol.json")
                or floor != config["coverage_floor"]
                or filtered != config.get("filter_input", False)
            ):
                raise ValueError(f"Checkpoint metadata differs from frozen arm: {path}")
        model = initialize_from_checkpoint(checkpoint, floor, filtered).to(device)
        key = (tensor_sha256(model), floor, filtered)
        if key not in cache:
            cache[key] = score_checkpoint(model, data, p, cases, plan["validation_case_ids"], plan["phase_ids"])
        metrics = cache[key]
        if reference is None:
            reference = metrics
        candidate = {
            "arm": arm,
            "epoch": epoch,
            "checkpoint": str(path.resolve()),
            "checkpoint_sha256": sha256(path),
            "state_tensor_sha256": key[0],
            "coverage_floor": floor,
            "filter_input": filtered,
            "metrics": metrics,
            "gates": candidate_gates(reference, metrics, plan, p["delta_std"]),
        }
        if arm == "original_B":
            baseline = candidate
        else:
            candidates.append(candidate)
        print(
            json.dumps(
                {
                    "arm": arm,
                    "epoch": epoch,
                    "validation_passed": candidate["gates"]["validation_passed"],
                    "score": candidate["gates"]["score"],
                }
            ),
            flush=True,
        )
    result = choose_candidate(candidates, baseline)
    result.update(
        {
            "validation_plan_sha256": sha256(plan_path),
            "protocol_sha256": sha256(root / "protocol.json"),
            "data_lock_sha256": data.data_lock_sha256,
            "scored_candidates": len(candidates),
            "unique_scored_models": len(cache),
        }
    )
    write_json(root / "validation-metrics.json", {"baseline": baseline, "candidates": candidates, "selection": result})
    result["validation_metrics_sha256"] = sha256(root / "validation-metrics.json")
    write_json(root / "selection.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "select"])
    parser.add_argument("experiment", type=Path)
    args = parser.parse_args()
    (prepare if args.action == "prepare" else select)(args.experiment)


if __name__ == "__main__":
    main()
