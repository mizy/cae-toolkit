"""Validation selection and raw held-out tests for full-pool online jitter.

The original B pair readout supplies all Rim delta results. Absolute values use
an added, train-only warmed readout and do not establish original B absolute
prediction capability. Image translations here are a separate synthetic probe;
Rim phase results always consume the actual re-rendered depth/mask caches.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from . import joint_sampling as experiment
from . import sampling_experiment as rim_experiment
from .delta_experiment import sha256, tensor_sha256, write_json
from .image_perturbation import translate_views
from .phase_views import phase_id
from .sampling_evaluation import regression_pair
from .sampling_validation import candidate_gates, choose_candidate, score_checkpoint

FIXED_OFFSETS = ((0.25, 0.0), (-0.25, 0.0), (0.0, 0.25))


def absolute_scores(truth, prediction, sources, phase_predictions=None):
    """Score only real finite labels, preserving source and target denominators."""
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    if truth.shape != prediction.shape or truth.ndim != 2 or truth.shape[1] != 2:
        raise ValueError("Require matching Nx2 physical targets/predictions")
    if not np.isfinite(prediction).all() or len(sources) != len(truth):
        raise ValueError("Nonfinite predictions or mismatched sources")
    sources = np.asarray(sources)
    groups = {"all": np.ones(len(truth), dtype=bool)}
    groups.update({s: sources == s for s in sorted(set(sources))})
    groups["aircraft"] = np.isin(sources, ("cocoanet", "hilift"))
    groups["vehicle"] = ~groups["aircraft"]
    result = {}
    for source, selected in groups.items():
        row = {"case_count": int(selected.sum()), "targets": {}}
        for j, name in enumerate(("Cd", "Cl")):
            valid = selected & np.isfinite(truth[:, j])
            error = prediction[valid, j] - truth[valid, j]
            values = {
                "label_count": int(valid.sum()),
                "mae": None,
                "bias": None,
                "p95_abs": None,
            }
            if valid.any():
                values.update(
                    mae=float(np.abs(error).mean()),
                    bias=float(error.mean()),
                    p95_abs=float(np.quantile(np.abs(error), 0.95)),
                )
            if phase_predictions is not None:
                phased = np.asarray(phase_predictions)
                if (
                    phased.ndim != 3
                    or phased.shape[1:] != truth.shape
                    or not np.isfinite(phased).all()
                ):
                    raise ValueError("Require finite KxNx2 shifted predictions")
                drift = phased[:, selected, j] - prediction[None, selected, j]
                values["sampling_drift_mae"] = (
                    float(np.abs(drift).mean()) if drift.size else None
                )
                values["sampling_drift_p95_abs"] = (
                    float(np.quantile(np.abs(drift), 0.95)) if drift.size else None
                )
                values["shifted_mae"] = (
                    float(np.abs(phased[:, valid, j] - truth[None, valid, j]).mean())
                    if valid.any()
                    else None
                )
            row["targets"][name] = values
        result[source] = row
    return result


def evaluate_full(model, readout, data, p, split, device, offsets=(), arrays_path=None):
    """@entry Evaluate auxiliary physical Cd/Cl without consuming another split."""
    indices = [i for i, c in enumerate(data.cases) if c["split"] == split]
    if not indices:
        raise ValueError(f"No cases in {split}")
    model.eval()
    readout.eval()
    predictions, shifted = [], [[] for _ in offsets]
    scale = np.asarray(p["normalization"]["target_std"])
    mean = np.asarray(p["normalization"]["target_mean"])
    with torch.inference_mode():
        for start in range(0, len(indices), 8):
            current = indices[start : start + 8]
            views, params, _ = experiment.batch_cases(data, current, device)
            predictions.append(
                readout(model.encode(views), params).cpu().numpy() * scale + mean
            )
            for k, offset in enumerate(offsets):
                delta = torch.tensor(offset, dtype=torch.float32, device=device).expand(
                    len(current), 2
                )
                shifted[k].append(
                    readout(model.encode(translate_views(views, delta)), params)
                    .cpu()
                    .numpy()
                    * scale
                    + mean
                )
    prediction = np.concatenate(predictions)
    phase_prediction = (
        np.stack([np.concatenate(rows) for rows in shifted]) if offsets else None
    )
    truth = data.targets[indices].cpu().numpy()
    sources = [data.cases[i]["source"] for i in indices]
    result = {
        "split": split,
        "metrics": absolute_scores(truth, prediction, sources, phase_prediction),
        "offsets_pixels": list(offsets),
        "scope": "Train-only warmed auxiliary readout on shared B CNN, not original B absolute outputs. Synthetic image shifts do not replace raw Rim re-rendered phase tests.",
    }
    if arrays_path is not None:
        arrays = {
            "case_ids": np.asarray([data.cases[i]["case_id"] for i in indices]),
            "sources": np.asarray(sources),
            "truth": truth,
            "prediction": prediction,
        }
        if offsets:
            arrays.update(
                shifted_prediction=phase_prediction, offsets_pixels=np.asarray(offsets)
            )
        np.savez_compressed(arrays_path, **arrays)
        result.update(
            arrays=str(Path(arrays_path).resolve()), arrays_sha256=sha256(arrays_path)
        )
    return result


def absolute_guard(reference, candidate, tolerance):
    """Protect each source's finite-label MAE; pooled improvements cannot hide loss."""
    reasons = []
    for source, row in reference["metrics"].items():
        if source in ("all", "vehicle", "aircraft"):
            continue
        for target, values in row["targets"].items():
            other = candidate["metrics"][source]["targets"][target]
            if other["label_count"] != values["label_count"]:
                raise ValueError("Absolute evaluation label denominator changed")
            if values["mae"] is not None and other["mae"] > values["mae"] * (
                1 + tolerance
            ):
                reasons.append(
                    f"{source}/{target} absolute MAE exceeds reference by {tolerance:.0%}"
                )
    return {"pass": not reasons, "reasons": reasons}


def freeze_val_plan(experiment_root):
    """@entry Freeze validation scope and gates before scoring any candidate."""
    root = Path(experiment_root)
    p, cases = experiment.load(root)
    sp, rim = rim_experiment.load(p["sampling_parent"])
    phase_path = Path(sp["phase_root"]) / "plan.json"
    ids = json.loads(phase_path.read_text())["selection"]["val"]
    by_id = {c["case_id"]: c for c in rim}
    if len(ids) != 12 or any(by_id[i]["split"] != "val" for i in ids):
        raise ValueError("Require 12 fixed validation geometries")
    plan = {
        "validation_case_ids": ids,
        "phase_ids": sp["phase_ids"],
        "anchors": sp["anchors"],
        "full_validation_case_ids": [
            c["case_id"] for c in cases if c["split"] == "val"
        ],
        "preservation": {
            "tolerance": 0.05,
            "min_amplitude_fraction": 0.9,
            "direction_slack": 0.03,
        },
        "stability": {"mae_over_delta_std": 0.05, "p95_over_delta_std": 0.10},
        "absolute_mae_tolerance": 0.05,
        "absolute_reference": "Each finite source/target versus initial warmed auxiliary head; online also versus control at the same epoch",
        "epochs": list(range(p["epochs"] + 1)),
        "arms": list(p["arms"]),
        "selection": "Adopt only an all-gates passing candidate; otherwise retain original B and report provisional and diagnostic candidates explicitly",
        "test_offsets_pixels": FIXED_OFFSETS,
        "source_sha256": {
            str(path.resolve()): sha256(path)
            for path in (root / "protocol.json", phase_path)
        },
        "code_sha256": {
            str(path.resolve()): sha256(path)
            for path in (
                Path(__file__),
                Path(__file__).with_name("sampling_validation.py"),
                Path(__file__).with_name("phase_metrics.py"),
                Path(__file__).with_name("delta_metrics.py"),
            )
        },
    }
    path = root / "validation-plan.json"
    # Round-trip tuples so comparison uses exactly the serialized plan.
    plan = json.loads(json.dumps(plan))
    if path.exists() and json.loads(path.read_text()) != plan:
        raise ValueError("Frozen validation plan differs")
    if not path.exists():
        write_json(path, plan)
    return plan


def completed_checkpoints(root, p):
    """Require both completed six-epoch arms with identical actual exposure."""
    paths, reports = [], []
    train_count = 4121
    for arm in p["arms"]:
        directory = root / arm
        report = json.loads((directory / "training.json").read_text())
        if [row["epoch"] for row in report["history"]] != list(
            range(1, p["epochs"] + 1)
        ) or report["optimizer_steps"] != p["training_steps"]:
            raise ValueError(f"All six epochs must finish before selection: {arm}")
        for actual, expected in (
            ("executed_full_trace_sha256", "full_trace_sha256"),
            ("executed_pair_sha256", "pair_trace_sha256"),
            ("initial_tensor_sha256", "initial_tensor_sha256"),
            ("initial_absolute_sha256", "initial_absolute_sha256"),
        ):
            if report[actual] != p[expected]:
                raise ValueError(
                    f"Training trace or initialization differs: {arm}/{actual}"
                )
        if len(report["full_case_exposures"]) != train_count or set(
            report["full_case_exposures"].values()
        ) != {p["epochs"]}:
            raise ValueError(f"Full pool was not visited exactly once per epoch: {arm}")
        if sum(report["rim_endpoint_exposures"].values()) != 2 * 720 * p["epochs"]:
            raise ValueError(f"Rim pair exposure budget differs: {arm}")
        if arm == "control" and any(report["perturbed_full_case_exposures"].values()):
            raise ValueError("Control received perturbed full-pool images")
        if (
            arm == "online"
            and report["perturbed_full_case_exposures"] != report["full_case_exposures"]
        ):
            raise ValueError("Online perturbations did not cover the complete pool")
        for epoch in range(p["epochs"] + 1):
            path = directory / f"epoch-{epoch:02d}.pt"
            if not path.is_file():
                raise ValueError(f"Missing retained epoch: {path}")
            paths.append((arm, epoch, path))
        reports.append(report)
    for name in ("full_case_exposures", "rim_endpoint_exposures"):
        if any(r[name] != reports[0][name] for r in reports[1:]):
            raise ValueError(f"Arm exposure counts differ: {name}")
    return paths


def select(experiment_root):
    """@entry Score all retained validation epochs, then seal the test entrypoint."""
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
    sp, rim_cases = rim_experiment.load(p["sampling_parent"])
    rim_data, full_data = (
        rim_experiment.SamplingCases(sp, rim_cases),
        experiment.FullCases(p, cases),
    )
    if [c["case_id"] for c in cases if c["split"] == "val"] != plan[
        "full_validation_case_ids"
    ]:
        raise ValueError("Full-pool validation membership changed")
    torch.set_num_threads(2)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    candidates, cache, baseline = [], {}, None
    for arm, epoch, path in [("original_B", None, root / "initial.pt"), *checkpoints]:
        model, readout = experiment.load_checkpoint(path, device)
        ck = torch.load(path, map_location="cpu", weights_only=True)
        if arm != "original_B" and (
            ck["arm"] != arm
            or ck["epoch"] != epoch
            or ck["protocol_sha256"] != sha256(root / "protocol.json")
        ):
            raise ValueError(f"Checkpoint metadata changed: {path}")
        identity = (tensor_sha256(model), tensor_sha256(readout))
        if identity not in cache:
            cache[identity] = {
                "metrics": score_checkpoint(
                    model,
                    rim_data,
                    sp,
                    rim_cases,
                    plan["validation_case_ids"],
                    plan["phase_ids"],
                ),
                "full_absolute": evaluate_full(
                    model, readout, full_data, p, "val", device
                ),
            }
        entry = {
            "arm": arm,
            "epoch": epoch,
            "checkpoint": str(path.resolve()),
            "checkpoint_sha256": sha256(path),
            "state_tensor_sha256": identity[0],
            "absolute_tensor_sha256": identity[1],
            **cache[identity],
        }
        if baseline is None:
            baseline = entry
        entry["gates"] = candidate_gates(
            baseline["metrics"], entry["metrics"], plan, p["delta_std"]
        )
        references = {"initial_auxiliary": baseline}
        if arm == "online":
            references["matched_control"] = next(
                c for c in candidates if c["arm"] == "control" and c["epoch"] == epoch
            )
        absolute = {
            name: absolute_guard(
                ref["full_absolute"],
                entry["full_absolute"],
                plan["absolute_mae_tolerance"],
            )
            for name, ref in references.items()
        }
        entry["gates"]["absolute_guards"] = absolute
        if not all(g["pass"] for g in absolute.values()):
            entry["gates"].update(nominal_preserved=False, validation_passed=False)
            entry["gates"]["reasons"] += [
                f"absolute/{name}: {reason}"
                for name, gate in absolute.items()
                for reason in gate["reasons"]
            ]
        if arm != "original_B":
            candidates.append(entry)
        print(
            json.dumps(
                {
                    "arm": arm,
                    "epoch": epoch,
                    "score": entry["gates"]["score"],
                    "passed": entry["gates"]["validation_passed"],
                }
            ),
            flush=True,
        )
        del model, readout
        if device.type == "mps":
            torch.mps.empty_cache()
    provisional = choose_candidate(candidates, baseline)
    result = (
        provisional
        if provisional["validation_passed"]
        else choose_candidate([], baseline)
    )
    result.update(
        provisional_recommendation=provisional,
        validation_plan_sha256=sha256(plan_path),
        protocol_sha256=sha256(root / "protocol.json"),
        scored_candidates=len(candidates),
        unique_scored_models=len(cache),
    )
    write_json(
        root / "validation-metrics.json",
        {"baseline": baseline, "candidates": candidates, "selection": result},
    )
    result["validation_metrics_sha256"] = sha256(root / "validation-metrics.json")
    write_json(root / "selection.json", result)
    return result


def test_representatives(validation, selection):
    """Resolve every model from validation only, retaining nonzero rejected attempts."""
    baseline, candidates = validation["baseline"], validation["candidates"]
    entries = [baseline, *candidates]
    result = {"original_B": baseline}
    for arm in ("control", "online"):
        options = [c for c in candidates if c["arm"] == arm]
        chosen = choose_candidate(options, baseline)
        result[arm] = next(
            c for c in entries if c["checkpoint"] == chosen["selected_checkpoint"]
        )
        result[arm + "-diagnostic"] = min(
            (c for c in options if c["epoch"] > 0),
            key=lambda c: (c["gates"]["score"], c["checkpoint"]),
        )
    matched_epoch = result["online-diagnostic"]["epoch"]
    result["control-matched"] = next(
        c for c in candidates if c["arm"] == "control" and c["epoch"] == matched_epoch
    )
    result["selected"] = next(
        c for c in entries if c["checkpoint"] == selection["selected_checkpoint"]
    )
    return result


def evaluate(experiment_root):
    """@entry Test sealed representatives; no test reranking or training occurs."""
    root = Path(experiment_root)
    if (root / "comparison.json").exists():
        raise FileExistsError("Comparison already recorded")
    p, cases = experiment.load(root)
    selection = json.loads((root / "selection.json").read_text())
    if (
        sha256(root / "validation-metrics.json")
        != selection["validation_metrics_sha256"]
        or sha256(root / "validation-plan.json") != selection["validation_plan_sha256"]
    ):
        raise ValueError("Sealed validation artifacts changed")
    validation = json.loads((root / "validation-metrics.json").read_text())
    plan = json.loads((root / "validation-plan.json").read_text())
    representatives = test_representatives(validation, selection)
    sp, rim_cases = rim_experiment.load(p["sampling_parent"])
    phase_plan = json.loads((Path(sp["phase_root"]) / "plan.json").read_text())
    test_ids, subset = (
        phase_plan["selection"]["all_test"],
        phase_plan["selection"]["test"],
    )
    main_phases = [phase_id((0.25, 0))]
    extra_phases = [phase_id((0.125, 0)), phase_id((0.5, 0)), phase_id((0, 0.25))]
    if (
        len(test_ids) != 92
        or len(subset) != 24
        or sum(c["split"] == "test" for c in cases) != 819
    ):
        raise ValueError(
            "Frozen held-out coverage differs from 92/24 Rim and 819 full-pool geometries"
        )
    for entry in representatives.values():
        if sha256(entry["checkpoint"]) != entry["checkpoint_sha256"]:
            raise ValueError("Representative checkpoint changed after validation")
    write_json(
        root / "test-plan.json",
        {
            "models": {
                n: {
                    k: c[k] for k in ("checkpoint", "checkpoint_sha256", "arm", "epoch")
                }
                for n, c in representatives.items()
            },
            "main_case_ids": test_ids,
            "main_pair_count": math.comb(len(test_ids), 2),
            "main_phase_ids": main_phases,
            "extra_case_ids": subset,
            "extra_phase_ids": extra_phases,
            "fullpool_offsets_pixels": FIXED_OFFSETS,
            "test_selection_used": False,
            "scope": "Raw re-rendered phases for original B pair readout; separate synthetic-shift absolute auxiliary diagnostic. Failed attempts remain diagnostic.",
        },
    )
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    torch.set_num_threads(2)
    rim_data, full_data = (
        rim_experiment.SamplingCases(sp, rim_cases),
        experiment.FullCases(p, cases),
    )
    result, cache = {}, {}
    for name, entry in representatives.items():
        identity = entry["checkpoint_sha256"]
        if identity not in cache:
            model, readout = experiment.load_checkpoint(entry["checkpoint"], device)
            main = score_checkpoint(
                model, rim_data, sp, rim_cases, test_ids, main_phases, split="test"
            )
            extra = score_checkpoint(
                model, rim_data, sp, rim_cases, subset, extra_phases, split="test"
            )
            cache[identity] = {
                "checkpoint": entry["checkpoint"],
                "chosen_arm": entry["arm"],
                "epoch": entry["epoch"],
                "main": main,
                "extra": extra,
                "regression": regression_pair(
                    model,
                    rim_data,
                    sp,
                    "drivaerrim__rim-0001",
                    "drivaerrim__rim-0002",
                    main_phases[0],
                    device,
                ),
                "full_absolute": evaluate_full(
                    model,
                    readout,
                    full_data,
                    p,
                    "test",
                    device,
                    FIXED_OFFSETS,
                    root / f"test-absolute-{name}.npz",
                ),
            }
            del model, readout
            if device.type == "mps":
                torch.mps.empty_cache()
        result[name] = dict(cache[identity])
        print(
            json.dumps(
                {"evaluated": name, "arm": entry["arm"], "epoch": entry["epoch"]}
            ),
            flush=True,
        )
    for name, record in result.items():
        for scope in ("main", "extra"):
            record["test_" + scope + "_gates"] = candidate_gates(
                result["original_B"][scope], record[scope], plan, p["delta_std"]
            )
        record["absolute_guard_vs_initial"] = absolute_guard(
            result["original_B"]["full_absolute"],
            record["full_absolute"],
            plan["absolute_mae_tolerance"],
        )
        control_name = "control-matched" if name == "online-diagnostic" else "control"
        record["absolute_control_reference"] = control_name
        record["absolute_guard_vs_control"] = absolute_guard(
            result[control_name]["full_absolute"],
            record["full_absolute"],
            plan["absolute_mae_tolerance"],
        )
    summary = {
        "selection": selection,
        "models": result,
        "scope": "Validation-only selection. Rim pair/phase rows are correlated repeated measurements, not independent CFD. Absolute original_B row means original B CNN plus train-only warmed auxiliary head. No Radiant rerun.",
    }
    write_json(root / "comparison.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "select", "test"])
    parser.add_argument("experiment", type=Path)
    args = parser.parse_args()
    {"prepare": freeze_val_plan, "select": select, "test": evaluate}[args.action](
        args.experiment
    )


if __name__ == "__main__":
    main()
