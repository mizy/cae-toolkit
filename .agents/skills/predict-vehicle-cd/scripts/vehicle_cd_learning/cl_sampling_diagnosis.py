"""Read-only validation diagnosis of frozen Cl camera-phase sensitivity.

Crossed feature/readout owners are interventions on network components only.
Feature bases may co-adapt; they are not aerodynamic force contributions.
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from . import sampling_experiment as experiment
from .delta_metrics import score_anchors
from .model import VIEWS
from .paired_delta import PairedDelta
from .phase_metrics import sampling_scores
from .sampling_validation import PATTERNS, candidate_gates

DEFAULT_OLD = Path("outputs/vehicle-sampling-consistency-20261117")
DEFAULT_OUT = Path("outputs/aerodynamic-cl-sampling-diagnosis-20261117")
CHECKPOINTS = {
    "B": "outputs/vehicle-delta-factorial-20261117/B/model.pt",
    "control5": "outputs/aerodynamic-online-jitter-20261117/control/epoch-05.pt",
    "E5": "outputs/aerodynamic-online-jitter-20261117/retained-online.pt",
}


def array_summary(values):
    """Signed distribution on frozen correlated rows, not confidence intervals."""
    values = np.asarray(values, dtype=float)
    return {
        name: {
            "mae": float(np.abs(values[..., i]).mean()),
            "signed_mean": float(values[..., i].mean()),
            "std": float(values[..., i].std()),
            "p95_abs": float(np.quantile(np.abs(values[..., i]), 0.95)),
            "positive_fraction": float((values[..., i] > 0).mean()),
        }
        for i, name in enumerate(("Cd", "Cl"))
    }


def phase_distribution(values, phases):
    """Group saved prediction rows without re-running inference."""
    values = np.asarray(values)
    groups = {
        "horizontal": (0, 1),
        "vertical": (2, 3),
        "positive": (0, 2),
        "negative": (1, 3),
    }
    return {
        "pooled": array_summary(values),
        "by_phase": {phase: array_summary(row) for phase, row in zip(phases, values)},
        "by_direction": {
            name: array_summary(values[list(indices)])
            for name, indices in groups.items()
        },
    }


def mix_views(nominal, shifted, view):
    """Select cached features and matching masks from one physical raw phase."""
    if view == "all5":
        return shifted
    return {
        name: shifted[name] if name in (view, view + "_mask") else value
        for name, value in nominal.items()
    }


def predict(model, cache, pairs, parameters, scale):
    outputs, device = [], next(model.parameters()).device
    for start in range(0, len(pairs), 64):
        current = pairs[start : start + 64]
        a, b = zip(*current)
        left = {
            n: torch.stack([cache[k][n] for k in a]).to(device) for n in cache[a[0]]
        }
        right = {
            n: torch.stack([cache[k][n] for k in b]).to(device) for n in cache[b[0]]
        }
        params = parameters[[k[0] for k in a]].to(device)
        outputs.append(model.predict_pair(left, right, params).cpu().numpy() * scale)
    return np.concatenate(outputs)


def score_from_features(model, cache, data, p, cases, identities, phases):
    """Nine component crosses consume the same frozen encoder feature caches."""
    by_id = {c["case_id"]: c for c in cases}
    queries = sorted(
        c["case_id"]
        for c in cases
        if c["source"] == "drivaerrim" and c["split"] == "val"
    )
    scale, index = np.asarray(p["delta_std"]), data.index

    def infer(pairs):
        return predict(model, cache, pairs, data.parameters, scale)

    anchor_pairs = [
        ((index[a], None), (index[q], None)) for a in p["anchors"] for q in queries
    ]
    values = infer(anchor_pairs).reshape(len(p["anchors"]), len(queries), 2)
    anchors = {
        a: {q: v.tolist() for q, v in zip(queries, row)}
        for a, row in zip(p["anchors"], values)
    }
    pairs = list(itertools.combinations(identities, 2))
    truth = np.asarray(
        [np.asarray(by_id[b]["targets"]) - by_id[a]["targets"] for a, b in pairs]
    )
    nominal = infer([((index[a], None), (index[b], None)) for a, b in pairs])
    self_phase = np.stack(
        [
            infer([((index[i], None), (index[i], phase)) for i in identities])
            for phase in phases
        ]
    )
    predictions = {name: [] for name in PATTERNS}
    for phase in phases:
        for name in PATTERNS:
            left = phase if name in ("common", "reference_shifted") else None
            right = phase if name in ("common", "query_shifted") else None
            predictions[name].append(
                infer([((index[a], left), (index[b], right)) for a, b in pairs])
            )
    return {
        "nominal_anchors": score_anchors(
            cases, p["anchors"], anchors, "val", p["delta_std"]
        ),
        "nominal_anchor_predictions": anchors,
        "patterns": {
            name: sampling_scores(truth, nominal, rows, self_phase, scale)
            for name, rows in predictions.items()
        },
        "phase_scoring": {
            "case_ids": identities,
            "phase_ids": phases,
            "pairs": pairs,
            "truth_pairs": truth.tolist(),
            "nominal_pred": nominal.tolist(),
            "self_phase_pred": self_phase.tolist(),
            **{
                name + "_pred": np.asarray(rows).tolist()
                for name, rows in predictions.items()
            },
        },
        "encoded_image_count": len(cache),
        "split": "val",
    }


def view_scores(model, cache, data, p, identities, phases):
    """Reuse encoded views; no mixed image is run through either encoder again."""
    results, index = {}, data.index
    for view in (*VIEWS, "all5"):
        mixed = dict(cache)
        for phase in phases:
            for identity in identities:
                key = index[identity], phase
                mixed[key] = mix_views(cache[(index[identity], None)], cache[key], view)
        rows = np.stack(
            [
                predict(
                    model,
                    mixed,
                    [((index[i], None), (index[i], phase)) for i in identities],
                    data.parameters,
                    np.asarray(p["delta_std"]),
                )
                for phase in phases
            ]
        )
        results[view] = {
            **phase_distribution(rows, phases),
            "predictions": rows.tolist(),
        }
    return results


def stage_name(name):
    if name.startswith("head.2."):
        return "head_final"
    if name.startswith("head."):
        return "head_hidden"
    return name.split(".")[0]


def parameter_changes(states):
    """Weight movement is descriptive; norms alone do not diagnose a defect."""
    output = {}
    for left, right in itertools.combinations(states, 2):
        groups = {}
        for name, tensor in states[left].items():
            group = stage_name(name)
            groups.setdefault(group, []).append(
                (tensor.double().flatten(), states[right][name].double().flatten())
            )
            if group == "head_final":
                for i, target in enumerate(("Cd", "Cl")):
                    groups.setdefault("head_final_" + target + "_row", []).append(
                        (
                            tensor[i].double().flatten(),
                            states[right][name][i].double().flatten(),
                        )
                    )
        output[left + "_to_" + right] = {}
        for group, pairs in groups.items():
            a, b = (torch.cat([pair[i] for pair in pairs]) for i in (0, 1))
            norm, change = (
                float(torch.linalg.vector_norm(a)),
                float(torch.linalg.vector_norm(b - a)),
            )
            output[left + "_to_" + right][group] = {
                "reference_l2": norm,
                "change_l2": change,
                "relative_l2": change / norm if norm else None,
            }
    return output


def feature_changes(caches, data, identities, phases):
    """Measure phase-direction feature gains and alignment against B per view."""
    output = {}
    for owner, cache in caches.items():
        output[owner] = {}
        for view in VIEWS:
            differences, reference = [], []
            for phase in phases:
                for identity in identities:
                    i = data.index[identity]
                    differences.append(
                        (cache[i, phase][view] - cache[i, None][view]).numpy().ravel()
                    )
                    reference.append(
                        (caches["B"][i, phase][view] - caches["B"][i, None][view])
                        .numpy()
                        .ravel()
                    )
            a, b = (
                np.asarray(reference, dtype=np.float64),
                np.asarray(differences, dtype=np.float64),
            )
            an, bn = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
            cosine = (a * b).sum(1) / np.maximum(an * bn, 1e-30)
            output[owner][view] = {
                "phase_rms": float(np.sqrt(np.mean(b * b))),
                "rms_gain_vs_B": float(np.linalg.norm(b) / np.linalg.norm(a)),
                "mean_phase_direction_cosine_vs_B": float(cosine.mean()),
            }
    return output


def compact(metrics):
    return {
        "nominal_anchors": metrics["nominal_anchors"]["pooled_correlated_pairs"],
        "patterns": {name: row["targets"] for name, row in metrics["patterns"].items()},
    }


def run(output=DEFAULT_OUT, old_root=DEFAULT_OLD):
    """@entry Freeze inputs before validation-only, inference-only diagnostics."""
    output, old_root = Path(output), Path(old_root)
    if output.exists():
        raise FileExistsError("Diagnosis output must be fresh")
    p, cases = experiment.load(old_root)
    phase_plan = json.loads((Path(p["phase_root"]) / "plan.json").read_text())
    identities, phases = phase_plan["selection"]["val"], p["phase_ids"]
    if len(identities) != 12 or len(phases) != 4:
        raise ValueError("Require fixed 12 validation geometries and four phases")
    by_id = {c["case_id"]: c for c in cases}
    if any(by_id[i]["split"] != "val" for i in identities):
        raise ValueError("Validation-only identities required")
    paths = [Path(v) for v in CHECKPOINTS.values()]
    paths += list(Path(__file__).parent.glob("*.py"))
    paths += [
        old_root / "protocol.json",
        old_root / "data-lock.json",
        Path(p["phase_root"]) / "plan.json",
    ]
    source_hashes = {str(path.resolve()): experiment.sha256(path) for path in paths}
    plan = {
        "checkpoints": CHECKPOINTS,
        "validation_case_ids": identities,
        "phase_ids": phases,
        "anchors": p["anchors"],
        "source_sha256": source_hashes,
        "preservation": {
            "tolerance": 0.05,
            "min_amplitude_fraction": 0.9,
            "direction_slack": 0.03,
        },
        "stability": {"mae_over_delta_std": 0.05, "p95_over_delta_std": 0.10},
        "scope": "No training, test scoring, selection, checkpoint publication or original writes. 9 crosses exclude auxiliary absolute heads. Neural feature bases may co-adapt; crossed readouts cannot identify aerodynamic contributions. Single-view responses are nonlinear and cannot be added as forces. Repeated phases/pairs are correlated.",
        "phase_groups": {
            "horizontal": phases[:2],
            "vertical": phases[2:],
            "positive": phases[::2],
            "negative": phases[1::2],
        },
    }
    output.mkdir(parents=True)
    experiment.write_json(output / "plan.json", plan)
    data = experiment.SamplingCases(p, cases)
    data.nominal.limit = 8
    torch.set_num_threads(2)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    queries = sorted(
        c["case_id"]
        for c in cases
        if c["source"] == "drivaerrim" and c["split"] == "val"
    )
    requests = [(data.index[i], None) for i in dict.fromkeys(p["anchors"] + queries)]
    requests += [(data.index[i], phase) for phase in phases for i in identities]
    models, states, caches, normalization = {}, {}, {}, None
    with torch.inference_mode():
        for owner, path in CHECKPOINTS.items():
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            stats = {
                name: np.asarray(checkpoint[name])
                for name in ("delta_std", "parameter_mean", "parameter_std")
            }
            if normalization is not None and any(
                not np.array_equal(stats[name], normalization[name]) for name in stats
            ):
                raise ValueError("Checkpoint normalization differs")
            normalization = stats
            if not np.allclose(stats["delta_std"], p["delta_std"], rtol=1e-7, atol=0):
                raise ValueError("Protocol delta scaling differs")
            model = PairedDelta(8, 36)
            model.load_state_dict(checkpoint["model_state"], strict=True)
            states[owner] = {k: v.clone() for k, v in model.state_dict().items()}
            models[owner] = model.to(device).eval()
            cache = {}
            for start in range(0, len(requests), 4):
                current = requests[start : start + 4]
                views, _, _ = experiment.batch_views(data, current, device)
                features = model.encode(views)
                cache.update(
                    {
                        key: {n: value[k].cpu() for n, value in features.items()}
                        for k, key in enumerate(current)
                    }
                )
            caches[owner] = cache
            data.nominal.cache.clear()
            data.cache.clear()
            print(
                json.dumps(
                    {
                        "encoded_owner": owner,
                        "images": len(cache),
                        "device": str(device),
                    }
                ),
                flush=True,
            )
        crosses = {}
        for encoder, cache in caches.items():
            for readout, model in models.items():
                key = encoder + "__" + readout
                crosses[key] = score_from_features(
                    model, cache, data, p, cases, identities, phases
                )
                print(
                    json.dumps(
                        {
                            "cross": key,
                            "Cl_self_mae": crosses[key]["patterns"]["common"][
                                "targets"
                            ]["Cl"]["self_phase_mae"],
                        }
                    ),
                    flush=True,
                )
        views = {
            owner: view_scores(models[owner], cache, data, p, identities, phases)
            for owner, cache in caches.items()
        }
    reference = crosses["B__B"]
    gates = {
        key: candidate_gates(reference, value, plan, p["delta_std"])
        for key, value in crosses.items()
    }
    feature_report = feature_changes(caches, data, identities, phases)
    experiment.write_json(
        output / "predictions.json", {"crosses": crosses, "single_view": views}
    )
    summary = {
        "crosses": {key: compact(value) for key, value in crosses.items()},
        "gates": gates,
        "single_view": {
            owner: {
                view: {k: v for k, v in row.items() if k != "predictions"}
                for view, row in value.items()
            }
            for owner, value in views.items()
        },
        "phase_distributions": {
            key: {
                "self": phase_distribution(
                    value["phase_scoring"]["self_phase_pred"], phases
                ),
                **{
                    name + "_drift": phase_distribution(
                        np.asarray(value["phase_scoring"][name + "_pred"])
                        - np.asarray(value["phase_scoring"]["nominal_pred"])[None],
                        phases,
                    )
                    for name in PATTERNS
                },
            }
            for key, value in crosses.items()
        },
        "parameter_changes": parameter_changes(states),
        "feature_changes": feature_report,
        "encoded_images_per_owner": len(requests),
        "device": str(device),
        "scope": plan["scope"],
    }
    for path, expected in source_hashes.items():
        if experiment.sha256(path) != expected:
            raise ValueError("Frozen source changed during diagnosis: " + path)
    summary["source_unchanged"] = True
    experiment.write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--old-root", type=Path, default=DEFAULT_OLD)
    args = parser.parse_args()
    run(args.output, args.old_root)


if __name__ == "__main__":
    main()
