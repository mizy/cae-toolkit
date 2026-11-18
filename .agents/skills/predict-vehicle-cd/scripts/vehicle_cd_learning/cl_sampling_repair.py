"""Train-only ridge repair of the retained epoch-five Cl output row.

All CNN, adapter, conditional hidden features, Cd weights and auxiliary absolute
weights remain fixed. Actual cached re-renders supply sampling nuisance features.
Validation seals the choice before any held-out feature encoding or test score.
"""

import argparse
import copy
import itertools
import json
import time
from pathlib import Path

import numpy as np
import torch

from . import sampling_experiment as sampling
from .delta_experiment import sha256, write_json
from .delta_metrics import score_anchors
from .paired_delta import PairedDelta
from .phase_metrics import preservation_gate, sampling_scores
from .phase_views import phase_id
from .sampling_validation import PATTERNS, candidate_gates

EXPECTED_CHECKPOINT = "842dcf8edc942ce843a18343e394b8eac512441c3c96861a57a946a144f12230"


def capture_features(model, left, right, parameters):
    """Capture the existing 64 hidden units; add no inference parameters or input."""
    hidden = []
    hook = model.head[-1].register_forward_pre_hook(
        lambda module, args: hidden.append(args[0].detach())
    )
    try:
        prediction = model.predict_pair(left, right, parameters)
    finally:
        hook.remove()
    if len(hidden) != 2:
        raise ValueError("Expected positive and negative antisymmetric head calls")
    z = (hidden[0] - hidden[1]) * 0.5
    reconstruction = z @ model.head[-1].weight.T
    if not torch.allclose(prediction, reconstruction, atol=2e-6, rtol=2e-5):
        raise ValueError("Captured features do not reproduce the unchanged pair head")
    return z, float((prediction - reconstruction).abs().max().cpu())


def fit_row(
    design, truth, fidelity, nuisance, original, alpha, penalty, ridge_ratio=1e-6
):
    """Solve a sample-normalized ridge objective centered on the original row."""
    design, truth, fidelity, nuisance, original = [
        np.asarray(a, np.float64) for a in (design, truth, fidelity, nuisance, original)
    ]
    covariance = design.T @ design / len(design)
    fidelity_covariance = fidelity.T @ fidelity / len(fidelity)
    noise_covariance = nuisance.T @ nuisance / len(nuisance)
    ridge = ridge_ratio * np.trace(covariance) / len(original)
    matrix = (
        covariance
        + alpha * fidelity_covariance
        + penalty * noise_covariance
        + ridge * np.eye(len(original))
    )
    rhs = (
        design.T @ truth / len(design)
        + alpha * fidelity_covariance @ original
        + ridge * original
    )
    return np.linalg.solve(matrix, rhs), float(ridge)


def changed_tensor_proof(original, candidate):
    """Require exact identity of every checkpoint tensor except the Cl row."""
    changed = []
    for section in ("model_state", "absolute_state"):
        if original[section].keys() != candidate[section].keys():
            raise ValueError("Checkpoint tensor membership changed")
        for name, tensor in original[section].items():
            other = candidate[section][name]
            if torch.equal(tensor, other):
                continue
            if (
                section != "model_state"
                or name != "head.2.weight"
                or not torch.equal(tensor[0], other[0])
            ):
                raise ValueError(f"Forbidden tensor change: {section}/{name}")
            changed.append(section + "/" + name + "[Cl]")
    return {
        "changed": changed,
        "changed_parameter_count": 64 if changed else 0,
        "Cd_row_exact": True,
        "other_pair_tensors_exact": True,
        "absolute_tensors_exact": True,
        "scope": "The auxiliary readout and its CNN inputs are identical for every input, including aircraft. This repair makes no claim about aircraft auxiliary Cl noise.",
    }


def encode_requests(model, data, requests, device):
    """Encode each unique geometry/actual phase once, retaining small CPU latents."""
    cache = {}
    requests = list(dict.fromkeys(requests))
    with torch.inference_mode():
        for start in range(0, len(requests), 8):
            current = requests[start : start + 8]
            views, _, _ = sampling.batch_views(data, current, device)
            encoded = model.encode(views)
            for k, key in enumerate(current):
                cache[key] = {n: value[k].cpu() for n, value in encoded.items()}
    return cache


def pair_features(model, data, cache, pairs, device):
    values, worst = [], 0.0
    with torch.inference_mode():
        for start in range(0, len(pairs), 64):
            current = pairs[start : start + 64]
            a, b = zip(*current)
            left = {
                n: torch.stack([cache[key][n] for key in a]).to(device)
                for n in cache[a[0]]
            }
            right = {
                n: torch.stack([cache[key][n] for key in b]).to(device)
                for n in cache[b[0]]
            }
            params = data.parameters[[key[0] for key in a]].to(device)
            z, error = capture_features(model, left, right, params)
            values.append(z.cpu().numpy())
            worst = max(worst, error)
    return np.concatenate(values), worst


def training_features(model, data, sp, cases, device, root):
    """Consume only nominal training images and the 32x4 real training phases."""
    selected = [data.index[i] for i in sp["aux_train_ids"]]
    phase_ids = sp["phase_ids"]
    with np.load(Path(sp["experiment"]) / "draws.npz", allow_pickle=False) as stored:
        left, right = stored["left"][:6].ravel(), stored["right"][:6].ravel()
    real_pairs = [((int(a), None), (int(b), None)) for a, b in zip(left, right)]
    local_pairs = list(itertools.combinations(selected, 2))
    pairs = real_pairs + [((a, None), (b, None)) for a, b in local_pairs]
    for phase in phase_ids:
        for pattern in PATTERNS:
            pa = phase if pattern in ("common", "reference_shifted") else None
            pb = phase if pattern in ("common", "query_shifted") else None
            pairs.extend(((a, pa), (b, pb)) for a, b in local_pairs)
    noise_pairs = [((i, None), (i, phase)) for phase in phase_ids for i in selected]
    requests = list(dict.fromkeys(key for pair in pairs + noise_pairs for key in pair))
    train_ids = {i for i, c in enumerate(cases) if c["split"] == "train"}
    if (
        len(train_ids) != 720
        or len(selected) != 32
        or len(phase_ids) != 4
        or any(i not in train_ids for i, _ in requests)
    ):
        raise ValueError("Require only the fixed 720 train and 32x4 train phases")
    cache = encode_requests(model, data, requests, device)
    design, err = pair_features(model, data, cache, pairs, device)
    nuisance, noise_err = pair_features(model, data, cache, noise_pairs, device)
    truth = np.array(
        [
            (cases[b[0]]["targets"][1] - cases[a[0]]["targets"][1]) / sp["delta_std"][1]
            for a, b in pairs
        ]
    )
    trace = {
        "design_pairs": [
            [[cases[i]["case_id"], phase] for i, phase in pair] for pair in pairs
        ],
        "nuisance_pairs": [
            [[cases[i]["case_id"], phase] for i, phase in pair] for pair in noise_pairs
        ],
        "encoded_requests": [[cases[i]["case_id"], phase] for i, phase in requests],
        "fit_split": "train",
        "real_schedule_rows": len(real_pairs),
        "design_rows": len(design),
        "nuisance_rows": len(nuisance),
        "encoded_nominal_count": sum(phase is None for _, phase in requests),
        "encoded_actual_phase_count": sum(phase is not None for _, phase in requests),
        "max_standardized_linear_reconstruction_error": max(err, noise_err),
    }
    fidelity = design[: len(real_pairs) + len(local_pairs)]
    trace["nominal_fidelity_rows"] = len(fidelity)
    np.savez_compressed(
        root / "train-features.npz",
        design=design,
        fidelity=fidelity,
        nuisance=nuisance,
        truth=truth,
    )
    write_json(root / "train-trace.json", trace)
    return design, truth, fidelity, nuisance, trace


def split_features(
    model, data, sp, cases, identities, phases, split, device, root, name
):
    """Cache every raw phase/pair feature needed to score all Cl rows cheaply."""
    queries = sorted(c["case_id"] for c in cases if c["split"] == split)
    by_id = {c["case_id"]: c for c in cases}
    if any(by_id[i]["split"] != split for i in identities):
        raise ValueError("Phase geometry split mismatch")
    index = data.index
    nominal_ids = list(dict.fromkeys(sp["anchors"] + queries))
    requests = [(index[i], None) for i in nominal_ids] + [
        (index[i], phase) for phase in phases for i in identities
    ]
    cache = encode_requests(model, data, requests, device)
    geometry_pairs = list(itertools.combinations(identities, 2))
    groups = {
        "anchors": [
            ((index[a], None), (index[b], None)) for a in sp["anchors"] for b in queries
        ],
        "nominal": [((index[a], None), (index[b], None)) for a, b in geometry_pairs],
        "self": [
            ((index[i], None), (index[i], phase))
            for phase in phases
            for i in identities
        ],
    }
    for pattern in PATTERNS:
        groups[pattern] = [
            (
                (
                    index[a],
                    phase if pattern in ("common", "reference_shifted") else None,
                ),
                (index[b], phase if pattern in ("common", "query_shifted") else None),
            )
            for phase in phases
            for a, b in geometry_pairs
        ]
    arrays, worst = {}, 0.0
    for group, pairs in groups.items():
        arrays["z_" + group], error = pair_features(model, data, cache, pairs, device)
        worst = max(worst, error)
    arrays["truth_pairs"] = np.array(
        [
            np.asarray(by_id[b]["targets"]) - by_id[a]["targets"]
            for a, b in geometry_pairs
        ]
    )
    metadata = {
        "split": split,
        "queries": queries,
        "case_ids": identities,
        "phase_ids": phases,
        "pairs": geometry_pairs,
        "encoded_image_count": len(cache),
        "max_standardized_linear_reconstruction_error": worst,
    }
    np.savez_compressed(root / f"{name}-features.npz", **arrays)
    write_json(root / f"{name}-trace.json", metadata)
    return arrays, metadata


def score_features(arrays, metadata, weights, sp, cases):
    """Use the original pair equations and inherited physical delta metrics."""
    scale = np.asarray(sp["delta_std"])
    predictions = {
        name[2:]: np.asarray(z, np.float64) @ weights.T * scale
        for name, z in arrays.items()
        if name.startswith("z_")
    }
    anchors = predictions["anchors"].reshape(
        len(sp["anchors"]), len(metadata["queries"]), 2
    )
    anchor_predictions = {
        a: {q: value.tolist() for q, value in zip(metadata["queries"], rows)}
        for a, rows in zip(sp["anchors"], anchors)
    }
    phases, count, pair_count = (
        len(metadata["phase_ids"]),
        len(metadata["case_ids"]),
        len(metadata["pairs"]),
    )
    self_prediction = predictions["self"].reshape(phases, count, 2)
    patterns = {
        name: sampling_scores(
            arrays["truth_pairs"],
            predictions["nominal"],
            predictions[name].reshape(phases, pair_count, 2),
            self_prediction,
            scale,
        )
        for name in PATTERNS
    }
    return {
        "nominal_anchors": score_anchors(
            cases, sp["anchors"], anchor_predictions, metadata["split"], scale
        ),
        "patterns": patterns,
    }, predictions


def repair_gates(reference, candidate, plan, delta_std, cd_exact):
    """Guard each reference response; expose the old strict stability gates too."""
    inherited = candidate_gates(reference, candidate, plan, delta_std)
    phase_response = {
        name: preservation_gate(
            reference["patterns"][name]["phase_score"],
            candidate["patterns"][name]["phase_score"],
            **plan["preservation"],
        )
        for name in PATTERNS
    }
    cl_score = max(
        (
            candidate["patterns"][name]["targets"]["Cl"]["self_phase_mae"]
            + candidate["patterns"][name]["targets"]["Cl"]["common_phase_drift_mae"]
        )
        / delta_std[1]
        for name in PATTERNS
    )
    passed = (
        inherited["nominal_preserved"]
        and all(g["pass"] for g in phase_response.values())
        and cd_exact
    )
    return {
        "response_preserved": passed,
        "Cd_prediction_exact": cd_exact,
        "Cl_sampling_score": cl_score,
        "phase_response_gates_vs_epoch5": phase_response,
        "strict_inherited_gates": inherited,
    }


def save_predictions(root, name, predictions):
    np.savez_compressed(root / f"{name}-predictions.npz", **predictions)


def run(checkpoint, sampling_parent, output, device_name=None):
    """@entry Fit train-only rows, seal validation selection, then run held-out tests."""
    checkpoint, output = Path(checkpoint), Path(output)
    if output.exists():
        raise FileExistsError("Use a fresh repair output directory")
    if sha256(checkpoint) != EXPECTED_CHECKPOINT:
        raise ValueError("Retained epoch-five checkpoint checksum differs")
    sp, cases = sampling.load(sampling_parent)
    output.mkdir(parents=True)
    started = time.monotonic()
    ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = PairedDelta(**ck["model_config"])
    model.load_state_dict(ck["model_state"])
    model.eval().requires_grad_(False)
    torch.set_num_threads(2)
    device = torch.device(
        device_name or ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    model.to(device)
    grid = [
        {"alpha": a, "penalty": n}
        for a in (1, 10, 100)
        for n in (0.01, 0.1, 1, 10, 100)
    ]
    plan = {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
        "sampling_parent": str(Path(sampling_parent).resolve()),
        "seed": 42,
        "anchors": sp["anchors"],
        "grid": grid,
        "ridge_ratio": 1e-6,
        "train_nominal_geometries": 720,
        "train_actual_phase_geometries": 32,
        "train_design": "4320 inherited real train nominal pairs, all496 auxiliary train nominal pairs, 4phase x 3patterns x496 true train pairs; fidelity only on 4320+496 nominal real geometry rows; never reward old phase errors",
        "nuisance": "32 training geometries x4 actual re-rendered nominal-to-phase self pairs, target zero",
        "objective": "mean(Dw-y)^2 +alpha*mean(F(w-w0))^2, F nominal-only +penalty*mean(Nw)^2 +ridge*||w-w0||^2",
        "preservation": {
            "tolerance": 0.05,
            "min_amplitude_fraction": 0.9,
            "direction_slack": 0.03,
        },
        "stability": {"mae_over_delta_std": 0.05, "p95_over_delta_std": 0.10},
        "selection": "Nominal each-anchor/pooled/all66 and corresponding actual-phase response preservation versus epoch5; exact Cd. Minimize max-pattern normalized Cl self+drift; require strict improvement or retain epoch5. Old all-target stability gates reported separately.",
        "test": "Only after validation selection sealed: epoch5 and selected improvement, otherwise best diagnostic; main92 at+.25x plus extra24 at+.125x,+.5x,+.25y. No test rerank.",
        "source_sha256": {
            str(Path(sampling_parent).resolve() / "protocol.json"): sha256(
                Path(sampling_parent) / "protocol.json"
            ),
            str(Path(__file__).resolve()): sha256(__file__),
        },
    }
    write_json(output / "plan.json", plan)
    data = sampling.SamplingCases(sp, cases)
    design, truth, fidelity, nuisance, train_trace = training_features(
        model, data, sp, cases, device, output
    )
    weights0 = ck["model_state"]["head.2.weight"].numpy().astype(np.float64)
    candidates = [("epoch5", weights0.copy(), {})]
    for k, config in enumerate(grid):
        weights = weights0.copy()
        weights[1], ridge = fit_row(
            design,
            truth,
            fidelity,
            nuisance,
            weights0[1],
            config["alpha"],
            config["penalty"],
            plan["ridge_ratio"],
        )
        # Checkpoints deploy float32 rows, so validation must score those exact values.
        weights[1] = weights[1].astype(np.float32).astype(np.float64)
        candidates.append((f"row-{k:02d}", weights, {**config, "ridge": ridge}))
    np.savez_compressed(
        output / "candidate-rows.npz",
        names=np.array([r[0] for r in candidates]),
        weights=np.stack([r[1] for r in candidates]),
    )
    train_lock = {
        "plan_sha256": sha256(output / "plan.json"),
        "train_features_sha256": sha256(output / "train-features.npz"),
        "train_trace_sha256": sha256(output / "train-trace.json"),
        "candidate_rows_sha256": sha256(output / "candidate-rows.npz"),
        "elapsed_s": time.monotonic() - started,
        "trace": train_trace,
    }
    write_json(output / "train-lock.json", train_lock)
    print(json.dumps({"stage": "train_features_sealed", **train_lock}), flush=True)
    # No validation image has been decoded or encoded before the training lock.
    phase_plan = json.loads((Path(sp["phase_root"]) / "plan.json").read_text())
    arrays, metadata = split_features(
        model,
        data,
        sp,
        cases,
        phase_plan["selection"]["val"],
        sp["phase_ids"],
        "val",
        device,
        output,
        "validation",
    )
    entries, baseline, baseline_predictions = [], None, None
    for name, weights, config in candidates:
        metrics, predictions = score_features(arrays, metadata, weights, sp, cases)
        if baseline is None:
            baseline, baseline_predictions = metrics, predictions
        cd_exact = all(
            np.array_equal(predictions[key][:, 0], baseline_predictions[key][:, 0])
            for key in predictions
        )
        gates = repair_gates(baseline, metrics, plan, sp["delta_std"], cd_exact)
        entry = {"name": name, "config": config, "metrics": metrics, "gates": gates}
        entries.append(entry)
        save_predictions(output, "validation-" + name, predictions)
    original, attempts = entries[0], entries[1:]
    eligible = [
        r
        for r in attempts
        if r["gates"]["response_preserved"]
        and r["gates"]["Cl_sampling_score"] < original["gates"]["Cl_sampling_score"]
    ]
    best_diagnostic = min(
        attempts, key=lambda r: (r["gates"]["Cl_sampling_score"], r["name"])
    )
    selected = (
        min(eligible, key=lambda r: (r["gates"]["Cl_sampling_score"], r["name"]))
        if eligible
        else original
    )
    diagnostic = selected if eligible else best_diagnostic
    selected_weights = next(
        weights for name, weights, _ in candidates if name == selected["name"]
    )
    candidate_ck = copy.deepcopy(ck)
    candidate_ck["model_state"]["head.2.weight"][1] = torch.tensor(
        selected_weights[1], dtype=torch.float32
    )
    proof = changed_tensor_proof(ck, candidate_ck)
    candidate_ck["cl_row_repair"] = {
        "parent_sha256": sha256(checkpoint),
        "candidate": selected["name"],
        "plan_sha256": sha256(output / "plan.json"),
    }
    torch.save(candidate_ck, output / "candidate.pt")
    write_json(output / "tensor-proof.json", proof)
    write_json(
        output / "validation.json", {"baseline": original, "candidates": attempts}
    )
    selection = {
        "selected": selected["name"],
        "improved": bool(eligible),
        "baseline_retained": not eligible,
        "baseline_Cl_sampling_score": original["gates"]["Cl_sampling_score"],
        "selected_Cl_sampling_score": selected["gates"]["Cl_sampling_score"],
        "test_candidate": diagnostic["name"],
        "test_candidate_is_diagnostic": not eligible,
        "candidate_sha256": sha256(output / "candidate.pt"),
        "validation_sha256": sha256(output / "validation.json"),
        "train_lock_sha256": sha256(output / "train-lock.json"),
        "proof": proof,
    }
    write_json(output / "selection.json", selection)
    print(json.dumps({"stage": "validation_selection_sealed", **selection}), flush=True)
    test_weights = next(
        weights for name, weights, _ in candidates if name == diagnostic["name"]
    )
    comparison = {
        "selection": selection,
        "models": {"epoch5": {}, diagnostic["name"]: {}},
        "scope": "Frozen epoch5 feature/readout-row experiment; correlated Rim true deltas and actual phase re-renders. Absolute auxiliary outputs are unchanged by structure, not newly validated aircraft noise improvement.",
    }
    write_json(
        output / "test-plan.json",
        {
            "selection_sha256": sha256(output / "selection.json"),
            "models": ["epoch5", diagnostic["name"]],
            "main_case_ids": phase_plan["selection"]["all_test"],
            "extra_case_ids": phase_plan["selection"]["test"],
            "test_reranking": False,
        },
    )
    scopes = (
        ("main", phase_plan["selection"]["all_test"], [phase_id((0.25, 0))]),
        (
            "extra",
            phase_plan["selection"]["test"],
            [phase_id((0.125, 0)), phase_id((0.5, 0)), phase_id((0, 0.25))],
        ),
    )
    for scope, identities, phases in scopes:
        arrays, metadata = split_features(
            model,
            data,
            sp,
            cases,
            identities,
            phases,
            "test",
            device,
            output,
            "test-" + scope,
        )
        reference = None
        for name, weights in (("epoch5", weights0), (diagnostic["name"], test_weights)):
            metrics, predictions = score_features(arrays, metadata, weights, sp, cases)
            if reference is None:
                reference, reference_predictions = metrics, predictions
            exact = all(
                np.array_equal(predictions[key][:, 0], reference_predictions[key][:, 0])
                for key in predictions
            )
            comparison["models"][name][scope] = {
                "metrics": metrics,
                "gates": repair_gates(reference, metrics, plan, sp["delta_std"], exact),
            }
            save_predictions(output, f"test-{scope}-{name}", predictions)
    comparison["elapsed_s"] = time.monotonic() - started
    write_json(output / "comparison.json", comparison)
    write_json(
        output / "verification.json",
        {
            "retained_epoch5_unchanged": sha256(checkpoint) == EXPECTED_CHECKPOINT,
            "plan_sha256": sha256(output / "plan.json"),
            "selection_sha256": sha256(output / "selection.json"),
            "candidate_sha256": sha256(output / "candidate.pt"),
            "fitted_geometry_split": "train",
            "device": str(device),
        },
    )
    print(
        json.dumps(
            {
                "stage": "complete",
                "output": str(output.resolve()),
                "selection": selection,
                "elapsed_s": comparison["elapsed_s"],
            }
        ),
        flush=True,
    )
    return comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/aerodynamic-online-jitter-20261117/retained-online.pt"),
    )
    parser.add_argument(
        "--sampling-parent",
        type=Path,
        default=Path("outputs/vehicle-sampling-consistency-20261117"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/vehicle-cl-row-repair-20261117")
    )
    parser.add_argument("--device", choices=("mps", "cpu"))
    args = parser.parse_args()
    run(args.checkpoint, args.sampling_parent, args.output, args.device)
