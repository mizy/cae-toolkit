"""Bound the epoch-five Cl row repair inside a hard parameter trust region.

This numerical correction reuses the previous experiment's fixed 64D features.
The historical test results have already been inspected; this is a development
check on those same held-out cases, not a new blind generalization experiment.
"""

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from . import cl_sampling_repair as repair
from .delta_experiment import sha256, write_json
from .phase_metrics import preservation_gate, sampling_scores


def fit_trust_region(
    design, truth, fidelity, nuisance, original, alpha, penalty, radius_fraction
):
    """Solve the convex train objective with ||w-w0|| <= fraction*||w0||."""
    design, truth, fidelity, nuisance, original = [
        np.asarray(a, np.float64) for a in (design, truth, fidelity, nuisance, original)
    ]
    covariance = design.T @ design / len(design)
    fidelity_covariance = fidelity.T @ fidelity / len(fidelity)
    noise_covariance = nuisance.T @ nuisance / len(nuisance)
    ridge = 1e-6 * np.trace(covariance) / len(original)
    matrix = (
        covariance
        + alpha * fidelity_covariance
        + penalty * noise_covariance
        + ridge * np.eye(len(original))
    )
    gradient = (
        design.T @ truth / len(design)
        - covariance @ original
        - penalty * noise_covariance @ original
    )
    values, vectors = np.linalg.eigh(matrix)
    coefficients = vectors.T @ gradient
    radius = radius_fraction * np.linalg.norm(original)
    if radius <= 0 or values.min() <= 0:
        raise ValueError(
            "Require positive trust radius and positive-definite ridge matrix"
        )

    def displacement(multiplier):
        return vectors @ (coefficients / (values + multiplier))

    unconstrained = displacement(0)
    multiplier = 0.0
    if np.linalg.norm(unconstrained) > radius:
        lower, upper = 0.0, max(float(np.trace(matrix) / len(original)), 1e-12)
        while np.linalg.norm(displacement(upper)) > radius:
            upper *= 2
        for _ in range(80):
            middle = (lower + upper) * 0.5
            if np.linalg.norm(displacement(middle)) > radius:
                lower = middle
            else:
                upper = middle
        multiplier = upper
    row = original + displacement(multiplier)
    # Validation consumes the deployed float32 row; leave rounding margin at the boundary.
    delta = row - original
    if np.linalg.norm(delta) >= radius * (1 - 1e-5):
        row = original + delta * (radius * (1 - 1e-5) / np.linalg.norm(delta))
    row = row.astype(np.float32).astype(np.float64)
    actual = np.linalg.norm(row - original)
    if actual > radius * (1 + 1e-6):
        raise ValueError("Float32 deployed row escaped its hard trust region")
    return row, {
        "lagrange_multiplier": multiplier,
        "base_ridge": float(ridge),
        "radius": float(radius),
        "actual_displacement": float(actual),
        "displacement_over_original_norm": float(actual / np.linalg.norm(original)),
        "weight_norm_over_original_norm": float(
            np.linalg.norm(row) / np.linalg.norm(original)
        ),
        "unconstrained_displacement_over_original_norm": float(
            np.linalg.norm(unconstrained) / np.linalg.norm(original)
        ),
    }


def per_phase_scores(arrays, metadata, predictions, delta_std):
    """Keep every phase and pattern explicit instead of averaging away failures."""
    phase_count, pair_count, case_count = (
        len(metadata["phase_ids"]),
        len(metadata["pairs"]),
        len(metadata["case_ids"]),
    )
    self_prediction = predictions["self"].reshape(phase_count, case_count, 2)
    result = {}
    for k, phase in enumerate(metadata["phase_ids"]):
        result[phase] = {}
        for pattern in repair.PATTERNS:
            phased = predictions[pattern].reshape(phase_count, pair_count, 2)[k : k + 1]
            result[phase][pattern] = sampling_scores(
                arrays["truth_pairs"],
                predictions["nominal"],
                phased,
                self_prediction[k : k + 1],
                delta_std,
            )
    return result


def phase_guards(reference, candidate, preservation, tolerance=0.05):
    """Protect each phase's Cl self/drift and its actual aerodynamic response."""
    reasons, response = [], {}
    score = 0.0
    for phase, patterns in reference.items():
        response[phase] = {}
        for pattern, baseline in patterns.items():
            current = candidate[phase][pattern]
            old_cl, cl = baseline["targets"]["Cl"], current["targets"]["Cl"]
            for metric in ("self_phase_mae", "common_phase_drift_mae"):
                if cl[metric] > old_cl[metric] * (1 + tolerance):
                    reasons.append(
                        f"{phase}/{pattern}/Cl/{metric} worsens by more than {tolerance:.0%}"
                    )
            gate = preservation_gate(
                baseline["phase_score"], current["phase_score"], **preservation
            )
            response[phase][pattern] = gate
            reasons.extend(f"{phase}/{pattern}: {reason}" for reason in gate["reasons"])
            score = max(score, cl["self_phase_mae"] + cl["common_phase_drift_mae"])
    return {
        "pass": not reasons,
        "reasons": reasons,
        "true_response_gates": response,
        "worst_phase_Cl_self_plus_drift": score,
    }


def load_features(parent, name):
    with np.load(parent / f"{name}-features.npz", allow_pickle=False) as stored:
        arrays = {key: stored[key].copy() for key in stored.files}
    metadata = json.loads((parent / f"{name}-trace.json").read_text())
    return arrays, metadata


def evaluate_rows(rows, arrays, metadata, sp, cases, plan, output, scope):
    """Evaluate fixed rows against epoch5; preserve original physical predictions."""
    entries, baseline, baseline_predictions, baseline_phases = [], None, None, None
    for name, weights, config in rows:
        metrics, predictions = repair.score_features(
            arrays, metadata, weights, sp, cases
        )
        phases = per_phase_scores(arrays, metadata, predictions, sp["delta_std"])
        if baseline is None:
            baseline, baseline_predictions, baseline_phases = (
                metrics,
                predictions,
                phases,
            )
        exact = all(
            np.array_equal(predictions[key][:, 0], baseline_predictions[key][:, 0])
            for key in predictions
        )
        gates = repair.repair_gates(baseline, metrics, plan, sp["delta_std"], exact)
        individual = phase_guards(
            baseline_phases, phases, plan["preservation"], plan["per_phase_tolerance"]
        )
        gates["per_phase"] = individual
        gates["all_repair_guards_passed"] = (
            gates["response_preserved"] and individual["pass"]
        )
        gates["worst_phase_normalized_Cl_score"] = (
            individual["worst_phase_Cl_self_plus_drift"] / sp["delta_std"][1]
        )
        entries.append(
            {
                "name": name,
                "config": config,
                "metrics": metrics,
                "per_phase_metrics": phases,
                "gates": gates,
            }
        )
        repair.save_predictions(output, scope + "-" + name, predictions)
    return entries


def run(parent, output):
    """@entry Freeze numerical correction, fit train only, seal choice, audit known test."""
    parent, output = Path(parent), Path(output)
    if output.exists():
        raise FileExistsError("Use a fresh trust-region experiment directory")
    old_plan = json.loads((parent / "plan.json").read_text())
    old_lock = json.loads((parent / "train-lock.json").read_text())
    checkpoint = Path(old_plan["checkpoint"])
    if (
        sha256(checkpoint) != repair.EXPECTED_CHECKPOINT
        or sha256(parent / "train-features.npz") != old_lock["train_features_sha256"]
    ):
        raise ValueError("Frozen epoch5 or train feature source changed")
    output.mkdir(parents=True)
    started = time.monotonic()
    grid = [
        {"radius_fraction": radius, "alpha": alpha, "penalty": penalty}
        for radius in (0.05, 0.1, 0.25)
        for alpha in (1, 10, 100)
        for penalty in (0.01, 0.1, 1, 10, 100)
    ]
    plan = {
        "parent": str(parent.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "seed": 42,
        "grid": grid,
        "reason": "The earlier train-only unconstrained row solve produced a 77.8x weight norm, revealing ill-conditioning. Add a hard parameter trust region independently of candidate test ranking.",
        "objective": "mean(Dw-y)^2+alpha*mean(Fnom(w-w0))^2+penalty*mean(Nw)^2+base_ridge*||w-w0||^2; same frozen train-only D/F/N as prior experiment",
        "constraint": "||w-w0|| <= radius_fraction*||w0||; centered ridge Lagrange multiplier solved by bisection. Deployed float32 row must obey constraint.",
        "anchors": old_plan["anchors"],
        "preservation": old_plan["preservation"],
        "stability": old_plan["stability"],
        "per_phase_tolerance": 0.05,
        "selection": "Require original nominal each-anchor/pooled/all66 preservation and corresponding real-phase true-response preservation; Cd exact. Every validation phase Cl self and each common/query/reference drift <= epoch5*1.05. Minimize worst individual phase Cl self+drift, require improvement; otherwise retain epoch5 and expose best diagnostic without adoption.",
        "test_disclosure": "Historical main and extra test outcomes have already been inspected. This is development evaluation on the same known test, not new blind evidence. Grid is fixed before reading validation features; never rerank candidates on test.",
        "source_sha256": {
            str(path.resolve()): sha256(path)
            for path in (
                Path(__file__),
                Path(repair.__file__),
                parent / "plan.json",
                parent / "train-lock.json",
                parent / "train-features.npz",
                checkpoint,
            )
        },
    }
    write_json(output / "plan.json", plan)
    ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
    original = ck["model_state"]["head.2.weight"].numpy().astype(np.float64)
    with np.load(parent / "train-features.npz", allow_pickle=False) as stored:
        design, truth, fidelity, nuisance = [
            stored[key].copy() for key in ("design", "truth", "fidelity", "nuisance")
        ]
    rows = [("epoch5", original.copy(), {})]
    for k, config in enumerate(grid):
        weights = original.copy()
        weights[1], fit = fit_trust_region(
            design,
            truth,
            fidelity,
            nuisance,
            original[1],
            config["alpha"],
            config["penalty"],
            config["radius_fraction"],
        )
        rows.append((f"trust-{k:02d}", weights, {**config, **fit}))
    np.savez_compressed(
        output / "candidate-rows.npz",
        names=np.array([r[0] for r in rows]),
        weights=np.stack([r[1] for r in rows]),
    )
    write_json(
        output / "train-lock.json",
        {
            "plan_sha256": sha256(output / "plan.json"),
            "candidate_rows_sha256": sha256(output / "candidate-rows.npz"),
            "train_source_sha256": old_lock["train_features_sha256"],
            "design_rows": len(design),
            "nominal_fidelity_rows": len(fidelity),
            "nuisance_rows": len(nuisance),
            "maximum_weight_norm_ratio": max(
                r[2].get("weight_norm_over_original_norm", 1) for r in rows
            ),
        },
    )
    # Train fitting and all 45 rows are sealed before opening validation feature arrays.
    sp, cases = repair.sampling.load(old_plan["sampling_parent"])
    arrays, metadata = load_features(parent, "validation")
    validation = evaluate_rows(
        rows, arrays, metadata, sp, cases, plan, output, "validation"
    )
    baseline, attempts = validation[0], validation[1:]
    baseline_score = baseline["gates"]["worst_phase_normalized_Cl_score"]
    eligible = [
        row
        for row in attempts
        if row["gates"]["all_repair_guards_passed"]
        and row["gates"]["worst_phase_normalized_Cl_score"] < baseline_score
    ]
    diagnostic = min(
        attempts,
        key=lambda row: (row["gates"]["worst_phase_normalized_Cl_score"], row["name"]),
    )
    chosen = (
        min(
            eligible,
            key=lambda row: (
                row["gates"]["worst_phase_normalized_Cl_score"],
                row["name"],
            ),
        )
        if eligible
        else baseline
    )
    test_candidate = chosen if eligible else diagnostic
    proofs = {}
    for file_name, entry in (("candidate", chosen), ("diagnostic", diagnostic)):
        candidate = copy.deepcopy(ck)
        weights = next(weights for name, weights, _ in rows if name == entry["name"])
        candidate["model_state"]["head.2.weight"][1] = torch.tensor(
            weights[1], dtype=torch.float32
        )
        proofs[file_name] = repair.changed_tensor_proof(ck, candidate)
        candidate["cl_trust_region"] = {
            "parent_sha256": repair.EXPECTED_CHECKPOINT,
            "name": entry["name"],
            "config": entry["config"],
        }
        torch.save(candidate, output / f"{file_name}.pt")
    write_json(output / "tensor-proof.json", proofs)
    write_json(
        output / "validation.json", {"baseline": baseline, "candidates": attempts}
    )
    selection = {
        "selected": chosen["name"],
        "baseline_retained": not bool(eligible),
        "improved": bool(eligible),
        "eligible_candidates": len(eligible),
        "baseline_score": baseline_score,
        "selected_score": chosen["gates"]["worst_phase_normalized_Cl_score"],
        "test_candidate": test_candidate["name"],
        "test_candidate_is_diagnostic": not bool(eligible),
        "best_diagnostic": diagnostic["name"],
        "candidate_sha256": sha256(output / "candidate.pt"),
        "validation_sha256": sha256(output / "validation.json"),
        "train_lock_sha256": sha256(output / "train-lock.json"),
        "test_disclosure": plan["test_disclosure"],
    }
    write_json(output / "selection.json", selection)
    print(json.dumps({"stage": "selection_sealed", **selection}), flush=True)
    write_json(
        output / "test-plan.json",
        {
            "selection_sha256": sha256(output / "selection.json"),
            "models": ["epoch5", test_candidate["name"]],
            "known_test": True,
            "test_reranking": False,
        },
    )
    test_rows = [rows[0], next(row for row in rows if row[0] == test_candidate["name"])]
    comparison = {
        "selection": selection,
        "models": {row[0]: {} for row in test_rows},
        "test_disclosure": plan["test_disclosure"],
    }
    for scope in ("main", "extra"):
        arrays, metadata = load_features(parent, "test-" + scope)
        entries = evaluate_rows(
            test_rows, arrays, metadata, sp, cases, plan, output, "test-" + scope
        )
        for entry in entries:
            comparison["models"][entry["name"]][scope] = entry
    comparison["elapsed_s"] = time.monotonic() - started
    write_json(output / "comparison.json", comparison)
    write_json(
        output / "verification.json",
        {
            "source_hashes_preserved": all(
                sha256(path) == expected
                for path, expected in plan["source_sha256"].items()
            ),
            "epoch5_unchanged": sha256(checkpoint) == repair.EXPECTED_CHECKPOINT,
            "candidate_sha256": sha256(output / "candidate.pt"),
            "Cl_only": proofs,
            "no_CNN_encoding": True,
        },
    )
    print(
        json.dumps(
            {
                "stage": "complete",
                "elapsed_s": comparison["elapsed_s"],
                "selection": selection,
            }
        ),
        flush=True,
    )
    return comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parent", type=Path, default=Path("outputs/vehicle-cl-row-repair-20261117")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/vehicle-cl-row-trust-region-20261117"),
    )
    args = parser.parse_args()
    run(args.parent, args.output)
