"""Evaluate held-out coefficients, family errors, paired changes, and mean baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from .data import CaseDataset, load_dataset
from .predict import load_model, predict


def metrics(truth, prediction, names):
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if not np.isfinite(prediction).all():
        raise ValueError("Predictions must be finite even for missing target labels")
    error = prediction - truth
    result = {}
    for index, name in enumerate(names):
        observed = error[np.isfinite(truth[:, index]), index]
        count = len(observed)
        result[name] = {
            "observed_count": count,
            "mae": float(np.abs(observed).mean()) if count else None,
            "rmse": float(np.sqrt((observed**2).mean())) if count else None,
            "p95_absolute_error": float(np.quantile(np.abs(observed), 0.95))
            if count
            else None,
        }
    return result


def save_scatter(path, truth, prediction, names):
    image = Image.new("RGB", (900, 460), "white")
    draw = ImageDraw.Draw(image)
    for index, name in enumerate(names):
        observed = np.isfinite(truth[:, index])
        x, y = truth[observed, index], prediction[observed, index]
        if not observed.any():
            draw.text(
                (60 + 450 * index, 45), f"{name}: no observed labels", fill="black"
            )
            continue
        lo, hi = float(min(x.min(), y.min())), float(max(x.max(), y.max()))
        span = max(hi - lo, 1e-8)
        left, top, size = 60 + 450 * index, 45, 330
        draw.text((left, 15), f"{name}: prediction (y) vs truth (x)", fill="black")
        draw.rectangle((left, top, left + size, top + size), outline="black")
        draw.line((left, top + size, left + size, top), fill="gray")
        for a, b in zip(x, y):
            px, py = left + (a - lo) / span * size, top + size - (b - lo) / span * size
            draw.ellipse((px - 3, py - 3, px + 3, py + 3), fill="blue")
        draw.text((left, top + size + 12), f"range [{lo:.5g}, {hi:.5g}]", fill="black")
    image.save(path)


def evaluate(
    checkpoint_path,
    prepared_dir,
    output_dir,
    split="test",
    pair_tolerance=0.0,
    diagnostic=False,
    condition_profile=None,
):
    dataset = load_dataset(prepared_dir)
    model, checkpoint = load_model(checkpoint_path)
    fingerprint = hashlib.sha256(
        (prepared_dir / "dataset.json").read_bytes()
    ).hexdigest()
    if fingerprint != checkpoint["dataset_sha256"]:
        raise ValueError(
            "Dataset differs from checkpoint; use the original frozen split"
        )
    if condition_profile is not None and condition_profile not in dataset.get(
        "condition_profiles", {}
    ):
        raise ValueError(f"Unknown condition profile: {condition_profile}")
    cases = [
        case
        for case in dataset["cases"]
        if case["split"] == split
        and (
            condition_profile is None
            or case.get("condition_profile") == condition_profile
        )
    ]
    if not cases:
        raise ValueError(f"No {split} cases")
    if split != "train":
        trained_hashes = set(checkpoint["train_geometry_hashes"])
        if any(case["geometry_hash"] in trained_hashes for case in cases):
            raise ValueError("Evaluation geometry overlaps training")
    truth = np.asarray([case["targets"] for case in cases], dtype=np.float64)
    predictions, latencies, default_rejections = [], [], []
    if diagnostic:
        evaluation_set = CaseDataset(
            prepared_dir,
            cases,
            checkpoint["parameter_mean"],
            checkpoint["parameter_std"],
            checkpoint["target_mean"],
            checkpoint["target_std"],
            model.variant,
        )
        target_mean = np.asarray(checkpoint["target_mean"])
        target_std = np.asarray(checkpoint["target_std"])
        for index, case in enumerate(cases):
            views, parameters, _ = evaluation_set[index]
            arrays = {name: value.numpy() for name, value in views.items()}
            try:
                predict(model, checkpoint, arrays, case["parameters"])
                default_rejections.append(None)
            except ValueError as error:
                default_rejections.append(str(error))

            start = time.perf_counter()
            with torch.inference_mode():
                standardized = model(
                    {name: value.unsqueeze(0) for name, value in views.items()},
                    parameters.unsqueeze(0),
                ).numpy()[0]
            latencies.append(time.perf_counter() - start)
            if not np.isfinite(standardized).all():
                raise ValueError(
                    f"Nonfinite raw model output for diagnostic case {case['case_id']}"
                )
            prediction = standardized * target_std + target_mean
            if not np.isfinite(prediction).all():
                raise ValueError(
                    f"Nonfinite raw coefficient for diagnostic case {case['case_id']}"
                )
            predictions.append(prediction)
    else:
        for case in cases:
            with np.load(prepared_dir / case["views"], allow_pickle=False) as archive:
                views = {key: archive[key] for key in archive.files}
            predict(
                model,
                checkpoint,
                views,
                case["parameters"],
                allow_geometry_extrapolation=True,
            )
            start = time.perf_counter()
            predictions.append(
                predict(
                    model,
                    checkpoint,
                    views,
                    case["parameters"],
                    allow_geometry_extrapolation=True,
                )
            )
            latencies.append(time.perf_counter() - start)
    prediction = np.asarray(predictions)
    names = checkpoint["target_names"]
    per_family = {}
    for family in sorted({case["family_id"] for case in cases}):
        indices = [i for i, case in enumerate(cases) if case["family_id"] == family]
        per_family[family] = metrics(truth[indices], prediction[indices], names)
    train_families = {
        case["family_id"] for case in dataset["cases"] if case["split"] == "train"
    }
    seen_family_indices = [
        i for i, case in enumerate(cases) if case["family_id"] in train_families
    ]
    unseen_family_indices = [
        i for i, case in enumerate(cases) if case["family_id"] not in train_families
    ]
    split_policy = dataset.get("split_policy", "family_disjoint")
    if split == "train":
        family_evaluation_scope = "training cases are seen families"
    elif split_policy == "geometry_disjoint":
        family_evaluation_scope = (
            "seen training families measure within-family geometry interpolation; "
            "unseen training families measure unseen-family generalization"
        )
    else:
        family_evaluation_scope = "held-out families are unseen by construction"
    per_condition = {}
    for condition in sorted(
        {
            (case.get("condition_profile", ""), case["reynolds"], case["yaw_deg"])
            for case in cases
        }
    ):
        indices = [
            i
            for i, case in enumerate(cases)
            if (case.get("condition_profile", ""), case["reynolds"], case["yaw_deg"])
            == condition
        ]
        prefix = f"{condition[0]}: " if condition[0] else ""
        label = f"{prefix}Re={condition[1]:.17g}, yaw={condition[2]:.17g}"
        per_condition[label] = metrics(truth[indices], prediction[indices], names)
    pairs = []
    for a, b in combinations(range(len(cases)), 2):
        first, second = cases[a], cases[b]
        if first.get("pair_id") and first.get("pair_id") == second.get("pair_id"):
            if first.get("condition_profile") != second.get("condition_profile"):
                raise ValueError("pair_id cannot cross condition profiles")
            # Body lengths derived from float32 STL coordinates can differ by
            # rounding even when the source CFD conditions are identical.
            if first["yaw_deg"] != second["yaw_deg"] or not math.isclose(
                first["reynolds"], second["reynolds"], rel_tol=1e-7
            ):
                raise ValueError(
                    "pair_id must identify designs under the same conditions"
                )
            pairs.append((truth[b] - truth[a], prediction[b] - prediction[a]))
    pair_report = {}
    if pairs:
        actual, estimated = (
            np.asarray([p[0] for p in pairs]),
            np.asarray([p[1] for p in pairs]),
        )
        pair_report = metrics(actual, estimated, names)
        for i, name in enumerate(names):
            valid = np.abs(actual[:, i]) > pair_tolerance
            pair_report[name]["direction_count"] = int(valid.sum())
            pair_report[name]["direction_accuracy"] = (
                float(
                    (np.sign(actual[valid, i]) == np.sign(estimated[valid, i])).mean()
                )
                if valid.any()
                else None
            )
    case_results = []
    for i, case in enumerate(cases):
        ranges = (
            checkpoint["condition_profiles"][case["condition_profile"]][
                "parameter_ranges"
            ]
            if "condition_profiles" in checkpoint
            else checkpoint["parameter_ranges"]
        )
        case_results.append(
            {
                "case_id": case["case_id"],
                "family_id": case["family_id"],
                "seen_training_family": case["family_id"] in train_families,
                "truth": [
                    float(value) if np.isfinite(value) else None for value in truth[i]
                ],
                "prediction": prediction[i].tolist(),
                "observed_targets": [
                    name for name, value in zip(names, truth[i]) if np.isfinite(value)
                ],
                "views": case["views"],
                **(
                    {"condition_profile": case["condition_profile"]}
                    if "condition_profile" in case
                    else {}
                ),
                "geometry_extrapolation": {
                    name: value
                    for name, value in zip(
                        checkpoint["parameter_names"][:2], case["parameters"][:2]
                    )
                    if name in ranges
                    and not ranges[name][0] - 1e-6 <= value <= ranges[name][1] + 1e-6
                },
            }
        )
    if diagnostic:
        for case_result, rejection in zip(case_results, default_rejections):
            case_result["default_rejection"] = rejection
    report = {
        "data_kind": dataset["data_kind"],
        "split": split,
        **(
            {"condition_profile": condition_profile}
            if condition_profile is not None
            else {}
        ),
        "split_policy": split_policy,
        "family_evaluation_scope": family_evaluation_scope,
        "seen_training_family_count": len(
            {cases[i]["family_id"] for i in seen_family_indices}
        ),
        "unseen_training_family_count": len(
            {cases[i]["family_id"] for i in unseen_family_indices}
        ),
        "seen_training_family_metrics": metrics(
            truth[seen_family_indices], prediction[seen_family_indices], names
        )
        if seen_family_indices
        else {},
        "unseen_training_family_metrics": metrics(
            truth[unseen_family_indices], prediction[unseen_family_indices], names
        )
        if unseen_family_indices
        else {},
        "count": len(cases),
        "metrics": metrics(truth, prediction, names),
        "mean_baseline": metrics(
            truth, np.tile(checkpoint["target_mean"], (len(cases), 1)), names
        ),
        "per_family": per_family,
        "per_profile": {
            profile: metrics(
                truth[
                    [
                        i
                        for i, c in enumerate(cases)
                        if c.get("condition_profile") == profile
                    ]
                ],
                prediction[
                    [
                        i
                        for i, c in enumerate(cases)
                        if c.get("condition_profile") == profile
                    ]
                ],
                names,
            )
            for profile in dataset.get("condition_profiles", {})
            if any(c.get("condition_profile") == profile for c in cases)
        },
        "per_condition": per_condition,
        "paired_changes": pair_report,
        "pair_tolerance": pair_tolerance,
        "applicability": "training fit only; not independent validation"
        if split == "train"
        else "raw model diagnostics include cases rejected by default inference"
        if diagnostic
        else "exploratory only"
        if dataset["data_kind"] in ("synthetic", "reference_assumed")
        else "heldout labeled cases",
        "prediction_mode": "raw_diagnostic" if diagnostic else "guarded_evaluation",
        "inference_timing_scope": "validated CaseDataset tensors through raw model forward"
        if diagnostic
        else "prediction checks, normalization, tensor conversion and model",
        "median_inference_seconds": float(np.median(latencies)),
        "cases": case_results,
    }
    if diagnostic:
        report["default_rejection_count"] = sum(
            rejection is not None for rejection in default_rejections
        )
    report["geometry_extrapolation_count"] = sum(
        bool(case["geometry_extrapolation"]) for case in report["cases"]
    )
    worst = np.argsort(
        np.nanmax(
            np.abs(prediction - truth) / np.maximum(checkpoint["target_std"], 1e-8),
            axis=1,
        )
    )[::-1][:5]
    report["worst_cases"] = [report["cases"][int(i)] for i in worst]
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "evaluation.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    save_scatter(output_dir / "scatter.png", truth, prediction, names)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("prepared_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--pair-tolerance", type=float, default=0.0)
    parser.add_argument(
        "--condition-profile",
        help="Evaluate one declared profile without changing the frozen dataset",
    )
    parser.add_argument(
        "--diagnostic",
        action="store_true",
        help="Report raw model outputs for cases rejected by default inference checks",
    )
    args = parser.parse_args()
    if args.pair_tolerance < 0:
        parser.error("pair-tolerance must be nonnegative")
    torch.set_num_threads(2)
    report = evaluate(
        args.checkpoint,
        args.prepared_dir,
        args.output_dir,
        args.split,
        args.pair_tolerance,
        args.diagnostic,
        args.condition_profile,
    )
    print(
        json.dumps({"metrics": report["metrics"], "count": report["count"]}, indent=2)
    )


if __name__ == "__main__":
    main()
