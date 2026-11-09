"""Compare completed paired projection-width runs on frozen train/validation data."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .data import CaseDataset, load_dataset
from .evaluate import metrics
from .predict import load_model


def evaluate_run(folder, prepared, protocol):
    model, checkpoint = load_model(folder / "model.pt")
    report = json.loads((folder / "training_report.json").read_text())
    if (
        len(report["history"]) != protocol["epochs"]
        or report["history"][-1]["epoch"] != protocol["epochs"]
    ):
        raise ValueError("Training has not completed the declared budget")
    if (
        checkpoint["epoch"] != report["best_epoch"]
        or checkpoint["seed"] != protocol["seed"]
    ):
        raise ValueError("Checkpoint selection or seed differs")
    if checkpoint["dataset_sha256"] != protocol["sources"]["dataset"]["sha256"]:
        raise ValueError("Checkpoint dataset differs")
    if (
        checkpoint["encoder_initialization"]["mode"]
        != "shared_pyramid_before_projection"
    ):
        raise ValueError("Expected matched pre-projection transfer")
    if checkpoint["augmentation"] != protocol["augmentation"]:
        raise ValueError("Augmentation differs")
    dataset = load_dataset(prepared)
    cases = [c for c in dataset["cases"] if c["split"] in ("train", "val")]
    trained = set(checkpoint["train_geometry_hashes"])
    if any(c["split"] == "val" and c["geometry_hash"] in trained for c in cases):
        raise ValueError("Validation geometry overlaps training")
    loader = CaseDataset(
        prepared,
        cases,
        checkpoint["parameter_mean"],
        checkpoint["parameter_std"],
        checkpoint["target_mean"],
        checkpoint["target_std"],
    )
    rows = []
    with torch.inference_mode():
        for case, (views, parameters, _) in zip(cases, loader):
            pred = model({k: v[None] for k, v in views.items()}, parameters[None])[
                0
            ].numpy()
            pred = (
                pred * np.asarray(checkpoint["target_std"]) + checkpoint["target_mean"]
            )
            if not np.isfinite(pred).all():
                raise ValueError("Nonfinite prediction")
            rows.append(
                {
                    "case_id": case["case_id"],
                    "split": case["split"],
                    "profile": case["condition_profile"],
                    "truth": case["targets"],
                    "prediction": [
                        float(p) if t is not None else None
                        for p, t in zip(pred, case["targets"])
                    ],
                }
            )
    return checkpoint, rows


def summarize(rows):
    truth = np.asarray([r["truth"] for r in rows], dtype=float)
    pred = np.asarray([r["prediction"] for r in rows], dtype=float)
    # metrics excludes missing targets; supply an arbitrary finite value only there.
    result = metrics(truth, np.where(np.isfinite(truth), pred, 0), ["cd", "cl"])
    valid = np.isfinite(truth[:, 1])
    if valid.any():
        t, p = truth[valid], pred[valid]
        if np.any(t[:, 0] <= 0) or np.any(p[:, 0] <= 0):
            raise ValueError("Lift/drag ratio needs positive drag in this experiment")
        result["signed_cl_over_cd_mae"] = float(
            abs(p[:, 1] / p[:, 0] - t[:, 1] / t[:, 0]).mean()
        )
        result["cl_sign_errors"] = int((np.sign(p[:, 1]) != np.sign(t[:, 1])).sum())
        result["cl_sign_errors_abs_truth_ge_0_05"] = int(
            ((np.sign(p[:, 1]) != np.sign(t[:, 1])) & (abs(t[:, 1]) >= 0.05)).sum()
        )
    return result


def paired_bootstrap(first, second):
    rows_a = [r for r in first if r["split"] == "val"]
    rows_b = [r for r in second if r["split"] == "val"]
    if any(
        any(a[k] != b[k] for k in ("case_id", "truth", "profile"))
        for a, b in zip(rows_a, rows_b, strict=True)
    ):
        raise ValueError("Validation case alignment differs")
    truth = np.asarray([r["truth"] for r in rows_a])
    errors = []
    for rows in (rows_a, rows_b):
        pred = np.asarray([r["prediction"] for r in rows])
        errors.append(
            np.column_stack(
                (
                    abs(pred - truth),
                    abs(pred[:, 1] / pred[:, 0] - truth[:, 1] / truth[:, 0]),
                )
            )
        )
    difference = errors[1] - errors[0]
    profiles = np.asarray([r["profile"] for r in rows_a])
    rng = np.random.default_rng(20261109)
    boot = np.zeros((10000, 3))
    for profile in sorted(set(profiles)):
        idx = np.flatnonzero(profiles == profile)
        boot += difference[rng.choice(idx, (10000, len(idx)), replace=True)].sum(1)
    return {
        "definition": "128 minus 32 MAE; negative favors 128; 10000 source-stratified paired case resamples; conditional on fixed models/reused validation, excluding training/family uncertainty",
        "columns": ["cd", "cl", "signed_cl_over_cd"],
        "difference": difference.mean(0).tolist(),
        "ci95": np.quantile(boot / len(rows_a), [0.025, 0.975], axis=0).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    args = parser.parse_args()
    root = args.experiment
    protocol = json.loads((root / "protocol.json").read_text())
    for item in protocol["sources"].values():
        if (
            hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest()
            != item["sha256"]
        ):
            raise ValueError("Frozen source changed")
    prepared = Path(protocol["sources"]["dataset"]["path"]).parent
    torch.set_num_threads(2)
    runs, states = {}, []
    for width in protocol["widths"]:
        folder = root / f"width-{width}-seed-{protocol['seed']}"
        checkpoint, rows = evaluate_run(folder, prepared, protocol)
        if checkpoint["model_config"]["view_feature_size"] != width:
            raise ValueError("Unexpected model width")
        states.append(checkpoint)
        groups = {
            "train": [r for r in rows if r["split"] == "train"],
            "val": [r for r in rows if r["split"] == "val"],
        }
        groups.update(
            {
                f"val_{p}": [
                    r for r in rows if r["split"] == "val" and r["profile"] == p
                ]
                for p in ("drivaerml", "ahmedml", "windsorml")
            }
        )
        groups["builtin_train"] = [r for r in rows if r["profile"] == "builtin_assumed"]
        runs[str(width)] = {
            "sha256": hashlib.sha256((folder / "model.pt").read_bytes()).hexdigest(),
            "best_epoch": checkpoint["epoch"],
            "scores": {n: summarize(rr) for n, rr in groups.items()},
            "cases": rows,
        }
    for key in (
        "common_initial_sha256",
        "train_ids",
        "train_source_counts",
        "parameter_mean",
        "parameter_std",
        "target_mean",
        "target_std",
    ):
        if states[0][key] != states[1][key]:
            raise ValueError(f"Paired initialization/data mismatch: {key}")
    result = {
        "scope": "One paired seed, same repeatedly-used development validation; clean train scores are fit only. No historical test used; no new-family accuracy claim.",
        "protocol": protocol,
        "common_initial_sha256": states[0]["common_initial_sha256"],
        "runs": runs,
        "paired_bootstrap": paired_bootstrap(runs["32"]["cases"], runs["128"]["cases"]),
    }
    output = root / "comparison.json"
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                w: {"best_epoch": r["best_epoch"], "scores": r["scores"]}
                for w, r in runs.items()
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
