"""Evaluate completed geometry experiments and expose causes, not just averages."""

import argparse
import gc
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from aircraft_learning.compare_scenes import grouped_scores
from aircraft_learning.scene_experiment import normalization, sampling_weights
from torch.utils.data import DataLoader

from .geometry_experiment import (
    check_inputs,
    datasets,
    select_cases,
    train_epoch,
)
from .geometry_report import plot, write_report
from .point_baseline import make_model
from .train import move_batch


def pairwise_ranking(rows):
    """Within-source test ordering; pairs are dependent, not independent trials."""
    result = {}
    for source in sorted({r["source"] for r in rows if r["split"] == "test"}):
        subset = [r for r in rows if r["split"] == "test" and r["source"] == source]
        truth = np.array([r["truth"][0] for r in subset])
        pred = np.array([r["prediction"][0] for r in subset])
        a, b = np.triu_indices(len(subset), 1)
        actual, predicted = truth[b] - truth[a], pred[b] - pred[a]
        result[source] = {}
        for threshold in (0, 0.001, 0.003, 0.005):
            valid = (abs(actual) >= threshold) & (abs(actual) > 1e-12)
            result[source][str(threshold)] = {
                "pairs": int(valid.sum()),
                "accuracy": float(
                    (np.sign(actual[valid]) == np.sign(predicted[valid])).mean()
                )
                if valid.any()
                else None,
                "ties": int((abs(predicted[valid]) < 1e-12).sum()),
                "delta_cd_mae": float(abs(actual[valid] - predicted[valid]).mean())
                if valid.any()
                else None,
            }
    return result


def spread_scores(rows):
    result = {}
    for split in ("train", "val", "test"):
        for source in sorted({r["source"] for r in rows if r["split"] == split}):
            subset = [r for r in rows if r["source"] == source and r["split"] == split]
            truth = np.asarray([r["truth"] for r in subset], dtype=float)
            pred = np.asarray([r["prediction"] for r in subset], dtype=float)
            result[f"{split}/{source}"] = {}
            for i, target in enumerate(("cd", "cl")):
                observed = np.isfinite(truth[:, i])
                y, p = truth[observed, i], pred[observed, i]
                sse = ((y - y.mean()) ** 2).sum() if len(y) else 0
                result[f"{split}/{source}"][target] = {
                    "truth_std": float(y.std()) if len(y) else None,
                    "prediction_std": float(p.std()) if len(y) else None,
                    "r2": float(1 - ((y - p) ** 2).sum() / sse) if sse > 0 else None,
                }
    return result


def predict_rows(model, root, cases, stats, points, mode, device):
    dataset = datasets(root, cases, stats, points, mode)
    predictions = []
    model.eval()
    with torch.inference_mode():
        for batch in DataLoader(dataset, batch_size=32):
            geometry, parameters, _ = move_batch(batch[:3], device)
            output = model(geometry, parameters).cpu().numpy()
            predictions.extend(output * stats["target_std"] + stats["target_mean"])
    rows = [
        {
            **{k: c[k] for k in ("case_id", "geometry_id", "source", "scene", "split")},
            "truth": c["targets"],
            "prediction": p.tolist(),
        }
        for c, p in zip(cases, predictions)
    ]
    del dataset
    return rows


def cached_latency(model, sample, device):
    geometry = {k: v[None].to(device) for k, v in sample[0].items()}
    parameters = sample[1][None].to(device)
    elapsed = []
    with torch.inference_mode():
        for i in range(25):
            if device.type == "mps":
                torch.mps.synchronize()
            started = time.perf_counter()
            model(geometry, parameters)
            if device.type == "mps":
                torch.mps.synchronize()
            if i >= 5:
                elapsed.append(1000 * (time.perf_counter() - started))
    return {
        "median_ms": float(np.median(elapsed)),
        "p95_ms": float(np.quantile(elapsed, 0.95)),
        "scope": "cached batch1, same machine/device; excludes loading/preparation",
    }


def constant_baselines(cases):
    train = [c for c in cases if c["split"] == "train"]
    y = np.asarray([c["targets"] for c in train], dtype=float)
    global_mean = np.nanmean(y, axis=0)
    result = {}
    for method in ("mean", "median"):
        values = {}
        for source in {c["source"] for c in train}:
            subset = np.asarray(
                [c["targets"] for c in train if c["source"] == source], float
            )
            reduce = np.mean if method == "mean" else np.median
            values[source] = [
                float(reduce(a[np.isfinite(a)]))
                if np.isfinite(a).any()
                else float(global_mean[i])
                for i, a in enumerate(subset.T)
            ]
        rows = [
            {
                **{
                    k: c[k]
                    for k in ("case_id", "geometry_id", "source", "scene", "split")
                },
                "truth": c["targets"],
                "prediction": values.get(c["source"], global_mean.tolist()),
            }
            for c in cases
        ]
        result[method] = {
            "metrics": grouped_scores(rows),
            "spread": spread_scores(rows),
            "prediction_by_source": values,
        }
    return result


def train_bias_correction(rows):
    """Diagnose calibration using train residuals; source identity is audit-only."""
    bias = {}
    for source in {r["source"] for r in rows if r["split"] == "train"}:
        subset = [r for r in rows if r["source"] == source and r["split"] == "train"]
        errors = np.asarray([r["prediction"] for r in subset]) - np.asarray(
            [r["truth"] for r in subset], dtype=float
        )
        bias[source] = [
            float(a[np.isfinite(a)].mean()) if np.isfinite(a).any() else 0
            for a in errors.T
        ]
    corrected = [
        {
            **r,
            "prediction": (
                np.asarray(r["prediction"]) - bias.get(r["source"], [0, 0])
            ).tolist(),
        }
        for r in rows
    ]
    return {
        "bias_from_train_only": bias,
        "metrics": grouped_scores(corrected),
        "scope": "Diagnostic only, requires known source; not new predictor weights",
    }


def paired_intervals(results, samples=2000):
    """Source-stratified geometry bootstrap, conditional on these fixed models."""
    baseline = {
        r["case_id"]: r for r in results["views"]["rows"] if r["split"] == "test"
    }
    result = {}
    for name in ("points", "parameters", "views-source-loss", "views-legacy"):
        other = {r["case_id"]: r for r in results[name]["rows"] if r["split"] == "test"}
        rng = np.random.default_rng(42)
        source_samples, source_means = [], []
        for source in sorted({r["source"] for r in baseline.values()}):
            ids = sorted(k for k, r in baseline.items() if r["source"] == source)
            truth = np.asarray([baseline[k]["truth"] for k in ids])
            a = np.asarray([baseline[k]["prediction"] for k in ids])
            b = np.asarray([other[k]["prediction"] for k in ids])
            difference = abs(a - truth) - abs(b - truth)
            source_means.append(difference.mean(0))
            indices = rng.integers(0, len(ids), (samples, len(ids)))
            source_samples.append(difference[indices].mean(1))
        draws, means = np.mean(source_samples, axis=0), np.mean(source_means, axis=0)
        result[name] = {
            target: {
                "mae_improvement_over_views": float(means[i]),
                "conditional_interval95": np.quantile(
                    draws[:, i], [0.025, 0.975]
                ).tolist(),
            }
            for i, target in enumerate(("cd", "cl"))
        }
    return {
        "results": result,
        "scope": "Positive means lower MAE. Conditional geometry resampling within four known sources; excludes training-seed and unseen-family uncertainty.",
    }


def small_fit(protocol, data, points, device):
    """Fit train-only cases to check whether a small subset is learnable."""
    cases = [
        c
        for c in data["cases"]
        if c["source"] == "drivaerrim" and c["split"] == "train"
    ][:32]
    root, stats = Path(protocol["prepared"]), normalization(cases)
    result = {}
    for mode in ("views", "points"):
        dataset = datasets(root, cases, stats, points, mode)
        values = [dataset[i] for i in range(len(dataset))]
        model = make_model(mode, protocol["seed"]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=protocol["learning_rate"])
        losses = []
        for _ in range(256):
            losses.append(
                train_epoch(
                    model, values, np.arange(len(values)), None, optimizer, device, 32
                )
            )
        rows = predict_rows(model, root, cases, stats, points, mode, device)
        result[mode] = {
            "train_ids": [c["case_id"] for c in cases],
            "initial_training_loss": losses[0],
            "final_training_loss": losses[-1],
            "final_eval_metrics": grouped_scores(rows),
            "normalization": stats,
        }
        del model, dataset, values
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()
    return result


def input_audit(data):
    groups = defaultdict(list)
    geometry_angles = defaultdict(set)
    for c in data["cases"]:
        groups[(c["numerical_views_sha256"], tuple(c["parameters"]))].append(c)
        geometry_angles[c["geometry_hash"]].add(c["conditions"].get("beta_deg"))
    collisions = [[c["case_id"] for c in g] for g in groups.values() if len(g) > 1]
    sources = {}
    for source in data["source_counts"]:
        y = np.asarray(
            [
                c["targets"]
                for c in data["cases"]
                if c["source"] == source and c["split"] == "train"
            ],
            float,
        )
        sources[source] = {
            target: {
                "train_std": float(a[np.isfinite(a)].std())
                if np.isfinite(a).any()
                else None
            }
            for target, a in zip(("cd", "cl"), y.T)
        }
    return {
        "identical_full_model_input_groups": collisions,
        "source_target_scales": sources,
        "same_vehicle_multi_beta_groups": sum(
            len(v) > 1 for v in geometry_angles.values()
        ),
        "limits": "No exact input collision is not proof of adequate detail resolution. No modified-mirror CFD labels. Source-heldout test also changes conditions/solver distribution.",
    }


def input_extrapolation(cases, stats, names):
    train = np.asarray([c["parameters"] for c in cases if c["split"] == "train"], float)
    test = np.asarray([c["parameters"] for c in cases if c["split"] == "test"], float)
    result = {}
    for i, name in enumerate(names):
        a, b = train[np.isfinite(train[:, i]), i], test[np.isfinite(test[:, i]), i]
        if not len(a) or not len(b):
            continue
        lo, hi = float(a.min()), float(a.max())
        result[name] = {
            "train_range": [lo, hi],
            "test_range": [float(b.min()), float(b.max())],
            "test_outside_train_range": int(((b < lo - 1e-6) | (b > hi + 1e-6)).sum()),
            "max_test_abs_standardized": float(
                np.max(
                    abs((b - stats["parameter_mean"][i]) / stats["parameter_std"][i])
                )
            ),
        }
    return result


def verify_runs(experiment, protocol, data, legacy):
    common, sampling, view_initial = {}, {}, {}
    result = []
    for run in protocol["runs"]:
        folder = experiment / run["name"]
        report = json.loads((folder / "training_report.json").read_text())
        checkpoint = torch.load(
            folder / "model.pt", map_location="cpu", weights_only=True
        )
        cases = select_cases(data, run, legacy)
        train = [c for c in cases if c["split"] == "train"]
        if len(report["history"]) != protocol["epochs"]:
            raise ValueError(f"Incomplete budget: {folder}")
        if (
            checkpoint["epoch"]
            != int(np.argmin([h["val_mse"] for h in report["history"]])) + 1
        ):
            raise ValueError("Checkpoint was not selected by validation")
        if set(checkpoint["train_ids"]) != {c["case_id"] for c in train}:
            raise ValueError("Training members differ")
        expected_sha = protocol["input_sha256"][
            str((Path(protocol["prepared"]) / "dataset.json").resolve())
        ]
        if checkpoint["dataset_sha256"] != expected_sha or checkpoint["run"] != run:
            raise ValueError("Checkpoint input contract differs")
        if (
            checkpoint["normalization"]
            != protocol["normalization"][f"{run['fold']}/{run['physics']}"]
        ):
            raise ValueError("Normalization differs")
        if (
            min(report["sampling_counts"].values()) <= 0
            or sum(report["sampling_counts"].values())
            != len(train) * protocol["epochs"]
        ):
            raise ValueError("Invalid actual sampling exposure")
        identity = (report["sampling_counts"], report["sampling_rng_sha256"])
        if run["fold"] in sampling and sampling[run["fold"]] != identity:
            raise ValueError("Matched arms sampled different cases")
        sampling[run["fold"]] = identity
        rng = torch.Generator().manual_seed(protocol["seed"] + 3)
        expected = np.zeros(len(train), dtype=int)
        for _ in range(protocol["epochs"]):
            drawn = torch.multinomial(
                sampling_weights(train), len(train), replacement=True, generator=rng
            ).numpy()
            expected += np.bincount(drawn, minlength=len(train))
        if (
            report["sampling_counts"]
            != dict(zip((c["case_id"] for c in train), expected.tolist()))
            or report["sampling_rng_sha256"]
            != hashlib.sha256(rng.get_state().numpy().tobytes()).hexdigest()
        ):
            raise ValueError("Actual sample sequence does not match frozen RNG replay")
        if run["mode"] == "views":
            if (
                run["fold"] in view_initial
                and view_initial[run["fold"]] != checkpoint["initial_model_sha256"]
            ):
                raise ValueError("Image arms did not start from identical weights")
            view_initial[run["fold"]] = checkpoint["initial_model_sha256"]
        if run["mode"] != "parameters":
            key = run["fold"]
            if key in common and common[key] != checkpoint["shared_initial_sha256"]:
                raise ValueError("Shared physical encoder/head initialization differs")
            common[key] = checkpoint["shared_initial_sha256"]
        result.append((run, checkpoint, report))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    protocol, data, legacy, points = check_inputs(args.experiment)
    torch.set_num_threads(2)
    runs = verify_runs(args.experiment, protocol, data, legacy)
    device, root = torch.device(args.device), Path(protocol["prepared"])
    results = {}
    for run, ck, report in runs:
        cases = select_cases(data, run, legacy)
        stats = ck["normalization"]
        model = make_model(run["mode"], protocol["seed"])
        model.load_state_dict(ck["model_state"], strict=True)
        model.to(device).eval()
        rows = predict_rows(model, root, cases, stats, points, run["mode"], device)
        sample = datasets(root, cases[:1], stats, points, run["mode"])[0]
        results[run["name"]] = {
            "run": run,
            "parameter_count": ck["parameter_count"],
            "best_epoch": ck["epoch"],
            "metrics": grouped_scores(rows),
            "ranking": pairwise_ranking(rows),
            "spread": spread_scores(rows),
            "latency": cached_latency(model, sample, device),
            "rows": rows,
            "train_bias_correction": train_bias_correction(rows),
            "input_extrapolation": input_extrapolation(
                cases, stats, data["parameter_names"]
            ),
        }
        print(
            json.dumps(
                {
                    "evaluated": run["name"],
                    "test": results[run["name"]]["metrics"]["test/vehicle"],
                }
            ),
            flush=True,
        )
        del model
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()
    baselines = {
        fold: constant_baselines(
            select_cases(data, {"fold": fold, "physics": "declared"}, legacy)
        )
        for fold in ("primary", "ahmed_holdout")
    }
    result = {
        "protocol": protocol,
        "input_audit": input_audit(data),
        "runs": results,
        "constant_baselines": baselines,
        "small_train_fit": small_fit(protocol, data, points, device),
        "paired_intervals": paired_intervals(results),
    }
    (args.experiment / "comparison.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    plot(args.experiment, runs, results)
    write_report(args.experiment, result)
    preserved = protocol["preserved_artifacts"]
    if not all(
        hashlib.sha256(Path(p).read_bytes()).hexdigest() == digest
        for p, digest in preserved.items()
    ):
        raise ValueError("A previous frozen artifact changed")
    (args.experiment / "final-verification.json").write_text(
        json.dumps(
            {
                "runs_complete": len(runs),
                "all_preserved_artifact_hashes_match": True,
                "comparison_sha256": hashlib.sha256(
                    (args.experiment / "comparison.json").read_bytes()
                ).hexdigest(),
                "default_weights_changed": False,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
