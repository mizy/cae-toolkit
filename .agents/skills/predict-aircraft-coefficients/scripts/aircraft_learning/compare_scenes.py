"""Compare completed, frozen A/B/C scene runs without selecting on test results."""

import argparse
import hashlib
import json
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from vehicle_cd_learning.evaluate import metrics
from vehicle_cd_learning.model import CoefficientModel

from .scene_experiment import SceneCases, verify_data

SCENES = {1: "vehicle", 2: "aircraft"}


def check_runs(experiment, protocol, data):
    """Reject every incomplete or mismatched run before any evaluation starts."""
    expected_ids = {c["case_id"] for c in data["cases"] if c["split"] == "train"}
    expected_hashes = {
        c["geometry_hash"] for c in data["cases"] if c["split"] == "train"
    }
    paired, runs = {}, []
    for seed in protocol["seeds"]:
        for mode, scene_mode in protocol["modes"].items():
            folder = experiment / f"{mode}-seed-{seed}"
            report = json.loads((folder / "training_report.json").read_text())
            proof = json.loads((folder / "sampling-proof.json").read_text())
            ck = torch.load(folder / "model.pt", map_location="cpu", weights_only=True)
            history = report["history"]
            if len(history) != protocol["epochs"] or [
                r["epoch"] for r in history
            ] != list(range(1, protocol["epochs"] + 1)):
                raise ValueError(f"Incomplete training budget: {folder}")
            if (
                ck["seed"] != seed
                or ck["epoch"] != report["best_epoch"]
                or ck["epoch"] != int(np.argmin([r["val_mse"] for r in history])) + 1
            ):
                raise ValueError(f"Checkpoint selection differs: {folder}")
            if (
                len(ck["train_ids"]) != len(expected_ids)
                or set(ck["train_ids"]) != expected_ids
                or set(ck["train_geometry_hashes"]) != expected_hashes
            ):
                raise ValueError(f"Training membership differs: {folder}")
            if (
                ck["dataset_sha256"] != protocol["dataset_sha256"]
                or ck["parameter_names"] != protocol["parameter_names"]
            ):
                raise ValueError(f"Dataset contract differs: {folder}")
            if ck["model_config"].get("scene_mode", "none") != scene_mode:
                raise ValueError(f"Unexpected model variant: {folder}")
            for key in ("parameter_mean", "parameter_std", "target_mean", "target_std"):
                if ck[key] != protocol["normalization"][key]:
                    raise ValueError(f"Normalization differs: {folder}")
            counts = proof["counts"]
            if (
                not proof["all_members_sampled"]
                or set(counts) != expected_ids
                or min(counts.values()) <= 0
                or sum(counts.values()) != protocol["epochs"] * len(expected_ids)
            ):
                raise ValueError(f"Invalid sampling proof: {folder}")
            identity = (
                ck["scene_common_initial_sha256"],
                proof["final_rng_sha256"],
                counts,
            )
            if seed in paired and paired[seed] != identity:
                raise ValueError(f"Initialization/sampling not paired for seed {seed}")
            paired[seed] = identity
            runs.append((mode, seed, folder, ck, report["parameter_count"]))
    return runs


def score(rows, prediction_key="prediction"):
    truth = np.asarray([r["truth"] for r in rows], dtype=float)
    pred = np.asarray([r[prediction_key] for r in rows], dtype=float)
    result = metrics(truth, pred, ["cd", "cl"])
    cl = np.isfinite(truth[:, 1])
    cd = np.isfinite(truth[:, 0]) & (truth[:, 0] > 1e-6)
    observed = cl & cd
    valid = observed & (pred[:, 0] > 1e-6)
    result.update(
        cases=len(rows),
        geometries=len({r["geometry_id"] for r in rows}),
        cd_mape=float(np.mean(abs(pred[cd, 0] - truth[cd, 0]) / truth[cd, 0]))
        if cd.any()
        else None,
        cl_sign_errors=int((np.sign(pred[cl, 1]) != np.sign(truth[cl, 1])).sum()),
        cl_sign_errors_abs_truth_ge_0_05=int(
            (
                (np.sign(pred[cl, 1]) != np.sign(truth[cl, 1]))
                & (abs(truth[cl, 1]) >= 0.05)
            ).sum()
        ),
        ratio_valid_count=int(valid.sum()),
        ratio_invalid_count=int(observed.sum() - valid.sum()),
        nonpositive_cd=int((pred[:, 0] <= 0).sum()),
        signed_cl_over_cd_mae=float(
            abs(
                pred[valid, 1] / pred[valid, 0] - truth[valid, 1] / truth[valid, 0]
            ).mean()
        )
        if valid.any()
        else None,
    )
    return result


def flat(scores):
    return {
        "cd_mae": scores["cd"]["mae"],
        "cl_mae": scores["cl"]["mae"],
        "cd_p95": scores["cd"]["p95_absolute_error"],
        "cl_p95": scores["cl"]["p95_absolute_error"],
        **{
            k: scores[k]
            for k in (
                "cd_mape",
                "signed_cl_over_cd_mae",
                "cl_sign_errors",
                "cl_sign_errors_abs_truth_ge_0_05",
                "ratio_valid_count",
                "ratio_invalid_count",
                "nonpositive_cd",
            )
        },
    }


def grouped_scores(rows, prediction_key="prediction"):
    groups = defaultdict(list)
    for row in rows:
        split, scene = row["split"], SCENES[row["scene"]]
        groups[f"{split}/{scene}"].append(row)
        groups[f"{split}/source/{row['source']}"].append(row)
    result = {key: flat(score(value, prediction_key)) for key, value in groups.items()}
    for split in ("val", "test"):
        for scene, scene_name in SCENES.items():
            sources = {
                r["source"] for r in rows if r["split"] == split and r["scene"] == scene
            }
            if not sources:
                continue
            values = [result[f"{split}/source/{source}"] for source in sorted(sources)]
            result[f"{split}_source_macro/{scene_name}"] = {
                key: float(np.mean([x[key] for x in values if x[key] is not None]))
                if any(x[key] is not None for x in values)
                else None
                for key in ("cd_mae", "cl_mae", "signed_cl_over_cd_mae")
            }
    return result


def measure_latency(model, sample, device):
    views, x, _, scene = sample
    views = {k: v[None].to(device) for k, v in views.items()}
    x, scene = x[None].to(device), scene[None].to(device)
    elapsed = []
    with torch.inference_mode():
        for i in range(25):
            if device == "mps":
                torch.mps.synchronize()
            started = time.perf_counter()
            model(views, x, scene)
            if device == "mps":
                torch.mps.synchronize()
            if i >= 5:
                elapsed.append(time.perf_counter() - started)
    return {
        "device": device,
        "batch_size": 1,
        "warmup": 5,
        "samples": 20,
        "median_ms": float(np.median(elapsed) * 1000),
        "p95_ms": float(np.quantile(elapsed, 0.95) * 1000),
        "scope": "Cached full views and conditions already on device; network forward only, no geometry preprocessing",
    }


def evaluate_run(ck, cases, root, device):
    model = CoefficientModel(**ck["model_config"])
    model.load_state_dict(ck["model_state"], strict=True)
    model.to(device).eval()
    dataset = SceneCases(root, cases, ck)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    predictions, unknown = [], []
    with torch.inference_mode():
        for views, x, _, scene in loader:
            views, x, scene = (
                {k: v.to(device) for k, v in views.items()},
                x.to(device),
                scene.to(device),
            )
            pred = model(views, x, scene).cpu().numpy()
            hidden = (
                model(views, x, torch.zeros_like(scene)).cpu().numpy()
                if model.scene_mode != "none"
                else pred
            )
            predictions.extend(pred * np.asarray(ck["target_std"]) + ck["target_mean"])
            unknown.extend(hidden * np.asarray(ck["target_std"]) + ck["target_mean"])
    rows = [
        {
            **{k: c[k] for k in ("case_id", "geometry_id", "source", "scene", "split")},
            "truth": c["targets"],
            "prediction": p.tolist(),
            "unknown_scene_prediction": u.tolist(),
        }
        for c, p, u in zip(cases, predictions, unknown, strict=True)
    ]
    latency = measure_latency(model, dataset[0], device)
    del model, loader, dataset
    if device == "mps":
        torch.mps.empty_cache()
    return {
        "scores": grouped_scores(rows),
        "unknown_scene_scores": grouped_scores(rows, "unknown_scene_prediction"),
        "latency": latency,
        "predictions": rows,
    }


def aggregate(runs):
    grouped = defaultdict(lambda: defaultdict(list))
    for run in runs:
        for group, scores in run["scores"].items():
            for metric, value in scores.items():
                if value is not None:
                    grouped[(run["mode"], group)][metric].append(value)
    result = defaultdict(dict)
    for (mode, group), scores in grouped.items():
        result[mode][group] = {
            metric: {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "values": values,
                "seeds": len(values),
            }
            for metric, values in scores.items()
        }
    return dict(result)


def paired_bootstrap(predictions, seeds, resamples=10000):
    """Source-stratified geometry clusters; seed-mean paired Cd/Cl errors."""
    rng, result = np.random.default_rng(20261111), {}
    modes = sorted(predictions)
    for first, second in combinations(modes, 2):
        ref = [r for r in predictions[first][seeds[0]] if r["split"] == "test"]
        ref.sort(key=lambda r: r["case_id"])
        source_results = {}
        for source in sorted({r["source"] for r in ref}):
            rows = [r for r in ref if r["source"] == source]
            ids = [r["case_id"] for r in rows]
            truth = np.asarray([r["truth"] for r in rows], dtype=float)
            errors = []
            for mode in (first, second):
                per_seed = []
                for seed in seeds:
                    mapped = {r["case_id"]: r for r in predictions[mode][seed]}
                    if any(
                        mapped[i]["truth"] != row["truth"]
                        or mapped[i]["geometry_id"] != row["geometry_id"]
                        for i, row in zip(ids, rows, strict=True)
                    ):
                        raise ValueError("Paired evaluation cases differ")
                    pred = np.asarray([mapped[i]["prediction"] for i in ids])
                    per_seed.append(abs(pred - truth))
                errors.append(np.mean(per_seed, axis=0))
            delta = errors[0] - errors[1]
            geometry = np.asarray([r["geometry_id"] for r in rows])
            sums = np.asarray(
                [np.nansum(delta[geometry == g], axis=0) for g in sorted(set(geometry))]
            )
            counts = np.asarray(
                [
                    np.isfinite(delta[geometry == g]).sum(0)
                    for g in sorted(set(geometry))
                ]
            )
            draw = rng.integers(0, len(sums), size=(resamples, len(sums)))
            denominator = counts[draw].sum(1)
            boot = np.divide(
                sums[draw].sum(1),
                denominator,
                out=np.full((resamples, 2), np.nan),
                where=denominator > 0,
            )
            point = np.divide(
                np.nansum(delta, axis=0),
                np.isfinite(delta).sum(0),
                out=np.full(2, np.nan),
                where=np.isfinite(delta).sum(0) > 0,
            )
            source_results[source] = (rows[0]["scene"], point, boot)
        result[f"{first}_vs_{second}"] = {}
        for scene, scene_name in SCENES.items():
            pieces = [v for v in source_results.values() if v[0] == scene]
            if not pieces:
                continue
            point = np.mean([v[1] for v in pieces], axis=0)
            draws = np.mean([v[2] for v in pieces], axis=0)
            result[f"{first}_vs_{second}"][scene_name] = {
                name: {
                    "improvement": float(point[i]) if np.isfinite(point[i]) else None,
                    "ci95": np.quantile(draws[:, i], [0.025, 0.975]).tolist()
                    if np.isfinite(draws[:, i]).all()
                    else None,
                }
                for i, name in enumerate(("cd", "cl"))
            }
    return {
        "definition": "Positive favors second model; equal source mean, paired geometry-cluster resampling within source; seed-mean errors; conditional on these trained models and reused source families, no training/new-family uncertainty",
        "resamples": resamples,
        "pairs": result,
    }


def write_report(experiment, comparison):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    aggregate_scores = comparison["aggregate"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    rows = []
    for i, scene in enumerate(SCENES.values()):
        for j, target in enumerate(("cd", "cl")):
            metric = f"{target}_mae"
            values = [
                aggregate_scores[m][f"test_source_macro/{scene}"][metric]
                for m in ("A", "B", "C")
            ]
            ax = axes[i, j]
            ax.bar(
                ("A", "B", "C"),
                [v["mean"] for v in values],
                yerr=[v["std"] for v in values],
                capsize=5,
                color=("#64748b", "#3b82f6", "#f97316"),
            )
            ax.set(
                title=f"{scene.title()} {target.upper()} MAE",
                ylabel="Equal-source macro MAE",
            )
            ax.grid(axis="y", alpha=0.2)
    fig.suptitle(
        "A: no scene · B: scene concatenation · C: scene FiLM\nError bars: standard deviation across training seeds"
    )
    fig.savefig(experiment / "comparison.png", dpi=160)
    plt.close(fig)
    for mode in ("A", "B", "C"):
        values = []
        for scene in SCENES.values():
            group = aggregate_scores[mode][f"test_source_macro/{scene}"]
            values.extend(
                f"{group[key]['mean']:.6f} ± {group[key]['std']:.6f}"
                for key in ("cd_mae", "cl_mae", "signed_cl_over_cd_mae")
            )
        rows.append(f"| {mode} | " + " | ".join(values) + " |")
    latency_rows = []
    for mode in ("A", "B", "C"):
        runs = [r for r in comparison.get("runs", []) if r["mode"] == mode]
        if runs:
            latency_rows.append(
                f"| {mode} | {runs[0]['parameter_count']} | {np.mean([r['latency']['median_ms'] for r in runs]):.3f} | {np.mean([r['latency']['p95_ms'] for r in runs]):.3f} |"
            )
    report = (
        """# Scene-conditioning A/B/C comparison

All nine prespecified runs completed the frozen training budget. Shared initialization, training members, normalization, sampled-case counts and sampling RNG were checked per seed before evaluation. Only validation selected checkpoints. Existing historical geometry splits are reused; this is not a new blind test or an independent family holdout.

Primary scores below give each source equal weight within its domain. Each cell is mean ± training-seed standard deviation. Pooled per-case and per-source MAE/RMSE/P95, sign errors, ratio coverage, unknown-scene diagnostics, raw predictions and pure network latency are in the JSON artifacts.

| Mode | Vehicle Cd | Vehicle Cl | Vehicle Cl/Cd | Aircraft Cd | Aircraft Cl | Aircraft Cl/Cd |
|---|---:|---:|---:|---:|---:|---:|
"""
        + "\n".join(rows)
        + """

A uses geometry and physical conditions; B adds scene embedding; C adds scene FiLM. C adds parameters as well as modulation; this three-arm experiment does not isolate capacity gains with a fourth control.

Ratio MAE includes only observed positive truth Cd and positive predicted Cd (>1e-6); invalid prediction counts are reported and cannot be interpreted as successful ratio predictions. Cd MAPE uses positive truth Cd. Bootstrap intervals resample source-stratified geometry clusters conditional on these fitted models, not new configurations or future training runs. Unknown-scene scores are a robustness diagnostic, not proof of arbitrary-object generalization. Timing excludes reading, rendering and geometry processing. No old checkpoint is replaced automatically.
"""
    )
    if latency_rows:
        report += (
            "\n| Mode | Parameters | Mean median forward ms | Mean P95 forward ms |\n|---|---:|---:|---:|\n"
            + "\n".join(latency_rows)
            + "\n"
        )
    (experiment / "README.md").write_text(report)


def main():
    """@entry Verify all frozen runs, then evaluate and summarize the three arms."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    args = parser.parse_args()
    experiment = args.experiment
    if (experiment / "comparison.json").exists():
        raise FileExistsError("Comparison already exists")
    protocol = json.loads((experiment / "protocol.json").read_text())
    root = Path(protocol["prepared"])
    if (
        hashlib.sha256((root / "dataset.json").read_bytes()).hexdigest()
        != protocol["dataset_sha256"]
    ):
        raise ValueError("Frozen dataset changed")
    for filename, expected in protocol["code_sha256"].items():
        if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen code changed: {filename}")
    data = json.loads((root / "dataset.json").read_text())
    verify_data(root, data)
    checked = check_runs(experiment, protocol, data)
    cases = [
        c
        for c in data["cases"]
        if c["split"] in ("val", "test") or c["source"] == "builtin_assumed"
    ]
    torch.set_num_threads(2)
    runs, predictions = [], defaultdict(dict)
    for mode, seed, folder, checkpoint, parameter_count in checked:
        report = evaluate_run(checkpoint, cases, root, args.device)
        report.update(
            mode=mode,
            seed=seed,
            best_epoch=checkpoint["epoch"],
            parameter_count=parameter_count,
            checkpoint_sha256=hashlib.sha256(
                (folder / "model.pt").read_bytes()
            ).hexdigest(),
        )
        predictions[mode][seed] = report["predictions"]
        (folder / "evaluation.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
        runs.append({k: v for k, v in report.items() if k != "predictions"})
        print(
            json.dumps({"evaluated": len(runs), "mode": mode, "seed": seed}), flush=True
        )
    unknown_runs = [{**r, "scores": r["unknown_scene_scores"]} for r in runs]
    result = {
        "scope": protocol["test_policy"],
        "protocol": protocol,
        "runs": runs,
        "aggregate": aggregate(runs),
        "unknown_scene_aggregate": aggregate(unknown_runs),
        "paired_bootstrap": paired_bootstrap(predictions, protocol["seeds"]),
    }
    write_report(experiment, result)
    (experiment / "comparison.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                m: {
                    k: v
                    for k, v in values.items()
                    if k.startswith("test_source_macro/")
                }
                for m, values in result["aggregate"].items()
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
