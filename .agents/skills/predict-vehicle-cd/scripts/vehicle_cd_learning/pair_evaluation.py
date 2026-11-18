"""Evaluate fixed-anchor detail changes only after both continuation arms finish."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from aircraft_learning.scene_experiment import SceneCases

from .dimensionless import load_model
from .pair_experiment import load, predictions


def delta_metrics(truth, pred, base_truth, base_pred):
    actual, estimated = truth - base_truth, pred - base_pred
    error = abs(estimated - actual)
    direction, counts = [], []
    for column in range(2):
        active = abs(actual[:, column]) > 1e-10
        counts.append(int(active.sum()))
        direction.append(
            float(
                np.mean(
                    np.sign(estimated[active, column])
                    == np.sign(actual[active, column])
                )
            )
            if active.any()
            else None
        )
    return {
        "n": len(truth),
        "delta_mae": error.mean(0).tolist(),
        "delta_p95": np.quantile(error, 0.95, axis=0).tolist(),
        "direction_accuracy_nonzero": direction,
        "direction_denominator": counts,
        "predicted_delta_std": estimated.std(0).tolist(),
        "true_delta_std": actual.std(0).tolist(),
        "anchored_mae": abs(base_truth + estimated - truth).mean(0).tolist(),
        "absolute_mae": abs(pred - truth).mean(0).tolist(),
        "cd_delta_bins": [
            {
                "low_inclusive": low,
                "high_exclusive": high,
                "count": int(mask.sum()),
                "delta_cd_mae": float(error[mask, 0].mean()) if mask.any() else None,
            }
            for low, high in ((0, 0.001), (0.001, 0.003), (0.003, None))
            for mask in [
                (abs(actual[:, 0]) >= low)
                & ((abs(actual[:, 0]) < high) if high is not None else True)
            ]
        ],
    }


def verify_training(experiment, protocol, data):
    train = [c for c in data["cases"] if c["split"] == "train"]
    with np.load(experiment / "pairs.npz", allow_pickle=False) as schedule:
        left, right = schedule["left"], schedule["right"]
    counts, trace = np.zeros(len(train), int), hashlib.sha256()
    steps = 0
    for a, b in zip(left, right):
        for start in range(0, len(a), protocol["batch_size"]):
            ids = np.concatenate(
                (
                    a[start : start + protocol["batch_size"]],
                    b[start : start + protocol["batch_size"]],
                )
            )
            trace.update(ids.astype(np.int32).tobytes())
            counts += np.bincount(ids, minlength=len(train))
            steps += 1
    for arm in protocol["arms"]:
        report = json.loads((experiment / arm / "training_report.json").read_text())
        if (
            len(report["history"]) != protocol["epochs"]
            or report["optimizer_steps"] != steps
        ):
            raise ValueError(f"Incomplete matched training: {arm}")
        if (
            report["endpoint_trace_sha256"] != trace.hexdigest()
            or list(report["endpoint_exposures"].values()) != counts.tolist()
        ):
            raise ValueError("Different training endpoints/exposures")
        ck = torch.load(
            experiment / arm / "model.pt", weights_only=True, map_location="cpu"
        )
        selected = int(np.argmin([h["val_score"] for h in report["history"]])) + 1
        if (
            ck["epoch"] != selected
            or ck["continued_from_sha256"]
            != protocol["input_sha256"][protocol["checkpoint"]]
        ):
            raise ValueError("Initialization or validation selection mismatch")
    return {
        "optimizer_steps_per_arm": steps,
        "endpoint_exposures_per_arm": int(counts.sum()),
        "matched_endpoint_trace_sha256": trace.hexdigest(),
        "minimum_training_geometry_exposures": int(counts.min()),
        "same_initialization_and_endpoints": True,
        "no_test_selection": True,
    }


def evaluate_model(path, cases, anchors, root, protocol, device):
    model, ck = load_model(path)
    model.to(device)
    stats = protocol["normalization"]
    values = SceneCases(root, cases + anchors, stats)
    result = (
        predictions(model, values, device) * np.asarray(stats["target_std"])
        + stats["target_mean"]
    )
    pred, base = result[: len(cases)], result[len(cases) :]
    truth = np.asarray([c["targets"] for c in cases])
    rim = np.array([c["source"] == "drivaerrim" for c in cases])
    scores = {
        a["case_id"]: delta_metrics(truth[rim], pred[rim], np.asarray(a["targets"]), b)
        for a, b in zip(anchors, base)
    }
    absolute = {
        s: {
            "n": int(mask.sum()),
            "mae": abs(pred[mask] - truth[mask]).mean(0).tolist(),
            "p95": np.quantile(abs(pred[mask] - truth[mask]), 0.95, axis=0).tolist(),
        }
        for s in sorted({c["source"] for c in cases})
        for mask in [np.array([c["source"] == s for c in cases])]
    }
    sample = SceneCases(root, [cases[0]], stats)
    predictions(model, sample, device)
    durations = []
    for _ in range(10):
        if device.type == "mps":
            torch.mps.synchronize()
        start = time.perf_counter()
        predictions(model, sample, device)
        if device.type == "mps":
            torch.mps.synchronize()
        durations.append(time.perf_counter() - start)
    repeat = predictions(
        model, SceneCases(root, [anchors[0], anchors[0]], stats), device
    )
    controls = {
        "same_geometry_repeat_max_abs": float(abs(repeat[0] - repeat[1]).max()),
        "zero_change_delta": [0.0, 0.0],
        "reverse_residual": float(abs((base[1] - base[0]) + (base[0] - base[1])).max()),
        "three_geometry_loop_residual": float(
            abs((base[1] - base[0]) + (base[2] - base[1]) + (base[0] - base[2])).max()
        ),
        "limit": "Difference-of-absolute-head algebraic controls; passing is not evidence of aerodynamic accuracy.",
    }
    output = {
        "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "epoch": ck["epoch"],
        "anchor_scores": scores,
        "absolute_scores": absolute,
        "controls": controls,
        "warm_cached_inference_median_seconds": float(np.median(durations)),
        "latency_scope": "One complete cached five-view case, transfer/loader+model; excludes mesh loading/rasterization and checkpoint load.",
        "anchors": [
            {"case_id": a["case_id"], "truth": a["targets"], "prediction": b.tolist()}
            for a, b in zip(anchors, base)
        ],
        "rows": [
            {
                "case_id": c["case_id"],
                "source": c["source"],
                "truth": c["targets"],
                "prediction": v.tolist(),
            }
            for c, v in zip(cases, pred)
        ],
    }
    del model, values
    if device.type == "mps":
        torch.mps.empty_cache()
    return output


def compare(experiment, device_name="mps"):
    """@entry Evaluate every historical query; keep deterministic display cases."""
    protocol, data = load(experiment)
    proof = verify_training(experiment, protocol, data)
    cases = [c for c in data["cases"] if c["split"] == "test"]
    anchors = sorted(
        (
            c
            for c in data["cases"]
            if c["source"] == "drivaerrim" and c["split"] == "train"
        ),
        key=lambda c: c["case_id"],
    )[:3]
    if anchors[0]["case_id"] != protocol["anchor_id"]:
        raise ValueError("Anchor rule changed")
    device = torch.device(device_name)
    torch.set_num_threads(2)
    models = {}
    for name, path in (
        ("A0", Path(protocol["checkpoint"])),
        ("A1", experiment / "A1/model.pt"),
        ("A2", experiment / "A2/model.pt"),
    ):
        models[name] = evaluate_model(
            path, cases, anchors, Path(protocol["prepared"]), protocol, device
        )
        print(
            json.dumps(
                {
                    "evaluated": name,
                    "Rim": models[name]["anchor_scores"][protocol["anchor_id"]],
                }
            ),
            flush=True,
        )
    source = json.loads((Path(protocol["prepared"]) / "dataset.json").read_text())[
        "source_prepared"
    ]
    native = json.loads(Path(source).read_text())
    old = {c["case_id"]: c for c in native["cases"]}
    models["A0_original_inputs"] = evaluate_model(
        Path(protocol["checkpoint"]),
        [old[c["case_id"]] for c in cases],
        [old[c["case_id"]] for c in anchors],
        Path(source).parent,
        protocol,
        device,
    )
    truth = np.asarray(
        [r["truth"] for r in models["A1"]["rows"] if r["source"] == "drivaerrim"]
    )
    errors = []
    for name in ("A1", "A2"):
        p = np.asarray(
            [
                r["prediction"]
                for r in models[name]["rows"]
                if r["source"] == "drivaerrim"
            ]
        )
        anchor = models[name]["anchors"][0]
        errors.append(abs((p - anchor["prediction"]) - (truth - anchor["truth"])))
    rng = np.random.default_rng(42)
    draws = rng.integers(len(truth), size=(2000, len(truth)))
    gains = (errors[0][draws] - errors[1][draws]).mean(1)
    result = {
        "protocol": protocol,
        "training_proof": proof,
        "models": models,
        "A2_vs_A1_delta_mae_reduction_query_bootstrap_95": np.quantile(
            gains, [0.025, 0.975], axis=0
        ).tolist(),
        "bootstrap_limit": "Resamples92 historical queries with one fixed anchor; not CFD noise, seed or new-family uncertainty.",
        "scope": "Known-family wheel-assembly change, same fixed source conditions; source CFD, not physical wind-tunnel truth. No attention-to-force interpretation.",
    }
    (experiment / "comparison.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    compare(args.experiment, args.device)
