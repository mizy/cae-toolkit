"""Probe transferable Rim detail information in the complete frozen views."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .conditioning_experiment import load
from .model import VIEWS


def vector(root, case):
    with np.load(root / case["views"], allow_pickle=False) as arrays:
        return np.concatenate([arrays[name].ravel() for name in VIEWS])


def distances(a, b):
    return np.maximum(
        (a * a).mean(1)[:, None] + (b * b).mean(1)[None] - 2 * (a @ b.T) / a.shape[1], 0
    )


def kernel(a, b, kind, scale):
    return (
        a @ b.T / a.shape[1]
        if kind == "linear"
        else np.exp(-distances(a, b) / (2 * scale**2))
    )


def fit(experiment):
    """Fit source-specialized information probes using train/val only."""
    p, data = load(experiment)
    out = experiment / "detail-kernel"
    if out.exists():
        raise FileExistsError("Kernel probe already exists")
    out.mkdir()
    root = Path(p["prepared"])
    train, val = (
        [
            c
            for c in data["cases"]
            if c["source"] == "drivaerrim" and c["split"] == split
        ]
        for split in ("train", "val")
    )
    lo = hi = None
    for c in train:
        v = vector(root, c)
        lo, hi = (
            (v.copy(), v.copy())
            if lo is None
            else (np.minimum(lo, v), np.maximum(hi, v))
        )
    indices = np.flatnonzero(hi - lo > 1e-5)
    x = np.stack([vector(root, c)[indices] for c in train]).astype(float)
    xv = np.stack([vector(root, c)[indices] for c in val]).astype(float)
    mean, std = x.mean(0), x.std(0)
    keep = std > 1e-6
    indices, mean, std = indices[keep], mean[keep], std[keep]
    x, xv = (x[:, keep] - mean) / std, (xv[:, keep] - mean) / std
    y, yv = (np.asarray([c["targets"] for c in group], float) for group in (train, val))
    ym, ys = y.mean(0), y.std(0)
    normalized = (y - ym) / ys
    selection_std = np.asarray(p["selection_std"]["full"])
    candidates, best = [], float("inf")
    for kind, scale in (("linear", 1), ("rbf", 0.25), ("rbf", 1), ("rbf", 4)):
        gram, validation = kernel(x, x, kind, scale), kernel(xv, x, kind, scale)
        eigen, basis = np.linalg.eigh(gram)
        projected = basis.T @ normalized
        for alpha in (0.0001, 0.001, 0.01, 0.1, 1, 10):
            weights = basis @ (projected / (np.maximum(eigen, 0)[:, None] + alpha))
            pred = (validation @ weights) * ys + ym
            score = float(np.mean(((pred - yv) / selection_std) ** 2))
            row = {
                "kind": kind,
                "scale": scale,
                "alpha": alpha,
                "val_mse": score,
                "val_mae": abs(pred - yv).mean(0).tolist(),
            }
            candidates.append(row)
            if score < best:
                best, selected = score, row
                np.savez_compressed(
                    out / "model.npz",
                    train_x=x,
                    indices=indices,
                    mean=mean,
                    std=std,
                    weights=weights,
                    target_mean=ym,
                    target_std=ys,
                )
    train_distance = distances(x, x)
    np.fill_diagonal(train_distance, np.inf)
    nearest = np.sqrt(train_distance.min(1))
    report = {
        "selected": selected,
        "candidates": candidates,
        "train_ids": [c["case_id"] for c in train],
        "val_ids": [c["case_id"] for c in val],
        "variable_features": len(indices),
        "complete_input_features": len(lo),
        "train_nearest_distance_quartiles": np.quantile(
            nearest, [0.25, 0.5, 0.75]
        ).tolist(),
        "scope": "Training-only variance selection across all complete views, no manual part crop or label-based pixel selection. Source-specialized convex probe, not a generic predictor or matched neural compute budget; test deferred until all8 runs finish.",
    }
    report["model_sha256"] = hashlib.sha256(
        (out / "model.npz").read_bytes()
    ).hexdigest()
    (out / "fit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "kernel_fit_complete": True,
                "selected": selected,
                "features": len(indices),
            }
        ),
        flush=True,
    )


def evaluate_probe(experiment, test_rows):
    """Attach input-distance and kernel errors after all prespecified runs finish."""
    p, data = load(experiment)
    for run in p["runs"]:
        history = json.loads(
            (experiment / run["name"] / "training_report.json").read_text()
        )["history"]
        if len(history) != p["epochs"]:
            raise ValueError("Do not read test scores before complete training")
    out = experiment / "detail-kernel"
    report = json.loads((out / "fit.json").read_text())
    if (
        hashlib.sha256((out / "model.npz").read_bytes()).hexdigest()
        != report["model_sha256"]
    ):
        raise ValueError("Kernel model changed after validation selection")
    root = Path(p["prepared"])
    cases = [
        c for c in data["cases"] if c["source"] == "drivaerrim" and c["split"] == "test"
    ]
    with np.load(out / "model.npz", allow_pickle=False) as m:
        x = np.stack([vector(root, c)[m["indices"]] for c in cases]).astype(float)
        x = (x - m["mean"]) / m["std"]
        selected = report["selected"]
        pred = (
            kernel(x, m["train_x"], selected["kind"], selected["scale"]) @ m["weights"]
        )
        pred = pred * m["target_std"] + m["target_mean"]
        distance = distances(x, m["train_x"])
    nearest = np.argmin(distance, axis=1)
    values = np.sqrt(distance[np.arange(len(cases)), nearest])
    truth = np.asarray([c["targets"] for c in cases])
    rows = [
        {
            **{k: c[k] for k in ("case_id", "geometry_id", "source", "scene", "split")},
            "truth": c["targets"],
            "prediction": value.tolist(),
            "nearest_train_id": report["train_ids"][i],
            "geometry_distance": float(dist),
        }
        for c, value, i, dist in zip(cases, pred, nearest, values)
    ]
    base = {r["case_id"]: r for r in test_rows}
    bins = np.digitize(values, report["train_nearest_distance_quartiles"])
    coverage = []
    for group in range(4):
        mask = bins == group
        a = [
            abs(base[c["case_id"]]["prediction"][0] - c["targets"][0])
            for c, inside in zip(cases, mask)
            if inside
        ]
        coverage.append(
            {
                "distance_bin": group,
                "cases": int(mask.sum()),
                "A_cd_mae": float(np.mean(a)) if a else None,
                "kernel_cd_mae": float(abs(pred[mask, 0] - truth[mask, 0]).mean())
                if mask.any()
                else None,
            }
        )
    return {
        **report,
        "rows": rows,
        "coverage": coverage,
        "test_geometry_distance_median": float(np.median(values)),
        "test_mae": abs(pred - truth).mean(0).tolist(),
        "distance_limit": "Distance in standardized projection pixels, not independent 3D or aerodynamic distance; association is diagnostic, not coverage causality.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    args = parser.parse_args()
    fit(args.experiment)


if __name__ == "__main__":
    main()
