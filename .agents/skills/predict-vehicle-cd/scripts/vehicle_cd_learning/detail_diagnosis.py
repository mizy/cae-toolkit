"""Locate Rim predictive information before pooling, projection, fusion and the head."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from aircraft_learning.scene_experiment import SceneCases
from torch.utils.data import DataLoader

from .detail_kernel import kernel, vector
from .dimensionless import load_model, read_prepared
from .geometry_diagnostics import pairwise_ranking
from .model import VIEWS
from .pair_evaluation import delta_metrics
from .train import move_batch


def extract(root, cases, checkpoint, device):
    model, ck = load_model(checkpoint)
    model.to(device).eval()
    values = SceneCases(root, cases, ck)
    collected = {name: [] for name in ("pooled", "projected", "fused", "prediction")}
    with torch.inference_mode():
        for batch in DataLoader(values, batch_size=24, shuffle=False):
            views, parameters, _ = move_batch(batch[:3], device)
            body = model.image_encoder.pooled_features(
                torch.cat([views[n] for n in VIEWS[:4]])
            )
            parts = dict(zip(VIEWS[:4], body.chunk(4)))
            parts["bottom"] = model.bottom_encoder.pooled_features(views["bottom"])
            pooled = torch.cat([parts[n].flatten(1) for n in VIEWS], dim=1)
            projected = list(model.image_encoder.project(body).chunk(4))
            projected.append(model.bottom_encoder.project(parts["bottom"]))
            fused = model.encode_pooled(parts, parameters)
            pred = model.head(fused)
            if not collected["prediction"]:
                torch.testing.assert_close(
                    pred, model(views, parameters), rtol=1e-5, atol=1e-5
                )
            collected["pooled"].append(pooled.cpu().numpy())
            collected["projected"].append(torch.cat(projected, dim=1).cpu().numpy())
            collected["fused"].append(fused.cpu().numpy())
            collected["prediction"].append(pred.cpu().numpy())
    result = {k: np.concatenate(v) for k, v in collected.items()}
    result["prediction"] = (
        result["prediction"] * np.asarray(ck["target_std"]) + ck["target_mean"]
    )
    del model, values
    if device.type == "mps":
        torch.mps.empty_cache()
    return result


def raw_features(root, cases, train):
    lo = hi = None
    for c, selected in zip(cases, train):
        if not selected:
            continue
        x = vector(root, c)
        lo, hi = (
            (x.copy(), x.copy())
            if lo is None
            else (np.minimum(lo, x), np.maximum(hi, x))
        )
    indices = np.flatnonzero(hi - lo > 1e-5)
    return np.stack([vector(root, c)[indices] for c in cases]), indices


def fit_probe(features, targets, train, val, output):
    """Train-only moments and a fixed validation grid; test values never select a model."""
    mean, std = features[train].mean(0), features[train].std(0)
    keep = std > 1e-7
    x = (features[:, keep].astype(float) - mean[keep]) / std[keep]
    ym, ys = targets[train].mean(0), targets[train].std(0)
    y = (targets[train] - ym) / ys
    best, candidates, chosen = float("inf"), [], None
    for kind, scale in (("linear", 1.0), ("rbf", 0.25), ("rbf", 1.0), ("rbf", 4.0)):
        gram = kernel(x[train], x[train], kind, scale)
        eigen, basis = np.linalg.eigh(gram)
        projected = basis.T @ y
        validation = kernel(x[val], x[train], kind, scale)
        for alpha in (0.0001, 0.001, 0.01, 0.1, 1.0, 10.0):
            weights = basis @ (projected / (np.maximum(eigen, 0)[:, None] + alpha))
            pred = validation @ weights
            score = float(np.mean((pred - (targets[val] - ym) / ys) ** 2))
            row = {
                "kind": kind,
                "scale": scale,
                "alpha": alpha,
                "val_standardized_mse": score,
            }
            candidates.append(row)
            if score < best:
                best, chosen = score, row
                np.savez_compressed(
                    output,
                    train_x=x[train],
                    weights=weights,
                    indices=np.flatnonzero(keep),
                    mean=mean[keep],
                    std=std[keep],
                    target_mean=ym,
                    target_std=ys,
                )
    with np.load(output, allow_pickle=False) as model:
        pred = (
            kernel(x, model["train_x"], chosen["kind"], chosen["scale"])
            @ model["weights"]
        )
        pred = pred * model["target_std"] + model["target_mean"]
    return pred, {
        "selected": chosen,
        "grid": candidates,
        "input_features": features.shape[1],
        "retained_features": int(keep.sum()),
        "model_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }


def scores(cases, pred, anchor_id):
    result = {}
    truth = np.asarray([c["targets"] for c in cases])
    anchor = next(i for i, c in enumerate(cases) if c["case_id"] == anchor_id)
    for split in ("train", "val", "test"):
        mask = np.array([c["split"] == split for c in cases])
        y, p = truth[mask], pred[mask]
        a, b = np.triu_indices(len(y), 1)
        rows = [
            {**c, "truth": c["targets"], "prediction": v.tolist()}
            for c, v in zip(np.asarray(cases, dtype=object)[mask], p)
        ]
        result[split] = {
            "n": int(mask.sum()),
            "absolute_mae": abs(p - y).mean(0).tolist(),
            "predicted_std_over_true_std": (p.std(0) / y.std(0)).tolist(),
            "pair_delta_mae": abs((p[b] - p[a]) - (y[b] - y[a])).mean(0).tolist(),
            "anchor": delta_metrics(y, p, truth[anchor], pred[anchor]),
        }
        if split == "test":
            result[split]["ranking"] = pairwise_ranking(rows)["drivaerrim"]
    return result


def run(prepared, source_experiment, output, device_name="mps"):
    """@entry Freeze/extract stage features, then evaluate prespecified information probes."""
    if output.exists():
        raise FileExistsError("Use a fresh diagnostic directory")
    data = read_prepared(prepared)
    cases = [c for c in data["cases"] if c["source"] == "drivaerrim"]
    train = np.array([c["split"] == "train" for c in cases])
    val = np.array([c["split"] == "val" for c in cases])
    source = json.loads((source_experiment / "protocol.json").read_text())
    checkpoints = {
        "A0": Path(source["checkpoint"]),
        "A2": source_experiment / "A2/model.pt",
    }
    inputs = [prepared / "dataset.json", *checkpoints.values(), Path(__file__)]
    output.mkdir(parents=True)
    protocol = {
        "prepared": str(prepared.resolve()),
        "source_experiment": str(source_experiment.resolve()),
        "source_sha256": {
            str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs
        },
        "anchor_id": source["anchor_id"],
        "stages": [
            "raw",
            "A0/pooled",
            "A0/projected",
            "A0/fused",
            "A2/pooled",
            "A2/projected",
            "A2/fused",
        ],
        "selection": "Fixed24 linear/RBF candidates; train-only moments; minimum val standardizedCd/Cl MSE",
        "scope": "Frozen-feature, Rim-specialized diagnostic; not same-budget architecture ranking, new independent data or physical attribution.",
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    torch.set_num_threads(2)
    device = torch.device(device_name)
    features = {}
    native = {}
    for name, path in checkpoints.items():
        found = extract(prepared, cases, path, device)
        native[name] = scores(cases, found.pop("prediction"), source["anchor_id"])
        np.savez_compressed(output / f"{name}-features.npz", **found)
        features.update({f"{name}-{k}": v for k, v in found.items()})
        print(json.dumps({"extracted": name, "native": native[name]}), flush=True)
    features["raw"], indices = raw_features(prepared, cases, train)
    np.save(output / "raw-feature-indices.npy", indices)
    targets = np.asarray([c["targets"] for c in cases])
    probes = {}
    for name, x in features.items():
        pred, fit = fit_probe(x, targets, train, val, output / f"{name}-probe.npz")
        probes[name] = {
            "fit": fit,
            "scores": scores(cases, pred, source["anchor_id"]),
            "rows": [
                {
                    "case_id": c["case_id"],
                    "split": c["split"],
                    "truth": c["targets"],
                    "prediction": v.tolist(),
                }
                for c, v in zip(cases, pred)
            ],
        }
        print(
            json.dumps({"probe": name, "test": probes[name]["scores"]["test"]}),
            flush=True,
        )
    result = {
        "protocol": protocol,
        "native": native,
        "probes": probes,
        "raw_selected_features": len(indices),
    }
    (output / "comparison.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared", type=Path)
    parser.add_argument("source_experiment", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", default="mps")
    a = parser.parse_args()
    run(a.prepared, a.source_experiment, a.output, a.device)
