"""Measure current train-side feature variation and source/target gradient conflicts."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from aircraft_learning.scene_experiment import SceneCases
from torch.utils.data import default_collate

from . import dimensionless as dim
from .end_to_end_experiment import load
from .kernel_experiment import load as load_features
from .readout_experiment import PooledCases, coefficient_loss


def group(name):
    if name.startswith("head."):
        return "head"
    if ".project." in name:
        return "projection"
    if name.startswith(("image_encoder.", "bottom_encoder.")):
        return "cnn"
    return "conditions_and_fusion"


def cosine(a, b):
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denominator) if denominator > 1e-12 else None


def measure(experiment):
    """@entry Diagnose training records without optimizer updates or test selection."""
    p, data = load(experiment)
    output = experiment / "diagnosis.json"
    if output.exists():
        raise FileExistsError("Diagnosis already recorded")
    _, _, features = load_features(Path(p["parent"]))
    train = np.array([c["split"] == "train" for c in data["cases"]])
    cases = [c for c in data["cases"] if c["split"] == "train"]
    sources = sorted({c["source"] for c in cases if c["source"] != "builtin_assumed"})
    rng = np.random.default_rng(20261116)
    variation = {}
    for stage in ("pooled", "fused"):
        x = features[stage][train, :-36].astype(float)
        x = (x - x.mean(0)) / np.maximum(x.std(0), 0.05)
        centers, entries = {}, {}
        for source in sources:
            indices = np.flatnonzero([c["source"] == source for c in cases])
            pairs = rng.choice(indices, size=(512, 2), replace=True)
            d = np.sqrt(((x[pairs[:, 0]] - x[pairs[:, 1]]) ** 2).mean(1))
            centers[source] = x[indices].mean(0)
            entries[source] = {
                "n": len(indices),
                "within_pair_rms_distance_median": float(np.median(d)),
                "within_variance_trace_per_feature": float(x[indices].var(0).mean()),
            }
        between = {
            f"{a}/{b}": float(np.sqrt(((centers[a] - centers[b]) ** 2).mean()))
            for i, a in enumerate(sources)
            for b in sources[i + 1 :]
        }
        variation[stage] = {
            "by_source": entries,
            "between_source_centroid_rms_distance": between,
        }
    model, ck = dim.load_model(Path(p["checkpoint"]))
    torch.set_num_threads(2)
    values = SceneCases(Path(p["prepared"]), cases, p["normalization"])
    named = list(model.named_parameters())
    parameters = [v for _, v in named]
    std = torch.tensor(ck["target_std"], dtype=torch.float32)
    groups = sorted({group(n) for n, _ in named})
    collected = {s: {g: [] for g in groups} for s in sources}
    losses = {}
    for source in sources:
        indices = np.flatnonzero([c["source"] == source for c in cases])
        rng = np.random.default_rng(20261116)
        batches, rows = rng.choice(indices, (6, 16), replace=False), []
        for batch in batches:
            views, params, targets, _ = default_collate([values[int(i)] for i in batch])
            _, target_losses, _ = coefficient_loss(
                model(views, params), targets, std, parameters, False
            )
            gradients = [
                torch.autograd.grad(
                    loss, parameters, retain_graph=j == 0, allow_unused=True
                )
                for j, loss in enumerate(target_losses)
            ]
            per_group = {}
            for g in groups:
                vectors = [
                    np.concatenate(
                        [
                            np.zeros(v.numel(), np.float32)
                            if derivative is None
                            else derivative.detach().numpy().ravel()
                            for (name, v), derivative in zip(named, parts)
                            if group(name) == g
                        ]
                    )
                    for parts in gradients
                ]
                collected[source][g].append(vectors[0] + vectors[1])
                per_group[g] = {
                    "cd_norm": float(np.linalg.norm(vectors[0])),
                    "cl_norm": float(np.linalg.norm(vectors[1])),
                    "cd_cl_cosine": cosine(*vectors),
                }
            rows.append(
                {
                    "train_case_ids": [cases[i]["case_id"] for i in batch],
                    "cd_mae": float(target_losses[0].detach()),
                    "cl_mae": float(target_losses[1].detach()),
                    "gradients": per_group,
                }
            )
            values.cache.clear()
        losses[source] = rows
        print(
            json.dumps({"diagnosed_source": source, "batches": len(rows)}), flush=True
        )
    conflicts = {}
    for g in groups:
        means = {s: np.mean(collected[s][g], axis=0) for s in sources}
        conflicts[g] = {
            f"{a}/{b}": {
                "mean_gradient_cosine": cosine(means[a], means[b]),
                "six_paired_batch_cosines": [
                    cosine(x, y) for x, y in zip(collected[a][g], collected[b][g])
                ],
            }
            for i, a in enumerate(sources)
            for b in sources[i + 1 :]
        }
    if any(
        not torch.equal(value, model.state_dict()[name])
        for name, value in ck["model_state"].items()
    ):
        raise ValueError("Read-only diagnosis changed weights")
    result = {
        "checkpoint": p["checkpoint"],
        "checkpoint_sha256": hashlib.sha256(
            Path(p["checkpoint"]).read_bytes()
        ).hexdigest(),
        "feature_variation": variation,
        "source_batches": losses,
        "source_gradient_conflicts": conflicts,
        "scope": "Train-only local diagnostic at the retained baseline:6 batches of16 per CFD source, no optimizer step. Coordinates are representation distances, not physical contributions. Negative cosine shows local update conflict, not a unique global cause.",
    }
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


def permute(experiment):
    """Hold conditions fixed while permuting complete five-view geometries within source."""
    p, data = load(experiment)
    output = experiment / "geometry-permutation.json"
    if output.exists():
        raise FileExistsError("Permutation diagnostic already recorded")
    paths = [Path(name) for name in p["sha256"] if name.endswith("/pooled.npz")]
    if len(paths) != 1:
        raise ValueError("Expected one verified baseline CNN cache")
    with np.load(paths[0], allow_pickle=False) as cache:
        pooled = torch.from_numpy(cache["pooled"])
    indices = [i for i, c in enumerate(data["cases"]) if c["split"] == "train"]
    values = PooledCases(pooled, data["cases"], p["normalization"], indices)
    model, ck = dim.load_model(Path(p["checkpoint"]))
    torch.set_num_threads(2)
    rng = np.random.default_rng(20261116)
    results = {}
    for source in sorted({c["source"] for c in values.cases}):
        if source == "builtin_assumed":
            continue
        selected = np.flatnonzero([c["source"] == source for c in values.cases])
        scores, predictions = [], []
        for repeat in range(11):
            shuffled = selected if repeat == 0 else rng.permutation(selected)
            result = []
            with torch.inference_mode():
                for start in range(0, len(selected), 64):
                    batch = [values[int(i)] for i in shuffled[start : start + 64]]
                    views, _, _ = default_collate(batch)
                    params = default_collate(
                        [values[int(i)][1] for i in selected[start : start + 64]]
                    )
                    result.append(
                        model.head(model.encode_pooled(views, params)).numpy()
                    )
            pred = (
                np.concatenate(result) * np.asarray(ck["target_std"])
                + ck["target_mean"]
            )
            truth = np.asarray([values.cases[i]["targets"] for i in selected])
            scores.append(abs(pred - truth).mean(0))
            if repeat == 0:
                predictions = pred
        scores = np.asarray(scores)
        results[source] = {
            "train_cases": len(selected),
            "original_cd_cl_mae": scores[0].tolist(),
            "shuffled_mean_cd_cl_mae": scores[1:].mean(0).tolist(),
            "shuffled_minus_original_mae": (scores[1:].mean(0) - scores[0]).tolist(),
            "ten_shuffle_maes": scores[1:].tolist(),
            "original_prediction_std": np.std(predictions, axis=0).tolist(),
            "true_target_std": np.std(truth, axis=0).tolist(),
        }
    output.write_text(
        json.dumps(
            {
                "sources": results,
                "training_only": True,
                "condition_vectors_unchanged": True,
                "scope": "Diagnostic permutation of all complete-view pooled features within a training source; no optimization, no label or condition shuffling. Evaluates dependence in this trained model, not physical aerodynamic contribution.",
                "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("experiment", type=Path)
    p.add_argument("--permute", action="store_true")
    a = p.parse_args()
    permute(a.experiment) if a.permute else measure(a.experiment)
