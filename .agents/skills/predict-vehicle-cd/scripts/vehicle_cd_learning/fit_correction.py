"""Fit one train-only multiscale correction to the validation-selected repaired ensemble."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from aircraft_learning.scene_experiment import sampling_weights

from .corrected_coefficients import CONTRACT
from .detail_kernel import distances
from .head_experiment import physical
from .kernel_experiment import load as load_features
from .multiscale_readout import SCALES, mixture
from .readout_experiment import absolute_scores
from .repair_experiment import load


def fit(experiment):
    """@entry Share the geometry kernel; regularize the two observed output columns separately."""
    if (experiment / "correction-selection.json").exists():
        raise FileExistsError("Integrated correction already selected")
    p, data = load(experiment)
    selection = json.loads((experiment / "neural-selection.json").read_text())
    if (
        hashlib.sha256((experiment / "neural-predictions.npz").read_bytes()).hexdigest()
        != selection["prediction_cache_sha256"]
    ):
        raise ValueError("Neural predictions changed")
    fp, feature_data, features = load_features(Path(p["feature_experiment"]))
    if [c["case_id"] for c in data["cases"]] != [
        c["case_id"] for c in feature_data["cases"]
    ]:
        raise ValueError("Feature and neural case order differ")
    if tuple(p["correction"]["scales"]) != SCALES:
        raise ValueError("Kernel scales differ from the frozen protocol")
    with np.load(experiment / "neural-predictions.npz", allow_pickle=False) as arrays:
        base = arrays[selection["selected"]].astype(float)
    manifest = json.loads((experiment / "base-ensemble.json").read_text())
    sources = [
        Path(__file__),
        Path(__file__).with_name("corrected_coefficients.py"),
        Path(__file__).with_name("multiscale_readout.py"),
        experiment / "neural-selection.json",
        experiment / "neural-predictions.npz",
        *[Path(f) for f in manifest["checkpoints"]],
    ]
    (experiment / "correction-protocol.json").write_text(
        json.dumps(
            {
                "configuration": p["correction"],
                "base": selection["selected"],
                "sha256": {
                    str(f.resolve()): hashlib.sha256(f.read_bytes()).hexdigest()
                    for f in sources
                },
            },
            indent=2,
        )
        + "\n"
    )
    train = np.array([c["split"] == "train" for c in data["cases"]])
    val = np.array([c["split"] == "val" for c in data["cases"]])
    val_cases = [c for c in data["cases"] if c["split"] == "val"]
    limits = absolute_scores(val_cases, physical(base[val], p["normalization"]))
    x = features["pooled"].astype(float)
    mean, scale = x[train].mean(0), np.maximum(x[train].std(0), 0.05)
    count = x.shape[1] - 36
    scale[:count] *= np.sqrt(count / x.shape[1])
    scale[count:] *= np.sqrt(36 / x.shape[1])
    x = (x - mean) / scale
    weights = sampling_weights(
        [c for c in data["cases"] if c["split"] == "train"]
    ).numpy()
    residual = features["targets"][train].astype(float) - base[train]
    gram, kv = (
        mixture(distances(x[train], x[train])),
        mixture(distances(x[val], x[train])),
    )
    dual, bias = np.zeros((int(train.sum()), 2)), np.zeros(2)
    choices, candidates = {}, []
    for j, name in enumerate(("cd", "cl")):
        key = name + "_mae"
        source = "drivaerrim" if name == "cd" else "windsorml"
        observed = np.isfinite(residual[:, j])
        root = np.sqrt(weights[observed] / weights[observed].mean())
        center = np.average(residual[observed, j], weights=weights[observed])
        eigen, vectors = np.linalg.eigh(
            gram[np.ix_(observed, observed)] * root[:, None] * root[None]
        )
        projected = vectors.T @ ((residual[observed, j] - center) * root)
        best = limits["macro"][key]
        choices[name] = {"alpha": None, "validation": limits, "zero_correction": True}
        for alpha in p["correction"]["alphas"]:
            value = root * (vectors @ (projected / (np.maximum(eigen, 0) + alpha)))
            pred = base[val].copy()
            pred[:, j] += kv[:, observed] @ value + center
            score = absolute_scores(val_cases, physical(pred, p["normalization"]))
            eligible = (
                score["macro"][key] <= limits["macro"][key]
                and score["sources"][source][key] <= limits["sources"][source][key]
            )
            record = {
                "target": name,
                "alpha": alpha,
                "eligible": eligible,
                "validation": score,
            }
            candidates.append(record)
            if eligible and score["macro"][key] < best:
                best = score["macro"][key]
                dual[:, j] = 0
                dual[observed, j], bias[j] = value, center
                choices[name] = record
    reference = (
        x[train] if any(c["alpha"] is not None for c in choices.values()) else x[:0]
    )
    if len(reference) == 0:
        dual = dual[:0]
    path = experiment / "correction.npz"
    np.savez_compressed(
        path, mean=mean, scale=scale, reference=reference, dual=dual, bias=bias
    )
    components = [
        Path(fp["checkpoint"]),
        path,
        *[Path(f) for f in manifest["checkpoints"]],
    ]
    integrated = {
        "input_contract": CONTRACT,
        "scales": SCALES,
        "base_checkpoints": manifest["checkpoints"],
        "feature_checkpoint": fp["checkpoint"],
        "correction": str(path.resolve()),
        "sha256": {
            str(f.resolve()): hashlib.sha256(f.read_bytes()).hexdigest()
            for f in components
        },
    }
    (experiment / "model.json").write_text(json.dumps(integrated, indent=2) + "\n")
    result = {
        "base": selection["selected"],
        "targets": choices,
        "candidates": candidates,
        "test_used": False,
        "external_used": False,
        "model_manifest_sha256": hashlib.sha256(
            (experiment / "model.json").read_bytes()
        ).hexdigest(),
    }
    (experiment / "correction-selection.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "base": selection["selected"],
                "selected_alpha": {k: v["alpha"] for k, v in choices.items()},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("experiment", type=Path)
    fit(p.parse_args().experiment)
