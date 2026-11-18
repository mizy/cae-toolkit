"""Compare every repair and the integrated predictor after validation choices are frozen."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .detail_kernel import distances
from .end_to_end_evaluation import radiant_scores
from .head_experiment import physical
from .kernel_experiment import bootstrap
from .kernel_experiment import load as load_features
from .multiscale_readout import mixture
from .readout_experiment import absolute_scores
from .repair_experiment import load


def evaluate(experiment):
    """@entry Score unchanged historical data; never reselect after seeing test/Radiant errors."""
    if (experiment / "comparison.json").exists():
        raise FileExistsError("Repair comparison already recorded")
    p, data = load(experiment)
    for file, sha in json.loads(
        (experiment / "preserved-artifacts.json").read_text()
    ).items():
        if hashlib.sha256(Path(file).read_bytes()).hexdigest() != sha:
            raise ValueError("Preserved experiment changed")
    selection = json.loads((experiment / "neural-selection.json").read_text())
    correction_choice = json.loads(
        (experiment / "correction-selection.json").read_text()
    )
    correction_protocol = json.loads(
        (experiment / "correction-protocol.json").read_text()
    )
    for file, sha in correction_protocol["sha256"].items():
        if hashlib.sha256(Path(file).read_bytes()).hexdigest() != sha:
            raise ValueError("Correction inputs changed after fitting")
    if (
        hashlib.sha256((experiment / "model.json").read_bytes()).hexdigest()
        != correction_choice["model_manifest_sha256"]
    ):
        raise ValueError("Integrated manifest changed")
    spec = json.loads((experiment / "model.json").read_text())
    for file, sha in spec["sha256"].items():
        if hashlib.sha256(Path(file).read_bytes()).hexdigest() != sha:
            raise ValueError("Integrated component changed")
    with np.load(experiment / "neural-predictions.npz", allow_pickle=False) as arrays:
        normalized = {k: arrays[k] for k in arrays.files}
    _, _, features = load_features(Path(p["feature_experiment"]))
    base = normalized[selection["selected"]]
    with np.load(experiment / "correction.npz", allow_pickle=False) as correction:
        x = (features["pooled"].astype(float) - correction["mean"]) / correction[
            "scale"
        ]
        if len(correction["reference"]):
            kernel = mixture(distances(x, correction["reference"]))
            normalized["integrated"] = (
                base + kernel @ correction["dual"] + correction["bias"]
            ).astype(np.float32)
        else:
            normalized["integrated"] = base.copy()
    outputs = {k: physical(v, p["normalization"]) for k, v in normalized.items()}
    parent = Path(p["parent"])
    previous = json.loads((parent / "comparison.json").read_text())
    old_decision = json.loads((parent / "decision.json").read_text())
    old_multi = json.loads(
        (
            Path(old_decision["rim_detail_candidate"]).parent / "comparison.json"
        ).read_text()
    )
    for name, rows in (
        ("original-fixed", previous["models"]["baseline"]["rows"]),
        ("previous-multiscale", old_multi["rows"]),
    ):
        if [r["case_id"] for r in rows] != [c["case_id"] for c in data["cases"]]:
            raise ValueError("Prior result case order differs")
        outputs[name] = np.array([r["prediction"] for r in rows])
    external = json.loads(Path(p["external"]).read_text())
    models = {}
    for name, pred in outputs.items():
        if not np.isfinite(pred).all():
            raise ValueError("Nonfinite predictions")
        scores = {}
        for split in ("train", "val", "test"):
            mask = np.array([c["split"] == split for c in data["cases"]])
            scores[split] = absolute_scores(
                [c for c in data["cases"] if c["split"] == split], pred[mask]
            )
        models[name] = {
            "scores": scores,
            "rows": [
                {
                    "case_id": c["case_id"],
                    "source": c["source"],
                    "split": c["split"],
                    "truth": c["targets"],
                    "prediction": v.tolist(),
                }
                for c, v in zip(data["cases"], pred)
            ],
        }
        if name in (
            "prior",
            "control-raw",
            "control-ema",
            "projected-raw",
            "projected-ema",
            "integrated",
            "previous-multiscale",
        ):
            models[name]["radiant"] = radiant_scores(data["cases"], pred, external)
    test = np.array([c["split"] == "test" for c in data["cases"]])
    uncertainty = bootstrap(
        [c for c in data["cases"] if c["split"] == "test"],
        outputs["prior"][test],
        outputs["integrated"][test],
    )
    paired = {}
    for policy in ("raw", "ema"):
        paired[policy] = [
            {
                "seed": seed,
                "control": models[f"control-{policy}-{seed}"]["scores"]["test"][
                    "macro"
                ],
                "projected": models[f"projected-{policy}-{seed}"]["scores"]["test"][
                    "macro"
                ],
            }
            for seed in p["seeds"]
        ]
    comparison = {
        "neural_selection": selection,
        "correction_selection": correction_choice,
        "models": models,
        "paired_seeds": paired,
        "integrated_vs_prior": uncertainty,
        "scope": "Historical known-family development data. Radiant comparison uses identical92 Rim queries and three known training references; no externalCl or independent-family claim.",
    }
    (experiment / "comparison.json").write_text(
        json.dumps(comparison, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "base": selection["selected"],
                "selected_alpha": {
                    k: v["alpha"] for k, v in correction_choice["targets"].items()
                },
                "test": {
                    n: models[n]["scores"]["test"]["macro"]
                    for n in (
                        "prior",
                        "control-raw",
                        "control-ema",
                        "projected-raw",
                        "projected-ema",
                        "integrated",
                        "previous-multiscale",
                    )
                },
                "integrated_radiant": {
                    k: v
                    for k, v in models["integrated"]["radiant"].items()
                    if k != "by_anchor"
                },
                "uncertainty": uncertainty,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("experiment", type=Path)
    evaluate(p.parse_args().experiment)
