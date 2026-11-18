"""Choose across the paired neural and fixed multiscale experiments using validation only."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from aircraft_learning.scene_experiment import SceneCases

from .end_to_end_experiment import load
from .head_experiment import physical
from .pair_experiment import predictions
from .pooled_residual import load_model
from .readout_experiment import absolute_scores


def select(experiment, multiscale):
    """@entry Freeze one primary candidate before either experiment reads test errors."""
    output = experiment / "accuracy-selection.json"
    if output.exists():
        raise FileExistsError("Overall validation selection already frozen")
    if (experiment / "comparison.json").exists() or (
        multiscale / "comparison.json"
    ).exists():
        raise ValueError("Overall selection must precede current test evaluation")
    p, data = load(experiment)
    torch.set_num_threads(2)
    cases = [c for c in data["cases"] if c["split"] == "val"]
    values = SceneCases(Path(p["prepared"]), cases, p["normalization"])
    choices, checkpoints = (
        {"baseline": p["baseline_validation"]},
        {"baseline": [p["checkpoint"]]},
    )
    for arm in p["arms"]:
        outputs, paths = [], []
        for seed in p["seeds"]:
            run = experiment / f"{arm}-{seed}"
            report = json.loads((run / "training_report.json").read_text())
            if len(report["history"]) != p["epochs"]:
                raise ValueError("Wait for all six training runs")
            path = run / "model.pt"
            model, _ = load_model(path)
            outputs.append(
                physical(
                    predictions(model, values, torch.device("cpu")), p["normalization"]
                )
            )
            paths.append(str(path.resolve()))
        name = arm + "-ensemble"
        choices[name] = absolute_scores(cases, np.mean(outputs, axis=0))
        checkpoints[name] = paths
    kernel = json.loads((multiscale / "selection.json").read_text())
    if (
        hashlib.sha256((multiscale / "model.pt").read_bytes()).hexdigest()
        != kernel["model_sha256"]
    ):
        raise ValueError("Multiscale selection changed")
    choices["multiscale"] = (
        kernel["selected"]["validation"]
        if kernel["selected"]
        else p["baseline_validation"]
    )
    checkpoints["multiscale"] = [str((multiscale / "model.pt").resolve())]
    limits = p["baseline_validation"]["macro"]
    eligible = [
        n
        for n, scores in choices.items()
        if n == "baseline" or all(scores["macro"][k] <= limits[k] for k in limits)
    ]
    winner = min(eligible, key=lambda n: sum(choices[n]["macro"].values()))
    result = {
        "selected": winner,
        "validation": choices,
        "checkpoints": checkpoints,
        "checkpoint_sha256": {
            f: hashlib.sha256(Path(f).read_bytes()).hexdigest()
            for paths in checkpoints.values()
            for f in paths
        },
        "test_errors_used": False,
        "radiant_errors_used": False,
        "selection_rule": "Both validation source-macro absolute MAEs no worse than frozen baseline, then minimum Cd MAE+Cl MAE; no test/reference-specific tuning",
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "selected": winner,
                "validation": {n: s["macro"] for n, s in choices.items()},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("experiment", type=Path)
    p.add_argument("multiscale", type=Path)
    a = p.parse_args()
    select(a.experiment, a.multiscale)
