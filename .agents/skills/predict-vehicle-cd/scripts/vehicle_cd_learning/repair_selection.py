"""Freeze the neural base using validation-only comparisons after every repair run."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from aircraft_learning.scene_experiment import SceneCases

from .head_experiment import physical
from .pair_experiment import predictions
from .pooled_residual import load_model
from .readout_experiment import absolute_scores
from .repair_experiment import load


def select(experiment):
    """@entry Infer/cache predictions without scoring test; select by fixed validation gates."""
    p, data = load(experiment)
    if (experiment / "neural-selection.json").exists():
        raise FileExistsError("Neural selection already frozen")
    paths, proof = {}, {}
    for seed in p["seeds"]:
        paths[f"prior-{seed}"] = Path(p["initial_checkpoints"][str(seed)])
        draws = np.load(experiment / f"draws-{seed}.npy", allow_pickle=False)
        trace = hashlib.sha256(draws.tobytes()).hexdigest()
        counts = np.bincount(draws.ravel(), minlength=draws.shape[1]).tolist()
        for arm in p["arms"]:
            run = experiment / f"{arm}-{seed}"
            report = json.loads((run / "training_report.json").read_text())
            expected = p["epochs"] * (
                (draws.shape[1] + p["batch_size"] - 1) // p["batch_size"]
            )
            if (
                len(report["history"]) != p["epochs"]
                or report["optimizer_steps"] != expected
                or report["draw_trace_sha256"] != trace
                or report["exposures"] != counts
            ):
                raise ValueError("Incomplete or unmatched repair budget")
            proof[run.name] = {
                "steps": expected,
                "exposures": sum(counts),
                "projections": report["projection_count"],
            }
            for policy in ("raw", "ema"):
                paths[f"{arm}-{policy}-{seed}"] = run / f"{policy}.pt"
    torch.set_num_threads(2)
    device = torch.device("mps")
    values = SceneCases(Path(p["prepared"]), data["cases"], p["normalization"])
    arrays, cache, checkpoints = {}, {}, {}
    for name, path in paths.items():
        model, ck = load_model(path)
        if any(ck[k] != v for k, v in p["normalization"].items()):
            raise ValueError("Checkpoint normalization changed")
        digest = hashlib.sha256()
        for value in ck["model_state"].values():
            digest.update(value.numpy().tobytes())
        identity = digest.hexdigest()
        if identity not in cache:
            cache[identity] = predictions(model.to(device), values, device)
        arrays[name] = cache[identity]
        checkpoints[name] = {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "epoch": ck["epoch"],
        }
        del model
        torch.mps.empty_cache()
        print(json.dumps({"predicted": name, "cases": len(values)}), flush=True)
    ensemble_paths = {}
    for arm in (
        "prior",
        *[f"{a}-{policy}" for a in p["arms"] for policy in ("raw", "ema")],
    ):
        arrays[arm] = np.mean([arrays[f"{arm}-{s}"] for s in p["seeds"]], axis=0)
        ensemble_paths[arm] = [str(paths[f"{arm}-{s}"].resolve()) for s in p["seeds"]]
    val = np.array([c["split"] == "val" for c in data["cases"]])
    cases = [c for c in data["cases"] if c["split"] == "val"]
    scores = {
        n: absolute_scores(cases, physical(arrays[n][val], p["normalization"]))
        for n in ensemble_paths
    }
    limits = scores["prior"]
    eligible = []
    for name, score in scores.items():
        allowed = all(score["macro"][k] <= limits["macro"][k] for k in limits["macro"])
        allowed &= (
            score["sources"]["drivaerrim"]["cd_mae"]
            <= limits["sources"]["drivaerrim"]["cd_mae"]
        )
        allowed &= (
            score["sources"]["windsorml"]["cl_mae"]
            <= limits["sources"]["windsorml"]["cl_mae"]
        )
        if allowed:
            eligible.append(name)
    winner = min(eligible, key=lambda n: sum(scores[n]["macro"].values()))
    np.savez_compressed(experiment / "neural-predictions.npz", **arrays)
    result = {
        "selected": winner,
        "validation": scores,
        "eligible": eligible,
        "checkpoints": checkpoints,
        "proof": proof,
        "prediction_cache_sha256": hashlib.sha256(
            (experiment / "neural-predictions.npz").read_bytes()
        ).hexdigest(),
        "test_scored": False,
        "radiant_scored": False,
    }
    (experiment / "neural-selection.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    (experiment / "base-ensemble.json").write_text(
        json.dumps(
            {"checkpoints": ensemble_paths[winner], "selection": winner}, indent=2
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "selected": winner,
                "validation": {n: s["macro"] for n, s in scores.items()},
                "eligible": eligible,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("experiment", type=Path)
    select(p.parse_args().experiment)
