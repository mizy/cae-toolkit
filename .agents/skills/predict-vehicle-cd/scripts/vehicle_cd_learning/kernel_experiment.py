"""Validate compressed versus pre-compression readout against one frozen candidate."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from aircraft_learning.scene_experiment import sampling_weights
from torch.utils.data import DataLoader

from . import dimensionless as dim
from .detail_kernel import distances
from .head_experiment import physical
from .kernel_readout import CONTRACT, initialize, load_model
from .readout_experiment import PooledCases, absolute_scores
from .readout_experiment import load as load_parent

STAGES = ("fused", "pooled")


def prepare(parent, checkpoint, output):
    """@entry Freeze one baseline and the validation protocol before fitting."""
    if output.exists():
        raise FileExistsError("Use a new kernel experiment directory")
    p, data, pooled = load_parent(parent)
    _, ck = dim.load_model(checkpoint)
    original = torch.load(p["checkpoint"], map_location="cpu", weights_only=True)
    for key, value in ck["model_state"].items():
        if (
            key.startswith(("image_encoder.", "bottom_encoder."))
            and ".project." not in key
            and not torch.equal(value, original["model_state"][key])
        ):
            raise ValueError("Cached CNN features belong to different weights")
    stats = {k: ck[k] for k in p["normalization"]}
    if stats != p["normalization"]:
        raise ValueError("Baseline normalization differs from frozen cache")
    if set(ck["train_ids"]) != {
        c["case_id"] for c in data["cases"] if c["split"] == "train"
    }:
        raise ValueError("Baseline supervised different training identities")
    torch.set_num_threads(2)
    values = PooledCases(pooled, data["cases"], stats, range(len(data["cases"])))
    features, base = {}, []
    for stage in STAGES:
        model = initialize(ck, stage, (5120 if stage == "pooled" else 168) + 36)
        arrays = []
        with torch.inference_mode():
            for part, parameters, _ in DataLoader(
                values, batch_size=128, shuffle=False
            ):
                x, pred = model.features(part, parameters)
                arrays.append(x.numpy())
                if stage == STAGES[0]:
                    base.append(pred.numpy())
        features[stage] = np.concatenate(arrays)
    base = np.concatenate(base)
    output.mkdir(parents=True)
    np.savez_compressed(
        output / "features.npz", **features, base=base, targets=values.y.numpy()
    )
    val = np.array([c["split"] == "val" for c in data["cases"]])
    files = {
        **p["input_sha256"],
        **p["code_sha256"],
        **{
            str(f.resolve()): hashlib.sha256(f.read_bytes()).hexdigest()
            for f in (
                checkpoint,
                parent / "protocol.json",
                output / "features.npz",
                Path(__file__),
                Path(__file__).with_name("kernel_readout.py"),
                Path(__file__).with_name("head_experiment.py"),
                Path(__file__).with_name("detail_kernel.py"),
            )
        },
    }
    protocol = {
        "prepared": p["prepared"],
        "checkpoint": str(checkpoint.resolve()),
        "stages": STAGES,
        "normalization": stats,
        "scales": [0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
        "alphas": [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
        "modes": ["direct", "residual"],
        "baseline_validation": absolute_scores(
            [c for c in data["cases"] if c["split"] == "val"],
            physical(base[val], stats),
        ),
        "selection": "Validation source-macro Cd MAE + Cl MAE; both must not exceed the fixed baseline. No test selection.",
        "normalization_rule": "Train-only per-feature mean/std (floor 0.05), mean squared geometry and condition distances added with equal block weight.",
        "fit": "Deterministic source-weighted kernel ridge on observed labels; 72 candidates per stage; shared kernel hyperparameters for Cd/Cl.",
        "input": "Complete five-view features plus the existing36 normalized dimensionless values/masks; no source or vehicle-class input.",
        "scope": "Historical known-family development comparison; not a fresh blind test or matched neural/kernel compute budget.",
        "sha256": files,
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print(
        json.dumps(
            {
                "cases": len(base),
                "baseline_val": protocol["baseline_validation"],
                "stages": STAGES,
            }
        )
    )


def load(experiment):
    p = json.loads((experiment / "protocol.json").read_text())
    for path, sha in p["sha256"].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != sha:
            raise ValueError(f"Frozen input changed: {path}")
    data = json.loads((Path(p["prepared"]) / "dataset.json").read_text())
    with np.load(experiment / "features.npz", allow_pickle=False) as arrays:
        features = {k: arrays[k] for k in arrays.files}
    return p, data, features


def save(model, source, path, p, choice):
    ck = dict(source)
    for key in (
        "epoch",
        "pair_arm",
        "migration",
        "validation_selection_score",
        "validation_absolute_mae",
    ):
        ck.pop(key, None)
    ck.update(
        input_contract=CONTRACT,
        readout_stage=model.stage,
        model_state={
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        },
        source_checkpoint=p["checkpoint"],
        source_checkpoint_sha256=p["sha256"][p["checkpoint"]],
        kernel_selection=choice,
        training_mode="frozen_features_kernel_readout",
        provenance={
            "kernel_protocol": str((path.parent.parent / "protocol.json").resolve())
        },
    )
    torch.save(ck, path)


def fit(experiment, stage):
    p, data, features = load(experiment)
    out = experiment / stage
    out.mkdir()
    torch.set_num_threads(2)
    started = time.monotonic()
    source = torch.load(p["checkpoint"], map_location="cpu", weights_only=True)
    train = np.array([c["split"] == "train" for c in data["cases"]])
    val = np.array([c["split"] == "val" for c in data["cases"]])
    val_cases = [c for c in data["cases"] if c["split"] == "val"]
    x = features[stage].astype(float)
    mean = x[train].mean(0)
    scale = np.maximum(x[train].std(0), 0.05)
    geometry_size = x.shape[1] - source["model_config"]["parameter_count"]
    scale[:geometry_size] *= np.sqrt(geometry_size / x.shape[1])
    scale[geometry_size:] *= np.sqrt((x.shape[1] - geometry_size) / x.shape[1])
    x = (x - mean) / scale
    model = initialize(source, stage, x.shape[1])
    model.feature_mean.copy_(torch.from_numpy(mean))
    model.feature_scale.copy_(torch.from_numpy(scale))
    limits = p["baseline_validation"]["macro"]
    best, chosen, candidates = sum(limits.values()), None, []
    save(model, source, out / "model.pt", p, chosen)
    weights = sampling_weights(
        [c for c in data["cases"] if c["split"] == "train"]
    ).numpy()
    targets, base = (
        features["targets"][train].astype(float),
        features["base"].astype(float),
    )
    train_distance = distances(x[train], x[train])
    val_distance = distances(x[val], x[train])
    for bandwidth in p["scales"]:
        gram = np.exp(-train_distance / (2 * bandwidth**2))
        kv = np.exp(-val_distance / (2 * bandwidth**2))
        systems = []
        for j in range(2):
            mask = np.isfinite(targets[:, j])
            root = np.sqrt(weights[mask] / weights[mask].mean())
            responses = np.stack(
                (targets[mask, j], targets[mask, j] - base[train, j][mask]), 1
            )
            bias = np.average(responses, axis=0, weights=weights[mask])
            eigen, vectors = np.linalg.eigh(
                gram[np.ix_(mask, mask)] * root[:, None] * root[None]
            )
            projected = vectors.T @ ((responses - bias) * root[:, None])
            systems.append((mask, root, eigen, vectors, projected, bias))
        for alpha in p["alphas"]:
            dual = np.zeros((int(train.sum()), 2, 2))
            bias = np.zeros((2, 2))
            for j, (mask, root, eigen, vectors, projected, center) in enumerate(
                systems
            ):
                dual[mask, j] = root[:, None] * (
                    vectors @ (projected / (np.maximum(eigen, 0)[:, None] + alpha))
                )
                bias[j] = center
            for mode_index, mode in enumerate(p["modes"]):
                prediction = (
                    kv @ dual[:, :, mode_index]
                    + bias[:, mode_index]
                    + mode_index * base[val]
                )
                score = absolute_scores(
                    val_cases, physical(prediction, p["normalization"])
                )
                metrics = score["macro"]
                eligible = all(metrics[k] <= limits[k] for k in limits)
                choice = {
                    "bandwidth": bandwidth,
                    "alpha": alpha,
                    "mode": mode,
                    "validation": score,
                    "eligible": eligible,
                }
                candidates.append(choice)
                if eligible and sum(metrics.values()) < best:
                    best, chosen = sum(metrics.values()), choice
                    model.kernel_reference = torch.from_numpy(x[train].copy())
                    model.kernel_dual = torch.from_numpy(dual[:, :, mode_index].copy())
                    model.kernel_bias.copy_(torch.from_numpy(bias[:, mode_index]))
                    model.kernel_bandwidth.fill_(bandwidth)
                    model.base_weight.fill_(mode_index)
                    save(model, source, out / "model.pt", p, chosen)
        print(
            json.dumps(
                {"stage": stage, "bandwidth_finished": bandwidth, "best": chosen}
            ),
            flush=True,
        )
    report = {
        "candidates": candidates,
        "selected": chosen,
        "elapsed_seconds": time.monotonic() - started,
        "observed_train_targets": np.isfinite(targets).sum(0).tolist(),
        "model_sha256": hashlib.sha256((out / "model.pt").read_bytes()).hexdigest(),
        "test_targets_used": False,
    }
    (out / "fit.json").write_text(json.dumps(report, indent=2) + "\n")


def bootstrap(cases, baseline, prediction):
    truth = np.asarray([c["targets"] for c in cases], float)
    delta = abs(prediction - truth) - abs(baseline - truth)
    rng = np.random.default_rng(20261116)
    samples = []
    for source in sorted({c["source"] for c in cases}):
        indices = np.flatnonzero([c["source"] == source for c in cases])
        draws = rng.choice(indices, (5000, len(indices)), replace=True)
        samples.append(delta[draws].mean(1))
    return {
        "candidate_minus_baseline_mae_95_interval": np.quantile(
            np.mean(samples, axis=0), [0.025, 0.975], axis=0
        ).T.tolist(),
        "scope": "Paired resampling within four sources; uncertainty conditional on these cases/families, excluding development-selection bias.",
    }


def evaluate(experiment):
    """@entry Freeze validation choice first; evaluate the historical test afterward."""
    p, data, features = load(experiment)
    torch.set_num_threads(2)
    reports = {s: json.loads((experiment / s / "fit.json").read_text()) for s in STAGES}
    count = len(p["scales"]) * len(p["alphas"]) * len(p["modes"])
    for stage, report in reports.items():
        if (
            len(report["candidates"]) != count
            or hashlib.sha256(
                (experiment / stage / "model.pt").read_bytes()
            ).hexdigest()
            != report["model_sha256"]
        ):
            raise ValueError("Incomplete or changed kernel fit")
    choices = {
        s: r["selected"]["validation"] if r["selected"] else p["baseline_validation"]
        for s, r in reports.items()
    }
    winner = min(STAGES, key=lambda s: sum(choices[s]["macro"].values()))
    selection = {
        "validation_selected": winner,
        "selected_validation": choices[winner],
        "model_sha256": reports[winner]["model_sha256"],
    }
    (experiment / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    stats = p["normalization"]
    predictions = {"baseline": physical(features["base"], stats)}
    source = torch.load(p["checkpoint"], map_location="cpu", weights_only=True)
    for stage in STAGES:
        model, _ = load_model(experiment / stage / "model.pt")
        for key, value in source["model_state"].items():
            if not torch.equal(value, model.state_dict()[key]):
                raise ValueError("Frozen neural weights changed")
        values = []
        with torch.inference_mode():
            for start in range(0, len(data["cases"]), 128):
                part = slice(start, start + 128)
                values.append(
                    model.predict_features(
                        torch.from_numpy(features[stage][part]),
                        torch.from_numpy(features["base"][part]),
                    ).numpy()
                )
        predictions[stage] = physical(np.concatenate(values), stats)
    result = {}
    for name, pred in predictions.items():
        scores = {}
        for split in ("train", "val", "test"):
            mask = np.array([c["split"] == split for c in data["cases"]])
            scores[split] = absolute_scores(
                [c for c in data["cases"] if c["split"] == split], pred[mask]
            )
        result[name] = {
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
    test = np.array([c["split"] == "test" for c in data["cases"]])
    interval = bootstrap(
        [c for c in data["cases"] if c["split"] == "test"],
        predictions["baseline"][test],
        predictions[winner][test],
    )
    result = {"selection": selection, "models": result, "paired_bootstrap": interval}
    (experiment / "comparison.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "selected": winner,
                "test": {
                    n: m["scores"]["test"]["macro"] for n, m in result["models"].items()
                },
                "bootstrap": interval,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    a = sub.add_parser("prepare")
    a.add_argument("parent", type=Path)
    a.add_argument("checkpoint", type=Path)
    a.add_argument("output", type=Path)
    a = sub.add_parser("fit")
    a.add_argument("experiment", type=Path)
    a.add_argument("stage", choices=STAGES)
    a = sub.add_parser("evaluate")
    a.add_argument("experiment", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.parent, args.checkpoint, args.output)
    elif args.action == "fit":
        fit(args.experiment, args.stage)
    else:
        evaluate(args.experiment)
