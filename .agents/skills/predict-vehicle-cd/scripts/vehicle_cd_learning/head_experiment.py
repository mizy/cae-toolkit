"""Whole-vehicle frozen-encoder head comparison, selected by absolute Cd/Cl MAE."""

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
from .coefficient_head import CONTRACT, KINDS, initialize, load_model
from .readout_experiment import (
    PooledCases,
    absolute_scores,
    coefficient_loss,
)
from .readout_experiment import (
    load as load_parent,
)
from .train import seed_everything


def physical(pred, stats):
    return pred * np.asarray(stats["target_std"]) + stats["target_mean"]


def predictions(model, features):
    model.eval()
    result = []
    with torch.inference_mode():
        for start in range(0, len(features), 128):
            result.append(model.predict_features(features[start : start + 128]).numpy())
    return np.concatenate(result)


def prepare(parent, output):
    """@entry Freeze encoder+fusion; change only the shared absolute-coefficient head."""
    if output.exists():
        raise FileExistsError("Use a fresh head experiment")
    p, data, pooled = load_parent(parent)
    source, ck = dim.load_model(Path(p["checkpoint"]))
    torch.set_num_threads(2)
    values = PooledCases(
        pooled, data["cases"], p["normalization"], range(len(data["cases"]))
    )
    features = []
    with torch.inference_mode():
        for part, x, _ in DataLoader(values, batch_size=128, shuffle=False):
            features.append(source.encode_pooled(part, x).numpy())
    features = np.concatenate(features)
    training = [i for i, c in enumerate(data["cases"]) if c["split"] == "train"]
    mean = features[training].mean(0, dtype=np.float64).astype(np.float32)
    std = np.maximum(features[training].std(0, dtype=np.float64), 0.05).astype(
        np.float32
    )
    reference = source.head(torch.from_numpy(features)).detach().numpy()
    parity = {}
    for kind in KINDS:
        seed_everything(42)
        m = initialize(ck, kind, mean, std)
        actual = predictions(m, torch.from_numpy(features))
        gap = abs(actual - reference).max()
        if gap > 1e-6:
            raise ValueError("New head did not preserve initial function")
        parity[kind] = float(gap)
    mask = np.array([c["split"] == "val" for c in data["cases"]])
    validation = [c for c in data["cases"] if c["split"] == "val"]
    output.mkdir(parents=True)
    np.savez_compressed(
        output / "features.npz", features=features, targets=values.y.numpy()
    )
    np.save(output / "draws.npy", np.load(parent / "draws.npy", allow_pickle=False))
    code = [
        Path(__file__),
        Path(__file__).with_name("coefficient_head.py"),
        Path(__file__).with_name("readout_experiment.py"),
        Path(__file__).with_name("model.py"),
    ]
    code += [Path(name) for name in p["code_sha256"]]
    paths = [
        Path(p["checkpoint"]),
        Path(p["prepared"]) / "dataset.json",
        output / "features.npz",
        output / "draws.npy",
        parent / "protocol.json",
    ]
    paths += [Path(name) for name in p["input_sha256"]]
    protocol = {
        "parent": str(parent.resolve()),
        "prepared": p["prepared"],
        "checkpoint": p["checkpoint"],
        "normalization": p["normalization"],
        "epochs": p["epochs"],
        "batch_size": p["batch_size"],
        "seed": 42,
        "learning_rate": 1e-4,
        "head_kinds": KINDS,
        "feature_mean": mean.tolist(),
        "feature_std": std.tolist(),
        "initial_function_gap": parity,
        "initial_validation": absolute_scores(
            validation, physical(reference[mask], p["normalization"])
        ),
        "selection": "Validation source-equal Cd MAE+Cl MAE; both no worse than initial. No ratio or design-delta metric.",
        "head_semantics": "All heads predict absolute Cd/Cl for one geometry. Residual branches correct original prediction error, not two-geometry design changes.",
        "frozen": "Entire visual encoder, projections, condition encoder and fusion; only original head or a shared two-output correction is trained.",
        "rbf_grid": {
            "scales": [0.03, 0.1, 0.3, 1.0, 3.0],
            "alpha": [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
        },
        "budget_limit": "Neural heads use identical 40-epoch draws; RBF uses source-weighted observed-label kernel ridge and the same validation selector, not a matched-compute neural comparison.",
        "input_sha256": {
            str(x.resolve()): hashlib.sha256(x.read_bytes()).hexdigest() for x in paths
        },
        "code_sha256": {
            str(x.resolve()): hashlib.sha256(x.read_bytes()).hexdigest() for x in code
        },
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print(
        json.dumps(
            {
                "cases": len(features),
                "features": features.shape[1],
                "kinds": KINDS,
                "initial_gaps": parity,
            }
        )
    )


def load(experiment):
    p = json.loads((experiment / "protocol.json").read_text())
    for file, sha in {**p["input_sha256"], **p["code_sha256"]}.items():
        if hashlib.sha256(Path(file).read_bytes()).hexdigest() != sha:
            raise ValueError(f"Frozen head experiment changed:{file}")
    data = json.loads((Path(p["prepared"]) / "dataset.json").read_text())
    with np.load(experiment / "features.npz", allow_pickle=False) as f:
        x = torch.from_numpy(f["features"])
        y = torch.from_numpy(f["targets"])
    return p, data, x, y


def save(model, source, out, p, kind, epoch, metrics):
    ck = dict(source)
    for key in ("pair_arm", "migration", "validation_selection_score"):
        ck.pop(key, None)
    ck.update(
        input_contract=CONTRACT,
        head_kind=kind,
        model_state={
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        },
        epoch=epoch,
        training_mode="fixed_encoder_head",
        validation_absolute_mae=metrics,
        source_checkpoint=p["checkpoint"],
        source_checkpoint_sha256=p["input_sha256"][p["checkpoint"]],
        provenance={"head_protocol": str((out.parent / "protocol.json").resolve())},
    )
    torch.save(ck, out / "model.pt")


def validation(model, x, data, p):
    indices = [i for i, c in enumerate(data["cases"]) if c["split"] == "val"]
    return absolute_scores(
        [data["cases"][i] for i in indices],
        physical(predictions(model, x[indices]), p["normalization"]),
    )


def train(experiment, kind):
    """@entry Train a genuine head variant with encoder weights and features fixed."""
    if kind == "rbf":
        return fit_rbf(experiment)
    p, data, x, y = load(experiment)
    out = experiment / kind
    out.mkdir()
    seed_everything(p["seed"])
    torch.set_num_threads(2)
    source = torch.load(p["checkpoint"], map_location="cpu", weights_only=True)
    model = initialize(source, kind, p["feature_mean"], p["feature_std"])
    params = [v for v in model.parameters() if v.requires_grad]
    optimizer = torch.optim.Adam(params, lr=p["learning_rate"])
    indices = torch.tensor(
        [i for i, c in enumerate(data["cases"]) if c["split"] == "train"]
    )
    draws = np.load(experiment / "draws.npy", allow_pickle=False)
    std = torch.tensor(p["normalization"]["target_std"], dtype=torch.float32)
    limits = p["initial_validation"]["macro"]
    best = sum(limits.values())
    save(model, source, out, p, kind, 0, limits)
    history = []
    steps = 0
    trace = hashlib.sha256()
    counts = np.zeros(len(indices), int)
    started = time.monotonic()
    for epoch, row in enumerate(draws, 1):
        model.train()
        for start in range(0, len(row), p["batch_size"]):
            selected = row[start : start + p["batch_size"]]
            trace.update(selected.tobytes())
            counts += np.bincount(selected, minlength=len(indices))
            idx = indices[selected]
            optimizer.zero_grad(set_to_none=True)
            pred = model.predict_features(x[idx])
            loss, _, _ = coefficient_loss(pred, y[idx], std, params, False)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite head loss")
            loss.backward()
            optimizer.step()
            steps += 1
        score = validation(model, x, data, p)
        metrics = score["macro"]
        eligible = all(metrics[k] <= limits[k] for k in limits)
        if eligible and sum(metrics.values()) < best:
            best = sum(metrics.values())
            save(model, source, out, p, kind, epoch, metrics)
        history.append(
            {
                "epoch": epoch,
                "validation": score,
                "eligible": eligible,
                "elapsed_seconds": time.monotonic() - started,
            }
        )
        (out / "training_report.json").write_text(
            json.dumps(
                {
                    "history": history,
                    "optimizer_steps": steps,
                    "draw_trace_sha256": trace.hexdigest(),
                    "exposures": counts.tolist(),
                },
                indent=2,
            )
            + "\n"
        )
        print(json.dumps({"head": kind, "epoch": epoch, **metrics}), flush=True)


def rbf_kernel(a, b, scale):
    distance = np.maximum(
        (a * a).mean(1)[:, None] + (b * b).mean(1)[None] - 2 * (a @ b.T) / a.shape[1], 0
    )
    return np.exp(-distance / (2 * scale**2))


def fit_rbf(experiment):
    p, data, features, targets = load(experiment)
    out = experiment / "rbf"
    out.mkdir()
    torch.set_num_threads(2)
    source = torch.load(p["checkpoint"], map_location="cpu", weights_only=True)
    model = initialize(source, "rbf", p["feature_mean"], p["feature_std"])
    train = np.array([c["split"] == "train" for c in data["cases"]])
    val = np.array([c["split"] == "val" for c in data["cases"]])
    x = (features.numpy().astype(float) - np.asarray(p["feature_mean"])) / np.asarray(
        p["feature_std"]
    )
    base = predictions(model, features)
    residual = targets.numpy().astype(float) - base
    observed = np.isfinite(residual[train])
    weights = sampling_weights(
        [c for c in data["cases"] if c["split"] == "train"]
    ).numpy()
    bias = np.array(
        [
            np.average(
                residual[train, j][observed[:, j]], weights=weights[observed[:, j]]
            )
            for j in range(2)
        ]
    )
    limits = p["initial_validation"]["macro"]
    best = sum(limits.values())
    save(model, source, out, p, "rbf", 0, limits)
    candidates = []
    choice = None
    for scale in p["rbf_grid"]["scales"]:
        gram = rbf_kernel(x[train], x[train], scale)
        kv = rbf_kernel(x[val], x[train], scale)
        systems = []
        for j in range(2):
            mask = observed[:, j]
            root_weight = np.sqrt(weights[mask] / weights[mask].mean())
            eigen, basis = np.linalg.eigh(
                gram[np.ix_(mask, mask)] * root_weight[:, None] * root_weight[None]
            )
            projected = basis.T @ ((residual[train, j][mask] - bias[j]) * root_weight)
            systems.append((mask, eigen, basis, projected, root_weight))
        for alpha in p["rbf_grid"]["alpha"]:
            dual = np.zeros((train.sum(), 2))
            for j, (mask, eigen, basis, projected, root_weight) in enumerate(systems):
                dual[mask, j] = root_weight * (
                    basis @ (projected / (np.maximum(eigen, 0) + alpha))
                )
            pred = base[val] + kv @ dual + bias
            score = absolute_scores(
                [c for c in data["cases"] if c["split"] == "val"],
                physical(pred, p["normalization"]),
            )
            metrics = score["macro"]
            row = {"scale": scale, "alpha": alpha, "validation": score}
            candidates.append(row)
            if (
                all(metrics[k] <= limits[k] for k in limits)
                and sum(metrics.values()) < best
            ):
                best = sum(metrics.values())
                choice = row
                model.kernel_reference = torch.tensor(x[train], dtype=torch.float64)
                model.kernel_dual = torch.tensor(dual, dtype=torch.float64)
                model.kernel_bias = torch.tensor(bias, dtype=torch.float64)
                model.kernel_scale.fill_(scale)
                save(model, source, out, p, "rbf", 1, metrics)
        print(json.dumps({"rbf_scale_complete": scale, "selected": choice}), flush=True)
    (out / "training_report.json").write_text(
        json.dumps(
            {
                "candidates": candidates,
                "selected": choice,
                "observed_train_targets": observed.sum(0).tolist(),
                "test_labels_used": False,
            },
            indent=2,
        )
        + "\n"
    )


def evaluate(experiment):
    """Evaluate only after all neural budgets and kernel validation fits finish."""
    p, data, x, _ = load(experiment)
    torch.set_num_threads(2)
    result = {}
    source = torch.load(p["checkpoint"], map_location="cpu", weights_only=True)
    draws = np.load(experiment / "draws.npy", allow_pickle=False)
    trace = hashlib.sha256()
    for row in draws:
        for start in range(0, len(row), p["batch_size"]):
            trace.update(row[start : start + p["batch_size"]].tobytes())
    counts = np.bincount(draws.ravel(), minlength=draws.shape[1]).tolist()
    for kind in KINDS:
        report = json.loads((experiment / kind / "training_report.json").read_text())
        if kind != "rbf" and len(report["history"]) != p["epochs"]:
            raise ValueError("Incomplete neural head budget")
        if kind != "rbf" and (
            report["draw_trace_sha256"] != trace.hexdigest()
            or report["exposures"] != counts
            or report["optimizer_steps"]
            != len(draws) * ((draws.shape[1] + p["batch_size"] - 1) // p["batch_size"])
        ):
            raise ValueError("Neural heads used different training draws")
        if kind == "rbf" and len(report["candidates"]) != len(
            p["rbf_grid"]["scales"]
        ) * len(p["rbf_grid"]["alpha"]):
            raise ValueError("Incomplete kernel validation grid")
        model, ck = load_model(experiment / kind / "model.pt")
        for name, value in source["model_state"].items():
            if (kind != "original" or not name.startswith("head.")) and not torch.equal(
                value, model.state_dict()[name]
            ):
                raise ValueError("Frozen encoder changed")
        y = physical(predictions(model, x), p["normalization"])
        scores = {}
        for split in ("train", "val", "test"):
            mask = np.array([c["split"] == split for c in data["cases"]])
            scores[split] = absolute_scores(
                [c for c in data["cases"] if c["split"] == split], y[mask]
            )
        result[kind] = {
            "epoch": ck["epoch"],
            "scores": scores,
            "rows": [
                {
                    "case_id": c["case_id"],
                    "source": c["source"],
                    "split": c["split"],
                    "truth": c["targets"],
                    "prediction": v.tolist(),
                }
                for c, v in zip(data["cases"], y)
            ],
        }
    winner = min(KINDS, key=lambda n: sum(result[n]["scores"]["val"]["macro"].values()))
    (experiment / "comparison.json").write_text(
        json.dumps(
            {"protocol": p, "models": result, "validation_selected": winner},
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "validation_selected": winner,
                "scores": {
                    n: {s: r["scores"][s]["macro"] for s in ("val", "test")}
                    for n, r in result.items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    f = sub.add_parser("prepare")
    f.add_argument("parent", type=Path)
    f.add_argument("output", type=Path)
    t = sub.add_parser("train")
    t.add_argument("experiment", type=Path)
    t.add_argument("kind", choices=KINDS)
    e = sub.add_parser("evaluate")
    e.add_argument("experiment", type=Path)
    a = parser.parse_args()
    if a.action == "prepare":
        prepare(a.parent, a.output)
    elif a.action == "train":
        train(a.experiment, a.kind)
    else:
        evaluate(a.experiment)
