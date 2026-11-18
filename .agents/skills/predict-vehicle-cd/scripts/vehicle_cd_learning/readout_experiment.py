"""Controlled readout continuation selected only by absolute Cd and Cl MAE."""

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from aircraft_learning.scene_experiment import SceneCases, sampling_weights
from torch.utils.data import DataLoader, Dataset, default_collate

from . import dimensionless as dim
from .model import VIEWS
from .readout import CONTRACT, initialize
from .train import seed_everything

RUNS = {
    "absolute": {"scale": False, "balance": False},
    "features": {"scale": True, "balance": False},
    "balanced": {"scale": False, "balance": True},
    "features-balanced": {"scale": True, "balance": True},
}


class PooledCases(Dataset):
    def __init__(self, pooled, cases, stats, indices):
        self.pooled, self.indices = pooled, list(indices)
        self.cases = [cases[i] for i in indices]
        self.x = torch.tensor(
            np.stack(
                [dim.normalize_parameters(c["parameters"], stats) for c in self.cases]
            )
        )
        y = (
            np.asarray([c["targets"] for c in self.cases], float) - stats["target_mean"]
        ) / stats["target_std"]
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return (
            {n: self.pooled[self.indices[index], i] for i, n in enumerate(VIEWS)},
            self.x[index],
            self.y[index],
        )


def predict(model, values, stats):
    model.eval()
    result = []
    with torch.inference_mode():
        for pooled, x, _ in DataLoader(values, batch_size=128, shuffle=False):
            result.append(model.head(model.encode_pooled(pooled, x)).numpy())
    return (
        np.concatenate(result) * np.asarray(stats["target_std"]) + stats["target_mean"]
    )


def absolute_scores(cases, prediction):
    truth = np.asarray([c["targets"] for c in cases], float)
    errors = abs(prediction - truth)
    groups = {}
    for source in sorted({c["source"] for c in cases}):
        mask = np.array([c["source"] == source for c in cases])
        entry = {"n": int(mask.sum())}
        for i, name in enumerate(("cd", "cl")):
            observed = mask & np.isfinite(truth[:, i])
            entry[name + "_n"] = int(observed.sum())
            entry[name + "_mae"] = (
                float(errors[observed, i].mean()) if observed.any() else None
            )
        groups[source] = entry
    macro = {
        key: float(np.mean([s[key] for s in groups.values() if s[key] is not None]))
        for key in ("cd_mae", "cl_mae")
    }
    return {"sources": groups, "macro": macro}


def extract(prepared, cases, source, device_name):
    model, _ = dim.load_model(source)
    device = torch.device(device_name)
    model.to(device).eval()
    stats = torch.load(source, map_location="cpu", weights_only=True)
    values = SceneCases(prepared, cases, stats)
    arrays = []
    with torch.inference_mode():
        for batch in DataLoader(values, batch_size=24, shuffle=False):
            views = {n: v.to(device) for n, v in batch[0].items()}
            body = model.image_encoder.pooled_features(
                torch.cat([views[n] for n in VIEWS[:4]])
            )
            parts = list(body.chunk(4)) + [
                model.bottom_encoder.pooled_features(views["bottom"])
            ]
            arrays.append(torch.stack(parts, dim=1).cpu().numpy())
            values.cache.clear()
    del model, values
    if device.type == "mps":
        torch.mps.empty_cache()
    return np.concatenate(arrays)


def feature_statistics(pooled, training):
    result = {}
    selected = pooled[training]
    for prefix, values in (
        ("body", selected[:, :4].reshape(-1, 1024)),
        ("bottom", selected[:, 4].reshape(-1, 1024)),
    ):
        result[prefix + "_mean"] = values.mean(0, dtype=np.float64).tolist()
        result[prefix + "_std"] = np.maximum(
            values.std(0, dtype=np.float64), 0.05
        ).tolist()
    return result


def prepare(prepared, checkpoint, output, epochs=40, device_name="mps"):
    """@entry Freeze four same-budget absolute-only arms and bounded feature cache."""
    if output.exists():
        raise FileExistsError("Use a fresh experiment directory")
    data = dim.read_prepared(prepared)
    ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
    output.mkdir(parents=True)
    torch.set_num_threads(2)
    pooled = extract(prepared, data["cases"], checkpoint, device_name)
    np.savez_compressed(output / "pooled.npz", pooled=pooled)
    training = [i for i, c in enumerate(data["cases"]) if c["split"] == "train"]
    train = [data["cases"][i] for i in training]
    rng = torch.Generator().manual_seed(45)
    draws = (
        torch.multinomial(
            sampling_weights(train), epochs * len(train), True, generator=rng
        )
        .reshape(epochs, -1)
        .numpy()
        .astype(np.int32)
    )
    np.save(output / "draws.npy", draws)
    stats = {
        k: ck[k]
        for k in ("parameter_mean", "parameter_std", "target_mean", "target_std")
    }
    moments = feature_statistics(pooled, training)
    values = PooledCases(
        torch.from_numpy(pooled), data["cases"], stats, range(len(data["cases"]))
    )
    reference = predict(initialize(ck), values, stats)
    scaled = predict(initialize(ck, moments), values, stats)
    gap = abs(reference - scaled).max(0)
    if gap.max() > 1e-5:
        raise ValueError("Feature reparameterization changed the initial function")
    files = [
        Path(__file__),
        Path(__file__).with_name("readout.py"),
        Path(__file__).with_name("readout_evaluation.py"),
        Path(__file__).with_name("model.py"),
        Path(__file__).with_name("dimensionless.py"),
    ]
    paths = [
        prepared / "dataset.json",
        checkpoint,
        output / "pooled.npz",
        output / "draws.npy",
    ]
    validation = [c for c in data["cases"] if c["split"] == "val"]
    mask = np.array([c["split"] == "val" for c in data["cases"]])
    protocol = {
        "prepared": str(prepared.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "epochs": epochs,
        "batch_size": 64,
        "seed": 42,
        "learning_rate": 1e-4,
        "runs": RUNS,
        "normalization": stats,
        "feature_statistics": moments,
        "feature_std_floor": 0.05,
        "initial_max_absolute_prediction_difference": gap.tolist(),
        "initial_validation": absolute_scores(validation, reference[mask]),
        "selection": "Validation source-equal absolute Cd MAE+Cl MAE; both MAEs must be no worse than the initial checkpoint. No ratio/delta/ranking objective.",
        "loss": "Per-target MAE in original coefficient units. Balanced arms equalize Cd/Cl gradient norms with weights summing2, bounded0.25..1.75.",
        "frozen": "All visual convolution/pooling/attention parameters before projection; train projection, condition encoder, view fusion and shared Cd/Cl head.",
        "feature_moments": "Train cases only; ordinary4 views share moments to preserve shared projection; function-preserving initial reparameterization.",
        "source_counts": dict(
            Counter(f"{c['split']}/{c['source']}" for c in data["cases"])
        ),
        "input_sha256": {
            str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
        },
        "code_sha256": {
            str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in files
        },
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print(
        json.dumps(
            {
                "cache_cases": len(data["cases"]),
                "epochs": epochs,
                "initial_parity": gap.tolist(),
                "initial_val": protocol["initial_validation"],
            }
        ),
        flush=True,
    )


def load(experiment):
    p = json.loads((experiment / "protocol.json").read_text())
    for name, sha in {**p["input_sha256"], **p["code_sha256"]}.items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != sha:
            raise ValueError(f"Frozen experiment artifact changed:{name}")
    data = json.loads((Path(p["prepared"]) / "dataset.json").read_text())
    with np.load(experiment / "pooled.npz", allow_pickle=False) as f:
        pooled = torch.from_numpy(f["pooled"])
    return p, data, pooled


def coefficient_loss(prediction, targets, std, parameters, balanced):
    """Mask missing Cl; adjust only Cd/Cl training gradients, never target values."""
    observed = torch.isfinite(targets)
    losses = []
    for i in range(2):
        mask = observed[:, i]
        losses.append(
            abs((prediction[mask, i] - targets[mask, i]) * std[i]).mean()
            if mask.any()
            else prediction[:, i].sum() * 0
        )
    weights = [1.0, 1.0]
    if balanced and all(observed[:, i].any() for i in range(2)):
        norms = []
        for loss in losses:
            gradient = torch.autograd.grad(
                loss, parameters, retain_graph=True, allow_unused=True
            )
            norms.append(
                float(
                    torch.sqrt(
                        sum(
                            g.detach().square().sum() for g in gradient if g is not None
                        )
                    )
                )
            )
        if sum(norms) > 1e-12:
            weights[0] = float(np.clip(2 * norms[1] / sum(norms), 0.25, 1.75))
            weights[1] = 2 - weights[0]
    return sum(w * l for w, l in zip(weights, losses)), losses, weights


def save(model, original, output, p, arm, epoch, metrics):
    ck = dict(original)
    for key in (
        "pair_arm",
        "migration",
        "parameter_observed_counts",
        "validation_selection_score",
    ):
        ck.pop(key, None)
    ck.update(
        input_contract=CONTRACT,
        model_state={
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        },
        epoch=epoch,
        training_mode="absolute_readout",
        readout_arm=arm,
        source_checkpoint=p["checkpoint"],
        source_checkpoint_sha256=p["input_sha256"][p["checkpoint"]],
        continued_from_sha256=p["input_sha256"][p["checkpoint"]],
        validation_absolute_mae=metrics,
        provenance={
            "readout_protocol": str((output.parent / "protocol.json").resolve())
        },
    )
    torch.save(ck, output / "model.pt")


def train(experiment, arm):
    """@entry Train a matched cached-feature arm; select only by raw Cd/Cl MAE."""
    p, data, pooled = load(experiment)
    run = p["runs"][arm]
    output = experiment / arm
    output.mkdir()
    seed_everything(p["seed"])
    torch.set_num_threads(2)
    original = torch.load(p["checkpoint"], map_location="cpu", weights_only=True)
    model = initialize(original, p["feature_statistics"] if run["scale"] else None)
    parameters = [v for v in model.parameters() if v.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=p["learning_rate"])
    training = [i for i, c in enumerate(data["cases"]) if c["split"] == "train"]
    train_data = PooledCases(pooled, data["cases"], p["normalization"], training)
    validation = PooledCases(
        pooled,
        data["cases"],
        p["normalization"],
        [i for i, c in enumerate(data["cases"]) if c["split"] == "val"],
    )
    limits = p["initial_validation"]["macro"]
    best = sum(limits.values())
    save(model, original, output, p, arm, 0, limits)
    draws = np.load(experiment / "draws.npy", allow_pickle=False)
    std = torch.tensor(p["normalization"]["target_std"], dtype=torch.float32)
    history = []
    counts = np.zeros(len(training), int)
    trace = hashlib.sha256()
    steps = 0
    started = time.monotonic()
    for epoch, ids in enumerate(draws, 1):
        model.train()
        totals = np.zeros(4)
        batches = 0
        for start in range(0, len(ids), p["batch_size"]):
            selected = ids[start : start + p["batch_size"]]
            counts += np.bincount(selected, minlength=len(training))
            trace.update(selected.tobytes())
            features, x, y = default_collate([train_data[int(i)] for i in selected])
            optimizer.zero_grad(set_to_none=True)
            prediction = model.head(model.encode_pooled(features, x))
            loss, target_losses, weights = coefficient_loss(
                prediction, y, std, parameters, run["balance"]
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite coefficient loss")
            loss.backward()
            optimizer.step()
            steps += 1
            batches += 1
            totals += np.array([float(v.detach()) for v in target_losses] + weights)
        score = absolute_scores(
            validation.cases, predict(model, validation, p["normalization"])
        )
        metrics = score["macro"]
        value = sum(metrics.values())
        eligible = all(metrics[k] <= limits[k] for k in limits)
        if eligible and value < best:
            best = value
            save(model, original, output, p, arm, epoch, metrics)
        record = {
            "epoch": epoch,
            "validation": score,
            "eligible": eligible,
            "train_cd_mae": totals[0] / batches,
            "train_cl_mae": totals[1] / batches,
            "mean_cd_weight": totals[2] / batches,
            "mean_cl_weight": totals[3] / batches,
            "elapsed_seconds": time.monotonic() - started,
        }
        history.append(record)
        (output / "training_report.json").write_text(
            json.dumps(
                {
                    "history": history,
                    "optimizer_steps": steps,
                    "exposures": dict(
                        zip([c["case_id"] for c in train_data.cases], counts.tolist())
                    ),
                    "draw_trace_sha256": trace.hexdigest(),
                },
                indent=2,
            )
            + "\n"
        )
        print(
            json.dumps({"arm": arm, "epoch": epoch, **metrics, "eligible": eligible}),
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    f = sub.add_parser("prepare")
    f.add_argument("prepared", type=Path)
    f.add_argument("checkpoint", type=Path)
    f.add_argument("output", type=Path)
    f.add_argument("--epochs", type=int, default=40)
    f.add_argument("--device", default="mps")
    t = sub.add_parser("train")
    t.add_argument("experiment", type=Path)
    t.add_argument("arm", choices=RUNS)
    a = parser.parse_args()
    prepare(
        a.prepared, a.checkpoint, a.output, a.epochs, a.device
    ) if a.action == "prepare" else train(a.experiment, a.arm)
