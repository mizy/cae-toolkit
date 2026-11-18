"""Matched absolute/delta continuation with train-only, same-condition Rim pairs."""

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from aircraft_learning.scene_experiment import SceneCases, sampling_weights
from torch.utils.data import DataLoader, default_collate

from .dimensionless import load_model, parameter_ranges, read_prepared
from .train import masked_mse_loss, move_batch, seed_everything


def pair_schedule(cases, epochs, seed):
    """Sample identical endpoints for both arms; only Rim has distinct partners."""
    rng = torch.Generator().manual_seed(seed + 3)
    partners = np.random.default_rng(seed + 4)
    rim = np.array([i for i, c in enumerate(cases) if c["source"] == "drivaerrim"])
    if len(rim) < 2:
        raise ValueError("Need at least two training Rim geometries")
    left = (
        torch.multinomial(
            sampling_weights(cases), epochs * len(cases), True, generator=rng
        )
        .numpy()
        .reshape(epochs, -1)
    )
    right = left.copy()
    for a, b in zip(left, right):
        for j, i in enumerate(a):
            if cases[i]["source"] == "drivaerrim":
                k = int(partners.integers(len(rim) - 1))
                position = int(np.searchsorted(rim, i))
                b[j] = rim[k + (k >= position)]
    return left, right


def freeze(prepared, checkpoint, output, epochs=12):
    """@entry Freeze the dataset, endpoints, loss scale and selector before training."""
    if output.exists():
        raise FileExistsError("Use a fresh experiment directory")
    data = read_prepared(prepared)
    _, ck = load_model(checkpoint)
    train = [c for c in data["cases"] if c["split"] == "train"]
    rim = [c for c in train if c["source"] == "drivaerrim"]
    if len(rim) != 720 or any(c["parameters"] != rim[0]["parameters"] for c in rim):
        raise ValueError(
            "Expected720 Rim training cases with identical declared numerical conditions"
        )
    anchor = min(rim, key=lambda c: c["case_id"])
    left, right = pair_schedule(train, epochs, 42)
    target = np.asarray([c["targets"] for c in train], float)
    delta = target[right] - target[left]
    active = np.array(
        [[train[i]["source"] == "drivaerrim" for i in row] for row in left]
    )
    scale = delta[active].std(0)
    if not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("No usable train-only delta scale")
    output.mkdir(parents=True)
    np.savez_compressed(output / "pairs.npz", left=left, right=right)
    files = [
        Path(__file__),
        Path(__file__).with_name("pair_evaluation.py"),
        Path(__file__).with_name("model.py"),
        Path(__file__).with_name("dimensionless.py"),
        Path(__file__).with_name("train.py"),
    ]
    inputs = [prepared / "dataset.json", checkpoint, output / "pairs.npz"]
    protocol = {
        "prepared": str(prepared.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "seed": 42,
        "epochs": epochs,
        "batch_size": 32,
        "learning_rate": 1e-4,
        "arms": {"A1": 0.0, "A2": 0.1},
        "anchor_id": anchor["case_id"],
        "display_query_ids": sorted(
            c["case_id"]
            for c in data["cases"]
            if c["source"] == "drivaerrim" and c["split"] == "test"
        )[:3],
        "delta_std": scale.tolist(),
        "normalization": {
            k: ck[k]
            for k in ("parameter_mean", "parameter_std", "target_mean", "target_std")
        },
        "augmentation": "none in either arm; existing strong jitter suppressed Rim detail in prior experiments",
        "loss": "masked absolute MSE over both endpoints + lambda * Rim-pair delta MSE / train-pair delta variance",
        "selection": "val source-macro absolute MSE in frozen global target units +0.1*val Rim anchored-delta MSE in train-pair units",
        "anchor_selection": "Smallest training Rim case ID; display queries are first3 historical test IDs, never selected by error",
        "re_policy": "Use the same explicitly completed Re inputs for A0/A1/A2; also report frozen A0 on original missing-Re inputs. Fixed source conditions only; no Re sweep claim.",
        "scope": f"Single-seed,{epochs}-epoch continuation diagnostic; existing historical family splits, not independent vehicle-family testing. No pseudo Cl supervision.",
        "source_counts": dict(
            Counter(f"{c['split']}/{c['source']}" for c in data["cases"])
        ),
        "input_sha256": {
            str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs
        },
        "code_sha256": {
            str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in files
        },
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    return protocol


def load(experiment):
    protocol = json.loads((experiment / "protocol.json").read_text())
    for path, sha in {**protocol["input_sha256"], **protocol["code_sha256"]}.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != sha:
            raise ValueError(f"Frozen experiment input changed: {path}")
    return protocol, json.loads(
        (Path(protocol["prepared"]) / "dataset.json").read_text()
    )


def pair_loss(prediction, targets, rim, global_std, delta_std, weight):
    """Both endpoints stay differentiable; missing true labels stay masked."""
    count = len(rim)
    absolute = masked_mse_loss(prediction, targets)
    if not rim.any():
        return absolute, absolute.detach() * 0
    expected = targets[count:] - targets[:count]
    actual = prediction[count:] - prediction[:count]
    observed = rim[:, None] & torch.isfinite(expected)
    error = (actual - expected) * global_std / delta_std
    delta = error[observed].square().mean()
    return absolute + weight * delta, delta


def predictions(model, values, device, batch_size=32):
    model.eval()
    result = []
    with torch.inference_mode():
        for batch in DataLoader(values, batch_size=batch_size, shuffle=False):
            views, params, _ = move_batch(batch[:3], device)
            result.append(model(views, params).cpu().numpy())
    return np.concatenate(result)


def selection(model, validation, anchor, protocol, device):
    pred = predictions(model, validation, device)
    stats = protocol["normalization"]
    truth = (
        np.asarray([c["targets"] for c in validation.cases]) - stats["target_mean"]
    ) / stats["target_std"]
    source = np.array([c["source"] for c in validation.cases])
    absolute = float(
        np.mean(
            [
                np.nanmean((pred[source == s] - truth[source == s]) ** 2)
                for s in sorted(set(source))
            ]
        )
    )
    base = predictions(model, anchor, device)[0]
    base_true = (np.asarray(anchor.cases[0]["targets"]) - stats["target_mean"]) / stats[
        "target_std"
    ]
    mask = source == "drivaerrim"
    error = (
        ((pred[mask] - base) - (truth[mask] - base_true))
        * np.asarray(stats["target_std"])
        / protocol["delta_std"]
    )
    delta = float(np.mean(error**2))
    return absolute + 0.1 * delta, absolute, delta


def save_checkpoint(model, original, output, protocol, data, epoch, score, arm):
    ck = dict(original)
    ck.pop("validation_mse_standardized", None)
    ck.update(
        model_state={
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        },
        dataset_sha256=protocol["input_sha256"][
            str((Path(protocol["prepared"]) / "dataset.json").resolve())
        ],
        parameter_ranges=parameter_ranges(
            [c for c in data["cases"] if c["split"] == "train"]
        ),
        continued_from_sha256=protocol["input_sha256"][protocol["checkpoint"]],
        epoch=epoch,
        seed=protocol["seed"],
        validation_selection_score=score,
        training_mode="paired_continuation",
        pair_arm=arm,
        augmentation={"enabled": False},
        provenance={"pair_protocol": str((output.parent / "protocol.json").resolve())},
    )
    torch.save(ck, output / "model.pt")


def train_arm(experiment, arm, device_name="mps"):
    """@entry Continue a frozen arm without changing the original checkpoint."""
    p, data = load(experiment)
    if arm not in p["arms"]:
        raise ValueError("Unknown arm")
    out = experiment / arm
    out.mkdir()
    seed_everything(p["seed"])
    torch.set_num_threads(2)
    model, original = load_model(Path(p["checkpoint"]))
    device = torch.device(device_name)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=p["learning_rate"])
    root, stats = Path(p["prepared"]), p["normalization"]
    train = [c for c in data["cases"] if c["split"] == "train"]
    values = SceneCases(root, train, stats)
    validation = SceneCases(
        root, [c for c in data["cases"] if c["split"] == "val"], stats
    )
    anchor = SceneCases(
        root, [next(c for c in train if c["case_id"] == p["anchor_id"])], stats
    )
    with np.load(experiment / "pairs.npz", allow_pickle=False) as pairs:
        left, right = pairs["left"], pairs["right"]
    std = torch.tensor(stats["target_std"], dtype=torch.float32, device=device)
    delta_std = torch.tensor(p["delta_std"], dtype=torch.float32, device=device)
    history, best, steps = [], float("inf"), 0
    counts, trace = np.zeros(len(train), int), hashlib.sha256()
    started = time.monotonic()
    for epoch, (a, b) in enumerate(zip(left, right), 1):
        total, delta_total = 0.0, 0.0
        model.train()
        for start in range(0, len(a), p["batch_size"]):
            first, second = (
                a[start : start + p["batch_size"]],
                b[start : start + p["batch_size"]],
            )
            ids = np.concatenate((first, second))
            trace.update(ids.astype(np.int32).tobytes())
            counts += np.bincount(ids, minlength=len(train))
            batch = default_collate([values[int(i)] for i in ids])
            views, params, targets = move_batch(batch[:3], device)
            rim = torch.tensor(
                [train[i]["source"] == "drivaerrim" for i in first], device=device
            )
            optimizer.zero_grad(set_to_none=True)
            pred = model(views, params)
            loss, delta = pair_loss(pred, targets, rim, std, delta_std, p["arms"][arm])
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite paired training loss")
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
            delta_total += float(delta.detach())
            steps += 1
        score, absolute, paired = selection(model, validation, anchor, p, device)
        record = {
            "epoch": epoch,
            "val_score": score,
            "val_absolute": absolute,
            "val_delta": paired,
            "train_loss": total / (steps / epoch),
            "train_delta_loss": delta_total / (steps / epoch),
            "elapsed_seconds": time.monotonic() - started,
        }
        history.append(record)
        if score < best:
            best = score
            save_checkpoint(model, original, out, p, data, epoch, score, arm)
        (out / "training_report.json").write_text(
            json.dumps(
                {
                    "history": history,
                    "optimizer_steps": steps,
                    "endpoint_exposures": dict(
                        zip([c["case_id"] for c in train], counts.tolist())
                    ),
                    "endpoint_trace_sha256": trace.hexdigest(),
                    "initial_checkpoint_sha256": p["input_sha256"][p["checkpoint"]],
                },
                indent=2,
            )
            + "\n"
        )
        print(json.dumps({"arm": arm, **record}), flush=True)
    if device.type == "mps":
        torch.mps.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("prepared", type=Path)
    f.add_argument("checkpoint", type=Path)
    f.add_argument("output", type=Path)
    f.add_argument("--epochs", type=int, default=12)
    t = sub.add_parser("train")
    t.add_argument("experiment", type=Path)
    t.add_argument("arm", choices=("A1", "A2"))
    t.add_argument("--device", default="mps")
    args = parser.parse_args()
    if args.action == "freeze":
        print(
            json.dumps(
                freeze(args.prepared, args.checkpoint, args.output, args.epochs),
                indent=2,
            )
        )
    else:
        train_arm(args.experiment, args.arm, args.device)
