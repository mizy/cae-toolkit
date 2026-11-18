"""Frozen four-arm, single-seed complete-view paired ΔCd/ΔCl training."""

import argparse
import hashlib
import json
import time
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np
import torch

from .data import validate_views
from .delta_metrics import score_anchors
from .dimensionless import NAMES, PARAMETERS
from .model import VIEWS
from .paired_delta import initialize

ARMS = {
    "A": {"resolution": 256, "pool_size": 8},
    "B": {"resolution": 512, "pool_size": 8},
    "C": {"resolution": 256, "pool_size": 16},
    "D": {"resolution": 512, "pool_size": 16},
}
ANCHORS = [f"drivaerrim__rim-{i:04d}" for i in (1, 3, 4)]


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def tensor_sha256(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(str((tuple(value.shape), value.dtype)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def pair_schedule(indices, epochs=12, seed=42):
    """Every geometry appears once on each side per epoch; endpoints differ."""
    rng = np.random.default_rng(seed)
    left, right = [], []
    for _ in range(epochs):
        order = rng.permutation(indices)
        shift = int(rng.integers(1, len(order)))
        left.append(order)
        right.append(np.roll(order, shift))
    return np.asarray(left), np.asarray(right)


def freeze(prepared, prepared512, checkpoint, external, output):
    """@entry Freeze all available inputs; defer the genuine512 data lock."""
    prepared, prepared512, checkpoint, external, output = map(
        Path, (prepared, prepared512, checkpoint, external, output)
    )
    if output.exists():
        raise FileExistsError("Use a fresh experiment directory")
    data = json.loads((prepared / "dataset.json").read_text())
    cases = sorted(
        [c for c in data["cases"] if c["source"] == "drivaerrim"],
        key=lambda c: c["case_id"],
    )
    if len(cases) != 904 or Counter(c["split"] for c in cases) != {
        "train": 720,
        "val": 92,
        "test": 92,
    }:
        raise ValueError("Expected historical Rim720/92/92 membership")
    if len({c["case_id"] for c in cases}) != len(cases):
        raise ValueError("Repeated case IDs")
    indices = [i for i, c in enumerate(cases) if c["split"] == "train"]
    if any(c["parameters"] != cases[indices[0]]["parameters"] for c in cases):
        raise ValueError("All reference/query pair conditions must match")
    if not set(ANCHORS) <= {cases[i]["case_id"] for i in indices}:
        raise ValueError("Missing training anchors")
    ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if ck["parameter_names"] != NAMES or data["parameter_names"] != PARAMETERS:
        raise ValueError("Expected the existing dimensionless numeric contract")
    if not {cases[i]["case_id"] for i in indices} <= set(ck["train_ids"]):
        raise ValueError("Source CNN did not train on all720 training IDs")
    if {c["case_id"] for c in cases if c["split"] != "train"} & set(ck["train_ids"]):
        raise ValueError("Pretrained CNN supervised held-out Rim geometries")
    left, right = pair_schedule(indices)
    target = np.asarray([c["targets"] for c in cases], float)
    delta = target[right] - target[left]
    scale = np.sqrt(np.mean(delta**2, axis=(0, 1)))
    if (
        not np.isfinite(scale).all()
        or (scale <= 0).any()
        or not np.allclose(delta.mean((0, 1)), 0, atol=1e-14)
    ):
        raise ValueError("Invalid zero-mean train-only delta scale")
    output.mkdir(parents=True)
    np.savez_compressed(output / "pairs.npz", left=left, right=right)
    inventory = Path(
        "outputs/aerodynamic-expansion-20261109/raw/drivaerrim/inventory-geometry.json"
    ).resolve()
    code = [
        Path(__file__).with_name(n)
        for n in (
            "delta_experiment.py",
            "paired_delta.py",
            "delta_metrics.py",
            "model.py",
            "data.py",
            "dimensionless.py",
            "render.py",
            "rim_resolution.py",
            "prepare_drivaerrim.py",
        )
    ]
    inputs = [
        prepared / "dataset.json",
        checkpoint,
        external,
        output / "pairs.npz",
        inventory,
    ]
    initial = initialize(ck, 8, 42)
    p = {
        "experiment": str(output.resolve()),
        "prepared": str(prepared.resolve()),
        "prepared512": str(prepared512.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "external": str(external.resolve()),
        "seed": 42,
        "epochs": 12,
        "pairs_per_epoch": 720,
        "batch_size": 8,
        "arms": ARMS,
        "anchors": ANCHORS,
        "delta_std": scale.tolist(),
        "parameter_mean": ck["parameter_mean"],
        "parameter_std": ck["parameter_std"],
        "case_ids": [c["case_id"] for c in cases],
        "train_ids": [cases[i]["case_id"] for i in indices],
        "normalization_source": "Frozen source checkpoint; no new held-out moments",
        "loss": "standardized delta MSE only; physical delta=output*delta_std; no offset",
        "selection": "Both validation target MAEs <= zero baseline; minimize mean(MAE/delta_std)",
        "cnn_lr": 1e-4,
        "head_lr": 1e-3,
        "clip_norm": 10,
        "optimizer": "Adam",
        "augmentation": "none",
        "cache_cases": {"256": len(cases), "512": 384},
        "inventory": str(inventory),
        "pair_trace_sha256": sha256(output / "pairs.npz"),
        "initial_tensor_sha256": tensor_sha256(initial),
        "parameter_count": sum(v.numel() for v in initial.parameters()),
        "input_sha256": {str(f.resolve()): sha256(f) for f in inputs},
        "code_sha256": {str(f.resolve()): sha256(f) for f in code},
    }
    write_json(output / "protocol.json", p)
    return p


def load(experiment):
    """@entry Validate the frozen protocol and return its ordered Rim cases."""
    experiment = Path(experiment)
    p = json.loads((experiment / "protocol.json").read_text())
    for path, expected in {**p["input_sha256"], **p["code_sha256"]}.items():
        if sha256(path) != expected:
            raise ValueError(f"Frozen input changed: {path}")
    data = json.loads((Path(p["prepared"]) / "dataset.json").read_text())
    by_id = {c["case_id"]: c for c in data["cases"]}
    return p, [by_id[i] for i in p["case_ids"]]


def lock512(p, cases):
    """Lock complete genuine512 membership and every compressed cache SHA."""
    root = Path(p["prepared512"])
    manifest = root / "dataset.json"
    lock_path = Path(p["experiment"]) / "data512-lock.json"
    if lock_path.exists():
        lock = json.loads(lock_path.read_text())
        if sha256(manifest) != lock["dataset_sha256"]:
            raise ValueError("Locked512 manifest changed")
        return json.loads(manifest.read_text()), sha256(lock_path)
    data = json.loads(manifest.read_text())
    preparation = data["resolution_preparation"]
    if not preparation["train_ready"] or preparation["inventory_sha256"] != sha256(
        p["inventory"]
    ):
        raise ValueError("Genuine512 data not complete or inventory differs")
    if (
        preparation["source_dataset_sha256"]
        != p["input_sha256"][str(Path(p["prepared"]) / "dataset.json")]
    ):
        raise ValueError("Genuine512 source dataset differs")
    for path, expected in preparation["code_sha256"].items():
        if p["code_sha256"].get(path) != expected:
            raise ValueError("Genuine512 renderer differs from frozen code")
    by_id = {c["case_id"]: c for c in data["cases"]}
    if len(by_id) != len(cases) or set(by_id) != set(p["case_ids"]):
        raise ValueError("Genuine512 membership differs")
    hashes = {}
    for original in cases:
        current = by_id[original["case_id"]]
        for key in ("split", "targets", "parameters", "geometry_hash", "source"):
            if current[key] != original[key]:
                raise ValueError(f"Genuine512 case field differs: {key}")
        file = root / current["views"]
        if sha256(file) != current["views_sha256"]:
            raise ValueError(f"Genuine512 compressed cache differs: {file}")
        hashes[current["case_id"]] = current["views_sha256"]
    write_json(
        lock_path,
        {
            "dataset_sha256": sha256(manifest),
            "cache_sha256": hashes,
            "inventory_sha256": preparation["inventory_sha256"],
        },
    )
    return data, sha256(lock_path)


class DeltaCases:
    """Own the bounded cache; retain original full-view arrays without copying."""

    def __init__(self, p, cases, resolution):
        self.cases, self.resolution = cases, resolution
        self.index = {c["case_id"]: i for i, c in enumerate(cases)}
        self.delta_std = np.asarray(p["delta_std"], np.float32)
        self.cache, self.verified = OrderedDict(), set()
        self.limit = p["cache_cases"][str(resolution)]
        self.data_lock_sha256 = None
        if resolution == 512:
            data, self.data_lock_sha256 = lock512(p, cases)
            mapping = {c["case_id"]: c for c in data["cases"]}
            self.records = [mapping[c["case_id"]] for c in cases]
            self.root = Path(p["prepared512"])
        else:
            self.records, self.root = cases, Path(p["prepared"])
        raw = np.asarray([c["parameters"] for c in cases], float)
        known = np.isfinite(raw)
        normalized = np.where(
            known,
            (raw - np.asarray(p["parameter_mean"])[:18])
            / np.asarray(p["parameter_std"])[:18],
            0,
        )
        self.parameters = torch.from_numpy(
            np.concatenate((normalized, known), axis=1).astype(np.float32)
        )
        self.targets = torch.from_numpy(
            np.asarray([c["targets"] for c in cases], np.float32)
        )

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, index):
        if index not in self.cache:
            record = self.records[index]
            path = self.root / record["views"]
            if index not in self.verified:
                if sha256(path) != record["views_sha256"]:
                    raise ValueError(f"View cache SHA differs: {path}")
                self.verified.add(index)
            with np.load(path) as saved:
                arrays = {n: saved[n] for n in VIEWS}
            validate_views(
                arrays,
                "five",
                {"ordinary_size": self.resolution, "bottom_size": self.resolution},
            )
            if any(v.dtype != np.float32 for v in arrays.values()):
                raise ValueError("Expected float32 caches")
            self.cache[index] = {n: torch.from_numpy(v) for n, v in arrays.items()}
            if len(self.cache) > self.limit:
                self.cache.popitem(last=False)
        self.cache.move_to_end(index)
        return self.cache[index], self.parameters[index], self.targets[index]


def dataset_for(p, cases, resolution):
    return DeltaCases(p, cases, resolution)


def batch(dataset, indices, device):
    rows = [dataset[int(i)] for i in indices]
    return (
        {n: torch.stack([r[0][n] for r in rows]).to(device) for n in VIEWS},
        torch.stack([r[1] for r in rows]).to(device),
        torch.stack([r[2] for r in rows]).to(device),
    )


def predict_anchors(model, dataset, cases, anchor_ids, split, device):
    """@entry Encode each validation query and shared train anchor only once."""
    queries = [
        i
        for i, c in enumerate(cases)
        if c["source"] == "drivaerrim" and c["split"] == split
    ]
    predictions = {a: {} for a in anchor_ids}
    model.eval()
    with torch.inference_mode():
        av, _, _ = batch(dataset, [dataset.index[a] for a in anchor_ids], device)
        anchors = model.encode(av)
        for start in range(0, len(queries), 8):
            indices = queries[start : start + 8]
            views, parameters, _ = batch(dataset, indices, device)
            encoded = model.encode(views)
            for k, anchor in enumerate(anchor_ids):
                reference = {
                    n: v[k : k + 1].expand(len(indices), -1, -1, -1)
                    for n, v in anchors.items()
                }
                delta = (
                    model.predict_pair(reference, encoded, parameters).cpu().numpy()
                    * dataset.delta_std
                )
                predictions[anchor].update(
                    {cases[i]["case_id"]: d.tolist() for i, d in zip(indices, delta)}
                )
    return predictions


def save_model(path, model, p, epoch, score, arm, data_lock):
    torch.save(
        {
            "model_state": {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            },
            "model_config": {
                "pool_size": p["arms"][arm]["pool_size"],
                "parameter_count": 36,
            },
            "epoch": epoch,
            "arm": arm,
            "delta_std": p["delta_std"],
            "parameter_mean": p["parameter_mean"],
            "parameter_std": p["parameter_std"],
            "protocol_sha256": sha256(Path(p["experiment"]) / "protocol.json"),
            "data512_lock_sha256": data_lock,
            "validation": score,
        },
        path,
    )


def train(experiment, arm):
    """@entry Train one frozen arm; retain epoch0 if no eligible epoch improves it."""
    p, cases = load(experiment)
    config = p["arms"][arm]
    output = Path(experiment) / arm
    if output.exists():
        raise FileExistsError("Arm output already exists")
    dataset = dataset_for(p, cases, config["resolution"])
    ck = torch.load(p["checkpoint"], map_location="cpu", weights_only=True)
    model = initialize(ck, config["pool_size"], p["seed"])
    parity = tensor_sha256(model)
    if parity != p["initial_tensor_sha256"]:
        raise ValueError("Initial arm tensor parity differs")
    output.mkdir()
    torch.set_num_threads(2)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    cnn, new = [], []
    for name, value in model.named_parameters():
        (cnn if name.startswith(("image_encoder.", "bottom_encoder.")) else new).append(
            value
        )
    optimizer = torch.optim.Adam(
        [{"params": cnn, "lr": p["cnn_lr"]}, {"params": new, "lr": p["head_lr"]}]
    )
    with np.load(Path(experiment) / "pairs.npz") as pair:
        left, right = pair["left"], pair["right"]
    zeros = {
        a: {c["case_id"]: [0.0, 0.0] for c in cases if c["split"] == "val"}
        for a in p["anchors"]
    }
    initial = score_anchors(cases, p["anchors"], zeros, "val", p["delta_std"])
    save_model(
        output / "epoch0.pt", model, p, 0, initial, arm, dataset.data_lock_sha256
    )
    save_model(output / "model.pt", model, p, 0, initial, arm, dataset.data_lock_sha256)
    report = {
        "arm": arm,
        "initial_tensor_sha256": parity,
        "epoch0": initial,
        "history": [],
        "selected_epoch": 0,
        "optimizer_steps": 0,
        "pair_trace_sha256": p["pair_trace_sha256"],
        "data512_lock_sha256": dataset.data_lock_sha256,
        "device": str(device),
    }
    best = initial["normalized_mae"]
    executed = hashlib.sha256()
    exposures = np.zeros(len(cases), dtype=np.int64)
    scale = torch.tensor(p["delta_std"], dtype=torch.float32, device=device)
    started = time.monotonic()
    for epoch, (aa, bb) in enumerate(zip(left, right), 1):
        model.train()
        losses, clipped = [], 0
        for start in range(0, len(aa), p["batch_size"]):
            a, b = (
                aa[start : start + p["batch_size"]],
                bb[start : start + p["batch_size"]],
            )
            executed.update(np.stack((a, b), axis=1).astype(np.int32).tobytes())
            exposures += np.bincount(np.concatenate((a, b)), minlength=len(cases))
            views, parameters, target = batch(dataset, np.concatenate((a, b)), device)
            optimizer.zero_grad(set_to_none=True)
            pooled = model.encode(views)
            prediction = model.predict_pair(
                {n: v[: len(a)] for n, v in pooled.items()},
                {n: v[len(a) :] for n, v in pooled.items()},
                parameters[: len(a)],
            )
            loss = (
                (prediction - (target[len(a) :] - target[: len(a)]) / scale)
                .square()
                .mean()
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite train loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), p["clip_norm"], error_if_nonfinite=True
            )
            clipped += int(norm > p["clip_norm"])
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            report["optimizer_steps"] += 1
        predictions = predict_anchors(
            model, dataset, cases, p["anchors"], "val", device
        )
        score = score_anchors(cases, p["anchors"], predictions, "val", p["delta_std"])
        eligible = all(
            score["mean_anchors"][k]["mae"] <= initial["mean_anchors"][k]["mae"]
            for k in ("Cd", "Cl")
        )
        if eligible and score["normalized_mae"] < best:
            best, report["selected_epoch"] = score["normalized_mae"], epoch
            save_model(
                output / "model.pt",
                model,
                p,
                epoch,
                score,
                arm,
                dataset.data_lock_sha256,
            )
        row = {
            "epoch": epoch,
            "training_loss": float(np.mean(losses)),
            "validation": score,
            "eligible": eligible,
            "gradient_clip_count": clipped,
            "elapsed_s": time.monotonic() - started,
        }
        report["history"].append(row)
        report["executed_pair_sha256"] = executed.hexdigest()
        report["endpoint_exposures"] = {
            cases[i]["case_id"]: int(n) for i, n in enumerate(exposures) if n > 0
        }
        write_json(output / "training.json", report)
        print(
            json.dumps(
                {
                    "arm": arm,
                    "epoch": epoch,
                    "loss": row["training_loss"],
                    "val": score["normalized_mae"],
                    "eligible": eligible,
                    "selected": report["selected_epoch"],
                    "elapsed_s": row["elapsed_s"],
                }
            ),
            flush=True,
        )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    f = sub.add_parser("freeze")
    for name in ("prepared", "prepared512", "checkpoint", "external", "output"):
        f.add_argument("--" + name, type=Path, required=True)
    t = sub.add_parser("train")
    t.add_argument("experiment", type=Path)
    t.add_argument("--arm", choices=ARMS, required=True)
    args = parser.parse_args()
    if args.action == "freeze":
        print(
            json.dumps(
                freeze(
                    args.prepared,
                    args.prepared512,
                    args.checkpoint,
                    args.external,
                    args.output,
                ),
                indent=2,
            )
        )
    else:
        train(args.experiment, args.arm)


if __name__ == "__main__":
    main()
