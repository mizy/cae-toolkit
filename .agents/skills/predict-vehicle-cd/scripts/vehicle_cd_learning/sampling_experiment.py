"""Frozen, equal-compute sampling-consistency training from the original B weights."""

import argparse
import hashlib
import json
import time
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from . import delta_experiment as delta
from .data import validate_views
from .delta_metrics import score_anchors
from .model import VIEWS
from .phase_views import TRAIN_PHASES, phase_id
from .stable_delta import initialize_from_checkpoint

ARMS = {
    "control": {"coverage_floor": 0.0, "phases": False, "filter_input": False},
    "consistency": {"coverage_floor": 0.0, "phases": True, "filter_input": False},
    "coverage": {"coverage_floor": 1 / 16, "phases": True, "filter_input": True},
}
sha256, write_json = delta.sha256, delta.write_json


def trace_sha(*arrays):
    return hashlib.sha256(
        np.stack(arrays, axis=-1).astype(np.int32).tobytes()
    ).hexdigest()


def auxiliary_schedule(indices, epochs=6, steps=90, seed=42):
    rng = np.random.default_rng(seed)
    pairs = np.array(
        [rng.choice(indices, 2, replace=False) for _ in range(epochs * steps)]
    ).reshape(epochs, steps, 2)
    phases = rng.integers(len(TRAIN_PHASES), size=(epochs, steps))
    return pairs[..., 0], pairs[..., 1], phases


def teacher_predictions(parent, cases, checkpoint, left, right, output):
    """Encode the 720 TRAIN IDs once; use their cached features for frozen pairs."""
    dataset = delta.DeltaCases(parent, cases, 512)
    dataset.limit = 256
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = initialize_from_checkpoint(checkpoint, 0).eval().to(device)
    indices = [i for i, c in enumerate(cases) if c["split"] == "train"]
    if set(np.concatenate((left.ravel(), right.ravel())).tolist()) != set(indices):
        raise ValueError("Teacher schedule must use all and only training IDs")
    cache = {}
    with torch.inference_mode():
        for start in range(0, len(indices), 8):
            current = indices[start : start + 8]
            views, _, _ = delta.batch(dataset, current, device)
            encoded = model.encode(views)
            cache.update(
                {
                    i: {n: value[k].cpu() for n, value in encoded.items()}
                    for k, i in enumerate(current)
                }
            )
        predictions = []
        for aa, bb in zip(left, right):
            rows = []
            for start in range(0, len(aa), 64):
                a, b = aa[start : start + 64], bb[start : start + 64]
                pa = {
                    n: torch.stack([cache[int(i)][n] for i in a]).to(device)
                    for n in cache[int(a[0])]
                }
                pb = {
                    n: torch.stack([cache[int(i)][n] for i in b]).to(device)
                    for n in cache[int(b[0])]
                }
                rows.append(
                    model.predict_pair(pa, pb, dataset.parameters[a].to(device))
                    .cpu()
                    .numpy()
                )
            predictions.append(np.concatenate(rows))
    prediction = np.stack(predictions).astype(np.float32)
    if not np.isfinite(prediction).all():
        raise FloatingPointError("Nonfinite teacher predictions")
    np.savez_compressed(
        output, prediction=prediction, encoded_train_indices=np.array(indices, np.int32)
    )
    return {
        "encoded_case_ids": [cases[i]["case_id"] for i in indices],
        "encoded_geometry_count": len(indices),
        "device": str(device),
        "scope": "Original B standardized predictions on training pairs only; no held-out image encoding",
    }


def freeze(parent_factorial, phase_root, output):
    """@entry Freeze draws, original B teacher targets and all training sources."""
    parent_factorial, phase_root, output = map(
        Path, (parent_factorial, phase_root, output)
    )
    if output.exists():
        raise FileExistsError("Use a fresh experiment directory")
    parent, cases = delta.load(parent_factorial)
    phase_plan = json.loads((phase_root / "plan.json").read_text())
    if (
        phase_plan["smoke_training_count"] is not None
        or phase_plan["full_item_count"] != 358
    ):
        raise ValueError("Full phase preparation plan required")
    selected = phase_plan["selection"]["train"]
    by_id = {c["case_id"]: i for i, c in enumerate(cases)}
    if (
        len(selected) != 32
        or set(selected) & set(parent["anchors"])
        or any(cases[by_id[i]]["split"] != "train" for i in selected)
    ):
        raise ValueError("Auxiliary draws require the 32 non-anchor training IDs")
    checkpoint_path = parent_factorial / "B/model.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (
        checkpoint["model_config"] != {"pool_size": 8, "parameter_count": 36}
        or checkpoint["delta_std"] != parent["delta_std"]
    ):
        raise ValueError("Expected the unchanged original B model contract")
    with np.load(parent_factorial / "pairs.npz", allow_pickle=False) as pairs:
        left, right = pairs["left"][:6].copy(), pairs["right"][:6].copy()
    if left.shape != (6, 720) or right.shape != left.shape:
        raise ValueError("Expected six inherited 720-pair epochs")
    u, v, phases = auxiliary_schedule([by_id[i] for i in selected])
    output.mkdir(parents=True)
    np.savez_compressed(
        output / "draws.npz",
        left=left,
        right=right,
        aux_left=u,
        aux_right=v,
        aux_phase=phases,
    )
    torch.set_num_threads(2)
    started = time.monotonic()
    teacher = teacher_predictions(
        parent, cases, checkpoint, left, right, output / "teacher.npz"
    )
    teacher["seconds"] = time.monotonic() - started
    files = [
        parent_factorial / "protocol.json",
        parent_factorial / "pairs.npz",
        checkpoint_path,
        phase_root / "plan.json",
        output / "draws.npz",
        output / "teacher.npz",
    ]
    code = [Path(__file__)] + [
        Path(__file__).with_name(n)
        for n in (
            "stable_delta.py",
            "phase_views.py",
            "delta_experiment.py",
            "delta_metrics.py",
            "paired_delta.py",
            "model.py",
            "data.py",
            "dimensionless.py",
        )
    ]
    p = {
        "experiment": str(output.resolve()),
        "parent_experiment": str(parent_factorial.resolve()),
        "phase_root": str(phase_root.resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
        "seed": 42,
        "epochs": 6,
        "pairs_per_epoch": 720,
        "batch_size": 8,
        "cnn_lr": 5e-5,
        "head_lr": 3e-4,
        "clip_norm": 10,
        "arms": ARMS,
        "anchors": parent["anchors"],
        "case_ids": parent["case_ids"],
        "aux_train_ids": selected,
        "phase_ids": [phase_id(p) for p in TRAIN_PHASES],
        "delta_std": parent["delta_std"],
        "parameter_mean": parent["parameter_mean"],
        "parameter_std": parent["parameter_std"],
        "pair_trace_sha256": trace_sha(left, right),
        "aux_trace_sha256": trace_sha(u, v, phases),
        "initial_tensor_sha256": delta.tensor_sha256(
            initialize_from_checkpoint(checkpoint, 0)
        ),
        "source_sha256": {str(p.resolve()): sha256(p) for p in files},
        "code_sha256": {str(p.resolve()): sha256(p) for p in code},
        "teacher": teacher,
        "loss_weights": {
            "real": 1.0,
            "aux_phase": 0.5,
            "self": 0.5,
            "consistency": 0.5,
            "teacher": 1.0,
        },
        "selection": "No checkpoint selection during training; retain epochs00..06 for independent phase validation",
        "compute": "20 complete five-view images per step in all arms; control duplicates nominal auxiliary images",
        "auxiliary_supervision": "Mean losses over phaseu->phasev, nomu->phasev and phaseu->nomv; same CFD delta and nominal prediction consistency",
    }
    write_json(output / "protocol.json", p)
    return p


def load(experiment):
    p = json.loads((Path(experiment) / "protocol.json").read_text())
    for path, expected in {**p["source_sha256"], **p["code_sha256"]}.items():
        if sha256(path) != expected:
            raise ValueError(f"Frozen sampling source changed: {path}")
    _, cases = delta.load(p["parent_experiment"])
    if [c["case_id"] for c in cases] != p["case_ids"]:
        raise ValueError("Inherited case order changed")
    return p, cases


class SamplingCases:
    """Own bounded nominal/phase caches; camera phase never enters numeric inputs."""

    def __init__(self, p, cases, with_phases=True):
        parent, _ = delta.load(p["parent_experiment"])
        self.nominal = delta.DeltaCases(parent, cases, 512)
        self.nominal.limit = 256
        self.index = self.nominal.index
        self.parameters, self.targets = self.nominal.parameters, self.nominal.targets
        self.delta_std, self.cases = self.nominal.delta_std, cases
        self.root, self.records, self.cache = Path(p["phase_root"]), {}, OrderedDict()
        self.data_lock_sha256 = None
        if with_phases:
            self.bind_phases(p)

    def bind_phases(self, p):
        manifest_path, dataset_path = (
            self.root / "manifest.json",
            self.root / "dataset.json",
        )
        manifest, dataset = (
            json.loads(manifest_path.read_text()),
            json.loads(dataset_path.read_text()),
        )
        if (
            not manifest["stage_complete"]
            or not manifest["train_ready"]
            or manifest["item_count"] != 358
            or sha256(dataset_path) != manifest["dataset_sha256"]
        ):
            raise ValueError("Complete locked phase dataset required")
        if manifest["plan_sha256"] != p["source_sha256"][str(self.root / "plan.json")]:
            raise ValueError("Phase plan differs from frozen training plan")
        for path, expected in {
            **manifest["source_sha256"],
            **manifest["code_sha256"],
        }.items():
            if sha256(path) != expected:
                raise ValueError(f"Phase preparation source changed: {path}")
        if len(dataset["cases"]) != 358 or dataset["phase_preparation"] != {
            k: v for k, v in manifest.items() if k != "dataset_sha256"
        }:
            raise ValueError("Phase dataset and completion manifest disagree")
        plan = json.loads((self.root / "plan.json").read_text())
        expected = {(r["source_case_id"], r["phase_id"]): r for r in plan["items"]}
        hashes = {}
        for record in dataset["cases"]:
            item = expected[(record["source_case_id"], record["phase_id"])]
            if any(record[k] != item[k] for k in item):
                raise ValueError("Phase record differs from frozen plan")
            identity = record["source_case_id"]
            nominal = self.nominal.records[self.index[identity]]
            for key in (
                "split",
                "targets",
                "parameters",
                "geometry_hash",
                "geometry_id",
            ):
                if record[key] != nominal[key]:
                    raise ValueError(f"Phase metadata changed: {identity}: {key}")
            if record["nominal_views_sha256"] != nominal["views_sha256"]:
                raise ValueError("Phase nominal source changed")
            key = (self.index[identity], record["phase_id"])
            if key in self.records:
                raise ValueError("Repeated case-phase")
            path = self.root / record["views"]
            hashes[record["views"]] = sha256(path)
            if hashes[record["views"]] != record["views_sha256"]:
                raise ValueError("Phase compressed cache changed")
            self.records[key] = record
        for identity in p["aux_train_ids"]:
            for phase in p["phase_ids"]:
                record = self.records[(self.index[identity], phase)]
                if record["split"] != "train" or record["role"] != "train":
                    raise ValueError("Auxiliary phase source must have training role")
        lock = {
            "dataset_sha256": sha256(dataset_path),
            "manifest_sha256": sha256(manifest_path),
            "plan_sha256": manifest["plan_sha256"],
            "source_sha256": manifest["source_sha256"],
            "code_sha256": manifest["code_sha256"],
            "cache_sha256": hashes,
        }
        path = Path(p["experiment"]) / "data-lock.json"
        if path.exists() and json.loads(path.read_text()) != lock:
            raise ValueError("Phase data lock changed")
        if not path.exists():
            write_json(path, lock)
        self.data_lock_sha256 = sha256(path)

    def get(self, index, phase_id=None):
        index = int(index)
        if phase_id is None:
            return self.nominal[index]
        key = (index, phase_id)
        if key not in self.cache:
            path = self.root / self.records[key]["views"]
            if sha256(path) != self.records[key]["views_sha256"]:
                raise ValueError("Phase cache changed after locking")
            with np.load(path, allow_pickle=False) as stored:
                arrays = {n: stored[n] for n in VIEWS}
            validate_views(arrays, "five", {"ordinary_size": 512, "bottom_size": 512})
            if any(a.dtype != np.float32 for a in arrays.values()):
                raise ValueError("Expected float32 phase caches")
            self.cache[key] = {n: torch.from_numpy(a) for n, a in arrays.items()}
            if len(self.cache) > 64:
                self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        return self.cache[key], self.parameters[index], self.targets[index]


def batch_views(data, requests, device):
    rows = [data.get(int(i), phase) for i, phase in requests]
    return (
        {n: torch.stack([r[0][n] for r in rows]).to(device) for n in VIEWS},
        torch.stack([r[1] for r in rows]).to(device),
        torch.stack([r[2] for r in rows]).to(device),
    )


def train_objective(model, encoded, parameters, targets, scale, teacher, real_batch=8):
    """Supervise common/mixed auxiliary phases and nominal-to-phase self pairs."""
    n = real_batch

    def predict(a, b, params):
        return model.predict_pair(
            {k: v[a] for k, v in encoded.items()},
            {k: v[b] for k, v in encoded.items()},
            parameters[params],
        )

    real = predict(slice(0, n), slice(n, 2 * n), slice(0, n))
    nominal = predict(
        slice(2 * n, 2 * n + 1), slice(2 * n + 1, 2 * n + 2), slice(2 * n, 2 * n + 1)
    )
    phased = predict(
        [2 * n + 2, 2 * n, 2 * n + 2], [2 * n + 3, 2 * n + 3, 2 * n + 1], [2 * n] * 3
    )
    self_pair = predict(
        slice(2 * n, 2 * n + 2), slice(2 * n + 2, 2 * n + 4), slice(2 * n, 2 * n + 2)
    )
    losses = {
        "real": F.mse_loss(real, (targets[n : 2 * n] - targets[:n]) / scale),
        "aux_phase": F.mse_loss(
            phased,
            (
                (targets[2 * n + 1 : 2 * n + 2] - targets[2 * n : 2 * n + 1]) / scale
            ).expand_as(phased),
        ),
        "self": self_pair.square().mean(),
        "consistency": (phased - nominal.expand_as(phased)).square().mean(),
        "teacher": F.mse_loss(real, teacher),
    }
    total = (
        losses["real"]
        + 0.5 * (losses["aux_phase"] + losses["self"] + losses["consistency"])
        + losses["teacher"]
    )
    return total, losses


def save_checkpoint(path, model, p, arm, epoch, score, data):
    torch.save(
        {
            "model_state": {
                n: v.detach().cpu().clone() for n, v in model.state_dict().items()
            },
            "model_config": {"pool_size": 8, "parameter_count": 36},
            "coverage_floor": p["arms"][arm]["coverage_floor"],
            "filter_input": p["arms"][arm]["filter_input"],
            "arm": arm,
            "epoch": epoch,
            "delta_std": p["delta_std"],
            "parameter_mean": p["parameter_mean"],
            "parameter_std": p["parameter_std"],
            "source_sha256": p["source_sha256"],
            "protocol_sha256": sha256(Path(p["experiment"]) / "protocol.json"),
            "data_lock_sha256": data.data_lock_sha256,
            "data512_lock_sha256": data.nominal.data_lock_sha256,
            "validation": score,
        },
        path,
    )


def train(experiment, arm):
    """@entry Retain every epoch; independent validation owns subsequent selection."""
    p, cases = load(experiment)
    output = Path(experiment) / arm
    if output.exists():
        raise FileExistsError("Arm output already exists")
    data = SamplingCases(p, cases, p["arms"][arm]["phases"])
    torch.set_num_threads(2)
    torch.manual_seed(p["seed"])
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ck = torch.load(p["checkpoint"], map_location="cpu", weights_only=True)
    model = initialize_from_checkpoint(
        ck, p["arms"][arm]["coverage_floor"], p["arms"][arm]["filter_input"]
    ).to(device)
    if delta.tensor_sha256(model) != p["initial_tensor_sha256"]:
        raise ValueError("Arm initialization weights differ from original B")
    cnn, readout = [], []
    for name, value in model.named_parameters():
        (
            cnn if name.startswith(("image_encoder.", "bottom_encoder.")) else readout
        ).append(value)
    optimizer = torch.optim.Adam(
        [{"params": cnn, "lr": p["cnn_lr"]}, {"params": readout, "lr": p["head_lr"]}]
    )
    with np.load(Path(experiment) / "draws.npz", allow_pickle=False) as draws:
        left, right, uu, vv, phases = (
            draws[n].copy()
            for n in ("left", "right", "aux_left", "aux_right", "aux_phase")
        )
    with np.load(Path(experiment) / "teacher.npz", allow_pickle=False) as teacher:
        teachers = teacher["prediction"].copy()
    output.mkdir()
    initial = score_anchors(
        cases,
        p["anchors"],
        delta.predict_anchors(model, data.nominal, cases, p["anchors"], "val", device),
        "val",
        p["delta_std"],
    )
    save_checkpoint(output / "epoch-00.pt", model, p, arm, 0, initial, data)
    report = {
        "arm": arm,
        "history": [],
        "epoch0": initial,
        "optimizer_steps": 0,
        "device": str(device),
        "initial_tensor_sha256": p["initial_tensor_sha256"],
        "data_lock_sha256": data.data_lock_sha256,
        "pair_trace_sha256": p["pair_trace_sha256"],
        "aux_trace_sha256": p["aux_trace_sha256"],
        "selection": p["selection"],
    }
    real_trace, aux_trace = hashlib.sha256(), hashlib.sha256()
    endpoints, phase_exposures = Counter(), Counter()
    scale = torch.tensor(p["delta_std"], dtype=torch.float32, device=device)
    started = time.monotonic()
    for epoch, (aa, bb) in enumerate(zip(left, right), 1):
        model.train()
        losses, clipped = [], 0
        for step, start in enumerate(range(0, len(aa), p["batch_size"])):
            a, b = aa[start : start + 8], bb[start : start + 8]
            u, v, phase = (
                int(uu[epoch - 1, step]),
                int(vv[epoch - 1, step]),
                int(phases[epoch - 1, step]),
            )
            selected_phase = p["phase_ids"][phase] if p["arms"][arm]["phases"] else None
            requests = [(int(i), None) for i in np.concatenate((a, b))] + [
                (u, None),
                (v, None),
                (u, selected_phase),
                (v, selected_phase),
            ]
            real_trace.update(np.stack((a, b), axis=-1).astype(np.int32).tobytes())
            aux_trace.update(np.array([u, v, phase], np.int32).tobytes())
            endpoints.update(cases[i]["case_id"] for i, _ in requests)
            phase_exposures.update(
                cases[i]["case_id"]
                for i, phase_name in requests
                if phase_name is not None
            )
            views, parameters, targets = batch_views(data, requests, device)
            optimizer.zero_grad(set_to_none=True)
            total, parts = train_objective(
                model,
                model.encode(views),
                parameters,
                targets,
                scale,
                torch.from_numpy(teachers[epoch - 1, start : start + 8]).to(device),
            )
            if not torch.isfinite(total):
                raise FloatingPointError("Nonfinite sampling training loss")
            total.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), p["clip_norm"], error_if_nonfinite=True
            )
            clipped += int(norm > p["clip_norm"])
            optimizer.step()
            losses.append(
                {
                    "total": float(total.detach().cpu()),
                    **{k: float(v.detach().cpu()) for k, v in parts.items()},
                }
            )
            report["optimizer_steps"] += 1
        score = score_anchors(
            cases,
            p["anchors"],
            delta.predict_anchors(
                model, data.nominal, cases, p["anchors"], "val", device
            ),
            "val",
            p["delta_std"],
        )
        save_checkpoint(
            output / f"epoch-{epoch:02d}.pt", model, p, arm, epoch, score, data
        )
        report["history"].append(
            {
                "epoch": epoch,
                "loss_components": {
                    k: float(np.mean([r[k] for r in losses])) for k in losses[0]
                },
                "validation": score,
                "gradient_clip_count": clipped,
                "elapsed_s": time.monotonic() - started,
            }
        )
        report.update(
            {
                "executed_pair_sha256": real_trace.hexdigest(),
                "executed_aux_sha256": aux_trace.hexdigest(),
                "endpoint_exposures": dict(endpoints),
                "phase_geometry_exposures": dict(phase_exposures),
            }
        )
        write_json(output / "training.json", report)
        print(json.dumps({"arm": arm, **report["history"][-1]}), flush=True)
    if (
        report["executed_pair_sha256"] != p["pair_trace_sha256"]
        or report["executed_aux_sha256"] != p["aux_trace_sha256"]
    ):
        raise ValueError("Actual execution trace differs from frozen draws")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    freeze_parser = sub.add_parser("freeze")
    for name in ("parent_factorial", "phase_root", "output"):
        freeze_parser.add_argument(name, type=Path)
    train_parser = sub.add_parser("train")
    train_parser.add_argument("experiment", type=Path)
    train_parser.add_argument("--arm", choices=ARMS, required=True)
    args = parser.parse_args()
    if args.action == "freeze":
        freeze(args.parent_factorial, args.phase_root, args.output)
    else:
        train(args.experiment, args.arm)


if __name__ == "__main__":
    main()
