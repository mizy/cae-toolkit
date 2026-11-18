"""Full-pool coefficient supervision sharing the frozen Rim B spatial architecture."""

import argparse
import hashlib
import json
import math
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np
import torch
from aircraft_learning.scene_experiment import (
    normalization,
    sampling_weights,
    verify_data,
)
from torch import nn

from .delta_experiment import sha256, tensor_sha256, write_json
from .dimensionless import PARAMETERS
from .model import VIEWS, CoefficientModel
from .paired_delta import PairedDelta
from .sampling_experiment import load as load_sampling


class AbsoluteReadout(nn.Module):
    """A training auxiliary head consuming B's existing pooled CNN features."""

    def __init__(self):
        super().__init__()
        source = CoefficientModel(
            parameter_count=36,
            variant="five",
            size="small",
            encoder="pyramid_detail_attention",
            fusion="attention",
        )
        self.image_project = source.image_encoder.project
        self.bottom_project = source.bottom_encoder.project
        self.parameter_encoder = source.parameter_encoder
        self.view_attention = source.view_attention
        self.head = source.head

    def forward(self, pooled, parameters):
        body = self.image_project(torch.cat([pooled[n] for n in VIEWS[:4]]))
        features = [*body.chunk(4), self.bottom_project(pooled["bottom"])]
        condition = self.parameter_encoder(parameters)
        features, _ = self.view_attention(features, condition)
        return self.head(torch.cat((*features, condition), dim=1))


class FullCases:
    """Own the bounded full-pool image cache and its train-only normalization."""

    def __init__(self, p, cases):
        self.root, self.cases = Path(p["joint_prepared"]), cases
        self.index = {c["case_id"]: i for i, c in enumerate(cases)}
        self.cache = OrderedDict()
        stats = p["normalization"]
        raw = np.asarray([c["parameters"] for c in cases], float)
        known = np.isfinite(raw)
        values = np.where(
            known,
            (raw - np.asarray(stats["parameter_mean"][:18]))
            / np.asarray(stats["parameter_std"][:18]),
            0,
        )
        self.parameters = torch.tensor(
            np.concatenate((values, known.astype(float)), axis=1), dtype=torch.float32
        )
        self.targets = torch.tensor(
            np.asarray([c["targets"] for c in cases], float), dtype=torch.float32
        )
        self.standardized_targets = (
            self.targets - torch.tensor(stats["target_mean"])
        ) / torch.tensor(stats["target_std"])
        self.weights = torch.tensor(
            [p["source_case_weights"][c["source"]] for c in cases], dtype=torch.float32
        )

    def get(self, index):
        record = self.cases[int(index)]
        path = self.root / record["views"]
        if path not in self.cache:
            if sha256(path) != record["views_sha256"]:
                raise ValueError("Full-pool view checksum changed")
            with np.load(path, allow_pickle=False) as source:
                arrays = {n: source[n] for n in VIEWS}
            if any(
                a.shape != (2, 256, 256) or a.dtype != np.float32
                for a in arrays.values()
            ):
                raise ValueError("Expected native full-pool float32 256 views")
            self.cache[path] = {n: torch.from_numpy(a) for n, a in arrays.items()}
            if len(self.cache) > 512:
                self.cache.popitem(last=False)
        self.cache.move_to_end(path)
        return (
            self.cache[path],
            self.parameters[index],
            self.standardized_targets[index],
        )


def batch_cases(data, indices, device):
    rows = [data.get(int(i)) for i in indices]
    return (
        {n: torch.stack([r[0][n] for r in rows]).to(device) for n in VIEWS},
        torch.stack([r[1] for r in rows]).to(device),
        torch.stack([r[2] for r in rows]).to(device),
    )


def load_checkpoint(path, device="cpu"):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    model = PairedDelta(**ck["model_config"])
    model.load_state_dict(ck["model_state"], strict=True)
    readout = AbsoluteReadout()
    readout.load_state_dict(ck["absolute_state"], strict=True)
    return model.to(device), readout.to(device)


def load(root):
    p = json.loads((Path(root) / "protocol.json").read_text())
    for path, expected in {**p["source_sha256"], **p["code_sha256"]}.items():
        if sha256(path) != expected:
            raise ValueError(f"Frozen joint sampling source changed: {path}")
    cases = json.loads((Path(p["joint_prepared"]) / "dataset.json").read_text())[
        "cases"
    ]
    if [c["case_id"] for c in cases] != p["case_ids"]:
        raise ValueError("Joint record order changed")
    return p, cases


def freeze(prepared, sampling_parent, output, batch_size=16):
    """@entry Lock full-pool membership, real Rim pairs and all online offsets."""
    prepared, sampling_parent, output = map(Path, (prepared, sampling_parent, output))
    if output.exists():
        raise FileExistsError("Use a fresh experiment directory")
    data = json.loads((prepared / "dataset.json").read_text())
    if data["parameter_names"] != PARAMETERS:
        raise ValueError("Expected eighteen dimensionless values, with derived masks")
    verify_data(prepared, data)
    cases = data["cases"]
    train = [i for i, c in enumerate(cases) if c["split"] == "train"]
    if Counter(c["split"] for c in cases) != {"train": 4121, "val": 614, "test": 819}:
        raise ValueError("Expected the complete audited vehicle/aircraft pool")
    sp, rim_cases = load_sampling(sampling_parent)
    by_id = {c["case_id"]: c for c in cases}
    for record in rim_cases:
        original = by_id[record["case_id"]]
        if any(
            record[k] != original[k]
            for k in ("split", "targets", "parameters", "geometry_hash")
        ):
            raise ValueError("Rim records differ between full pool and frozen B")
    output.mkdir(parents=True)
    rng = np.random.default_rng(42)
    full_order = np.stack([rng.permutation(train) for _ in range(6)]).astype(np.int32)
    full_offsets = rng.uniform(-0.5, 0.5, (6, len(train), 2)).astype(np.float32)
    with np.load(sampling_parent / "draws.npz") as old:
        left, right = old["left"].copy(), old["right"].copy()
    rim_offsets = rng.uniform(-0.5, 0.5, (6, 720, 2, 2)).astype(np.float32)
    common = rng.random((6, 720)) < 0.5
    rim_offsets[:, :, 1] = np.where(
        common[..., None], rim_offsets[:, :, 0], rim_offsets[:, :, 1]
    )
    np.savez_compressed(
        output / "draws.npz",
        full_order=full_order,
        full_offsets=full_offsets,
        left=left,
        right=right,
        rim_offsets=rim_offsets,
        common=common,
    )
    train_cases = [cases[i] for i in train]
    weights = sampling_weights(train_cases).numpy() * len(train)
    model = PairedDelta(8, 36)
    model.load_state_dict(
        torch.load(sp["checkpoint"], map_location="cpu", weights_only=True)[
            "model_state"
        ]
    )
    p = {
        "experiment": str(output.resolve()),
        "joint_prepared": str(prepared.resolve()),
        "sampling_parent": str(sampling_parent.resolve()),
        "parent_experiment": sp["parent_experiment"],
        "checkpoint": sp["checkpoint"],
        "seed": 42,
        "epochs": 6,
        "batch_size": batch_size,
        "steps_per_epoch": math.ceil(len(train) / batch_size),
        "training_steps": 6 * math.ceil(len(train) / batch_size),
        "arms": ["control", "online"],
        "case_ids": [c["case_id"] for c in cases],
        "train_ids": [cases[i]["case_id"] for i in train],
        "source_counts": dict(Counter(c["source"] for c in train_cases)),
        "source_case_weights": {
            c["source"]: float(w) for c, w in zip(train_cases, weights)
        },
        "normalization": normalization(train_cases),
        "anchors": sp["anchors"],
        "delta_std": sp["delta_std"],
        "parameter_mean": sp["parameter_mean"],
        "parameter_std": sp["parameter_std"],
        "initial_tensor_sha256": tensor_sha256(model),
        "full_trace_sha256": hashlib.sha256(full_order.tobytes()).hexdigest(),
        "pair_trace_sha256": sp["pair_trace_sha256"],
        "cnn_lr": 5e-5,
        "head_lr": 3e-4,
        "clip_norm": 10,
        "warm_epochs": 20,
        "loss_weights": {
            "absolute_nominal": 0.25,
            "absolute_augmented": 0.25,
            "absolute_consistency": 0.1,
            "rim_nominal": 1.0,
            "rim_augmented": 0.5,
            "rim_self": 0.25,
            "rim_consistency": 0.25,
            "rim_teacher": 1.0,
        },
        "augmentation": "Online mask-normalized bilinear image translation, uniform[-0.5,+0.5] px per axis. Original images each time. Same offsets for five views; Rim common and independent endpoint shifts mixed 50/50.",
        "resolution": "Existing complete256 full-pool views; genuine512 for the original B Rim paired branch. No upsampling or whole-pool re-render.",
        "compute": "Identical records, pair draws, objectives and optimizer updates. Control reuses identical original-image CNN features within a step; online encodes both original and perturbed images. Logical exposure and actual CNN encoding counts are reported separately.",
        "supervision": "Every full-pool train record has masked real absolute coefficient supervision through an auxiliary head sharing B CNN; only matched-condition Rim pairs supervise B delta head. No arbitrary cross-source or aircraft-condition pair labels.",
        "warmup": "Fixed B CNN; train auxiliary head on train-only cached pooled features, then identical warm head for both arms. No validation/test selection in warmup.",
        "selection": "Validation-only joint sampling, true delta preservation, and per-source absolute guards. Failed new weights are diagnostic only; original B stays adopted unless all gates pass.",
    }
    files = [
        prepared / "dataset.json",
        sampling_parent / "protocol.json",
        sampling_parent / "teacher.npz",
        output / "draws.npz",
        Path(sp["checkpoint"]),
    ]
    p["source_sha256"] = {str(f.resolve()): sha256(f) for f in files}
    names = (
        "joint_sampling.py",
        "joint_sampling_train.py",
        "image_perturbation.py",
        "paired_delta.py",
        "model.py",
        "phase_metrics.py",
        "delta_metrics.py",
    )
    code = [Path(__file__).with_name(n) for n in names]
    code.extend(
        [
            Path(__file__).parents[3]
            / "predict-aircraft-coefficients/scripts/aircraft_learning/scene_experiment.py"
        ]
    )
    p["code_sha256"] = {str(f.resolve()): sha256(f) for f in code}
    protected = json.loads((sampling_parent / "preserved-artifacts.json").read_text())
    for directory in (sampling_parent, Path(sp["parent_experiment"])):
        for f in list(directory.glob("*.json")) + list(directory.glob("*/*.pt")):
            protected[str(f.resolve())] = sha256(f)
    write_json(output / "preserved-artifacts.json", protected)
    write_json(
        output / "data-lock.json",
        {
            "dataset_sha256": sha256(prepared / "dataset.json"),
            "view_sha256": {
                str((prepared / c["views"]).resolve()): c["views_sha256"] for c in cases
            },
        },
    )
    p["source_sha256"][str((output / "data-lock.json").resolve())] = sha256(
        output / "data-lock.json"
    )
    from .joint_sampling_train import warm_start

    warm_start(p, cases)
    initial = torch.load(output / "initial.pt", map_location="cpu", weights_only=True)
    p["initial_absolute_sha256"] = initial["absolute_tensor_sha256"]
    p["source_sha256"][str((output / "initial.pt").resolve())] = sha256(
        output / "initial.pt"
    )
    write_json(output / "protocol.json", p)
    return p


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared", type=Path)
    parser.add_argument("sampling_parent", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    print(
        json.dumps(
            freeze(args.prepared, args.sampling_parent, args.output, args.batch_size),
            indent=2,
        )
    )
