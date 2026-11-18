"""Conditional PointNet-basic adaptation for a matched vehicle diagnostic."""

import hashlib

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from .model import CoefficientModel
from .train import seed_everything


class PointRegressor(nn.Module):
    """PointNet-basic shared MLP + symmetric max pool, with the existing Cd/Cl head.

    Architecture reference: charlesq34/pointnet/models/pointnet_cls_basic.py (MIT).
    This is a small PyTorch regression adaptation, not a reproduction of an
    industrial paper's accuracy: channels32/32/64/96/256, projection160, fixed
    canonical axes, no learned rotation, and the same physical encoder/head as A.
    """

    def __init__(self, reference):
        super().__init__()
        self.parameter_count, self.target_count = 40, 2
        self.parameter_encoder, self.head = reference.parameter_encoder, reference.head
        layers = []
        for before, after in zip((3, 32, 32, 64, 96), (32, 32, 64, 96, 256)):
            layers.extend(
                (
                    nn.Conv1d(before, after, 1, bias=False),
                    nn.BatchNorm1d(after),
                    nn.ReLU(),
                )
            )
        self.point_encoder = nn.Sequential(*layers)
        self.project = nn.Sequential(nn.Linear(256, 160), nn.GELU())

    def forward(self, geometry, parameters, scene=None):
        features = self.point_encoder(geometry["points"]).amax(dim=2)
        return self.head(
            torch.cat(
                (self.project(features), self.parameter_encoder(parameters)), dim=1
            )
        )


def make_model(mode, seed=42):
    """Build diagnostic arms with identical compatible physical/head initialization."""
    seed_everything(seed)
    model = CoefficientModel(
        40,
        variant="parameters" if mode == "parameters" else "five",
        size="small",
        encoder="pyramid_detail_attention",
        fusion="concat" if mode == "parameters" else "attention",
    )
    return PointRegressor(model) if mode == "points" else model


def shared_initial_hash(model):
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        if key.startswith(("parameter_encoder.", "head.")):
            digest.update(key.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class PointCases(Dataset):
    """Read point inputs; preserve the existing SceneCases normalization/masks."""

    def __init__(self, root, cases, stats, point_manifest, read_points=True):
        self.root, self.cases, self.stats = root, cases, stats
        self.points, self.read_points, self.cache = point_manifest, read_points, {}

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, index):
        case = self.cases[index]
        geometry = {}
        if self.read_points:
            path = self.root / self.points[case["case_id"]]["path"]
            if path not in self.cache:
                with np.load(path, allow_pickle=False) as f:
                    self.cache[path] = torch.tensor(f["points"].T.copy())
            geometry["points"] = self.cache[path]
        raw = np.asarray(case["parameters"], dtype=float)
        known = np.isfinite(raw)
        mean, std = (
            np.asarray(self.stats[k][: len(raw)])
            for k in ("parameter_mean", "parameter_std")
        )
        x = np.concatenate(
            (np.where(known, (raw - mean) / std, 0), known.astype(float))
        )
        y = (
            np.asarray(case["targets"], dtype=float) - self.stats["target_mean"]
        ) / self.stats["target_std"]
        return (
            geometry,
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
            torch.tensor(case["scene"]),
        )
