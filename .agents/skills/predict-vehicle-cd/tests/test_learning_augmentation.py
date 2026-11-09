from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from vehicle_cd_learning.data import VIEWS, jitter_views, validate_views
from vehicle_cd_learning.model import CoefficientModel


def test_sampling_jitter_preserves_masks_inputs_and_opposite_depth_relation():
    views = {}
    for name in VIEWS:
        size = 64 if name == "bottom" else 32
        image = torch.zeros(2, 2, size, size)
        image[:, 0, size // 4 : 3 * size // 4, size // 4 : 3 * size // 4] = 0.5
        image[:, 1, size // 4 : 3 * size // 4, size // 4 : 3 * size // 4] = 1
        views[name] = image
    before = {name: value.clone() for name, value in views.items()}
    first = jitter_views(views, torch.Generator().manual_seed(21))
    repeat = jitter_views(views, torch.Generator().manual_seed(21))
    different = jitter_views(views, torch.Generator().manual_seed(22))
    for name in VIEWS:
        assert torch.equal(before[name], views[name])
        assert torch.equal(first[name], repeat[name])
        assert not torch.equal(first[name], different[name])
    for index in range(2):
        validate_views(
            {name: value[index].numpy() for name, value in first.items()},
            "five",
            {"ordinary_size": 32, "bottom_size": 64},
        )
    for a, b in (("front", "rear"), ("top", "bottom")):
        aa = first[a][:, 0, first[a].shape[-1] // 2, first[a].shape[-1] // 2]
        bb = first[b][:, 0, first[b].shape[-1] // 2, first[b].shape[-1] // 2]
        np.testing.assert_allclose((aa + bb).numpy(), 1, atol=5e-4, rtol=0)


def test_small_model_has_fewer_parameters_and_finite_gradients():
    torch.set_num_threads(2)
    standard = CoefficientModel(5, target_count=1)
    small = CoefficientModel(5, target_count=1, size="small")
    assert sum(p.numel() for p in small.parameters()) < 0.1 * sum(
        p.numel() for p in standard.parameters()
    )
    views = {name: torch.rand(2, 2, 32, 32) for name in VIEWS}
    prediction = small(views, torch.rand(2, 5))
    assert prediction.shape == (2, 1)
    prediction.square().mean().backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in small.parameters()
    )
