import sys
from pathlib import Path

import numpy as np
import pytest
import torch

SKILL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL / "scripts"))
sys.path.insert(0, str(SKILL.parent / "predict-aircraft-coefficients/scripts"))

from vehicle_cd_learning.model import VIEWS
from vehicle_cd_learning.paired_delta import PairedDelta
from vehicle_cd_learning.phase_decoder_scores import pair_grid, patterns
from vehicle_cd_learning.phase_decoder_train import constrain
from vehicle_cd_learning.subpixel_sampling import (
    acquisition_offset,
    area_average,
    capture,
    resolve,
)


def test_single_raster_has_ambiguous_subpixel_geometry_and_extra_rays_recover_it():
    # Both meshes share a large backing surface. One has a thin nearer triangle.
    vertices = np.array(
        [
            [2, 2, 0.8],
            [29, 2, 0.8],
            [29, 29, 0.8],
            [2, 29, 0.8],
            [12.39, 12.39, 0.2],
            [12.62, 12.39, 0.2],
            [12.5, 12.64, 0.2],
        ],
        dtype=np.float32,
    )
    body = np.array([[0, 1, 2], [0, 2, 3]], np.int32)
    edited = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6]], np.int32)

    def image(faces, p):
        return resolve(capture(vertices, faces, 32, 1, p, (0, 0)), 32, 1)

    np.testing.assert_array_equal(image(body, (0.25, 0)), image(edited, (0.25, 0)))
    nominal = image(edited, (0, 0))
    assert (
        nominal[0, 12, 12] == np.float32(0.2) and image(body, (0, 0))[0, 12, 12] > 0.79
    )
    for p in [(0.25, 0), (0.173, -0.317), (-0.362, 0.119), (0.41, 0.37), (-0.5, 0.5)]:
        residual = acquisition_offset(p, 4)
        dense = capture(vertices, edited, 32, 4, p, residual)
        np.testing.assert_array_equal(
            resolve(dense, 32, 4, np.asarray(p) - residual), nominal
        )
        sparse = capture(vertices, edited, 32, 1, p, p)
        np.testing.assert_array_equal(resolve(sparse, 32, 1), nominal)
    with pytest.raises(ValueError):
        acquisition_offset((np.nan, 0), 4)


def test_area_resolve_uses_visible_depth_and_fractional_coverage():
    vertices = np.array(
        [[5.5, 6.5, 0.4], [15.5, 6.5, 0.4], [5.5, 17.5, 0.4]], np.float32
    )
    fine = capture(vertices, np.array([[0, 1, 2]], np.int32), 32, 4, (0, 0), (0, 0))
    depth, coverage = area_average(fine, 32, 4)
    assert np.any((coverage > 0) & (coverage < 1))
    np.testing.assert_allclose(depth[coverage > 0], 0.4, rtol=0, atol=2e-7)
    assert not depth[coverage == 0].any()


def test_pair_grid_reproduces_original_nonlinear_head_with_independent_shifted_endpoints():
    torch.manual_seed(42)
    model = PairedDelta(8, 36).eval()
    ids = ["a", "b", "c"]
    bank = {n: torch.randn(3, 3, 16, 8, 8) for n in VIEWS}
    bank.update({n + "_mask": torch.rand(3, 3, 1, 8, 8) > 0.25 for n in VIEWS})
    bank = {k: v.float() for k, v in bank.items()}
    grid = patterns(3, 4)[::4]
    pairs = [("a", "b"), ("b", "a"), ("c", "c")]
    param = torch.randn(36)
    with torch.inference_mode():
        values = pair_grid(model, bank, ids, pairs, grid, param, [0.01, 0.03])
        for i, (a, b) in enumerate(grid):
            left = {
                k: bank[k][[0, 1, 2], s]
                for n, s in zip(VIEWS, a)
                for k in (n, n + "_mask")
            }
            right = {
                k: bank[k][[1, 0, 2], s]
                for n, s in zip(VIEWS, b)
                for k in (n, n + "_mask")
            }
            expected = model.predict_pair(left, right, param.expand(3, -1)).numpy() * [
                0.01,
                0.03,
            ]
            np.testing.assert_allclose(values[i], expected, atol=1e-7, rtol=1e-6)


def test_module_bound_applies_to_relative_motion_and_keeps_frozen_outputs():
    import copy

    model = PairedDelta(8, 36)
    start = copy.deepcopy(model)
    with torch.no_grad():
        for module in [model.adapter, model.head[0]]:
            for weight in module.parameters():
                weight.add_(torch.randn_like(weight) * 5)
    constrain(model, start, 0.10)
    for module, initial in [
        (model.adapter, start.adapter),
        (model.head[0], start.head[0]),
    ]:
        shift = torch.cat(
            [
                (a - b).flatten()
                for a, b in zip(module.parameters(), initial.parameters())
            ]
        )
        scale = torch.cat([v.flatten() for v in initial.parameters()])
        assert float((shift.norm() / scale.norm()).detach()) == pytest.approx(
            0.1, abs=2e-6
        )
    assert torch.equal(model.head[-1].weight, start.head[-1].weight)
