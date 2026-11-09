"""A body without tires must retain its declared distance from the ground."""

import sys
from pathlib import Path

import numpy as np
import pytest
import pyvista as pv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from vehicle_cd_learning.render import (
    canonical_points,
    render_five_views,
    render_geometry,
)


def test_ground_clearance_preserves_dimensions_and_changes_complete_views(tmp_path):
    mesh = pv.Box(bounds=(-0.5, 0.5, -0.2, 0.2, 0.05, 0.35)).triangulate()
    options = {"unit_scale_to_m": 1.0, "ordinary_size": 64, "bottom_size": 64}
    base, original = render_five_views(mesh, **options)
    lifted, dimensions = render_five_views(mesh, ground_clearance_m=0.05, **options)
    assert dimensions["height_m"] == original["height_m"]
    assert dimensions["ground_clearance_m"] == 0.05
    assert not np.array_equal(base["right"].mask, lifted["right"].mask)
    overlap = base["bottom"].mask & lifted["bottom"].mask
    np.testing.assert_allclose(
        lifted["bottom"].depth[overlap] - base["bottom"].depth[overlap],
        0.05 / 1.3,
        atol=1e-7,
    )
    path = tmp_path / "body.vtp"
    mesh.save(path)
    fresh, _ = render_geometry(
        path,
        {
            "unit_scale_to_m": 1,
            "front_axis": "-x",
            "up_axis": "+z",
            "ground_clearance_m": 0.05,
        },
        {"ordinary_size": 64, "bottom_size": 64, "window_l": 1.3},
    )
    np.testing.assert_array_equal(fresh["right"].depth, lifted["right"].depth)


def test_explicit_clearance_is_unit_and_translation_invariant():
    points = np.array([[-0.5, -0.2, 0.05], [0.5, 0.2, 0.35]])
    actual, _ = canonical_points(points, unit_scale_to_m=1, ground_clearance_m=0.05)
    other, _ = canonical_points(
        points * 1000 + 2500, unit_scale_to_m=0.001, ground_clearance_m=0.05
    )
    np.testing.assert_array_equal(actual, other)
    assert actual[:, 2].min() == pytest.approx(0.05)
    for value in (-0.01, np.nan, np.inf):
        with pytest.raises(ValueError, match="ground_clearance_m"):
            canonical_points(points, unit_scale_to_m=1, ground_clearance_m=value)
