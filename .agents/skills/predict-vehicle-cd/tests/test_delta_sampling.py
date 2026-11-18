import sys
from pathlib import Path

import numpy as np
import pyvista as pv

SKILL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL / "scripts"))

from vehicle_cd_learning.delta_sampling import shifted_views
from vehicle_cd_learning.rim_resolution import render_component


def test_camera_phase_preserves_depth_and_zero_phase_replays_original_renderer():
    mesh = pv.Plane(
        center=(0, 0, 0.3),
        direction=(0, 0, 1),
        i_size=0.7,
        j_size=0.7,
        i_resolution=2,
        j_resolution=2,
    ).triangulate()
    original = render_component(mesh, np.zeros(3), 1, 64)
    zero = shifted_views(mesh, np.zeros(3), 1, 64, 0)
    moved = shifted_views(mesh, np.zeros(3), 1, 64, 1)
    for name in original:
        np.testing.assert_array_equal(original[name].depth, zero[name].depth)
        np.testing.assert_array_equal(original[name].mask, zero[name].mask)
    assert np.any(original["top"].mask != moved["top"].mask)
    np.testing.assert_allclose(
        moved["top"].depth[moved["top"].mask], (1.3 - 0.3) / 1.3, atol=1e-6
    )
