import sys
from pathlib import Path

import numpy as np

SKILL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL / "scripts"))
sys.path.insert(0, str(SKILL.parent / "predict-aircraft-coefficients/scripts"))

from vehicle_cd_learning.camera_tolerance import camera_points
from vehicle_cd_learning.render import VIEW_AXES


def test_camera_turn_centres_a_landmark_on_the_new_line_of_sight():
    pivot = np.array([0.0, 0.0, 0.15])
    points = np.array([[0.2, -0.3, 0.18], [-0.5, 0.1, 0.02]], dtype=np.float32)
    for view, (h, v, d, sign) in VIEW_AXES.items():
        for axis, side in (("azimuth", h), ("elevation", v)):
            for degrees in (-1.0, -0.5, 0.5, 1.0):
                theta = np.deg2rad(degrees)
                landmark = pivot.copy()
                landmark[d] += sign * 0.2 * np.cos(theta)
                landmark[side] += 0.2 * np.sin(theta)
                seen = camera_points(landmark[None], pivot, view, axis, degrees)[0]
                np.testing.assert_allclose(seen[[h, v]], pivot[[h, v]], atol=1e-12)
                np.testing.assert_allclose(sign * (seen[d] - pivot[d]), 0.2, atol=1e-12)
                turned = camera_points(points, pivot, view, axis, degrees)
                recovered = camera_points(turned, pivot, view, axis, -degrees)
                np.testing.assert_allclose(recovered, points, rtol=0, atol=6e-8)
