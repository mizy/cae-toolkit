import sys
from pathlib import Path

import numpy as np

SKILL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL / "scripts"))
sys.path.insert(0, str(SKILL.parent / "predict-vehicle-cd/scripts"))

from vehicle_cd_learning.detail_sensitivity import edit_mirrors


def test_mirror_edit_preserves_body_wheels_and_fixed_roots():
    selection = {
        "x_core": [0.9, 1.1],
        "z_core": [0.65, 0.8],
        "padding_m": 0.04,
        "root_abs_y_m": 0.93,
    }
    points = np.array(
        [
            [1, 1.1, 0.72],
            [1, -1.1, 0.72],
            [1, 0.9, 0.72],
            [1, 0.93, 0.72],
            [0, 1.1, 0.1],
            [2, 1.1, 0.72],
        ]
    )
    np.testing.assert_array_equal(edit_mirrors(points, 1.0, selection), points)
    edited = edit_mirrors(points, 0.7, selection)
    np.testing.assert_array_equal(edited[:, [0, 2]], points[:, [0, 2]])
    np.testing.assert_array_equal(edited[2:], points[2:])
    np.testing.assert_allclose(edited[:2, 1], [0.93 + 0.7 * 0.17, -0.93 - 0.7 * 0.17])


def test_mirror_mapping_remains_monotone_across_both_roots():
    selection = {
        "x_core": [0.9, 1.1],
        "z_core": [0.65, 0.8],
        "padding_m": 0.04,
        "root_abs_y_m": 0.93,
    }
    y = np.linspace(-1.3, 1.3, 2001)
    points = np.column_stack((np.ones_like(y), y, np.full_like(y, 0.72)))
    changed = edit_mirrors(points, 0.7, selection)
    derivative = np.diff(changed[:, 1]) / np.diff(y)
    assert derivative.min() >= 0.7 - 1e-10
    assert derivative.max() <= 1 + 1e-10
