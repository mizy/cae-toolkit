import sys
from pathlib import Path

import numpy as np
import pytest
import pyvista as pv

SKILL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL / "scripts"))
sys.path.insert(0, str(SKILL.parent / "predict-aircraft-coefficients/scripts"))

from vehicle_cd_learning.canonical_sampling import canonical_views


def test_phase_changes_use_same_canonical_geometry_but_real_edits_invalidate_cache(
    tmp_path,
):
    file = tmp_path / "body.stl"
    pv.Box(bounds=(-0.4, 0.4, -0.2, 0.2, 0, 0.3)).triangulate().save(file)
    parts = [{"path": str(file), "mirror_y": False}]
    frame = {"origin_m": [0, 0, 0], "length_m": 1.0, "resolution": 64}
    first, a = canonical_views(parts, frame, tmp_path / "cache")
    second, b = canonical_views(parts, frame, tmp_path / "cache", (0.25, -0.25))
    assert a["key"] == b["key"] and b["reused_canonical_cache"]
    for name in first:
        np.testing.assert_array_equal(first[name], second[name])
    pv.Box(bounds=(-0.4, 0.4, -0.2, 0.2, 0, 0.35)).triangulate().save(file)
    changed, c = canonical_views(parts, frame, tmp_path / "cache", (0.25, -0.25))
    assert c["key"] != a["key"] and not c["reused_canonical_cache"]
    assert any(not np.array_equal(first[n], changed[n]) for n in first)


def test_canonical_cache_binds_camera_and_rejects_corruption(tmp_path):
    file = tmp_path / "body.stl"
    pv.Box(bounds=(-0.4, 0.4, -0.2, 0.2, 0, 0.3)).triangulate().save(file)
    parts = [{"path": str(file)}]
    frame = {"origin_m": [0, 0, 0], "length_m": 1.0, "resolution": 64}
    _, a = canonical_views(parts, frame, tmp_path / "cache")
    _, b = canonical_views(parts, {**frame, "resolution": 128}, tmp_path / "cache")
    assert a["key"] != b["key"]
    Path(a["canonical_views"]).write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="cache content"):
        canonical_views(parts, frame, tmp_path / "cache", (0.25, 0))
