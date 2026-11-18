import sys
from pathlib import Path

import numpy as np
import pytest
import pyvista as pv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from vehicle_cd_learning import rim_resolution as resolution
from vehicle_cd_learning.prepare_drivaerrim import full_frame, merge_depth
from vehicle_cd_learning.rim_resolution import prepare, render_component


def test_component_composition_reproduces_old_full_frame():
    a = pv.Box(bounds=(-0.4, 0.4, -0.15, 0.15, 0, 0.2)).triangulate()
    b = pv.Box(bounds=(-0.2, 0.2, -0.2, 0.2, 0, 0.1)).triangulate()
    origin = np.zeros(3)
    old = full_frame(a.append_polydata(b), origin, 1)
    current = merge_depth(
        render_component(a, origin, 1, 256), render_component(b, origin, 1, 256)
    )
    for name in old:
        np.testing.assert_array_equal(current[name].depth, old[name].depth)
        np.testing.assert_array_equal(current[name].mask, old[name].mask)
    high = render_component(a, origin, 1, 512)
    for name in high:
        assert high[name].depth.shape == (512, 512)
        assert high[name].mask.any()


def test_preparation_requires_fresh_directory(tmp_path):
    with pytest.raises(FileExistsError):
        prepare(tmp_path / "missing", tmp_path / "missing", tmp_path)


def test_frame_preserves_asymmetric_shared_body_bounds(monkeypatch, tmp_path):
    body = pv.Box(bounds=(-0.4, 0.4, -0.2, 0.21, 0, 0.3)).triangulate()
    fixed = pv.Box(bounds=(-0.1, 0.1, -0.1, 0.05, 0, 0.2)).triangulate()
    tyre = pv.Box(bounds=(-0.1, 0.1, -0.1, 0.1, -0.02, 0.1)).triangulate()
    entries = {f"Geometry/{n}/{n}.stl": {"sha256": "ok"} for n in resolution.SHARED}
    entries["Geometry/Wheel_FL/Wheel_FL_0001.stl"] = {"sha256": "ok"}
    monkeypatch.setattr(resolution, "sha256", lambda p: "ok")
    monkeypatch.setattr(
        resolution.pv,
        "read",
        lambda p: (
            body if p.name == "Car_Body.stl" else tyre if "0001" in p.name else fixed
        ).copy(),
    )
    monkeypatch.setattr(resolution, "render_component", lambda *args: object())
    monkeypatch.setattr(resolution, "merge_depth", lambda a, b: a)
    _, origin, dimensions, _ = resolution.shared_views(tmp_path, entries)
    assert origin[1] == pytest.approx(0.005)
    assert dimensions[1] == pytest.approx(0.41)
    assert origin[2] == pytest.approx(-0.02)
