import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import pyvista as pv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from vehicle_cd_learning import phase_views as phase
from vehicle_cd_learning.delta_sampling import shifted_views
from vehicle_cd_learning.render import VIEW_AXES


def all_cases():
    return [
        {
            "case_id": f"drivaerrim__rim-{i:04d}",
            "split": "test" if i == 2 or i > 813 else "val" if i > 721 else "train",
            "targets": [i, -i],
        }
        for i in range(1, 905)
    ]


def test_selection_frozen_before_geometry_and_target_blind():
    cases = all_cases()
    selected, items = phase.selection(cases)
    altered = copy.deepcopy(cases)
    for case in altered:
        case.pop("targets")
    assert phase.selection(altered) == (selected, items)
    assert phase.selection(list(reversed(cases))) == (selected, items)
    assert len(items) == 358
    assert [
        len(selected[s]) for s in ("train", "val", "test", "all_test", "anchors")
    ] == [32, 12, 24, 92, 3]
    assert not set(selected["train"]) & set(phase.ANCHORS)
    assert phase.TEST_QUERY in selected["test"]
    assert len({(i["source_case_id"], i["phase_id"]) for i in items}) == 358
    assert all(
        i["phase_pixels"] in [list(p) for p in phase.TRAIN_PHASES]
        for i in items
        if i["role"] in ("train", "val")
    )
    for identity in selected["all_test"]:
        current = [i for i in items if i["source_case_id"] == identity]
        assert len(current) == (4 if identity in selected["test"] else 1)
        assert current[0]["phase_pixels"] == list(phase.MAIN)


def test_two_axis_phase_keeps_each_view_depth_and_screen_sign(monkeypatch):
    mesh = pv.Box(bounds=(-0.3, 0.3, -0.2, 0.2, 0, 0.3)).triangulate()
    origin = np.array([0.1, 0.02, -0.1])
    calls = {}

    def raster(points, faces, *, view, size, window_l):
        calls[view] = points.copy()
        return view

    monkeypatch.setattr(phase, "_rasterize", raster)
    phase.camera_views(mesh, origin, 2, 512, (0.25, -0.25))
    nominal = ((mesh.points - origin) / 2).astype(np.float32)
    for view, (h, v, depth, _) in VIEW_AXES.items():
        expected = nominal.copy()
        expected[:, h] += np.float32(0.25 * 1.3 / 504)
        expected[:, v] += np.float32(0.25 * 1.3 / 504)
        np.testing.assert_array_equal(calls[view], expected)
        np.testing.assert_array_equal(calls[view][:, depth], nominal[:, depth])


def test_horizontal_phase_exactly_reuses_existing_sampling_renderer():
    mesh = pv.Box(bounds=(-0.3, 0.3, -0.2, 0.2, 0, 0.3)).triangulate()
    for pixels in (0, 0.25):
        reference = shifted_views(mesh, np.zeros(3), 1, 32, pixels)
        actual = phase.camera_views(mesh, np.zeros(3), 1, 32, (pixels, 0))
        for view in VIEW_AXES:
            np.testing.assert_array_equal(actual[view].depth, reference[view].depth)
            np.testing.assert_array_equal(actual[view].mask, reference[view].mask)


def test_vertical_positive_phase_moves_raster_down():
    mesh = pv.Plane(
        center=(0, 0, 0.3),
        direction=(0, 0, 1),
        i_size=0.4,
        j_size=0.4,
        i_resolution=2,
        j_resolution=2,
    ).triangulate()
    nominal = phase.camera_views(mesh, np.zeros(3), 1, 64, (0, 0))["top"]
    down = phase.camera_views(mesh, np.zeros(3), 1, 64, (0, 1))["top"]
    np.testing.assert_array_equal(down.mask[1:], nominal.mask[:-1])
    np.testing.assert_allclose(down.depth[down.mask], (1.3 - 0.3) / 1.3, atol=1e-6)


def test_shared_buffers_read_each_geometry_once_and_do_not_mirror_whole_body(
    monkeypatch, tmp_path
):
    read, rendered = [], []

    def checked(raw, relative, entries, hashes):
        read.append(relative)
        hashes[relative] = "verified"
        return pv.Box(bounds=(-0.3, 0.3, -0.1, 0.2, 0, 0.3)).triangulate()

    original = phase.camera_views

    def camera(mesh, *args):
        rendered.append(float(mesh.points[:, 1].mean()))
        return original(mesh, args[0], args[1], 16, args[3])

    monkeypatch.setattr(phase, "checked_mesh", checked)
    monkeypatch.setattr(phase, "camera_views", camera)
    buffers, hashes = phase.shared_buffers(
        tmp_path, {}, np.zeros(3), 1, [(0, 0), (0.25, 0)]
    )
    assert len(read) == 5 and len(set(read)) == 5 and len(hashes) == 5
    assert len(rendered) == 18
    assert all(v > 0 for v in rendered[:2])
    for views in buffers.values():
        for view in views.values():
            assert not view.depth.flags.writeable and not view.mask.flags.writeable


def test_preparation_rejects_existing_destination_and_bad_workers(tmp_path):
    with pytest.raises(FileExistsError):
        phase.prepare(tmp_path / "absent", tmp_path)
    with pytest.raises(ValueError):
        phase.prepare(tmp_path / "absent", tmp_path / "new", workers=3)
