from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pv = pytest.importorskip("pyvista")

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import vehicle_cd_learning.render as render_module
from vehicle_cd_learning.render import (
    VIEW_AXES,
    _rasterize,
    canonical_points,
    render_five_views,
    save_five_views,
    save_preview,
)


def box(length: float, width: float, height: float) -> pv.PolyData:
    points = np.array(
        [
            [-length / 2, -width / 2, 0],
            [length / 2, -width / 2, 0],
            [length / 2, width / 2, 0],
            [-length / 2, width / 2, 0],
            [-length / 2, -width / 2, height],
            [length / 2, -width / 2, height],
            [length / 2, width / 2, height],
            [-length / 2, width / 2, height],
        ],
        dtype=np.float64,
    )
    triangles = np.array(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ],
        dtype=np.int64,
    )
    packed = np.column_stack((np.full(len(triangles), 3), triangles)).reshape(-1)
    return pv.PolyData(points, packed)


def combine(*meshes: pv.PolyData) -> pv.PolyData:
    points: list[np.ndarray] = []
    triangles: list[np.ndarray] = []
    offset = 0
    for mesh in meshes:
        points.append(np.asarray(mesh.points))
        packed = np.asarray(mesh.faces).reshape(-1, 4)[:, 1:]
        triangles.append(packed + offset)
        offset += mesh.n_points
    faces = np.concatenate(triangles)
    packed_faces = np.column_stack((np.full(len(faces), 3), faces)).reshape(-1)
    return pv.PolyData(np.concatenate(points), packed_faces)


def slow_raster(
    points: np.ndarray, faces: np.ndarray, view: str, size: int, window_l: float
):
    """Brute-force reference retained to verify the vectorized candidate filter."""
    h_axis, v_axis, depth_axis, camera_sign = VIEW_AXES[view]
    margin = 4.0
    scale = (size - 2 * margin) / window_l
    h_min = 0.0 if h_axis == 2 else -window_l / 2
    v_min = 0.0 if v_axis == 2 else -window_l / 2
    px = (points[:, h_axis] - h_min) * scale + margin
    py = size - 1 - ((points[:, v_axis] - v_min) * scale + margin)
    depth = np.full((size, size), np.inf, dtype=np.float32)
    mask = np.zeros((size, size), dtype=bool)
    lo, hi = (-window_l / 2, window_l / 2) if depth_axis != 2 else (0, window_l)
    plane = hi if camera_sign > 0 else lo
    d = (
        (plane - points[:, depth_axis])
        if camera_sign > 0
        else (points[:, depth_axis] - plane)
    )
    d /= window_l
    for face in faces:
        verts = np.column_stack((px[face], py[face], d[face]))
        c0, c1 = (
            max(0, math.floor(verts[:, 0].min())),
            min(size - 1, math.ceil(verts[:, 0].max())),
        )
        r0, r1 = (
            max(0, math.floor(verts[:, 1].min())),
            min(size - 1, math.ceil(verts[:, 1].max())),
        )
        if c0 > c1 or r0 > r1:
            continue
        x0, y0 = verts[0, :2]
        x1, y1 = verts[1, :2]
        x2, y2 = verts[2, :2]
        denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(float(denom)) < 1e-8:
            continue
        gx, gy = np.meshgrid(
            np.arange(c0, c1 + 1, dtype=np.float32) + 0.5,
            np.arange(r0, r1 + 1, dtype=np.float32) + 0.5,
        )
        w0 = ((y1 - y2) * (gx - x2) + (x2 - x1) * (gy - y2)) / denom
        w1 = ((y2 - y0) * (gx - x2) + (x0 - x2) * (gy - y2)) / denom
        w2 = 1 - w0 - w1
        inside = (w0 >= -1e-5) & (w1 >= -1e-5) & (w2 >= -1e-5)
        tri_depth = w0 * verts[0, 2] + w1 * verts[1, 2] + w2 * verts[2, 2]
        window = depth[r0 : r1 + 1, c0 : c1 + 1]
        update = inside & (tri_depth < window)
        window[update] = tri_depth[update]
        mask[r0 : r1 + 1, c0 : c1 + 1][update] = True
    output = np.zeros_like(depth)
    output[mask] = depth[mask]
    return output, mask


def test_declared_units_produce_same_canonical_geometry() -> None:
    meters = box(4.5, 1.8, 1.5)
    millimeters = box(4500, 1800, 1500)
    m_points, m_meta = canonical_points(meters.points, unit_scale_to_m=1)
    mm_points, mm_meta = canonical_points(millimeters.points, unit_scale_to_m=0.001)
    np.testing.assert_allclose(m_points, mm_points)
    assert m_meta["length_m"] == pytest.approx(4.5)
    assert mm_meta["width_m"] == pytest.approx(1.8)
    assert m_meta["height_m"] == pytest.approx(1.5)


def test_explicit_front_and_up_declarations_map_to_canonical_axes() -> None:
    mesh = box(4.5, 1.8, 1.5)
    canonical, _ = canonical_points(
        mesh.points, unit_scale_to_m=1, front_axis="+x", up_axis="-z"
    )
    # Nose lies at raw +X; top lies at raw -Z. Canonical front is -X and top is +Z.
    nose_index = int(np.argmax(mesh.points[:, 0]))
    bottom_index = int(np.argmax(mesh.points[:, 2]))
    assert canonical[nose_index, 0] == pytest.approx(-0.5)
    assert canonical[bottom_index, 2] == pytest.approx(0)


def test_five_views_keep_fixed_scale_and_expected_resolutions() -> None:
    views, dims = render_five_views(box(4.5, 1.8, 1.5), unit_scale_to_m=1)
    assert tuple(views) == ("front", "rear", "right", "top", "bottom")
    assert views["front"].depth.shape == (256, 256)
    assert views["bottom"].depth.shape == (512, 512)
    assert all(view.mask.any() for view in views.values())
    assert dims["length_m"] == pytest.approx(4.5)
    # The front camera sees the near face; hidden faces do not overwrite its depth.
    front_values = views["front"].depth[views["front"].mask]
    assert front_values.min() == pytest.approx(0.5 - 0.5 / 1.3, abs=0.02)
    assert front_values.max() - front_values.min() < 1e-4


def test_front_and_rear_views_look_from_the_declared_vehicle_directions() -> None:
    body = box(4.5, 1.8, 1.5)
    nose = box(0.4, 0.45, 0.45)
    nose.points[:, 0] -= 2.3  # Small protrusion extends the canonical -X nose.
    views, _ = render_five_views(combine(body, nose), unit_scale_to_m=1)
    front_depth = views["front"].depth[views["front"].mask]
    rear_depth = views["rear"].depth[views["rear"].mask]
    # The nose marker covers only a small part of the front projection; from
    # the rear camera the broad tail face occupies most pixels at near depth.
    front_near = np.mean(front_depth < 0.13)
    rear_near = np.mean(rear_depth < 0.13)
    assert front_near < rear_near * 0.5


def test_bottom_depth_resolves_a_local_bump() -> None:
    body = box(4.5, 1.8, 1.5)
    bump = box(0.9, 0.45, 0.1)
    bump.points[:, 0] -= 0.6
    bump.points[:, 2] -= 0.1
    smooth, _ = render_five_views(body, unit_scale_to_m=1)
    uneven, _ = render_five_views(combine(body, bump), unit_scale_to_m=1)
    smooth_depth = smooth["bottom"].depth[smooth["bottom"].mask]
    uneven_depth = uneven["bottom"].depth[uneven["bottom"].mask]
    assert np.unique(smooth_depth).size == 1
    assert np.unique(uneven_depth).size > 1
    assert smooth["bottom"].depth.shape[0] == 512


def test_vectorized_triangle_filter_matches_brute_force_reference() -> None:
    mesh = box(4.5, 1.8, 1.5)
    points, _ = canonical_points(mesh.points, unit_scale_to_m=1)
    faces = np.asarray(mesh.faces).reshape(-1, 4)[:, 1:]
    for name in VIEW_AXES:
        actual = _rasterize(points, faces, view=name, size=48, window_l=1.3)
        expected_depth, expected_mask = slow_raster(points, faces, name, 48, 1.3)
        np.testing.assert_array_equal(actual.mask, expected_mask)
        np.testing.assert_array_equal(actual.depth, expected_depth)


@pytest.mark.parametrize("seed", [3, 41])
def test_batched_raster_matches_reference_for_dense_random_triangles(seed, monkeypatch):
    rng = np.random.default_rng(seed)
    points = rng.uniform((-0.65, -0.65, 0), (0.65, 0.65, 1.0), (90, 3)).astype(
        np.float32
    )
    faces = rng.integers(0, len(points), size=(220, 3), dtype=np.int64)
    # Degenerate and repeated faces are valid input boundaries and must be ignored safely.
    faces[:5, 2] = faces[:5, 1]
    monkeypatch.setattr(render_module, "_RASTER_PIXEL_BUDGET", 140)
    for name in VIEW_AXES:
        actual = _rasterize(points, faces, view=name, size=40, window_l=1.3)
        expected_depth, expected_mask = slow_raster(points, faces, name, 40, 1.3)
        np.testing.assert_array_equal(actual.mask, expected_mask)
        np.testing.assert_array_equal(actual.depth, expected_depth)


def test_raster_chunking_bounds_candidate_batches(monkeypatch):
    mesh = box(4.5, 1.8, 1.5)
    points, _ = canonical_points(mesh.points, unit_scale_to_m=1)
    faces = np.asarray(mesh.faces).reshape(-1, 4)[:, 1:]
    monkeypatch.setattr(render_module, "_RASTER_PIXEL_BUDGET", 32)
    actual = _rasterize(points, faces, view="top", size=64, window_l=1.3)
    expected_depth, expected_mask = slow_raster(points, faces, "top", 64, 1.3)
    np.testing.assert_array_equal(actual.mask, expected_mask)
    np.testing.assert_array_equal(actual.depth, expected_depth)


def test_renderer_rejects_empty_surfaces_and_nonfinite_window() -> None:
    empty = pv.PolyData(np.empty((0, 3)))
    with pytest.raises(ValueError, match="triangle"):
        render_five_views(empty, unit_scale_to_m=1)
    with pytest.raises(ValueError, match="window_l"):
        render_five_views(box(4.5, 1.8, 1.5), unit_scale_to_m=1, window_l=np.nan)


def test_saved_training_arrays_and_preview_are_readable() -> None:
    views, _ = render_five_views(box(4.5, 1.8, 1.5), unit_scale_to_m=1)
    with tempfile.TemporaryDirectory() as tmp:
        npz = Path(tmp) / "case.npz"
        preview = Path(tmp) / "case.png"
        save_five_views(npz, views)
        save_preview(preview, views)
        with np.load(npz) as data:
            assert set(data.files) == {"front", "rear", "right", "top", "bottom"}
            assert data["front"].shape == (2, 256, 256)
            assert data["bottom"].shape == (2, 512, 512)
            assert data["front"].dtype == np.float32
            np.testing.assert_array_equal(
                data["front"][1], views["front"].mask.astype(np.float32)
            )
        with Image.open(preview) as image:
            assert image.size == (768, 560)
