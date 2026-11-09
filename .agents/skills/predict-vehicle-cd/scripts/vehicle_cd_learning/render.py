"""Render canonical vehicle surfaces as fixed-scale depth and mask views."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from vehicle_cd_similarity_lib.mesh_io import read_mesh
from vehicle_cd_similarity_lib.types import SideArtifacts

_RASTER_PIXEL_BUDGET = 1_000_000

VIEW_AXES = {
    "front": (1, 2, 0, -1),
    "rear": (1, 2, 0, 1),
    "right": (0, 2, 1, 1),
    "top": (0, 1, 2, 1),
    "bottom": (0, 1, 2, -1),
}


def canonical_points(
    points: np.ndarray,
    *,
    unit_scale_to_m: float,
    front_axis: str = "-x",
    up_axis: str = "+z",
    ground_clearance_m: float = 0.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Map explicitly declared raw (length,width,up) axes to X/Y/Z, in car lengths.

    Axis declarations such as ``front_axis="-x"`` and ``up_axis="+z"`` identify
    the raw coordinate directions explicitly. The remaining axis is vehicle width.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Mesh points must be a finite N x 3 array")
    if not np.isfinite(unit_scale_to_m) or unit_scale_to_m <= 0:
        raise ValueError("unit_scale_to_m must be a positive finite scale")
    if not np.isfinite(ground_clearance_m) or ground_clearance_m < 0:
        raise ValueError("ground_clearance_m must be finite and nonnegative")

    def parse_axis(value: str) -> tuple[int, int]:
        if value not in {"+x", "-x", "+y", "-y", "+z", "-z"}:
            raise ValueError("Axis declarations must be one of +x, -x, +y, -y, +z, -z")
        return "xyz".index(value[1]), 1 if value[0] == "+" else -1

    length_axis, nose_sign = parse_axis(front_axis)
    up_index, vertical_sign = parse_axis(up_axis)
    if length_axis == up_index:
        raise ValueError("front_axis and up_axis must use different raw axes")
    width_axis = next(axis for axis in range(3) if axis not in {length_axis, up_index})
    axis_order = (length_axis, width_axis, up_index)

    raw = points * unit_scale_to_m
    out = raw[:, [length_axis, width_axis, up_index]].copy()
    out[:, 0] *= -nose_sign  # The nose always points toward canonical -X.
    out[:, 2] *= vertical_sign

    # Select Y sign so X cross Y points toward +Z, accounting for axis permutation.
    basis = np.eye(3)[list(axis_order)]
    basis[0] *= -nose_sign
    basis[2] *= vertical_sign
    if np.linalg.det(basis) < 0:
        out[:, 1] *= -1

    spans_m = np.ptp(out, axis=0)
    length_m = float(spans_m[0])
    if length_m <= 1e-12:
        raise ValueError("Vehicle has zero extent along its declared length axis")
    out /= length_m
    out[:, 0] -= 0.5 * (float(out[:, 0].min()) + float(out[:, 0].max()))
    out[:, 1] -= 0.5 * (float(out[:, 1].min()) + float(out[:, 1].max()))
    out[:, 2] -= float(out[:, 2].min())
    out[:, 2] += ground_clearance_m / length_m
    info = {
        "unit_scale_to_m": float(unit_scale_to_m),
        "axis_order": [int(axis) for axis in axis_order],
        "front_axis": front_axis,
        "up_axis": up_axis,
        "ground_clearance_m": float(ground_clearance_m),
        "length_m": length_m,
        "width_m": float(spans_m[1]),
        "height_m": float(spans_m[2]),
        "normalized_extents_l": [float(v) for v in np.ptp(out, axis=0)],
    }
    return out.astype(np.float32), info


def _rasterize(
    points: np.ndarray,
    faces: np.ndarray,
    *,
    view: str,
    size: int,
    window_l: float,
) -> SideArtifacts:
    h_axis, v_axis, depth_axis, camera_sign = VIEW_AXES[view]
    # Each view uses the same physical pixel scale and a fixed field of view.
    margin = 4.0
    scale = (size - 2 * margin) / window_l
    h = points[:, h_axis]
    v = points[:, v_axis]
    h_min = 0.0 if h_axis == 2 else -window_l / 2
    v_min = 0.0 if v_axis == 2 else -window_l / 2
    px = (h - h_min) * scale + margin
    py = size - 1 - ((v - v_min) * scale + margin)

    depth = np.full((size, size), np.inf, dtype=np.float32)
    mask = np.zeros((size, size), dtype=bool)
    lo, hi = -window_l / 2, window_l / 2
    if depth_axis == 2:
        lo = 0.0
        hi = window_l
    camera_plane = hi if camera_sign > 0 else lo
    d = (
        camera_plane - points[:, depth_axis]
        if camera_sign > 0
        else points[:, depth_axis] - camera_plane
    )
    d /= window_l

    triangles = np.stack((px[faces], py[faces], d[faces]), axis=2)
    min_col = triangles[:, :, 0].min(axis=1)
    max_col = triangles[:, :, 0].max(axis=1)
    min_row = triangles[:, :, 1].min(axis=1)
    max_row = triangles[:, :, 1].max(axis=1)
    pixel_tolerance = 2e-5 * np.maximum(max_col - min_col, max_row - min_row) + 1e-5
    c0s = np.maximum(0, np.ceil(min_col - 0.5 - pixel_tolerance).astype(np.int32))
    c1s = np.minimum(
        size - 1, np.floor(max_col - 0.5 + pixel_tolerance).astype(np.int32)
    )
    r0s = np.maximum(0, np.ceil(min_row - 0.5 - pixel_tolerance).astype(np.int32))
    r1s = np.minimum(
        size - 1, np.floor(max_row - 0.5 + pixel_tolerance).astype(np.int32)
    )

    x0, y0 = triangles[:, 0, 0], triangles[:, 0, 1]
    x1, y1 = triangles[:, 1, 0], triangles[:, 1, 1]
    x2, y2 = triangles[:, 2, 0], triangles[:, 2, 1]
    denoms = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    visible = (c0s <= c1s) & (r0s <= r1s) & (np.abs(denoms) >= 1e-8)

    # Rasterize bounded chunks of candidate pixels in NumPy. Dense triangles often
    # cover only a handful of pixel centers; batching them avoids a Python loop
    # and an allocation for every face while keeping peak temporary memory bounded.
    visible_faces = np.flatnonzero(visible)
    widths = c1s[visible_faces] - c0s[visible_faces] + 1
    heights = r1s[visible_faces] - r0s[visible_faces] + 1
    areas = widths.astype(np.int64) * heights
    cumulative = np.cumsum(areas)
    depth_flat = depth.ravel()
    mask_flat = mask.ravel()
    budget = _RASTER_PIXEL_BUDGET
    start = 0
    while start < len(visible_faces):
        end = (
            int(
                np.searchsorted(
                    cumulative, cumulative[start - 1] + budget, side="right"
                )
            )
            if start
            else int(np.searchsorted(cumulative, budget, side="right"))
        )
        end = min(max(end, start + 1), len(visible_faces))
        selected = visible_faces[start:end]
        local_areas = areas[start:end]
        offsets = np.cumsum(local_areas) - local_areas
        candidate_tri = np.repeat(np.arange(len(selected)), local_areas)
        pixel_offset = np.arange(int(local_areas.sum()), dtype=np.int64) - np.repeat(
            offsets, local_areas
        )
        tri_indices = selected[candidate_tri]
        cols = c0s[tri_indices] + pixel_offset % widths[start:end][candidate_tri]
        rows = r0s[tri_indices] + pixel_offset // widths[start:end][candidate_tri]

        verts = triangles[tri_indices]
        gx = cols.astype(np.float32) + 0.5
        gy = rows.astype(np.float32) + 0.5
        x0, y0 = verts[:, 0, 0], verts[:, 0, 1]
        x1, y1 = verts[:, 1, 0], verts[:, 1, 1]
        x2, y2 = verts[:, 2, 0], verts[:, 2, 1]
        denom = denoms[tri_indices]
        w0 = ((y1 - y2) * (gx - x2) + (x2 - x1) * (gy - y2)) / denom
        w1 = ((y2 - y0) * (gx - x2) + (x0 - x2) * (gy - y2)) / denom
        w2 = 1 - w0 - w1
        inside = (w0 >= -1e-5) & (w1 >= -1e-5) & (w2 >= -1e-5)
        tri_depth = w0 * verts[:, 0, 2] + w1 * verts[:, 1, 2] + w2 * verts[:, 2, 2]
        linear = rows[inside] * size + cols[inside]
        np.minimum.at(depth_flat, linear, tri_depth[inside])
        mask_flat[linear] = True
        start = end

    out = np.zeros_like(depth)
    out[mask] = depth[mask]
    return SideArtifacts(
        mask=mask, depth=out, normal=np.zeros((*out.shape, 3), dtype=np.float32)
    )


def render_five_views(
    mesh: Any,
    *,
    unit_scale_to_m: float,
    front_axis: str = "-x",
    up_axis: str = "+z",
    ground_clearance_m: float = 0.0,
    ordinary_size: int = 256,
    bottom_size: int = 512,
    window_l: float = 1.3,
) -> tuple[dict[str, SideArtifacts], dict[str, Any]]:
    """Render front/rear/right/top and high-resolution bottom depth/mask views.

    The input axes and units are explicit. Every view uses the same length-based
    metric scale; no per-image fit or depth min/max normalization is applied.
    """
    if (
        ordinary_size < 16
        or bottom_size < ordinary_size
        or not np.isfinite(window_l)
        or window_l <= 0
    ):
        raise ValueError(
            "Require ordinary_size >= 16, bottom_size >= ordinary_size, and window_l > 0"
        )
    # Reuse the established triangle parser and fail before normalization if the
    # input has no renderable surface.
    from vehicle_cd_similarity_lib.artifacts import triangle_faces

    faces = triangle_faces(mesh)
    points, normalization = canonical_points(
        np.asarray(mesh.points),
        unit_scale_to_m=unit_scale_to_m,
        front_axis=front_axis,
        up_axis=up_axis,
        ground_clearance_m=ground_clearance_m,
    )
    half_window = window_l / 2
    if (
        np.any(np.abs(points[:, :2]) > half_window)
        or points[:, 2].min() < 0
        or points[:, 2].max() > window_l
    ):
        raise ValueError(
            f"Vehicle exceeds the fixed {window_l:g} length-unit view window"
        )
    views = {
        name: _rasterize(
            points,
            faces,
            view=name,
            size=bottom_size if name == "bottom" else ordinary_size,
            window_l=window_l,
        )
        for name in VIEW_AXES
    }
    normalization["window_l"] = float(window_l)
    dimensions = {
        key: normalization[key] for key in ("length_m", "width_m", "height_m")
    }
    dimensions["normalized_extents_l"] = normalization["normalized_extents_l"]
    dimensions["window_l"] = float(window_l)
    dimensions["front_axis"] = front_axis
    dimensions["up_axis"] = up_axis
    dimensions["ground_clearance_m"] = float(ground_clearance_m)
    return views, dimensions


def save_five_views(path: Path, views: dict[str, SideArtifacts]) -> None:
    """Save model inputs as lossless float32 depth and uint8 masks in NPZ."""
    if set(views) != set(VIEW_AXES):
        raise ValueError(f"Expected views {tuple(VIEW_AXES)}, got {tuple(views)}")
    arrays: dict[str, np.ndarray] = {}
    for name, artifact in views.items():
        arrays[name] = np.stack(
            (
                np.asarray(artifact.depth, dtype=np.float32),
                np.asarray(artifact.mask, dtype=np.float32),
            ),
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def save_preview(path: Path, views: dict[str, SideArtifacts]) -> None:
    """Save a labeled grayscale depth/mask contact sheet for human inspection."""
    from PIL import ImageDraw

    names = tuple(VIEW_AXES)
    tile = 256
    canvas = Image.new("RGB", (tile * 3, tile * 2 + 48), "#20242a")
    draw = ImageDraw.Draw(canvas)
    positions = ((0, 0), (1, 0), (2, 0), (0, 1), (1, 1))
    for name, (col, row) in zip(names, positions):
        item = views[name]
        depth_img = Image.fromarray(
            (np.clip(item.depth, 0, 1) * 255).astype(np.uint8), mode="L"
        )
        depth_img = depth_img.resize((tile, tile), Image.Resampling.NEAREST).convert(
            "RGB"
        )
        mask = Image.fromarray(item.mask.astype(np.uint8) * 255, mode="L")
        mask = mask.resize((tile, tile), Image.Resampling.NEAREST)
        depth_img.paste((255, 80, 80), mask=Image.eval(mask, lambda px: 255 - px))
        x, y = col * tile, row * tile
        canvas.paste(depth_img, (x, y))
        draw.text((x + 6, y + 6), name, fill="white")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def render_geometry(path, geometry, render_config):
    mesh = read_mesh(Path(path))
    views, dimensions = render_five_views(
        mesh,
        unit_scale_to_m=geometry["unit_scale_to_m"],
        front_axis=geometry["front_axis"],
        up_axis=geometry["up_axis"],
        ground_clearance_m=geometry.get("ground_clearance_m", 0.0),
        **render_config,
    )
    return views, dimensions
