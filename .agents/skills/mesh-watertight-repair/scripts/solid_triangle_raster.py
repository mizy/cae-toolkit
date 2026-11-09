"""Axis-agnostic conservative solid-triangle rasterization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np


MAX_RASTER_CANDIDATES_PER_CHUNK = 4_000_000
DEFAULT_FACE_CHUNK = 12_000
DEPTH_EVALUATION_CHUNK = 250_000
ORTHOGRAPHIC_FRAMES = (
    ((1., 0., 0.), (0., 1., 0.)), ((-1., 0., 0.), (0., 1., 0.)),
    ((0., 0., 1.), (0., 1., 0.)), ((0., 0., -1.), (0., 1., 0.)),
    ((0., 1., 0.), (0., 0., -1.)), ((0., -1., 0.), (0., 0., 1.)),
    ((1., .4, 1.), (0., 1., 0.)), ((-1., .4, -1.), (0., 1., 0.)),
)


def projection_frame(points: np.ndarray, eye: tuple, up: tuple) -> tuple[np.ndarray, np.ndarray]:
    forward = -np.asarray(eye, dtype=float)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    upward = np.cross(right, forward)
    basis = np.column_stack((right, upward, forward))
    return points @ basis, basis


@dataclass(frozen=True)
class ViewSpec:
    name: str
    project_axes: tuple[int, int]
    depth_vector: tuple[float, float, float]


@dataclass(frozen=True)
class SolidViewRaster:
    view: ViewSpec
    silhouette: np.ndarray
    depth: np.ndarray
    first_hit_face: np.ndarray
    face_covered_pixels: np.ndarray
    face_first_hit_pixels: np.ndarray
    face_min_depth_gap: np.ndarray
    report: dict[str, Any]


@dataclass(frozen=True)
class SolidRasterEvidence:
    views: tuple[SolidViewRaster, ...]
    face_first_hit_view_count: np.ndarray
    face_first_hit_pixels: np.ndarray
    face_covered_pixels: np.ndarray
    face_min_depth_gap: np.ndarray
    report: dict[str, Any]


SIX_VIEWS = (
    ViewSpec("minus_x", (1, 2), (1.0, 0.0, 0.0)),
    ViewSpec("plus_x", (1, 2), (-1.0, 0.0, 0.0)),
    ViewSpec("minus_y", (0, 2), (0.0, 1.0, 0.0)),
    ViewSpec("plus_y", (0, 2), (0.0, -1.0, 0.0)),
    ViewSpec("minus_z", (0, 1), (0.0, 0.0, 1.0)),
    ViewSpec("plus_z", (0, 1), (0.0, 0.0, -1.0)),
)


# @entry six-direction conservative solid-triangle raster evidence.
def rasterize_solid_views(
    points: np.ndarray,
    faces: np.ndarray,
    *,
    grid_size: int,
    depth_tolerance: float = 0.0,
    collect_face_evidence: bool = True,
) -> SolidRasterEvidence:
    coordinates, triangles = _validate_mesh(points, faces)
    if grid_size < 2:
        raise ValueError("grid_size must be at least 2")
    if depth_tolerance < 0.0:
        raise ValueError("depth_tolerance must be non-negative")
    bbox_extent = float(np.max(np.ptp(coordinates, axis=0)))
    effective_tolerance = max(depth_tolerance, bbox_extent * 1.0e-12)
    views = tuple(
        rasterize_solid_view(
            coordinates,
            triangles,
            view,
            grid_size=grid_size,
            depth_tolerance=effective_tolerance,
            collect_face_evidence=collect_face_evidence,
        )
        for view in SIX_VIEWS
    )
    first_hit_pixels = np.stack([view.face_first_hit_pixels for view in views], axis=1)
    covered_pixels = np.stack([view.face_covered_pixels for view in views], axis=1)
    depth_gap = np.stack([view.face_min_depth_gap for view in views], axis=1)
    view_count = (
        np.count_nonzero(first_hit_pixels > 0, axis=1).astype(np.int8)
        if collect_face_evidence
        else np.full(triangles.shape[0], -1, dtype=np.int8)
    )
    return SolidRasterEvidence(
        views=views,
        face_first_hit_view_count=view_count,
        face_first_hit_pixels=first_hit_pixels,
        face_covered_pixels=covered_pixels,
        face_min_depth_gap=depth_gap,
        report={
            "method": "six_direction_conservative_solid_triangle_z_buffer",
            "grid_size_longest_axis": int(grid_size),
            "requested_depth_tolerance": float(depth_tolerance),
            "effective_depth_tolerance": float(effective_tolerance),
            "faces_with_first_hit": (
                int(np.count_nonzero(view_count)) if collect_face_evidence else None
            ),
            "face_evidence_collected": collect_face_evidence,
            "candidate_limit_per_chunk": MAX_RASTER_CANDIDATES_PER_CHUNK,
            "silhouette_contract": (
                "triangle_pixel_cell_intersection; no per-face centroid fallback"
            ),
            "views": [view.report for view in views],
        },
    )


# @entry one caller-selected view, useful for fixed-view regression checks.
def rasterize_solid_view(
    points: np.ndarray,
    faces: np.ndarray,
    view: ViewSpec,
    *,
    grid_size: int,
    depth_tolerance: float,
    collect_face_evidence: bool = True,
    projection_bounds: np.ndarray | None = None,
    sample_centers: bool = False,
) -> SolidViewRaster:
    if grid_size < 2:
        raise ValueError("grid_size must be at least 2")
    if depth_tolerance < 0.0:
        raise ValueError("depth_tolerance must be non-negative")
    coordinates, triangles = _validate_mesh(points, faces)
    rows, cols, projected = _project_points(coordinates, view, grid_size, projection_bounds)
    projected_triangles = projected[triangles]
    vertex_depth = coordinates @ np.asarray(view.depth_vector, dtype=np.float64)
    triangle_depth = vertex_depth[triangles]
    depth_buffer = np.full(rows * cols, np.inf, dtype=np.float64)
    first_pass_stats = _RasterStats()
    sampler = triangle_center_samples if sample_centers else _solid_triangle_samples
    for face_ids, pixels, sample_depth in sampler(
        projected_triangles, triangle_depth, rows, cols, first_pass_stats
    ):
        np.minimum.at(depth_buffer, pixels, sample_depth)

    face_count = triangles.shape[0]
    unavailable = -1 if not collect_face_evidence else 0
    covered_pixels = np.full(face_count, unavailable, dtype=np.int64)
    first_hit_pixels = np.full(face_count, unavailable, dtype=np.int64)
    min_depth_gap = np.full(
        face_count, np.nan if not collect_face_evidence else np.inf, dtype=np.float64
    )
    first_hit_face = np.full(rows * cols, face_count, dtype=np.int64)
    second_pass_stats = _RasterStats()
    if collect_face_evidence:
        for face_ids, pixels, sample_depth in sampler(
            projected_triangles, triangle_depth, rows, cols, second_pass_stats
        ):
            gap = np.maximum(0.0, sample_depth - depth_buffer[pixels])
            covered_pixels += np.bincount(face_ids, minlength=face_count)
            np.minimum.at(min_depth_gap, face_ids, gap)
            first_hit = gap <= depth_tolerance
            if np.any(first_hit):
                hit_faces = face_ids[first_hit]
                hit_pixels = pixels[first_hit]
                first_hit_pixels += np.bincount(hit_faces, minlength=face_count)
                np.minimum.at(first_hit_face, hit_pixels, hit_faces)

    silhouette = np.isfinite(depth_buffer)
    first_hit_face[first_hit_face == face_count] = -1
    occupied_pixels = int(np.count_nonzero(silhouette))
    return SolidViewRaster(
        view=view,
        silhouette=silhouette.reshape(rows, cols),
        depth=depth_buffer.reshape(rows, cols),
        first_hit_face=first_hit_face.reshape(rows, cols),
        face_covered_pixels=covered_pixels,
        face_first_hit_pixels=first_hit_pixels,
        face_min_depth_gap=min_depth_gap,
        report={
            "view": view.name,
            "method": "pixel_center_triangle_z_buffer" if sample_centers else "conservative_triangle_pixel_cell_min_depth_z_buffer",
            "grid": {"rows": rows, "cols": cols},
            "pixels": int(rows * cols),
            "occupied_pixels": occupied_pixels,
            "faces_with_coverage": (
                int(np.count_nonzero(covered_pixels)) if collect_face_evidence else None
            ),
            "faces_with_first_hit": (
                int(np.count_nonzero(first_hit_pixels))
                if collect_face_evidence
                else None
            ),
            "face_evidence_collected": collect_face_evidence,
            "candidate_samples": int(first_pass_stats.candidate_samples),
            "solid_samples": int(first_pass_stats.solid_samples),
            "chunks": int(first_pass_stats.chunks),
            "max_candidate_samples_in_chunk": int(
                first_pass_stats.max_candidate_samples
            ),
            "candidate_limit_per_chunk": MAX_RASTER_CANDIDATES_PER_CHUNK,
            "depth_tolerance": float(depth_tolerance),
            "subpixel_policy": "pixel_center_only" if sample_centers else "conservative_pixel_cell_intersection",
            "depth_policy": "linear_at_pixel_center" if sample_centers else "linear_minimum_on_triangle_pixel_cell_intersection",
        },
    )


@dataclass
class _RasterStats:
    candidate_samples: int = 0
    solid_samples: int = 0
    chunks: int = 0
    max_candidate_samples: int = 0


def triangle_center_samples(
    projected_triangles: np.ndarray,
    triangle_depth: np.ndarray,
    rows: int,
    cols: int,
    stats: _RasterStats | None = None,
):
    """Yield orthographic ray/triangle hits at pixel centers in bounded-memory chunks."""
    face_count = projected_triangles.shape[0]
    start = 0
    while start < face_count:
        end = min(face_count, start + DEFAULT_FACE_CHUNK)
        while True:
            triangles = projected_triangles[start:end]
            lower, widths, _heights, counts = _raster_bounds(triangles, rows, cols)
            candidate_count = int(np.sum(counts, dtype=np.int64))
            if candidate_count <= MAX_RASTER_CANDIDATES_PER_CHUNK or end - start <= 1:
                break
            end = start + max(1, (end - start) // 2)

        if stats is not None:
            stats.candidate_samples += candidate_count
            stats.chunks += 1
            stats.max_candidate_samples = max(stats.max_candidate_samples, candidate_count)
        local_faces = np.repeat(np.arange(end - start, dtype=np.int64), counts)
        if local_faces.size == 0:
            start = end
            continue
        begins = np.cumsum(counts, dtype=np.int64) - counts
        offsets = np.arange(local_faces.size, dtype=np.int64) - np.repeat(begins, counts)
        repeated_widths = widths[local_faces]
        col = lower[local_faces, 0] + offsets % repeated_widths
        row = lower[local_faces, 1] + offsets // repeated_widths

        triangles = projected_triangles[start:end]
        a = triangles[local_faces, 0]
        b = triangles[local_faces, 1]
        c = triangles[local_faces, 2]
        sample_x = col.astype(np.float64)
        sample_y = row.astype(np.float64)
        denominator = (
            (b[:, 1] - c[:, 1]) * (a[:, 0] - c[:, 0])
            + (c[:, 0] - b[:, 0]) * (a[:, 1] - c[:, 1])
        )
        denominator_epsilon = np.finfo(np.float64).eps * max(rows, cols) ** 2 * 16.0
        valid = np.abs(denominator) > denominator_epsilon
        weight_a = np.zeros_like(denominator)
        weight_b = np.zeros_like(denominator)
        weight_a[valid] = (
            (b[valid, 1] - c[valid, 1]) * (sample_x[valid] - c[valid, 0])
            + (c[valid, 0] - b[valid, 0]) * (sample_y[valid] - c[valid, 1])
        ) / denominator[valid]
        weight_b[valid] = (
            (c[valid, 1] - a[valid, 1]) * (sample_x[valid] - c[valid, 0])
            + (a[valid, 0] - c[valid, 0]) * (sample_y[valid] - c[valid, 1])
        ) / denominator[valid]
        weight_c = 1.0 - weight_a - weight_b
        barycentric_epsilon = 1.0e-9
        inside = (
            valid
            & (weight_a >= -barycentric_epsilon)
            & (weight_b >= -barycentric_epsilon)
            & (weight_c >= -barycentric_epsilon)
        )
        if np.any(inside):
            if stats is not None:
                stats.solid_samples += int(np.count_nonzero(inside))
            selected_local = local_faces[inside]
            depths = triangle_depth[start:end]
            sample_depth = (
                weight_a[inside] * depths[selected_local, 0]
                + weight_b[inside] * depths[selected_local, 1]
                + weight_c[inside] * depths[selected_local, 2]
            )
            yield (
                start + selected_local,
                row[inside] * cols + col[inside],
                sample_depth,
            )
        start = end




def _solid_triangle_samples(
    projected_triangles: np.ndarray,
    triangle_depth: np.ndarray,
    rows: int,
    cols: int,
    stats: _RasterStats,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    face_count = projected_triangles.shape[0]
    start = 0
    while start < face_count:
        end = min(face_count, start + DEFAULT_FACE_CHUNK)
        while True:
            lower, widths, heights, counts = _raster_bounds(
                projected_triangles[start:end], rows, cols
            )
            candidate_count = int(np.sum(counts, dtype=np.int64))
            if candidate_count <= MAX_RASTER_CANDIDATES_PER_CHUNK:
                break
            if end - start == 1:
                raise ValueError(
                    "raster_candidate_budget_exceeded_for_single_face: "
                    f"{candidate_count} > {MAX_RASTER_CANDIDATES_PER_CHUNK}"
                )
            end = start + max(1, (end - start) // 2)

        stats.candidate_samples += candidate_count
        stats.max_candidate_samples = max(stats.max_candidate_samples, candidate_count)
        stats.chunks += 1
        if candidate_count == 0:
            start = end
            continue
        local_faces = np.repeat(np.arange(end - start, dtype=np.int64), counts)
        begins = np.cumsum(counts, dtype=np.int64) - counts
        offsets = np.arange(candidate_count, dtype=np.int64) - np.repeat(begins, counts)
        repeated_widths = widths[local_faces]
        col = lower[local_faces, 0] + offsets % repeated_widths
        row = lower[local_faces, 1] + offsets // repeated_widths
        triangles = projected_triangles[start:end][local_faces]
        solid = _triangle_intersects_pixel(triangles, col, row)
        if np.any(solid):
            selected_faces = local_faces[solid]
            depth = _minimum_depth_on_triangle_pixel_intersection(
                triangles[solid],
                triangle_depth[start:end][selected_faces],
                col[solid],
                row[solid],
            )
            stats.solid_samples += int(selected_faces.size)
            yield (
                start + selected_faces,
                row[solid] * cols + col[solid],
                depth,
            )
        start = end


def _triangle_intersects_pixel(
    triangles: np.ndarray,
    col: np.ndarray,
    row: np.ndarray,
) -> np.ndarray:
    """Vectorized 2-D triangle/AABB separating-axis test."""
    centers = np.column_stack((col, row)).astype(np.float64)
    relative = triangles - centers[:, None, :]
    intersects = np.ones(triangles.shape[0], dtype=bool)
    epsilon = np.finfo(np.float64).eps * 64.0
    for edge_id in range(3):
        edge = relative[:, (edge_id + 1) % 3] - relative[:, edge_id]
        normal = np.column_stack((-edge[:, 1], edge[:, 0]))
        projection = np.einsum("nij,nj->ni", relative, normal)
        radius = 0.5 * (np.abs(normal[:, 0]) + np.abs(normal[:, 1]))
        intersects &= (np.min(projection, axis=1) <= radius + epsilon) & (
            np.max(projection, axis=1) >= -radius - epsilon
        )
    return intersects


def _minimum_depth_on_triangle_pixel_intersection(
    triangles: np.ndarray,
    depths: np.ndarray,
    col: np.ndarray,
    row: np.ndarray,
) -> np.ndarray:
    """Return the true linear minimum over each triangle/pixel-cell intersection.

    Conservative coverage includes cells whose centers lie outside the triangle,
    so center interpolation would extrapolate depth.  The intersection is convex
    and depth is linear; its minimum is therefore attained at a triangle vertex,
    a pixel-cell corner, or a triangle-edge/pixel-edge intersection.
    """

    result = np.empty(triangles.shape[0], dtype=np.float64)
    for start in range(0, triangles.shape[0], DEPTH_EVALUATION_CHUNK):
        end = min(triangles.shape[0], start + DEPTH_EVALUATION_CHUNK)
        result[start:end] = _minimum_depth_block(
            triangles[start:end], depths[start:end], col[start:end], row[start:end]
        )
    return result


def _minimum_depth_block(
    triangles: np.ndarray,
    depths: np.ndarray,
    col: np.ndarray,
    row: np.ndarray,
) -> np.ndarray:
    count = triangles.shape[0]
    minimum = np.full(count, np.inf, dtype=np.float64)
    left = col.astype(np.float64) - 0.5
    right = left + 1.0
    bottom = row.astype(np.float64) - 0.5
    top = bottom + 1.0
    coordinate_scale = max(1.0, float(np.max(np.abs(triangles), initial=0.0)))
    coordinate_tolerance = np.finfo(np.float64).eps * 256.0 * coordinate_scale
    barycentric_tolerance = np.finfo(np.float64).eps * 512.0

    def update(values: np.ndarray, selected: np.ndarray) -> None:
        np.minimum(minimum, np.where(selected, values, np.inf), out=minimum)

    # Triangle vertices contained by the pixel cell.
    for vertex_id in range(3):
        vertex = triangles[:, vertex_id]
        inside = (
            (vertex[:, 0] >= left - coordinate_tolerance)
            & (vertex[:, 0] <= right + coordinate_tolerance)
            & (vertex[:, 1] >= bottom - coordinate_tolerance)
            & (vertex[:, 1] <= top + coordinate_tolerance)
        )
        update(depths[:, vertex_id], inside)

    # Pixel corners contained by a non-degenerate projected triangle.
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    denominator = (b[:, 1] - c[:, 1]) * (a[:, 0] - c[:, 0]) + (c[:, 0] - b[:, 0]) * (
        a[:, 1] - c[:, 1]
    )
    denominator_tolerance = (
        np.finfo(np.float64).eps * 256.0 * coordinate_scale * coordinate_scale
    )
    stable = np.abs(denominator) > denominator_tolerance
    for corner_x in (left, right):
        for corner_y in (bottom, top):
            weight_a = np.zeros(count, dtype=np.float64)
            weight_b = np.zeros(count, dtype=np.float64)
            np.divide(
                (b[:, 1] - c[:, 1]) * (corner_x - c[:, 0])
                + (c[:, 0] - b[:, 0]) * (corner_y - c[:, 1]),
                denominator,
                out=weight_a,
                where=stable,
            )
            np.divide(
                (c[:, 1] - a[:, 1]) * (corner_x - c[:, 0])
                + (a[:, 0] - c[:, 0]) * (corner_y - c[:, 1]),
                denominator,
                out=weight_b,
                where=stable,
            )
            weight_c = 1.0 - weight_a - weight_b
            inside = (
                stable
                & (weight_a >= -barycentric_tolerance)
                & (weight_b >= -barycentric_tolerance)
                & (weight_c >= -barycentric_tolerance)
            )
            corner_depth = (
                weight_a * depths[:, 0]
                + weight_b * depths[:, 1]
                + weight_c * depths[:, 2]
            )
            update(corner_depth, inside)

    # Triangle edges crossing any of the four pixel-cell boundaries.
    for first_id, second_id in ((0, 1), (1, 2), (2, 0)):
        first = triangles[:, first_id]
        delta = triangles[:, second_id] - first
        depth_delta = depths[:, second_id] - depths[:, first_id]
        for boundary_x in (left, right):
            valid = np.abs(delta[:, 0]) > coordinate_tolerance
            parameter = np.zeros(count, dtype=np.float64)
            np.divide(
                boundary_x - first[:, 0],
                delta[:, 0],
                out=parameter,
                where=valid,
            )
            clipped = np.clip(parameter, 0.0, 1.0)
            crossing_y = first[:, 1] + clipped * delta[:, 1]
            intersects = (
                valid
                & (parameter >= -barycentric_tolerance)
                & (parameter <= 1.0 + barycentric_tolerance)
                & (crossing_y >= bottom - coordinate_tolerance)
                & (crossing_y <= top + coordinate_tolerance)
            )
            update(depths[:, first_id] + clipped * depth_delta, intersects)
        for boundary_y in (bottom, top):
            valid = np.abs(delta[:, 1]) > coordinate_tolerance
            parameter = np.zeros(count, dtype=np.float64)
            np.divide(
                boundary_y - first[:, 1],
                delta[:, 1],
                out=parameter,
                where=valid,
            )
            clipped = np.clip(parameter, 0.0, 1.0)
            crossing_x = first[:, 0] + clipped * delta[:, 0]
            intersects = (
                valid
                & (parameter >= -barycentric_tolerance)
                & (parameter <= 1.0 + barycentric_tolerance)
                & (crossing_x >= left - coordinate_tolerance)
                & (crossing_x <= right + coordinate_tolerance)
            )
            update(depths[:, first_id] + clipped * depth_delta, intersects)

    if not np.all(np.isfinite(minimum)):
        unresolved = int(np.count_nonzero(~np.isfinite(minimum)))
        raise RuntimeError(
            "solid raster found pixel-cell intersections without a depth vertex: "
            f"{unresolved} samples"
        )
    # Numerical tolerance at an edge must never create an extrapolated depth.
    return np.clip(minimum, np.min(depths, axis=1), np.max(depths, axis=1))


def _raster_bounds(
    triangles: np.ndarray,
    rows: int,
    cols: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lower = np.ceil(np.min(triangles, axis=1) - 0.5).astype(np.int64)
    upper = np.floor(np.max(triangles, axis=1) + 0.5).astype(np.int64)
    lower[:, 0] = np.clip(lower[:, 0], 0, cols - 1)
    upper[:, 0] = np.clip(upper[:, 0], 0, cols - 1)
    lower[:, 1] = np.clip(lower[:, 1], 0, rows - 1)
    upper[:, 1] = np.clip(upper[:, 1], 0, rows - 1)
    widths = np.maximum(0, upper[:, 0] - lower[:, 0] + 1)
    heights = np.maximum(0, upper[:, 1] - lower[:, 1] + 1)
    return lower, widths, heights, widths * heights


def _project_points(
    points: np.ndarray,
    view: ViewSpec,
    grid_size: int,
    projection_bounds: np.ndarray | None = None,
) -> tuple[int, int, np.ndarray]:
    projected = points[:, view.project_axes]
    if projection_bounds is None:
        mins = np.min(projected, axis=0)
        spans = np.ptp(projected, axis=0)
    else:
        bounds = np.asarray(projection_bounds, dtype=float)
        if bounds.shape != (2, 2) or not np.isfinite(bounds).all() or np.any(bounds[1] <= bounds[0]):
            raise ValueError("projection_bounds must contain finite 2-D minima and greater maxima")
        mins, spans = bounds[0], bounds[1] - bounds[0]
    longest = float(np.max(spans))
    if not np.isfinite(longest) or longest <= 0.0:
        raise ValueError("cannot rasterize a zero-size projection")
    shape = np.maximum(2, np.rint(grid_size * spans / longest).astype(np.int64))
    cols, rows = int(shape[0]), int(shape[1])
    safe_spans = np.where(spans > 0.0, spans, longest)
    normalized = (projected - mins) / safe_spans
    pixels = normalized * np.asarray((cols - 1, rows - 1), dtype=np.float64)
    return rows, cols, pixels


def _validate_mesh(
    points: np.ndarray,
    faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    coordinates = np.asarray(points, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3 or coordinates.shape[0] == 0:
        raise ValueError("points must have non-empty shape (N, 3)")
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("faces must have shape (M, 3)")
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("points must be finite; run polygon-soup cleanup first")
    if triangles.size and (
        int(np.min(triangles)) < 0 or int(np.max(triangles)) >= coordinates.shape[0]
    ):
        raise ValueError("faces contain out-of-range point indices")
    return np.ascontiguousarray(coordinates), np.ascontiguousarray(triangles)
