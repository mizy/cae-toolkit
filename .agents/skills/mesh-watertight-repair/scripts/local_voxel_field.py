"""Sample a bounded, locally signed voxel field and extract its zero surface."""

from __future__ import annotations

from typing import Any

import igl
import numpy as np
import pyvista as pv
from sdf_closure import bounded_sdf_smoothing, extract_zero_surface


def closest_distances(surface: pv.PolyData, queries: np.ndarray, *, chunk_size: int = 100_000) -> np.ndarray:
    """Query finite closest-cell distances or fail closed on invalid VTK output."""
    points = np.asarray(queries, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all() or chunk_size <= 0:
        raise ValueError("closest-cell queries must be finite N by 3 points")
    distances = np.empty(len(points), dtype=np.float64)
    for start in range(0, len(points), chunk_size):
        stop = min(start + chunk_size, len(points))
        chunk = points[start:stop]
        cell_ids, closest = surface.find_closest_cell(chunk, return_closest_point=True)
        cell_ids = np.asarray(cell_ids).reshape(-1)
        closest = np.asarray(closest, dtype=np.float64)
        if (
            cell_ids.shape != (len(chunk),)
            or np.any(cell_ids < 0)
            or closest.shape != chunk.shape
            or not np.isfinite(closest).all()
        ):
            raise ValueError("closest-cell query returned invalid cells or points")
        values = np.linalg.norm(chunk - closest, axis=1)
        if not np.isfinite(values).all():
            raise ValueError("closest-cell query returned non-finite distances")
        distances[start:stop] = values
    return distances


def build_local_zero_surface(
    points: np.ndarray,
    faces: np.ndarray,
    source_surface: pv.PolyData,
    bounds: np.ndarray,
    *,
    pitch: float,
    max_voxels: int,
    smoothing_sigma_voxels: float,
) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, Any], str | None, str]:
    """Build one zero surface from an ROI-only grid and a closed source mesh."""
    with np.errstate(over="ignore", invalid="ignore"):
        margin = 2.0 * pitch
        origin = bounds[0] - margin
        upper = bounds[1] + margin
        grid_extent = upper - origin
    if not np.isfinite(origin).all() or not np.isfinite(grid_extent).all():
        return None, None, {}, "invalid_local_grid_bounds", "rejected"
    shape = tuple(int(np.ceil(float(grid_extent[axis]) / pitch)) + 1 for axis in range(3))
    voxel_count = shape[0] * shape[1] * shape[2]
    report = {
        "origin": origin.tolist(),
        "shape": list(shape),
        "pitch": float(pitch),
        "voxels": voxel_count,
        "max_voxels": int(max_voxels),
        "sign_method": "fast_winding_with_direct_queries_near_half",
        "refined_winding_queries": 0,
    }
    if voxel_count > max_voxels:
        return None, None, report, "voxel_budget_exceeded", "budget_exceeded"
    try:
        axes = [origin[axis] + np.arange(shape[axis]) * pitch for axis in range(3)]
        grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
        phi = np.empty(len(grid), dtype=np.float32)
        for start in range(0, len(grid), 100_000):
            stop = min(start + 100_000, len(grid))
            queries = np.ascontiguousarray(grid[start:stop], dtype=np.float64)
            winding = np.asarray(
                igl.fast_winding_number(
                    np.ascontiguousarray(points),
                    np.ascontiguousarray(faces),
                    queries,
                ),
                dtype=np.float64,
            ).reshape(-1)
            if len(winding) != len(queries) or not np.isfinite(winding).all():
                return None, None, report, "fast_winding_number_failed", "rejected"
            ambiguous = np.abs(winding - 0.5) < 0.2
            if np.any(ambiguous):
                direct = np.asarray(igl.winding_number(points, faces, queries[ambiguous])).reshape(-1)
                if direct.size != int(ambiguous.sum()) or not np.isfinite(direct).all():
                    return None, None, report, "direct_winding_query_failed", "rejected"
                winding[ambiguous] = direct
                report["refined_winding_queries"] += int(ambiguous.sum())
            distances = closest_distances(source_surface, queries)
            phi[start:stop] = distances * np.where(winding > 0.5, -1.0, 1.0)
        field = phi.reshape(shape)
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] *= pitch
        transform[:3, 3] = origin
        smoothed, smoothing = bounded_sdf_smoothing(
            field,
            np.abs(field) <= 6.0 * pitch,
            pitch=pitch,
            sigma_voxels=smoothing_sigma_voxels,
            max_zero_shift_voxels=0.5,
        )
        zero_points, zero_faces = extract_zero_surface(smoothed, transform)
    except MemoryError:
        return None, None, report, "local_sdf_memory_allocation_failed", "rejected"
    except (RuntimeError, ValueError, TypeError):
        return None, None, report, "local_sdf_construction_failed", "rejected"
    report["smoothing"] = smoothing
    return zero_points, zero_faces, report, None, "ready"
