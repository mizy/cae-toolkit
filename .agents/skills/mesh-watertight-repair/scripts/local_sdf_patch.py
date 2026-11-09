"""Build a bounded SDF patch candidate inside a source-face ROI."""

from __future__ import annotations

from numbers import Integral
from typing import Any

import numpy as np
import pyvista as pv
import trimesh
from local_defect_regions import extract_defect_regions
from local_voxel_field import build_local_zero_surface, closest_distances
from mesh_metrics import edge_topology, face_component_labels
from source_patch_collar import connect_source_collar
from source_projected_closure import ProjectionThresholds, project_closure_to_source


def _boundary_loops(points: np.ndarray, faces: np.ndarray) -> list[list[int]]:
    regions = extract_defect_regions(
        points,
        faces,
        seed_faces=np.arange(len(faces), dtype=np.int64),
        halo_rings=0,
        max_region_faces=len(faces),
        max_total_faces=len(faces),
    )
    if regions["status"] != "computed" or len(regions["regions"]) != 1:
        return []
    region = regions["regions"][0]
    if region["boundary"]["status"] != "closed_loops":
        return []
    return region["boundary_loops"]


def _validate_inputs(
    points: np.ndarray,
    faces: np.ndarray,
    rows: np.ndarray,
    bounds: np.ndarray,
    pitch: float,
    max_voxels: int,
    max_patch_area: float,
    max_error: float,
    max_bridge_length: float,
    smoothing_sigma_voxels: float,
) -> tuple[
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    str | None,
]:
    try:
        source_points = np.asarray(points, dtype=np.float64)
        source_faces = np.asarray(faces, dtype=np.int64)
        face_rows = np.asarray(rows, dtype=np.int64).reshape(-1)
        roi = np.asarray(bounds, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        return None, None, None, None, "invalid_input_arrays"
    if (
        source_points.ndim != 2
        or source_points.shape[1] != 3
        or source_faces.ndim != 2
        or source_faces.shape[1] != 3
        or not len(source_points)
        or not len(source_faces)
        or not np.isfinite(source_points).all()
        or source_faces.min() < 0
        or source_faces.max() >= len(source_points)
    ):
        return None, None, None, None, "invalid_source_mesh"
    if (
        roi.shape != (2, 3)
        or not np.isfinite(roi).all()
        or np.any(roi[0] >= roi[1])
        or not np.isfinite(roi[1] - roi[0]).all()
    ):
        return None, None, None, None, "invalid_roi_bounds"
    if (
        face_rows.size == 0
        or face_rows.min() < 0
        or face_rows.max() >= len(source_faces)
        or len(np.unique(face_rows)) != len(face_rows)
    ):
        return None, None, None, None, "invalid_defect_face_rows"
    if (
        not np.isfinite(pitch)
        or pitch <= 0
        or not isinstance(max_voxels, Integral)
        or max_voxels <= 0
        or not np.isfinite(max_patch_area)
        or max_patch_area <= 0
        or not np.isfinite(max_error)
        or max_error < 0
        or not np.isfinite(max_bridge_length)
        or max_bridge_length <= 0
        or not np.isfinite(smoothing_sigma_voxels)
        or smoothing_sigma_voxels < 0
    ):
        return None, None, None, None, "invalid_patch_budget_or_resolution"
    return source_points, source_faces, face_rows, roi, None


def _source_region_loops(
    points: np.ndarray, faces: np.ndarray, rows: np.ndarray
) -> tuple[float | None, list[list[int]], str | None]:
    source = trimesh.Trimesh(vertices=points, faces=faces, process=False)
    if not source.is_volume or float(source.volume) <= 0.0:
        return None, [], "source_component_must_be_closed_consistent_and_positive"
    _topology, source_edges = edge_topology(faces)
    if len(np.unique(face_component_labels(len(faces), source_edges))) != 1:
        return None, [], "source_input_must_be_one_component"
    loops = _boundary_loops(points, faces[rows])
    if len(loops) != 1:
        return (
            float(source.volume),
            loops,
            "source_region_requires_one_simple_boundary_loop",
        )
    return float(source.volume), loops, None


def _prepare_source_region(
    points: np.ndarray,
    faces: np.ndarray,
    rows: np.ndarray,
    bounds: np.ndarray,
    pitch: float,
) -> tuple[
    float | None,
    list[list[int]],
    np.ndarray,
    pv.PolyData | None,
    float,
    str | None,
]:
    volume, loops, reason = _source_region_loops(points, faces, rows)
    if reason:
        return volume, loops, np.zeros(0, dtype=np.int64), None, 0.0, reason
    cut_vertices = np.unique(np.asarray(loops[0][:-1], dtype=np.int64))
    core_rows = rows[~np.any(np.isin(faces[rows], cut_vertices), axis=1)]
    if not len(core_rows):
        return volume, loops, core_rows, None, 0.0, "source_roi_has_no_interior_faces"
    tolerance = max(pitch * 1e-9, np.finfo(np.float64).eps * 128)
    defect_vertices = points[faces[rows].ravel()]
    if np.any(defect_vertices < bounds[0] - tolerance) or np.any(defect_vertices > bounds[1] + tolerance):
        return volume, loops, core_rows, None, tolerance, "defect_faces_outside_roi"
    surface = pv.PolyData(
        points,
        np.column_stack((np.full(len(faces), 3), faces)).ravel(),
    )
    return volume, loops, core_rows, surface, tolerance, None


def _select_inset_patch(
    zero_points: np.ndarray,
    zero_faces: np.ndarray,
    source_surface: pv.PolyData,
    source_core_rows: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None, list[int], str | None]:
    try:
        face_owners = np.asarray(
            source_surface.find_closest_cell(zero_points[zero_faces].mean(axis=1)),
            dtype=np.int64,
        )
        vertex_owners = np.asarray(source_surface.find_closest_cell(zero_points), dtype=np.int64)
    except (MemoryError, RuntimeError, ValueError, TypeError):
        return None, None, [], "source_face_ownership_query_failed"
    keep = np.isin(face_owners, source_core_rows)
    keep &= np.all(np.isin(vertex_owners[zero_faces], source_core_rows), axis=1)
    selected_faces = zero_faces[keep]
    if not len(selected_faces):
        return None, None, [], "no_zero_faces_owned_by_inset_source_roi"
    used = np.unique(selected_faces)
    remap = np.full(len(zero_points), -1, dtype=np.int64)
    remap[used] = np.arange(len(used), dtype=np.int64)
    return (
        zero_points[used],
        remap[selected_faces],
        np.unique(face_owners[keep]).astype(int).tolist(),
        None,
    )


def _check_patch(
    points: np.ndarray,
    faces: np.ndarray,
    source_loops: list[list[int]],
    source_surface: pv.PolyData,
    bounds: np.ndarray,
    *,
    max_patch_area: float,
    max_error: float,
    tolerance: float,
) -> tuple[dict[str, Any], str | None, str]:
    loops = _boundary_loops(points, faces)
    report: dict[str, Any] = {"patch_boundary_loops": loops}
    if len(loops) != len(source_loops):
        return (
            report,
            "source_and_sdf_boundary_loop_count_mismatch",
            "unsupported",
        )
    if len(loops) != 1:
        return (
            report,
            "sdf_region_requires_one_simple_boundary_loop",
            "unsupported",
        )
    if max(len(loops[0]), len(source_loops[0])) - 1 > 4096:
        return (
            report,
            "zipper_boundary_vertex_budget_exceeded",
            "budget_exceeded",
        )

    area = float(trimesh.Trimesh(vertices=points, faces=faces, process=False).area)
    report["patch_area"] = area
    if not np.isfinite(area) or area > max_patch_area:
        return report, "patch_area_exceeded", "budget_exceeded"
    if np.any(points < bounds[0] - tolerance) or np.any(points > bounds[1] + tolerance):
        return report, "sdf_patch_escaped_roi", "rejected"

    samples = np.vstack((points, points[faces].mean(axis=1)))
    try:
        errors = closest_distances(source_surface, samples)
    except ValueError:
        return report, "source_distance_query_failed", "rejected"
    maximum_error = float(errors.max(initial=0.0))
    report["source_distance"] = {
        "method": "vtk_closest_cell_to_full_source_component",
        "samples": len(samples),
        "max_error": maximum_error,
        "limit": float(max_error),
    }
    if maximum_error > max_error:
        return report, "source_distance_exceeded", "rejected"
    return report, None, "ready"


def _project_trusted_interior(
    points, faces, source_points, source_faces, trusted_rows, pitch, max_error, *, preserve_boundary_ring=True
):
    if trusted_rows is None or not len(trusted_rows):
        return points, {"status": "not_requested", "reason": "no declared trusted source support"}
    trusted_rows = np.asarray(trusted_rows, dtype=np.int64)
    if np.any(trusted_rows < 0) or np.any(trusted_rows >= len(source_faces)):
        raise ValueError("trusted projection source face rows are invalid")
    reliable = np.zeros(len(source_faces), dtype=bool)
    reliable[trusted_rows] = True
    _, edges = edge_topology(faces)
    rim = np.unique(
        np.asarray(
            [vertex for edge, incident in edges.items() if len(incident) == 1 for vertex in edge], dtype=np.int64
        )
    )
    rim_faces = np.asarray([incident[0] for incident in edges.values() if len(incident) == 1], dtype=np.int64)
    protected = np.unique(faces[rim_faces]) if preserve_boundary_ring else rim
    movable = np.ones(len(points), dtype=bool)
    movable[protected] = False
    projected, _, provenance, report = project_closure_to_source(
        points,
        faces,
        source_points,
        source_faces,
        reliable_source_face_mask=reliable,
        movable_vertex_mask=movable,
        thresholds=ProjectionThresholds(
            max_projection_distance=min(2 * pitch, max_error), source_edge_barycentric_margin=0.0
        ),
    )
    applied = np.asarray(provenance["point_data"]["source_projection_applied"], dtype=bool)
    rows = provenance["point_data"]["nearest_source_face_id"]
    triangles = source_points[source_faces[rows]]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1)[:, None], np.finfo(float).tiny)
    delta = projected - points
    normal_motion = np.einsum("ij,ij->i", delta, normals)
    tangent = np.linalg.norm(delta - normal_motion[:, None] * normals, axis=1)
    tolerance = max(float(np.linalg.norm(np.ptp(source_points, axis=0))) * 1e-12, 1e-15)
    rejected = applied & (tangent > 0.25 * np.abs(normal_motion) + tolerance)
    projected[rejected] = points[rejected]
    applied[rejected] = False
    return projected, {
        "status": "computed",
        "method": report["method"],
        "trusted_source_faces": len(trusted_rows),
        "projected_vertices": int(applied.sum()),
        "moved_vertices": int(np.count_nonzero(np.linalg.norm(projected - points, axis=1) > tolerance)),
        "fixed_boundary_vertices": len(rim),
        "tangential_clamp_rejected": int(rejected.sum()),
        "fixed_boundary_ring_vertices": len(protected),
        "max_projection_distance": float(np.linalg.norm(projected - points, axis=1).max(initial=0.0)),
        "boundary_unchanged": bool(np.array_equal(projected[rim], points[rim])),
    }


def _rebuild_common_contour(
    zero_points,
    zero_faces,
    source_surface,
    rows,
    source_loops,
    bounds,
    *,
    pitch,
    max_error,
    max_patch_area,
    max_bridge_length,
    tolerance,
    trusted_rows,
):
    from mesh_io import triangle_faces
    from source_contour_stitch import clip_source_contour, stitch_source_contour

    source_points, source_faces = np.asarray(source_surface.points), triangle_faces(source_surface)
    points, faces, cut = clip_source_contour(
        zero_points, zero_faces, source_points, source_faces, rows, source_loops[0]
    )
    points, projection = _project_trusted_interior(
        points, faces, source_points, source_faces, trusted_rows, pitch, max_error, preserve_boundary_ring=False
    )
    report, invalid, _ = _check_patch(
        points,
        faces,
        source_loops,
        source_surface,
        bounds,
        max_patch_area=max_patch_area,
        max_error=max_error,
        tolerance=tolerance,
    )
    if invalid:
        raise ValueError(invalid)
    points, faces, connection = stitch_source_contour(
        source_points,
        source_faces,
        rows,
        source_loops[0],
        points,
        faces,
        report["patch_boundary_loops"][0],
        max_displacement=min(2 * pitch, max_error, max_bridge_length),
    )
    report.update(
        connection,
        source_contour_cut=cut,
        source_projection=projection,
        method="local_sdf_shared_source_contour",
        status="ready",
        reason=None,
    )
    return points, faces, report


# @entry: propose a bounded implicit patch with a fixed collar; caller owns global acceptance.
def build_local_sdf_patch(
    points: np.ndarray,
    faces: np.ndarray,
    *,
    defect_face_rows: np.ndarray,
    roi_bounds: np.ndarray,
    pitch: float,
    max_voxels: int,
    max_patch_area: float,
    max_error: float,
    max_bridge_length: float,
    smoothing_sigma_voxels: float = 0.0,
    reliable_source_face_rows: np.ndarray | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, Any]]:
    """Return an open SDF patch and its source/patch loops, or no patch on failure.

    ``roi_bounds`` is ``((xmin, ymin, zmin), (xmax, ymax, zmax))``. The input
    must be one closed, consistently wound, positive-volume component. The
    selected source face rows lose their boundary-touching ring; zero faces
    must map to the remaining source rows before a bounded zipper is built.
    The function never mutates its source arrays.
    """
    report: dict[str, Any] = {
        "schema": "local_sdf_patch/v1",
        "status": "rejected",
        "reason": None,
        "patch_boundary_loops": [],
        "source_boundary_loops": [],
        "zipper_faces": [],
    }

    def reject(reason: str, status: str = "unsupported"):
        report.update(status=status, reason=reason)
        return None, None, report

    source_points, source_faces, rows, bounds, invalid = _validate_inputs(
        points,
        faces,
        defect_face_rows,
        roi_bounds,
        pitch,
        max_voxels,
        max_patch_area,
        max_error,
        max_bridge_length,
        smoothing_sigma_voxels,
    )
    if invalid:
        return reject(invalid)
    volume, source_loops, core_rows, source_surface, tolerance, invalid = _prepare_source_region(
        source_points, source_faces, rows, bounds, float(pitch)
    )
    report["source_boundary_loops"] = source_loops
    if invalid:
        return reject(invalid)
    report["source_volume"] = volume
    report["source_roi"] = {
        "face_rows": len(rows),
        "inset_face_rows": core_rows.astype(int).tolist(),
        "removed_boundary_ring_faces": len(rows) - len(core_rows),
    }
    source_patch_area = float(
        trimesh.Trimesh(
            vertices=source_points,
            faces=source_faces[rows],
            process=False,
        ).area
    )
    report["source_patch_area"] = source_patch_area
    if not np.isfinite(source_patch_area) or source_patch_area > max_patch_area:
        return reject("source_patch_area_exceeded", status="budget_exceeded")
    zero_points, zero_faces, grid, invalid, status = build_local_zero_surface(
        source_points,
        source_faces,
        source_surface,
        bounds,
        pitch=float(pitch),
        max_voxels=int(max_voxels),
        smoothing_sigma_voxels=float(smoothing_sigma_voxels),
    )
    report["grid"] = grid
    if invalid:
        return reject(invalid, status)
    patch_points, patch_faces, owner_rows, invalid = _select_inset_patch(
        zero_points,
        zero_faces,
        source_surface,
        core_rows,
    )
    if invalid:
        return reject(invalid)
    try:
        patch_points, projection = _project_trusted_interior(
            patch_points, patch_faces, source_points, source_faces, reliable_source_face_rows, pitch, max_error
        )
    except ValueError as exc:
        return reject(str(exc))
    report["source_projection"] = projection
    patch_report, invalid, status = _check_patch(
        patch_points,
        patch_faces,
        source_loops,
        source_surface,
        bounds,
        max_patch_area=float(max_patch_area),
        max_error=float(max_error),
        tolerance=tolerance,
    )
    report.update(patch_report)
    report["source_roi"]["owned_inset_face_rows"] = owner_rows
    if invalid:
        return reject(invalid, status)
    try:
        collar_points, _collar_faces, connection = connect_source_collar(
            source_points,
            source_faces,
            rows,
            source_loops[0],
            patch_points,
            patch_faces,
            report["patch_boundary_loops"][0],
            max_bridge_length=float(max_bridge_length),
        )
    except ValueError as exc:
        report["zipper"] = {
            "status": "rejected",
            "reason": str(exc),
            "max_normal_angle": 20.0,
            "max_bridge_length": float(max_bridge_length),
        }
        try:
            patch_points, patch_faces, common = _rebuild_common_contour(
                zero_points,
                zero_faces,
                source_surface,
                rows,
                source_loops,
                bounds,
                pitch=pitch,
                max_error=max_error,
                max_patch_area=max_patch_area,
                max_bridge_length=max_bridge_length,
                tolerance=tolerance,
                trusted_rows=reliable_source_face_rows,
            )
        except ValueError as common_error:
            report["source_contour_attempt"] = {"status": "rejected", "reason": str(common_error)}
            return reject(str(common_error))
        report["zipper_attempt"] = report.pop("zipper")
        report.update(common)
        return patch_points, patch_faces, report
    report.update(connection)
    patch_points = np.vstack((collar_points, patch_points))
    patch_faces = patch_faces + len(collar_points)
    report.update(
        status="ready",
        reason="local_sdf_patch_ready_for_zipper",
        method="local_fast_winding_sdf_zero_surface",
        smoothing=grid["smoothing"],
        patch={"points": len(patch_points), "triangles": len(patch_faces)},
    )
    return patch_points, patch_faces, report
