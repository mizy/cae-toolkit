"""Diagnostic SL03 projection using visible first-hit rays instead of closest points."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv
import vtk
from PIL import Image, ImageDraw

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from mesh_io import triangle_faces, write_vtp  # noqa: E402
from mesh_metrics import _pad_points_to_shared_bounds, signed_volume, triangle_quality  # noqa: E402
from solid_triangle_raster import SIX_VIEWS, rasterize_solid_view, rasterize_solid_views  # noqa: E402
from source_projected_closure import _vertex_normals  # noqa: E402
from source_projected_validation import exact_self_intersection_report, roundtrip_validation  # noqa: E402


DEFAULT_SOURCE = Path("outputs/sl03-internal-review-20261023-verified/source_no_shadow.vtp")
DEFAULT_TARGET = Path("outputs/sl03-internal-review-20261023-verified/exterior_proposal.vtp")
DEFAULT_CANDIDATE = Path("outputs/sl03-smooth-retry-20261026/round-840-native-faired/candidate.vtp")
DEFAULT_OUTPUT = Path("outputs/sl03-multiroute-20261026/directional")


def _polydata(points: np.ndarray, faces: np.ndarray) -> pv.PolyData:
    packed = np.empty((len(faces), 4), dtype=np.int64)
    packed[:, 0] = 3
    packed[:, 1:] = faces
    return pv.PolyData(points, packed.ravel())


def _write_json(path: Path, report: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _load_triangles(path: Path, *, remove_degenerate: bool) -> tuple[pv.PolyData, np.ndarray, np.ndarray, dict[str, int]]:
    mesh = pv.read(path)
    if isinstance(mesh, pv.MultiBlock):
        mesh = mesh.combine()
    if not isinstance(mesh, pv.PolyData):
        mesh = mesh.extract_surface()
    mesh = mesh.triangulate()
    points = np.asarray(mesh.points, dtype=np.float64)
    faces = triangle_faces(mesh)
    triangles = points[faces]
    areas = 0.5 * np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    valid = areas > 1.0e-16 if remove_degenerate else np.ones(len(faces), dtype=bool)
    result = _polydata(points, faces[valid])
    for name in mesh.cell_data.keys():
        values = np.asarray(mesh.cell_data[name])
        if len(values) == len(valid):
            result.cell_data[name] = values[valid]
    return result, points, faces[valid], {
        "raw_triangles": int(len(faces)),
        "valid_triangles": int(np.count_nonzero(valid)),
        "degenerate_triangles_removed": int(np.count_nonzero(~valid)),
    }


def _ray_hit(
    locator: vtk.vtkStaticCellLocator,
    point: np.ndarray,
    direction: np.ndarray,
    half_length: float,
    mutable: tuple[Any, ...],
) -> tuple[np.ndarray, int] | None:
    t, hit, pcoords, sub_id, cell_id, cell = mutable
    start = point + direction * half_length
    end = point - direction * half_length
    found = locator.IntersectWithLine(start, end, 1.0e-9, t, hit, pcoords, sub_id, cell_id, cell)
    if not found:
        return None
    return np.asarray(hit, dtype=np.float64), int(cell_id)


def _directional_points(
    points: np.ndarray,
    faces: np.ndarray,
    target: pv.PolyData,
    target_faces: np.ndarray,
    half_length: float,
    min_normal_dot: float,
    progress_path: Path,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    normals = _vertex_normals(points, faces)
    target_points = np.asarray(target.points, dtype=np.float64)
    triangles = target_points[target_faces]
    target_normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(target_normals, axis=1)
    target_normals = np.divide(
        target_normals,
        lengths[:, None],
        out=np.zeros_like(target_normals),
        where=lengths[:, None] > 1.0e-15,
    )

    locator = vtk.vtkStaticCellLocator()
    locator.SetDataSet(target)
    locator.BuildLocator()
    mutable = (
        vtk.mutable(0.0),
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        vtk.mutable(0),
        vtk.mutable(0),
        vtk.vtkGenericCell(),
    )

    output = points.copy()
    applied = np.zeros(len(points), dtype=np.uint8)
    ray_kind = np.zeros(len(points), dtype=np.uint8)  # 1=normal, 2=dominant axis, 0=fallback
    source_face = np.full(len(points), -1, dtype=np.int64)
    distance = np.zeros(len(points), dtype=np.float64)
    normal_dot = np.zeros(len(points), dtype=np.float32)
    rejected_no_hit = rejected_normal = 0
    best_distances: list[float] = []
    started = time.time()
    step = 50_000

    for begin in range(0, len(points), step):
        end = min(begin + step, len(points))
        for index in range(begin, end):
            normal = normals[index]
            normal_length = float(np.linalg.norm(normal))
            if normal_length <= 1.0e-12:
                rejected_no_hit += 1
                continue
            normal = normal / normal_length
            axis = np.zeros(3, dtype=np.float64)
            axis[int(np.argmax(np.abs(normal)))] = np.sign(normal[np.argmax(np.abs(normal))])
            hits = []
            for kind, direction in ((1, normal), (2, axis)):
                hit = _ray_hit(locator, points[index], direction, half_length, mutable)
                if hit is None:
                    continue
                hit_point, cell_id = hit
                alignment = abs(float(np.dot(normal, target_normals[cell_id])))
                displacement = float(np.linalg.norm(hit_point - points[index]))
                if alignment < min_normal_dot:
                    rejected_normal += 1
                    continue
                hits.append((displacement, kind, hit_point, cell_id, alignment))
            if not hits:
                rejected_no_hit += 1
                continue
            displacement, kind, hit_point, cell_id, alignment = min(hits, key=lambda row: row[0])
            output[index] = hit_point
            applied[index] = 1
            ray_kind[index] = kind
            source_face[index] = cell_id
            distance[index] = displacement
            normal_dot[index] = alignment
            best_distances.append(displacement)

        _write_json(progress_path, {
            "stage": "directional_first_hit_projection",
            "processed_vertices": end,
            "total_vertices": int(len(points)),
            "projected_vertices": int(applied[:end].sum()),
            "elapsed_seconds": time.time() - started,
            "ray_half_length_m": half_length,
        })
        print(
            f"ray projection {end}/{len(points)} vertices; projected={int(applied[:end].sum())}; "
            f"elapsed={time.time() - started:.1f}s",
            flush=True,
        )

    distance_stats = _percentiles(np.asarray(best_distances))
    point_data = {
        "directional_projection_applied": applied,
        "directional_projection_fallback": (1 - applied).astype(np.uint8),
        "directional_projection_source_face_id": source_face,
        "directional_projection_distance_m": distance,
        "directional_projection_ray_kind": ray_kind,
        "directional_projection_normal_dot_abs": normal_dot,
    }
    return output, point_data, {
        "method": "first_hit_from_outside_along_candidate_normal_and_dominant_axis",
        "visibility_rule": "cast from p + outward direction * half_length toward p - direction * half_length; choose nearest valid first hit",
        "normal_gate": {"absolute_dot_min": min_normal_dot},
        "ray_half_length_m": half_length,
        "vertices": int(len(points)),
        "projected_vertices": int(applied.sum()),
        "fallback_vertices": int(len(points) - applied.sum()),
        "projection_ratio": float(applied.mean()),
        "normal_ray_vertices": int(np.count_nonzero(ray_kind == 1)),
        "dominant_axis_ray_vertices": int(np.count_nonzero(ray_kind == 2)),
        "rejected_candidate_hits_normal_gate": int(rejected_normal),
        "fallback_no_accepted_hit": int(rejected_no_hit),
        "distance_m": distance_stats,
    }


def _percentiles(values: np.ndarray) -> dict[str, float | None]:
    values = np.asarray(values, dtype=np.float64)
    if not values.size:
        return {name: None for name in ("min", "p50", "p95", "p99", "max")}
    return {
        name: float(np.percentile(values, percentile))
        for name, percentile in (("min", 0), ("p50", 50), ("p95", 95), ("p99", 99), ("max", 100))
    }


def _weighted_p95(values: np.ndarray, weights: np.ndarray) -> float | None:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if not values.size or not np.any(weights > 0.0):
        return None
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    index = min(int(np.searchsorted(cumulative, cumulative[-1] * 0.95)), len(order) - 1)
    return float(values[order[index]])


def _surface_metrics(
    source_points: np.ndarray,
    source_faces: np.ndarray,
    candidate_points: np.ndarray,
    candidate_faces: np.ndarray,
    scale_mm_per_unit: float,
) -> dict[str, Any]:
    source_surface = _polydata(source_points, source_faces)
    candidate_surface = _polydata(candidate_points, candidate_faces)
    source_triangles = source_points[source_faces]
    candidate_triangles = candidate_points[candidate_faces]
    source_centroids = source_triangles.mean(axis=1)
    candidate_centroids = candidate_triangles.mean(axis=1)
    source_areas = 0.5 * np.linalg.norm(
        np.cross(source_triangles[:, 1] - source_triangles[:, 0], source_triangles[:, 2] - source_triangles[:, 1]), axis=1
    )
    candidate_areas = 0.5 * np.linalg.norm(
        np.cross(candidate_triangles[:, 1] - candidate_triangles[:, 0], candidate_triangles[:, 2] - candidate_triangles[:, 1]), axis=1
    )

    _, candidate_to_source = source_surface.find_closest_cell(candidate_centroids, return_closest_point=True)
    forward = np.linalg.norm(candidate_centroids - candidate_to_source, axis=1) * scale_mm_per_unit

    visible = rasterize_solid_views(source_points, source_faces, grid_size=384)
    visible_weights = np.sum(np.stack([view.face_first_hit_pixels for view in visible.views]), axis=0)
    visible_ids = np.flatnonzero(visible_weights > 0)
    reference_samples = source_centroids[visible_ids]
    _, source_to_candidate = candidate_surface.find_closest_cell(reference_samples, return_closest_point=True)
    reverse = np.linalg.norm(reference_samples - source_to_candidate, axis=1) * scale_mm_per_unit
    return {
        "method": "area_weighted_candidate_and_visible_source_triangle_centroids_with_384_grid_visibility",
        "scale_mm_per_unit_for_200mm": scale_mm_per_unit,
        "visible_source_face_count": int(len(visible_ids)),
        "visible_source_face_pixels": int(visible_weights[visible_ids].sum()),
        "forward_mm_at_200mm": {
            "p50": _weighted_percentile(forward, candidate_areas, 0.50),
            "p95": _weighted_percentile(forward, candidate_areas, 0.95),
            "p99": _weighted_percentile(forward, candidate_areas, 0.99),
            "sampled_max": float(forward.max()) if forward.size else None,
        },
        "reverse_visible_mm_at_200mm": {
            "p50": _weighted_percentile(reverse, source_areas[visible_ids], 0.50),
            "p95": _weighted_percentile(reverse, source_areas[visible_ids], 0.95),
            "p99": _weighted_percentile(reverse, source_areas[visible_ids], 0.99),
            "sampled_max": float(reverse.max()) if reverse.size else None,
        },
        "limitations": [
            "Discrete samples are not continuous Hausdorff or minimum-wall certificates.",
            "Visible-source sampling can miss subpixel apertures and thin spokes.",
            "Distance does not establish semantic ownership or preserve every wheel opening.",
        ],
    }


def _weighted_percentile(values: np.ndarray, weights: np.ndarray, percentile: float) -> float | None:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if not values.size or not np.any(weights > 0.0):
        return None
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    index = min(int(np.searchsorted(cumulative, cumulative[-1] * percentile)), len(order) - 1)
    return float(values[order[index]])


def _wheel_centres(target: pv.PolyData, target_faces: np.ndarray) -> list[dict[str, float]]:
    categories = (
        np.asarray(target.cell_data["review_category"])
        if "review_category" in target.cell_data
        else np.zeros(len(target_faces), dtype=np.uint8)
    )
    centers = target.points[target_faces].mean(axis=1)
    wheel = categories == 3
    result = []
    for side in (-1, 1):
        for end in (-1, 1):
            selected = wheel & (centers[:, 0] * side > 0.25) & (centers[:, 2] * end > 0.8)
            if not np.any(selected):
                continue
            mean = centers[selected].mean(axis=0)
            radial = np.linalg.norm(centers[selected, 1:3] - mean[1:3], axis=1)
            result.append({
                "side": int(side),
                "end": int(end),
                "x": float(mean[0]),
                "y": float(mean[1]),
                "z": float(mean[2]),
                "radius": float(np.percentile(radial, 95) * 1.2),
                "faces": int(np.count_nonzero(selected)),
            })
    return result


def _wheel_face_mask(
    points: np.ndarray,
    faces: np.ndarray,
    centres: list[dict[str, float]],
) -> np.ndarray:
    centroids = points[faces].mean(axis=1)
    mask = np.zeros(len(faces), dtype=bool)
    for wheel in centres:
        radial = np.hypot(centroids[:, 1] - wheel["y"], centroids[:, 2] - wheel["z"])
        mask |= (
            (np.abs(centroids[:, 0] - wheel["x"]) <= 0.30)
            & (radial <= wheel["radius"])
        )
    return mask


def _high_error_wheel_faces(
    target_points: np.ndarray,
    target_faces: np.ndarray,
    candidate_points: np.ndarray,
    candidate_faces: np.ndarray,
    centres: list[dict[str, float]],
    scale_mm_per_unit: float,
    threshold_mm: float,
    grid_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    ref_shared, candidate_shared = _pad_points_to_shared_bounds(target_points, candidate_points)
    wheel_faces = _wheel_face_mask(candidate_points, candidate_faces, centres)
    bad_pixels = np.zeros(len(candidate_faces), dtype=np.int64)
    view_reports = {}
    for view in SIX_VIEWS[:2]:
        reference = rasterize_solid_view(
            ref_shared, target_faces, view, grid_size=grid_size,
            depth_tolerance=0.0, collect_face_evidence=False,
        )
        candidate = rasterize_solid_view(
            candidate_shared, candidate_faces, view, grid_size=grid_size,
            depth_tolerance=0.0, collect_face_evidence=True,
        )
        common = reference.silhouette & candidate.silhouette
        error_mm = np.abs(reference.depth[common] - candidate.depth[common]) * scale_mm_per_unit
        high = np.zeros(common.shape, dtype=bool)
        high[common] = error_mm > threshold_mm
        face_ids = candidate.first_hit_face[high]
        face_ids = face_ids[face_ids >= 0]
        if face_ids.size:
            bad_pixels += np.bincount(face_ids, minlength=len(candidate_faces))
        view_reports[view.name] = {
            "common_pixels": int(common.sum()),
            "high_error_pixels": int(high.sum()),
            "depth_error_threshold_mm": threshold_mm,
            "depth_error_mm": _percentiles(error_mm),
        }
        del reference, candidate, common, error_mm, high, face_ids
    selected = wheel_faces & (bad_pixels >= 2)
    return selected, {
        "views": view_reports,
        "wheel_centres": centres,
        "wheel_zone_faces": int(wheel_faces.sum()),
        "selected_high_error_faces": int(selected.sum()),
        "selected_high_error_vertices": int(np.unique(candidate_faces[selected]).size) if np.any(selected) else 0,
        "minimum_high_error_pixels_per_candidate_face": 2,
    }


def _axis_first_hit_step(
    points: np.ndarray,
    faces: np.ndarray,
    selected_faces: np.ndarray,
    target: pv.PolyData,
    target_faces: np.ndarray,
    half_length: float,
    fraction: float,
    max_step_m: float,
    min_normal_dot: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    normals = _vertex_normals(points, faces)
    selected_vertices = np.zeros(len(points), dtype=bool)
    if np.any(selected_faces):
        selected_vertices[np.unique(faces[selected_faces])] = True
    target_triangles = np.asarray(target.points)[target_faces]
    target_normals = np.cross(
        target_triangles[:, 1] - target_triangles[:, 0],
        target_triangles[:, 2] - target_triangles[:, 0],
    )
    target_lengths = np.linalg.norm(target_normals, axis=1)
    target_normals = np.divide(
        target_normals,
        target_lengths[:, None],
        out=np.zeros_like(target_normals),
        where=target_lengths[:, None] > 1.0e-15,
    )
    locator = vtk.vtkStaticCellLocator()
    locator.SetDataSet(target)
    locator.BuildLocator()
    mutable = (
        vtk.mutable(0.0), [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
        vtk.mutable(0), vtk.mutable(0), vtk.vtkGenericCell(),
    )
    result = points.copy()
    moved = np.zeros(len(points), dtype=bool)
    face_ids = np.full(len(points), -1, dtype=np.int64)
    steps = np.zeros(len(points), dtype=np.float32)
    alignment = np.zeros(len(points), dtype=np.float32)
    hits = 0
    normal_rejected = 0
    for index in np.flatnonzero(selected_vertices):
        point = points[index]
        side = float(np.sign(point[0]))
        if side == 0.0 or normals[index, 0] * side < 0.25:
            continue
        hit = _ray_hit(locator, point, np.asarray([side, 0.0, 0.0]), half_length, mutable)
        if hit is None:
            continue
        hit_point, cell_id = hit
        normal_dot = abs(float(np.dot(normals[index], target_normals[cell_id])))
        if normal_dot < min_normal_dot:
            normal_rejected += 1
            continue
        delta = float(hit_point[0] - point[0])
        if abs(delta) < 5.0e-4:
            continue
        dx = float(np.clip(delta * fraction, -max_step_m, max_step_m))
        result[index, 0] += dx
        moved[index] = True
        face_ids[index] = cell_id
        steps[index] = dx
        alignment[index] = normal_dot
        hits += 1
    point_data = {
        "wheel_directional_moved": moved.astype(np.uint8),
        "wheel_directional_target_face_id": face_ids,
        "wheel_directional_step_m": steps,
        "wheel_directional_abs_normal_dot": alignment,
    }
    return result, moved, point_data, {
        "selected_vertices": int(selected_vertices.sum()),
        "accepted_ray_hits": hits,
        "moved_vertices": int(moved.sum()),
        "normal_gate_rejections": normal_rejected,
        "max_step_m": max_step_m,
        "fraction_toward_first_hit": fraction,
        "step_m": _percentiles(np.abs(steps[moved])),
    }


def _six_view_depth(
    reference_points: np.ndarray,
    reference_faces: np.ndarray,
    candidate_points: np.ndarray,
    candidate_faces: np.ndarray,
    scale_mm_per_unit: float,
    grid_size: int,
) -> dict[str, Any]:
    reference_shared, candidate_shared = _pad_points_to_shared_bounds(reference_points, candidate_points)
    reference = rasterize_solid_views(reference_shared, reference_faces, grid_size=grid_size, collect_face_evidence=False)
    candidate = rasterize_solid_views(candidate_shared, candidate_faces, grid_size=grid_size, collect_face_evidence=False)
    result = {}
    for ref_view, candidate_view in zip(reference.views, candidate.views, strict=True):
        common = ref_view.silhouette & candidate_view.silhouette
        delta = np.abs(ref_view.depth[common] - candidate_view.depth[common]) * scale_mm_per_unit
        union = np.count_nonzero(ref_view.silhouette | candidate_view.silhouette)
        intersection = np.count_nonzero(common)
        result[ref_view.view.name] = {
            "common_pixels": int(common.sum()),
            "silhouette_iou": float(intersection / union) if union else 1.0,
            "depth_error_mm_at_200mm": _percentiles(delta),
        }
    return {"grid_longest_axis": grid_size, "views": result}


def _certify_and_rollback(
    points: np.ndarray,
    baseline_points: np.ndarray,
    faces: np.ndarray,
    applied: np.ndarray,
    max_rounds: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    current = points.copy()
    active = applied.copy()
    rounds = []
    for round_index in range(max_rounds + 1):
        report = exact_self_intersection_report(current, faces, max_reported_pairs=100_000)
        rounds.append({key: report.get(key) for key in ("status", "passed", "intersection_pairs", "reported_pairs_truncated")})
        if report.get("passed") is True:
            break
        if report.get("status") != "computed" or report.get("reported_pairs_truncated"):
            break
        if round_index == max_rounds:
            break
        pairs = np.asarray(report.get("reported_pairs", []), dtype=np.int64).reshape(-1, 2)
        if not pairs.size:
            break
        affected = np.unique(faces[pairs].ravel())
        affected = affected[active[affected] != 0]
        if not affected.size:
            break
        current[affected] = baseline_points[affected]
        active[affected] = 0
    final = exact_self_intersection_report(current, faces, max_reported_pairs=100_000)
    final_summary = {
        "method": final.get("method"),
        "status": final.get("status"),
        "passed": final.get("passed"),
        "intersection_pairs": final.get("intersection_pairs"),
        "reported_pairs_truncated": final.get("reported_pairs_truncated"),
        "scan_scope": "all_faces",
        "kernel": final.get("kernel"),
        "rollback_rounds": rounds,
        "rolled_back_vertices": int(np.count_nonzero(applied & (active == 0))),
        "first_reported_pairs": final.get("reported_pairs", [])[:20],
        "failure_reason": final.get("failure_reason"),
    }
    return current, active, final_summary


def _cgal_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        key: report.get(key)
        for key in ("method", "status", "passed", "intersection_pairs", "reported_pairs_truncated", "kernel")
    }


def _run_wheel_continuation(
    args: argparse.Namespace,
    report: dict[str, Any],
    target: pv.PolyData,
    target_faces: np.ndarray,
    source_points: np.ndarray,
    baseline_points: np.ndarray,
    faces: np.ndarray,
) -> int:
    progress_path = args.output_dir / "progress.json"
    centres = _wheel_centres(target, target_faces)
    if not centres:
        raise ValueError("target has no review_category=3 wheel surfaces")
    current = baseline_points.copy()
    successful_stages = []
    last_path: Path | None = None
    last_points = current
    last_readback: dict[str, Any] | None = None
    scale = 200.0 / float(np.max(np.ptp(source_points, axis=0)))

    for stage, fraction in enumerate(args.local_step_fractions, start=1):
        selected, selection_report = _high_error_wheel_faces(
            target.points,
            target_faces,
            current,
            faces,
            centres,
            scale,
            args.depth_error_threshold_mm,
            args.depth_grid,
        )
        proposed, moved, point_data, ray_report = _axis_first_hit_step(
            current,
            faces,
            selected,
            target,
            target_faces,
            args.ray_half_length,
            fraction,
            args.max_step_m,
            args.min_normal_dot,
        )
        stage_report: dict[str, Any] = {
            "stage": stage,
            "fraction": fraction,
            "selection": selection_report,
            "ray_step": ray_report,
        }
        if ray_report["moved_vertices"] == 0:
            stage_report["status"] = "no_progress"
            successful_stages.append(stage_report)
            break

        cgal = exact_self_intersection_report(proposed, faces, max_reported_pairs=100_000)
        cgal_initial = _cgal_summary(cgal)
        rolled_back = 0
        if cgal.get("passed") is not True:
            if cgal.get("reported_pairs_truncated"):
                stage_report.update({"status": "rejected", "cgal_exact": cgal_initial})
                successful_stages.append(stage_report)
                break
            pairs = np.asarray(cgal.get("reported_pairs", []), dtype=np.int64).reshape(-1, 2)
            affected = np.unique(faces[pairs].ravel()) if pairs.size else np.zeros(0, dtype=np.int64)
            affected = affected[moved[affected]] if affected.size else affected
            if not affected.size or affected.size > max(1, int(moved.sum() * 0.5)):
                stage_report.update({"status": "rejected", "cgal_exact": cgal_initial, "local_rollback_vertices": int(affected.size)})
                successful_stages.append(stage_report)
                break
            proposed[affected] = current[affected]
            moved[affected] = False
            rolled_back = int(affected.size)
            cgal = exact_self_intersection_report(proposed, faces, max_reported_pairs=100_000)
            if cgal.get("passed") is not True:
                stage_report.update({"status": "rejected", "cgal_exact": _cgal_summary(cgal), "local_rollback_vertices": rolled_back})
                successful_stages.append(stage_report)
                break

        path = args.output_dir / f"wheel_continuation_step_{stage:02d}.vtp"
        point_data["wheel_directional_moved"] = moved.astype(np.uint8)
        point_data["wheel_directional_target_face_id"][~moved] = -1
        point_data["wheel_directional_step_m"][~moved] = 0.0
        point_data["wheel_directional_abs_normal_dot"][~moved] = 0.0
        write_vtp(path, proposed, faces, point_data=point_data)
        readback = roundtrip_validation(path, proposed, faces)
        stage_report.update({
            "status": "accepted_for_next_diagnostic_step" if readback.get("passed") else "readback_failed",
            "cgal_exact": _cgal_summary(cgal),
            "initial_cgal_exact": cgal_initial,
            "local_rollback_vertices": rolled_back,
            "moved_vertices_after_rollback": int(moved.sum()),
            "readback": readback,
            "candidate_vtp": str(path.resolve()),
        })
        successful_stages.append(stage_report)
        if readback.get("passed") is not True:
            break
        current = proposed
        last_points = proposed
        last_path = path
        last_readback = readback
        _write_json(progress_path, {
            "stage": "wheel_continuation_step_complete",
            "step": stage,
            "moved_vertices": int(moved.sum()),
            "cgal_passed": True,
            "readback_passed": True,
            "candidate": str(path.resolve()),
        })
        print(f"wheel step {stage}: moved={int(moved.sum())}, CGAL=0, readback=True", flush=True)

    report["method"] = "wheel-region-side-depth-error-guided-small-step-first-hit-continuation"
    report["steps"] = successful_stages
    report["outputs"] = {
        "candidate_vtp": str(last_path.resolve()) if last_path else None,
        "comparison_png": str((args.output_dir / "wheel_side_raster_comparison.png").resolve()) if last_path else None,
    }
    report["target_selection"] = {
        "view_scope": ["minus_x", "plus_x"],
        "depth_error_threshold_mm_at_200mm": args.depth_error_threshold_mm,
        "wheel_component_source": "target cell_data.review_category == 3, clustered by X side and Z axle",
        "geometry_scope": "selected high-depth-error first-hit faces inside the four target wheel zones",
    }
    if last_path is not None and last_readback is not None:
        final_cgal = next(
            (stage["cgal_exact"] for stage in reversed(successful_stages) if stage.get("status") == "accepted_for_next_diagnostic_step"),
            None,
        )
        report["final_candidate"] = str(last_path.resolve())
        report["readback"] = last_readback
        report["fidelity"] = _surface_metrics(target.points, target_faces, last_points, faces, scale)
        report["six_view_depth"] = _six_view_depth(
            target.points, target_faces, last_points, faces, scale, args.depth_grid
        )
        report["quality"] = triangle_quality(last_points, faces)
        report["signed_volume"] = signed_volume(last_points, faces)
        image_path = args.output_dir / "wheel_side_raster_comparison.png"
        _render_side_view(baseline_points, last_points, faces, image_path)
        report["geometry_gates"] = {
            "global_exact_cgal_passed": final_cgal is not None and final_cgal.get("passed") is True,
            "serialized_readback_passed": last_readback.get("passed") is True,
            "boundary_edges_zero": last_readback.get("raw", {}).get("boundary_edges") == 0,
            "non_manifold_edges_zero": last_readback.get("raw", {}).get("non_manifold_edges") == 0,
            "degenerate_faces_zero": last_readback.get("raw", {}).get("degenerate_faces") == 0,
            "appearance_and_printability_accepted": False,
        }
        report["status"] = "diagnostic"
    else:
        report["status"] = "no_accepted_geometry_change"
    report["limitations"] = [
        "Small-step movement retains the existing closed connectivity; it cannot create a missing aperture or change part ownership.",
        "Selected wheel depth pixels and sampled distances do not constitute continuous Hausdorff or wall-thickness certificates.",
        "The candidate remains diagnostic until smooth-shaded visual and print-specific geometry review passes.",
    ]
    report_path = args.output_dir / "directional_report.json"
    _write_json(report_path, report)
    _write_json(progress_path, {"stage": "complete", "report": str(report_path.resolve()), "candidate": report["outputs"]["candidate_vtp"]})
    return 0


def _render_side_view(
    baseline_points: np.ndarray,
    candidate_points: np.ndarray,
    candidate_faces: np.ndarray,
    output_path: Path,
) -> None:
    """Render an orthographic -X wheel crop with native reference normals."""
    native_path = Path("outputs/sl03-multiroute-20261026/reference_exterior_native.vtp")
    transferred_path = Path(
        "outputs/sl03-smooth-retry-20261026/round-840-native-faired/source_normals_candidate.vtp"
    )
    native = pv.read(native_path)
    native_points = np.asarray(native.points, dtype=np.float64)
    native_faces = triangle_faces(native)
    native_triangles = native_points[native_faces]
    native_areas = 0.5 * np.linalg.norm(
        np.cross(native_triangles[:, 1] - native_triangles[:, 0], native_triangles[:, 2] - native_triangles[:, 0]),
        axis=1,
    )
    native_valid = native_areas > 1.0e-16
    native_faces = native_faces[native_valid]
    native_normals = np.asarray(native.point_data["Normals"], dtype=np.float64)
    transferred = pv.read(transferred_path)
    candidate_normals = np.asarray(transferred.point_data["Normals"], dtype=np.float64)
    meshes = [
        (native_points, native_faces, native_normals, "Exterior + wheels (GLB NORMAL)"),
        (baseline_points, candidate_faces, candidate_normals, "Existing 840 (transferred normals)"),
        (candidate_points, candidate_faces, candidate_normals, "Directional candidate"),
    ]
    minimum = np.minimum.reduce([mesh_points.min(axis=0) for mesh_points, _, _, _ in meshes])
    maximum = np.maximum.reduce([mesh_points.max(axis=0) for mesh_points, _, _, _ in meshes])
    corners = np.asarray(
        [[x, y, z] for x in (minimum[0], maximum[0])
         for y in (minimum[1], maximum[1]) for z in (minimum[2], maximum[2])],
        dtype=np.float64,
    )
    view = SIX_VIEWS[0]  # -X camera: screen columns are Y; rows are Z.
    grid_size = 1024
    light = np.asarray([-0.72, 0.58, 0.36], dtype=np.float64)
    light /= np.linalg.norm(light)
    panels = []
    target_crop = None
    for points, faces, point_normals, label in meshes:
        shared_points = np.vstack((points, corners))
        evidence = rasterize_solid_view(
            shared_points,
            faces,
            view,
            grid_size=grid_size,
            depth_tolerance=0.0,
            collect_face_evidence=True,
        )
        ids = evidence.first_hit_face
        face_normals = point_normals[faces].mean(axis=1)
        lengths = np.linalg.norm(face_normals, axis=1)
        face_normals = np.divide(
            face_normals,
            lengths[:, None],
            out=np.zeros_like(face_normals),
            where=lengths[:, None] > 1.0e-15,
        )
        valid = ids >= 0
        intensity = np.zeros(ids.shape, dtype=np.float32)
        if np.any(valid):
            intensity[valid] = 0.27 + 0.73 * np.abs(face_normals[ids[valid]] @ light)
        base_color = np.asarray([137.0, 148.0, 156.0], dtype=np.float32)
        image = np.zeros((*ids.shape, 3), dtype=np.uint8)
        image[:] = np.asarray([22, 31, 39], dtype=np.uint8)
        image[valid] = np.clip(base_color[None, :] * intensity[valid, None], 0, 255).astype(np.uint8)

        if target_crop is None:
            axes = np.asarray(view.project_axes, dtype=np.int64)
            projected_min = minimum[axes]
            projected_span = np.maximum(maximum[axes] - projected_min, 1.0e-15)
            rows, cols = ids.shape
            # Crop the positive-Z wheel, its spokes, and the surrounding arch.
            col_min = int(np.floor((0.0 - projected_min[0]) / projected_span[0] * (cols - 1)))
            col_max = int(np.ceil((0.88 - projected_min[0]) / projected_span[0] * (cols - 1)))
            row_min = int(np.floor((1.00 - projected_min[1]) / projected_span[1] * (rows - 1)))
            row_max = int(np.ceil((2.10 - projected_min[1]) / projected_span[1] * (rows - 1)))
            target_crop = (max(0, row_min), min(rows, row_max + 1), max(0, col_min), min(cols, col_max + 1))
        row_min, row_max, col_min, col_max = target_crop
        crop = Image.fromarray(image[row_min:row_max, col_min:col_max])
        panels.append((crop.resize((720, 650), Image.Resampling.LANCZOS), label))
        del evidence, ids, face_normals, image

    canvas = Image.new("RGB", (2160, 700), (14, 21, 29))
    draw = ImageDraw.Draw(canvas)
    for index, (panel, label) in enumerate(panels):
        x = index * 720
        canvas.paste(panel, (x, 42))
        draw.text((x + 14, 12), label, fill=(226, 234, 238))
    canvas.save(output_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ray-half-length", type=float, default=0.12)
    parser.add_argument("--min-normal-dot", type=float, default=0.35)
    parser.add_argument("--cgal-rollback-rounds", type=int, default=6)
    parser.add_argument("--depth-grid", type=int, default=512)
    parser.add_argument("--mode", choices=("full", "wheel-continuation"), default="full")
    parser.add_argument("--depth-error-threshold-mm", type=float, default=0.5)
    parser.add_argument("--max-step-m", type=float, default=0.003)
    parser.add_argument("--local-step-fractions", type=float, nargs="+", default=[0.25, 0.5])
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_dir / "progress.json"
    report_path = args.output_dir / "directional_report.json"
    report: dict[str, Any] = {
        "status": "diagnostic",
        "source": str(args.source.resolve()),
        "target": str(args.target.resolve()),
        "baseline_candidate": str(args.candidate.resolve()),
        "outputs": {
            "candidate_vtp": str((args.output_dir / "candidate_directional_diagnostic.vtp").resolve()),
            "comparison_png": str((args.output_dir / "wheel_side_comparison.png").resolve()),
        },
        "acceptance": "diagnostic_only_until_visual, fidelity, topology, exact_CGAL, and serialized_readback_gates_pass",
    }
    started = time.time()
    _write_json(progress_path, {"stage": "loading_inputs", "started_unix": started})
    source_raw = pv.read(args.source)
    source_points = np.asarray(source_raw.points, dtype=np.float64)
    source_faces = triangle_faces(source_raw)
    target, _, target_faces, target_counts = _load_triangles(args.target, remove_degenerate=True)
    candidate_raw = pv.read(args.candidate)
    candidate_points = np.asarray(candidate_raw.points, dtype=np.float64)
    candidate_faces = triangle_faces(candidate_raw)
    candidate_triangles = candidate_points[candidate_faces]
    candidate_areas = 0.5 * np.linalg.norm(
        np.cross(candidate_triangles[:, 1] - candidate_triangles[:, 0], candidate_triangles[:, 2] - candidate_triangles[:, 0]),
        axis=1,
    )
    if np.any(candidate_areas <= 1.0e-16):
        raise ValueError("baseline candidate contains degenerate faces; refusing to mutate its topology")
    report["inputs"] = {
        "source_points": int(len(source_points)),
        "source_raw_triangles": int(len(source_faces)),
        "target": target_counts,
        "baseline_candidate_points": int(len(candidate_points)),
        "baseline_candidate_triangles": int(len(candidate_faces)),
    }
    _write_json(progress_path, {"stage": "inputs_loaded", **report["inputs"]})
    print(f"inputs ready: {report['inputs']}", flush=True)

    if args.mode == "wheel-continuation":
        return _run_wheel_continuation(
            args, report, target, target_faces, source_points, candidate_points, candidate_faces
        )

    projected, point_data, projection = _directional_points(
        candidate_points,
        candidate_faces,
        target,
        target_faces,
        args.ray_half_length,
        args.min_normal_dot,
        progress_path,
    )
    report["projection"] = projection
    applied = point_data["directional_projection_applied"].astype(bool)
    projected, applied, cgal = _certify_and_rollback(
        projected,
        candidate_points,
        candidate_faces,
        applied,
        args.cgal_rollback_rounds,
    )
    point_data["directional_projection_applied"] = applied.astype(np.uint8)
    point_data["directional_projection_fallback"] = (~applied).astype(np.uint8)
    point_data["directional_projection_source_face_id"][~applied] = -1
    point_data["directional_projection_distance_m"][~applied] = 0.0
    point_data["directional_projection_ray_kind"][~applied] = 0
    point_data["directional_projection_normal_dot_abs"][~applied] = 0.0
    report["cgal_exact"] = cgal
    candidate_path = args.output_dir / "candidate_directional_diagnostic.vtp"
    write_vtp(candidate_path, projected, candidate_faces, point_data=point_data)
    _write_json(progress_path, {"stage": "serialized_candidate", "cgal_passed": cgal["passed"], "candidate_vtp": str(candidate_path.resolve())})
    print(f"CGAL exact complete: passed={cgal['passed']} pairs={cgal['intersection_pairs']}", flush=True)

    readback = roundtrip_validation(candidate_path, projected, candidate_faces)
    report["readback"] = readback
    report["quality"] = {
        "candidate_triangle_quality": triangle_quality(projected, candidate_faces),
        "signed_volume": signed_volume(projected, candidate_faces),
        "bounds": {
            "minimum": projected.min(axis=0).tolist(),
            "maximum": projected.max(axis=0).tolist(),
            "extents": np.ptp(projected, axis=0).tolist(),
        },
        "roundtrip_topology": readback,
    }
    _write_json(progress_path, {"stage": "surface_metrics", "readback_passed": readback.get("passed")})

    scale_mm_per_unit = 200.0 / float(np.max(np.ptp(source_points, axis=0)))
    report["fidelity"] = _surface_metrics(target.points, target_faces, projected, candidate_faces, scale_mm_per_unit)
    report["six_view_depth"] = _six_view_depth(
        target.points, target_faces, projected, candidate_faces, scale_mm_per_unit, args.depth_grid
    )
    _write_json(progress_path, {"stage": "rendering", "fidelity_ready": True})

    image_path = args.output_dir / "wheel_side_comparison.png"
    _render_side_view(candidate_points, projected, candidate_faces, image_path)
    report["elapsed_seconds"] = time.time() - started
    report["geometry_gates"] = {
        "global_exact_cgal_passed": cgal.get("passed") is True,
        "serialized_readback_passed": readback.get("passed") is True,
        "boundary_edges_zero": readback.get("raw", {}).get("boundary_edges") == 0,
        "non_manifold_edges_zero": readback.get("raw", {}).get("non_manifold_edges") == 0,
        "degenerate_faces_zero": readback.get("raw", {}).get("degenerate_faces") == 0,
        "appearance_and_printability_accepted": False,
    }
    _write_json(report_path, report)
    _write_json(progress_path, {
        "stage": "complete",
        "report": str(report_path.resolve()),
        "candidate": str(candidate_path.resolve()),
        "cgal_passed": cgal.get("passed"),
        "readback_passed": readback.get("passed"),
        "elapsed_seconds": report["elapsed_seconds"],
    })
    print(
        "complete: "
        f"projection={projection['projection_ratio']:.4%}, "
        f"CGAL={cgal.get('passed')}, readback={readback.get('passed')}, "
        f"forward_p95={report['fidelity']['forward_mm_at_200mm']['p95']:.4f}mm, "
        f"reverse_p95={report['fidelity']['reverse_visible_mm_at_200mm']['p95']:.4f}mm",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
