from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv
from PIL import Image, ImageDraw

from cfd_surface_refine import source_target_sizes
from mesh_io import triangle_faces, write_vtp
from mesh_metrics import mesh_report
from solid_triangle_raster import SIX_VIEWS, rasterize_solid_view
from solid_triangle_raster import rasterize_solid_views
from source_projected_closure import ProjectionThresholds, project_closure_to_source
from source_projected_validation import (
    exact_self_intersection_report,
    repair_projected_self_intersections,
    roundtrip_validation,
    watertight_metrics,
)

ROOT = Path.cwd()
SOURCE_ROOT = ROOT / "outputs/sl03-internal-review-20261023-verified"
ROUTE_ROOT = ROOT / "outputs/sl03-multiroute-20261026/feature-remesh"
BASELINE = ROOT / "outputs/sl03-smooth-retry-20261026/round-840-native-faired/candidate.vtp"
FAIRING_REPORT = BASELINE.parent / "fairing_report.json"
SOURCE_TARGET = SOURCE_ROOT / "exterior_proposal.vtp"
SOURCE_REFERENCE = SOURCE_ROOT / "source_no_shadow.vtp"
SOURCE_DISPLAY = ROOT / "outputs/sl03-multiroute-20261026/reference_exterior_native.vtp"
MAX_SMOOTH_DISPLACEMENT_FACTOR = 0.5


def save_progress(stage: str, started: float, **extra: Any) -> None:
    row = {"stage": stage, "elapsed_seconds": round(time.time() - started, 1), **extra}
    (ROUTE_ROOT / "progress.json").write_text(
        json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(row, ensure_ascii=False), flush=True)


def triangle_areas(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    triangles = points[faces]
    return np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1
    ) * 0.5


def feature_constrained_smooth(
    points: np.ndarray, faces: np.ndarray, feature_fraction: np.ndarray, max_displacement: float
) -> tuple[np.ndarray, dict[str, Any]]:
    packed = np.empty((len(faces), 4), dtype=np.int64)
    packed[:, 0] = 3
    packed[:, 1:] = faces
    mesh = pv.PolyData(points, packed.ravel())
    smooth = mesh.smooth_taubin(
        n_iter=30,
        pass_band=0.1,
        edge_angle=15.0,
        feature_angle=45.0,
        boundary_smoothing=False,
        feature_smoothing=False,
        normalize_coordinates=True,
    )
    delta = np.asarray(smooth.points, dtype=np.float64) - points
    peak = float(np.linalg.norm(delta, axis=1).max())
    global_blend = min(1.0, max_displacement / max(peak, 1e-15))
    feature_strength = np.clip((feature_fraction - 0.2) / 0.6, 0.0, 1.0)
    mobility = 1.0 - feature_strength
    result = points + delta * (global_blend * mobility[:, None])
    moved = np.linalg.norm(result - points, axis=1)
    return result, {
        "method": "vtk_windowed_sinc_with_source_feature_weighted_displacement",
        "iterations": 30,
        "pass_band": 0.1,
        "edge_angle_degrees": 15.0,
        "feature_angle_degrees": 45.0,
        "source_feature_fraction_full_smoothing_below": 0.2,
        "source_feature_fraction_locked_above": 0.8,
        "vertices_smoothed": int(np.count_nonzero(mobility > 0.0)),
        "vertices_locked_by_source_features": int(np.count_nonzero(mobility == 0.0)),
        "raw_windowed_sinc_max_displacement_m": peak,
        "global_blend": global_blend,
        "max_actual_displacement_m": float(moved.max()),
        "displacement_limit_m": max_displacement,
        "connectivity_changed": False,
    }


def weighted_percentiles(values: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    cumulative = np.cumsum(weights[order])
    total = float(cumulative[-1])
    return {
        f"p{q}": float(
            sorted_values[
                min(int(np.searchsorted(cumulative, (q / 100.0) * total, side="left")), len(sorted_values) - 1)
            ]
        )
        for q in (50, 95, 99)
    }


def centroid_distances(
    points: np.ndarray, faces: np.ndarray, target: pv.PolyData, chunk_size: int = 10_000
) -> tuple[np.ndarray, np.ndarray]:
    distances, weights = [], []
    for start in range(0, len(faces), chunk_size):
        triangles = points[faces[start : start + chunk_size]]
        area = np.linalg.norm(
            np.cross(
                triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
            ),
            axis=1,
        ) * 0.5
        valid = area > 1e-16
        if not np.any(valid):
            continue
        centers = triangles[valid].mean(axis=1)
        query = pv.PolyData(np.asarray(centers, dtype=np.float64))
        sampled = query.compute_implicit_distance(target)
        distances.append(np.abs(np.asarray(sampled.point_data["implicit_distance"], dtype=np.float64)))
        weights.append(area[valid])
    return np.concatenate(distances), np.concatenate(weights)


def point_distances(points: np.ndarray, target: pv.PolyData, chunk_size: int = 10_000) -> np.ndarray:
    distances = []
    for start in range(0, len(points), chunk_size):
        query = pv.PolyData(np.asarray(points[start : start + chunk_size], dtype=np.float64))
        sampled = query.compute_implicit_distance(target)
        distances.append(np.abs(np.asarray(sampled.point_data["implicit_distance"], dtype=np.float64)))
    return np.concatenate(distances)


def render_comparison(
    source_path: Path, baseline_path: Path, candidate_path: Path, output_path: Path
) -> None:
    items = (
        ("Original exterior and wheels", source_path),
        ("Native faired baseline", baseline_path),
        ("Feature constrained projection", candidate_path),
    )
    views = (("Side", "minus_x"), ("Front", "plus_z"))
    meshes = []
    for label, path in items:
        mesh = pv.read(path).triangulate()
        points = np.asarray(mesh.points, dtype=np.float64)
        faces = triangle_faces(mesh).astype(np.int64, copy=False)
        area = np.linalg.norm(
            np.cross(points[faces[:, 1]] - points[faces[:, 0]], points[faces[:, 2]] - points[faces[:, 0]]),
            axis=1,
        )
        faces = faces[area > 1e-16]
        point_normals = np.asarray(mesh.point_data["Normals"], dtype=np.float64) if "Normals" in mesh.point_data else None
        meshes.append((label, points, faces, point_normals))
    rendered = []
    for label, view_name in views:
        row = []
        view = next(item for item in SIX_VIEWS if item.name == view_name)
        for name, points, faces, point_normals in meshes:
            raster = rasterize_solid_view(points, faces, view, grid_size=768, depth_tolerance=0.0)
            face_ids = raster.first_hit_face
            valid = face_ids >= 0
            triangles = points[faces]
            raw = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
            lengths = np.linalg.norm(raw, axis=1)
            normals = np.divide(raw, lengths[:, None], out=np.zeros_like(raw), where=lengths[:, None] > 0.0)
            if point_normals is not None and len(point_normals) == len(points):
                smooth_normals = point_normals[faces].mean(axis=1)
                normal_lengths = np.linalg.norm(smooth_normals, axis=1)
                normals = np.divide(smooth_normals, normal_lengths[:, None], out=normals.copy(), where=normal_lengths[:, None] > 1e-15)
            camera = -np.asarray(view.depth_vector, dtype=np.float64)
            light = camera + np.array([0.0, 0.65, 0.45])
            light /= np.linalg.norm(light)
            visible_normals = normals[face_ids[valid]]
            visible_normals[np.einsum("ij,j->i", visible_normals, camera) < 0.0] *= -1.0
            shade = 0.25 + 0.75 * np.maximum(0.0, np.einsum("ij,j->i", visible_normals, light))
            rgb = np.empty((*face_ids.shape, 3), dtype=np.uint8)
            rgb[:] = (237, 241, 244)
            rgb[valid] = np.clip(np.array([119.0, 136.0, 148.0])[None, :] * shade[:, None], 0, 255).astype(np.uint8)
            image = Image.fromarray(rgb)
            ImageDraw.Draw(image).text((10, 10), f"{label} · {name}", fill=(20, 30, 38))
            row.append(image)
        rendered.append(row)
    canvas = Image.new("RGB", (768 * 3, 768 * 2), (237, 241, 244))
    for row_index, row in enumerate(rendered):
        for col_index, image in enumerate(row):
            canvas.paste(image, (col_index * 768, row_index * 768))
    canvas.save(output_path)


def run() -> dict[str, Any]:
    ROUTE_ROOT.mkdir(parents=True, exist_ok=True)
    if (ROUTE_ROOT / "candidate.vtp").exists():
        raise FileExistsError(ROUTE_ROOT / "candidate.vtp")
    started = time.time()
    baseline = pv.read(BASELINE)
    points = np.asarray(baseline.points, dtype=np.float64)
    faces = triangle_faces(baseline).astype(np.int64, copy=False)
    fairing = json.loads(FAIRING_REPORT.read_text(encoding="utf-8"))
    if fairing.get("self_intersection", {}).get("passed") is not True:
        raise ValueError("native-faired baseline lacks a passing exact CGAL certificate")
    if fairing.get("readback", {}).get("passed") is not True:
        raise ValueError("native-faired baseline lacks a passing serialized readback")
    pitch = 2.0 * float(fairing["displacement_limit_m"])
    projection_limit = 0.5 * pitch
    save_progress(
        "baseline_loaded",
        started,
        points=int(len(points)),
        triangles=int(len(faces)),
        exact_intersections=int(fairing["self_intersection"]["intersection_pairs"]),
        pitch_m=pitch,
    )

    raw_source = pv.read(SOURCE_TARGET).triangulate()
    source_points = np.asarray(raw_source.points, dtype=np.float64)
    raw_faces = triangle_faces(raw_source).astype(np.int64, copy=False)
    source_valid = triangle_areas(source_points, raw_faces) > 1e-16
    source_faces = raw_faces[source_valid]
    source = pv.PolyData(
        source_points,
        np.column_stack((np.full(len(source_faces), 3, dtype=np.int64), source_faces)).ravel(),
    )
    target_sizes, nearest_source_face = source_target_sizes(source, points, faces, 0.004)
    source_feature_face = target_sizes < 0.004
    vertex_feature_count = np.bincount(
        faces[source_feature_face].ravel(), minlength=len(points)
    )
    vertex_face_count = np.bincount(faces.ravel(), minlength=len(points))
    feature_fraction = vertex_feature_count / np.maximum(vertex_face_count, 1)
    save_progress(
        "source_features_mapped",
        started,
        source_triangles=int(len(source_faces)),
        removed_degenerate_source_triangles=int(len(raw_faces) - len(source_faces)),
        mapped_feature_faces=int(np.count_nonzero(source_feature_face)),
        mapped_ordinary_faces=int(np.count_nonzero(~source_feature_face)),
        vertices_with_feature_fraction_ge_0_8=int(np.count_nonzero(feature_fraction >= 0.8)),
    )

    smoothing_attempts = []
    for attempt in range(4):
        smooth_points, smoothing = feature_constrained_smooth(
            points, faces, feature_fraction, projection_limit * (0.5**attempt)
        )
        smooth_metrics = mesh_report(smooth_points, faces)
        smooth_water = watertight_metrics(smooth_metrics)
        smooth_cgal = exact_self_intersection_report(smooth_points, faces)
        attempt_report = {
            "attempt": attempt + 1,
            "displacement_limit_m": smoothing["displacement_limit_m"],
            "watertight": smooth_water,
            "cgal_passed": smooth_cgal.get("passed") is True,
            "intersection_pairs": smooth_cgal.get("intersection_pairs"),
        }
        smoothing_attempts.append(attempt_report)
        save_progress("smoothing_geometry_certification", started, **attempt_report)
        if smooth_water and smooth_cgal.get("passed") is True:
            break
    else:
        failure = {
            "status": "diagnostic_only_no_candidate",
            "smoothing_attempts": smoothing_attempts,
            "reason": "No source-feature-constrained smoothing displacement passed topology and exact CGAL gates.",
        }
        (ROUTE_ROOT / "feature_projection_report.json").write_text(
            json.dumps(failure, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        raise ValueError(failure["reason"])
    scaffold_path = ROUTE_ROOT / "smooth_scaffold.vtp"
    write_vtp(
        scaffold_path,
        smooth_points,
        faces,
        {"feature_projection_source_feature_face_fraction": feature_fraction[faces].mean(axis=1).astype(np.float32)},
        point_data={"feature_projection_source_feature_fraction": feature_fraction.astype(np.float32)},
    )
    save_progress(
        "smooth_scaffold_certified",
        started,
        scaffold_path=str(scaffold_path),
        max_displacement_m=smoothing["max_actual_displacement_m"],
        cgal_passed=True,
        smoothing_attempts=smoothing_attempts,
        metrics={
            "boundary_edges": smooth_metrics["topology"]["boundary_edges"],
            "non_manifold_edges": smooth_metrics["topology"]["non_manifold_edges"],
            "degenerate_faces": smooth_metrics["quality"]["degenerate_faces"],
        },
    )

    thresholds = ProjectionThresholds(
        max_projection_distance=projection_limit,
        orientation_vote_min_margin=0.02,
        source_edge_barycentric_margin=0.0,
        defect_edge_barycentric_margin=0.0,
        cross_component_jump_ratio=0.5,
    )
    projected, projected_faces, provenance, projection = project_closure_to_source(
        smooth_points,
        faces,
        source_points,
        source_faces,
        thresholds=thresholds,
        movable_vertex_mask=np.ones(len(points), dtype=bool),
    )
    final_points, provenance, exact = repair_projected_self_intersections(
        projected,
        smooth_points,
        projected_faces,
        provenance,
        certified_proxy_report=smooth_cgal,
    )
    metrics = mesh_report(final_points, projected_faces)
    if not watertight_metrics(metrics):
        raise ValueError("source projected candidate failed watertightness, orientation, or volume gates")
    if exact.get("passed") is not True:
        raise ValueError("source projected candidate failed exact CGAL certification")

    cell_data = {
        "feature_projection_source_feature_face_fraction": feature_fraction[faces].mean(axis=1).astype(np.float32),
        "feature_projection_source_projected_vertex_fraction": provenance["cell_data"]["source_projected_vertex_fraction"],
    }
    point_data = {f"feature_projection_{name}": values for name, values in provenance["point_data"].items()}
    candidate = ROUTE_ROOT / "candidate.vtp"
    write_vtp(candidate, final_points, projected_faces, cell_data, point_data=point_data)
    readback = roundtrip_validation(candidate, final_points, projected_faces)

    reference = pv.read(SOURCE_REFERENCE)
    reference_points = np.asarray(reference.points, dtype=np.float64)
    reference_faces_all = triangle_faces(reference).astype(np.int64, copy=False)
    categories = np.asarray(reference.cell_data["review_category"])
    reference_faces = reference_faces_all[np.isin(categories, [0, 3])]
    reference_valid = triangle_areas(reference_points, reference_faces) > 1e-16
    reference_faces = reference_faces[reference_valid]
    reference_surface = pv.PolyData(
        reference_points,
        np.column_stack((np.full(len(reference_faces), 3, dtype=np.int64), reference_faces)).ravel(),
    )
    forward_distance, forward_area = centroid_distances(final_points, projected_faces, reference_surface)
    visible = rasterize_solid_views(
        reference_points, reference_faces, grid_size=384, depth_tolerance=0.0, collect_face_evidence=True
    )
    visible_faces = visible.face_first_hit_view_count > 0
    visible_reference_faces = reference_faces[visible_faces]
    reverse_area = triangle_areas(reference_points, visible_reference_faces)
    reverse_centers = reference_points[visible_reference_faces].mean(axis=1)
    reverse_distance = point_distances(reverse_centers, pv.read(candidate))
    scale = 200.0 / float(np.max(np.ptp(reference_points, axis=0)))
    fidelity = {
        "target": "source_no_shadow review_category 0 external_keep + 3 wheel_review; interior/support excluded",
        "forward_candidate_centroid_to_source_m": weighted_percentiles(forward_distance, forward_area),
        "forward_candidate_centroid_to_source_at_200mm": weighted_percentiles(forward_distance * scale, forward_area),
        "reverse_source_triangle_centroid_to_candidate_m": weighted_percentiles(reverse_distance, reverse_area),
        "reverse_source_triangle_centroid_to_candidate_at_200mm": weighted_percentiles(reverse_distance * scale, reverse_area),
        "reverse_visibility": {
            "method": "six_axis_first_hit_union_on_external_and_wheel_faces",
            "grid_size": 384,
            "visible_source_faces": int(len(visible_reference_faces)),
            "visible_area_m2": float(reverse_area.sum()),
        },
    }
    input_fidelity = json.loads((ROOT / "outputs/sl03-smooth-retry-20261026/final_fidelity_metrics.json").read_text())
    if len(visible_reference_faces) != input_fidelity["visible_source_face_samples"]:
        raise RuntimeError("six-view visible source sample no longer matches the baseline fidelity run")
    input_displacement = np.linalg.norm(final_points - points, axis=1)
    decision = {
        "status": "diagnostic_only_not_promoted",
        "baseline_cgal": fairing["self_intersection"],
        "scaffold_cgal": smooth_cgal,
        "projected_cgal": exact,
        "watertight": watertight_metrics(metrics),
        "readback": readback,
        "old_vertices_movable": True,
        "projection": projection,
        "smoothing": smoothing,
        "smoothing_cgal_attempts": smoothing_attempts,
        "source_feature_mapping": {
            "feature_faces": int(np.count_nonzero(source_feature_face)),
            "ordinary_faces": int(np.count_nonzero(~source_feature_face)),
            "vertices_with_feature_fraction_ge_0_8": int(np.count_nonzero(feature_fraction >= 0.8)),
        },
        "fidelity": fidelity,
        "baseline_fidelity": input_fidelity["metrics"]["round-840-native-faired"],
        "maximum_displacement_from_native_faired_m": float(input_displacement.max()),
        "surface_metrics": {
            "points": int(metrics["points"]),
            "triangles": int(metrics["triangles"]),
            "topology": metrics["topology"],
            "quality": metrics["quality"],
            "volume": metrics["volume"],
        },
        "candidate": str(candidate),
        "render": str(ROUTE_ROOT / "comparison.png"),
        "limitations": [
            "Projection uses the complete exterior and wheel target; surface ownership is still limited by source triangle coverage.",
            "Six-view source distance and smooth shading remain diagnostic and do not prove all-angle equivalence.",
            "The candidate is not promoted automatically.",
        ],
    }
    (ROUTE_ROOT / "feature_projection_report.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    save_progress(
        "candidate_validated",
        started,
        candidate=str(candidate),
        points=int(len(final_points)),
        triangles=int(len(projected_faces)),
        watertight=True,
        cgal_pairs=int(exact.get("intersection_pairs", 0)),
        readback_passed=readback.get("passed") is True,
        fidelity=fidelity,
    )
    render_comparison(SOURCE_DISPLAY, BASELINE, candidate, ROUTE_ROOT / "comparison.png")
    return decision


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False), flush=True)
