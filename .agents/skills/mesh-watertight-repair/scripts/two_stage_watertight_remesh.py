#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import trimesh
from scipy.ndimage import binary_fill_holes, minimum_filter

from mesh_io import compact_mesh, grid_shape, read_surface, triangle_faces, write_vtp
from mesh_metrics import edge_topology, face_component_labels, mesh_report, silhouette_drift_from_meshes
from html_report import write_html_report
from opening_policy_visuals import attach_policy_evidence_views
from repair_inventory import build_inventory
from repair_policy import build_policy_packet, load_policy_decisions, resolve_policy_decisions
from sealed_exterior_atlas import (
    build_sealed_exterior_atlas,
)
from sdf_closure import SdfClosureConfig, build_tsdf_closure, smooth_closure_scaffold
from source_projected_closure import ProjectionThresholds, interpolate_source_normals, project_closure_to_source
from source_projected_validation import (
    repair_closure_proxy_self_intersections,
    repair_projected_self_intersections,
    roundtrip_validation,
    update_projection_report,
)
from run_status import (
    initial_run_status,
    pitch_status,
    prepare_output_dir,
    record_failure,
    release_output_lease,
    run_parameters,
    topology_brief,
    update_run_status,
)
from two_stage_contract import build_report, record_cfd_refinement
from two_stage_outputs import (
    extraction_report,
    outputs_report,
    print_summary,
    write_previews,
    write_source_projected_watertight_candidate,
    write_source_preserving_candidate,
    write_stage1,
    write_visibility_labels,
)
from source_preserving_repair import prune_classified_internal_components, run_deterministic_repair
from solid_triangle_raster import triangle_center_samples as triangle_raster_samples


@dataclass(frozen=True)
class ViewSpec:
    name: str
    project_axes: tuple[int, int]
    depth_vector: tuple[float, float, float]


VIEWS = [
    ViewSpec("minus_x", (1, 2), (1.0, 0.0, 0.0)),
    ViewSpec("plus_x", (1, 2), (-1.0, 0.0, 0.0)),
    ViewSpec("minus_y", (0, 2), (0.0, 1.0, 0.0)),
    ViewSpec("plus_y", (0, 2), (0.0, -1.0, 0.0)),
    ViewSpec("plus_z", (0, 1), (0.0, 0.0, -1.0)),
    ViewSpec("minus_z", (0, 1), (0.0, 0.0, 1.0)),
]

DEFAULT_REMOVE_NAME_REGEX = (
    r"(^|[_:])int(ernal)?([_:]|$)|internal|interior|hidden|inside|cavity"
)

DEFAULT_DEPTH_TOLERANCE_BBOX_RATIO = 2.0e-4
DEFAULT_DEPTH_TOLERANCE_EDGE_RATIO = 0.25
TARGET_NAME = "watertight-exterior-shell"
MAX_DEPTH_TOLERANCE_BBOX_RATIO = 1.0e-3
DEFAULT_REGRESSION_PIXEL_TOLERANCE = 1
MAX_SOURCE_PROJECTION_DISTANCE_VOXELS = 3.0
RASTER_FACE_CHUNK = 12_000
MAX_RASTER_CANDIDATES_PER_CHUNK = 4_000_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build and audit a source-faithful watertight exterior shell from a mesh input; "
            "the closure proxy remains diagnostic guidance."
        )
    )
    parser.add_argument("input", type=Path, help="Input mesh: VTP, STL, OBJ, GLB, GLTF, or VTK.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--purpose", choices=("watertight", "cfd", "print"), default="watertight")
    parser.add_argument("--surface-strategy", choices=("projected-shell", "source-material"), default="projected-shell")
    parser.add_argument("--stitch-tolerance", type=float, default=0., help="Optional boundary seam tolerance in input units; zero disables approximate sewing")
    parser.add_argument("--repair-timeout-seconds", type=float, help="Wall-clock limit for projected-shell, including native calls (default 120 seconds)")
    parser.add_argument("--_projected-shell-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--source-max-error", type=float, help="Source-material geometric error budget, in input units")
    parser.add_argument("--source-boundary-mode", choices=("review", "wall", "cap"), default="review")
    parser.add_argument("--source-max-recovery-facets", type=int, help="Maximum source facets allowed in one recovery union (default 1000)")
    parser.add_argument("--source-union-backend", choices=("cgal", "manifold", "local", "planar"), help="Source-material union backend; planar is limited to reviewed near-coplanar open sheets (default cgal)")
    parser.add_argument("--source-assembly-context", type=Path, help="Unchanged assembly excluding the source part, in the same frame and units")
    parser.add_argument("--source-timeout-seconds", type=float, help="Wall-clock limit for source-material, including native calls (default 120 seconds)")
    parser.add_argument("--_source-material-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--certify-reverse-fidelity", action="store_true", help="Attempt a bounded source-to-candidate continuous surface check")
    from print_mesh import add_print_arguments, validate_print_arguments
    add_print_arguments(parser)
    parser.add_argument("--cfd-max-edge-length", type=float, help="CFD surface edge limit in input coordinate units.")
    parser.add_argument("--cfd-max-output-faces", type=int, default=2_000_000)
    parser.add_argument("--overwrite", action="store_true", help="Allow writing over files in an existing output directory.")
    parser.add_argument(
        "--group-source-gltf",
        type=Path,
        help=(
            "Optional GLTF/GLB source used only to diagnose geometry names and flattened face ranges. "
            "It never removes STL/VTP source triangles."
        ),
    )
    parser.set_defaults(target_name=TARGET_NAME)
    parser.add_argument(
        "--remove-name-regex",
        default=DEFAULT_REMOVE_NAME_REGEX,
        help="Case-insensitive GLTF group/geometry-name regex reported as a diagnostic removal candidate.",
    )
    parser.add_argument("--visibility-grid", type=int)
    parser.add_argument(
        "--depth-tolerance",
        type=float,
        default=0.0,
        help=(
            "Optional absolute ceiling for first-hit depth tolerance. Zero selects an automatic "
            "bbox/local-edge tolerance; oversized absolute values are clamped to that scale-aware value."
        ),
    )
    parser.add_argument(
        "--depth-tolerance-bbox-ratio",
        type=float,
        default=DEFAULT_DEPTH_TOLERANCE_BBOX_RATIO,
        help="Automatic first-hit tolerance as a fraction of the source bbox maximum extent.",
    )
    parser.add_argument(
        "--depth-tolerance-edge-ratio",
        type=float,
        default=DEFAULT_DEPTH_TOLERANCE_EDGE_RATIO,
        help="Automatic first-hit tolerance as a fraction of the source median triangle edge length.",
    )
    parser.add_argument(
        "--visibility-min-views",
        type=int,
        default=1,
        help=(
            "Diagnostic corroboration threshold. Any face hit by at least one view is always hard-kept; "
            "this value cannot narrow the hard-keep set."
        ),
    )
    parser.add_argument(
        "--outside-flood-grid",
        type=int,
        default=192,
        help="Longest-axis voxel resolution for the outside flood-fill evidence atlas.",
    )
    parser.add_argument(
        "--sealed-exterior-grid",
        type=int,
        default=192,
        help="Longest-axis voxel resolution for component-level sealed exterior evidence.",
    )
    parser.add_argument(
        "--sealed-exterior-radius-voxels",
        type=int,
        default=1,
        help="Voxel shell thickening radius used only to seal narrow assembly seams in the evidence atlas.",
    )
    parser.add_argument(
        "--sealed-exterior-band-voxels",
        type=float,
        default=1.5,
        help="Maximum voxel distance from far-field space for a face to receive sealed-exterior support.",
    )
    parser.add_argument("--dilate-rings", type=int, default=1)
    parser.add_argument(
        "--voxel-pitch",
        type=float,
        default=0.0,
        help=(
            "Explicit closure-proxy voxel pitch. Zero derives it from the candidate bbox; "
            "an explicit positive value takes precedence over --voxel-pitch-bbox-divisor."
        ),
    )
    parser.add_argument(
        "--voxel-pitch-bbox-divisor",
        type=float,
        default=192.0,
        help="Derive the global voxel/closure proxy pitch as candidate bbox max_extent divided by this value.",
    )
    parser.add_argument(
        "--sdf-band-voxels",
        type=float,
        default=6.0,
        help="Half-width of the exact signed-distance narrow band in closure voxel pitches.",
    )
    parser.add_argument(
        "--sdf-smoothing-sigma",
        type=float,
        default=0.5,
        help="Gaussian SDF smoothing sigma in voxels; bounded to at most half a pitch of near-zero field motion.",
    )
    parser.add_argument(
        "--scaffold-smoothing-iterations", type=int, default=0,
        help="Windowed-Sinc iterations before source projection; motion is bounded to half a pitch (0 disables).",
    )
    parser.add_argument(
        "--projection-target", type=Path,
        help="Optional AI-reviewed exterior surface in the input frame/units; supplies shape while the closure scaffold supplies topology.",
    )
    parser.add_argument(
        "--cross-component-jump-ratio", type=float,
        help="Allow source-panel seams only when displacement jumps stay below this fraction of scaffold edge length; default keeps strict component IDs.",
    )
    parser.add_argument(
        "--max-sdf-memory-gb",
        type=float,
        default=4.0,
        help="Explicit dense-SDF peak-memory budget; the run fails closed when the preflight estimate exceeds it.",
    )
    parser.add_argument(
        "--policy-decisions",
        type=Path,
        help="Optional JSON policy review decisions from a previous unresolved_policy_packet.",
    )
    parser.add_argument(
        "--policy-item-limit",
        type=int,
        default=500,
        help="Report-size hint only; geometry inventory and policy truth are always processed in full.",
    )
    parser.add_argument(
        "--exterior-component-support-threshold",
        type=float,
        default=0.25,
        help="Maximum direct exterior-atlas support for a contained component to be considered internal.",
    )
    parser.add_argument(
        "--sealed-exterior-support-threshold",
        type=float,
        default=0.01,
        help=(
            "Maximum fraction of faces in the sealed exterior band for a strictly contained small "
            "component to be an automatic internal-component removal candidate."
        ),
    )
    parser.add_argument(
        "--internal-component-max-face-ratio",
        type=float,
        default=0.01,
        help="Maximum candidate face ratio for automatic contained-internal component removal.",
    )
    parser.add_argument(
        "--internal-component-max-diameter-ratio",
        type=float,
        default=0.05,
        help=(
            "Maximum component bbox diagonal relative to the robust shell for automatic "
            "contained-internal removal. Larger low-poly surfaces are protected."
        ),
    )
    parser.add_argument(
        "--internal-component-max-projected-bbox-area-ratio",
        type=float,
        default=0.01,
        help=(
            "Maximum component projected bbox area relative to the robust shell for automatic "
            "contained-internal removal."
        ),
    )
    parser.add_argument(
        "--floating-fragment-max-face-ratio",
        type=float,
        default=1.0e-4,
        help="Maximum face ratio for a tiny component outside the robust shell envelope to be a flying fragment.",
    )
    parser.add_argument(
        "--floating-fragment-max-diameter-ratio",
        type=float,
        default=0.01,
        help="Maximum fragment diameter relative to the robust shell-envelope diagonal.",
    )
    parser.add_argument(
        "--shell-envelope-min-component-face-ratio",
        type=float,
        default=1.0e-3,
        help="Minimum component face ratio used to establish the robust exterior shell envelope.",
    )
    parser.add_argument(
        "--shell-envelope-margin-ratio",
        type=float,
        default=0.005,
        help="Margin around the robust shell envelope as a fraction of its diagonal.",
    )
    parser.add_argument("--preview-size", type=int, default=900)
    parser.add_argument("--skip-previews", action="store_true")
    args = parser.parse_args()
    if args.surface_strategy == "projected-shell" and args.purpose != "print":
        if args.repair_timeout_seconds is None:
            args.repair_timeout_seconds = 120.
        if not np.isfinite(args.repair_timeout_seconds) or args.repair_timeout_seconds <= 0:
            parser.error("--repair-timeout-seconds must be finite and positive")
    elif args.repair_timeout_seconds is not None or args._projected_shell_worker:
        parser.error("--repair-timeout-seconds applies to projected-shell watertight/CFD")
    if not np.isfinite(args.stitch_tolerance) or args.stitch_tolerance < 0:
        parser.error("--stitch-tolerance must be finite and nonnegative")
    if args.stitch_tolerance and (args.surface_strategy != "projected-shell" or args.purpose == "print"):
        parser.error("--stitch-tolerance applies to projected-shell watertight/CFD")
    if args.visibility_grid is None:
        args.visibility_grid = 128 if args.surface_strategy == "source-material" else 900
    if args.surface_strategy == "source-material":
        if args.purpose != "watertight":
            parser.error("source-material supports geometric watertight material; use the purpose-specific CFD/print workflow for engineering validation")
        if args.source_max_error is None or not np.isfinite(args.source_max_error) or args.source_max_error <= 0:
            parser.error("source-material requires a positive finite --source-max-error in input units")
        if args.visibility_grid < 32:
            parser.error("source-material --visibility-grid must be at least 32")
        if args.source_max_recovery_facets is None:
            args.source_max_recovery_facets = 1000
        if args.source_union_backend is None:
            args.source_union_backend = "cgal"
        if args.source_max_recovery_facets < 1:
            parser.error("--source-max-recovery-facets must be positive")
        if args.source_timeout_seconds is None:
            args.source_timeout_seconds = 120.0
        if not np.isfinite(args.source_timeout_seconds) or args.source_timeout_seconds <= 0:
            parser.error("--source-timeout-seconds must be finite and positive")
        if args.scaffold_smoothing_iterations or args.cross_component_jump_ratio is not None:
            parser.error("scaffold smoothing and cross-component projection controls apply to projected-shell")
        args.target_name = "watertight-material"
    elif args.source_max_error is not None or args.source_boundary_mode != "review" or args.certify_reverse_fidelity or args.source_max_recovery_facets is not None or args.source_union_backend is not None or args.source_assembly_context is not None or args.source_timeout_seconds is not None or args._source_material_worker:
        parser.error("source-material settings require --surface-strategy source-material")
    if args.scaffold_smoothing_iterations < 0:
        parser.error("--scaffold-smoothing-iterations must be nonnegative")
    if args.cross_component_jump_ratio is not None and (
        not np.isfinite(args.cross_component_jump_ratio) or args.cross_component_jump_ratio <= 0
    ):
        parser.error("--cross-component-jump-ratio must be finite and positive")
    if args.purpose == "cfd" and (
        args.cfd_max_edge_length is None or not np.isfinite(args.cfd_max_edge_length)
        or args.cfd_max_edge_length <= 0 or args.cfd_max_output_faces <= 0
    ):
        parser.error("--purpose cfd requires a positive finite --cfd-max-edge-length and a positive face budget")
    if args.purpose != "cfd" and args.cfd_max_edge_length is not None:
        parser.error("--cfd-max-edge-length requires --purpose cfd")
    if args.purpose == "print" and (args.scaffold_smoothing_iterations or args.cross_component_jump_ratio is not None or args.projection_target is not None):
        parser.error("scaffold smoothing and component projection controls require --purpose watertight or cfd")
    if args.purpose == "print":
        try:
            validate_print_arguments(args)
        except ValueError as exc:
            parser.error(str(exc))
    elif any(value.startswith("--print-") for value in sys.argv[1:]):
        parser.error("--print-* settings require --purpose print")
    return args


def gltf_group_keep_mask(
    gltf_path: Path,
    expected_faces: int,
    remove_name_regex: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    scene = trimesh.load(gltf_path, force="scene")
    remove_pattern = re.compile(remove_name_regex, re.IGNORECASE)
    keep = np.ones(expected_faces, dtype=bool)
    cursor = 0
    removed_groups = []
    kept_groups = []
    for name, geometry in scene.geometry.items():
        face_count = int(len(geometry.faces))
        start = cursor
        end = cursor + face_count
        cursor = end
        if end > expected_faces:
            raise ValueError("GLTF flattened face count exceeds input face count")
        if remove_pattern.search(name):
            keep[start:end] = False
            removed_groups.append({"name": name, "triangles": face_count})
        else:
            kept_groups.append({"name": name, "triangles": face_count})
    if cursor != expected_faces:
        raise ValueError(
            f"GLTF flattened face count {cursor} does not match input face count {expected_faces}"
        )
    return keep, group_filter_report(gltf_path, remove_pattern, removed_groups, kept_groups, expected_faces)


def group_filter_report(
    gltf_path: Path,
    remove_pattern: re.Pattern[str],
    removed_groups: list[dict[str, Any]],
    kept_groups: list[dict[str, Any]],
    expected_faces: int,
) -> dict[str, Any]:
    removed_triangles = sum(int(row["triangles"]) for row in removed_groups)
    return {
        "method": "gltf_geometry_name_diagnostic_by_flattened_face_ranges",
        "source": str(gltf_path),
        "role": "diagnostic_only",
        "geometry_filter_applied": False,
        "input_geometry_truth": "mesh_input",
        "assumption": "Diagnostic ranges are meaningful only if GLTF flatten order matches the mesh input triangle order.",
        "remove_pattern": remove_pattern.pattern,
        "removed_groups": len(removed_groups),
        "kept_groups": len(kept_groups),
        "diagnostic_remove_candidate_triangles": removed_triangles,
        "diagnostic_remove_candidate_ratio": float(removed_triangles / expected_faces),
        "removed_triangles": 0,
        "removed_triangle_ratio": 0.0,
        "largest_removed_groups": largest_groups(removed_groups),
        "largest_kept_groups": largest_groups(kept_groups),
    }


def largest_groups(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(groups, key=lambda row: row["triangles"], reverse=True)[:20]


def visible_faces_from_view(
    points: np.ndarray,
    faces: np.ndarray,
    view: ViewSpec,
    *,
    grid_size: int,
    depth_tolerance: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    if grid_size < 2:
        raise ValueError("visibility grid must be at least 2")
    if depth_tolerance < 0.0:
        raise ValueError("resolved depth tolerance must be non-negative")

    coords = points[:, view.project_axes]
    rows, cols, mins, spans = grid_shape(coords, grid_size)
    uv = np.clip((coords - mins) / spans, 0.0, 1.0)
    projected_points = uv * np.asarray([cols - 1, rows - 1], dtype=np.float64)
    projected_triangles = projected_points[faces]

    depth_vector = np.asarray(view.depth_vector, dtype=np.float64)
    vertex_depth = points @ depth_vector
    triangle_depth = vertex_depth[faces]
    min_depth = np.full(rows * cols, np.inf, dtype=np.float64)
    rasterized = np.zeros(faces.shape[0], dtype=bool)
    raster_samples = 0
    for face_ids, pixels, depth in triangle_raster_samples(
        projected_triangles,
        triangle_depth,
        rows,
        cols,
    ):
        rasterized[face_ids] = True
        raster_samples += int(face_ids.size)
        np.minimum.at(min_depth, pixels, depth)

    fallback_ids, fallback_pixels, fallback_depth = unrasterized_face_samples(
        projected_triangles,
        triangle_depth,
        rasterized,
        rows,
        cols,
    )
    if fallback_ids.size:
        np.minimum.at(min_depth, fallback_pixels, fallback_depth)

    visible = np.zeros(faces.shape[0], dtype=bool)
    front_samples = 0
    for face_ids, pixels, depth in triangle_raster_samples(
        projected_triangles,
        triangle_depth,
        rows,
        cols,
    ):
        first_hit = depth <= min_depth[pixels] + depth_tolerance
        visible[face_ids[first_hit]] = True
        front_samples += int(np.count_nonzero(first_hit))
    if fallback_ids.size:
        fallback_first_hit = fallback_depth <= min_depth[fallback_pixels] + depth_tolerance
        visible[fallback_ids[fallback_first_hit]] = True
        front_samples += int(np.count_nonzero(fallback_first_hit))

    occupied_pixels = int(np.count_nonzero(np.isfinite(min_depth)))
    return visible, {
        "view": view.name,
        "grid": {"rows": rows, "cols": cols},
        "method": "orthographic_triangle_raster_first_hit",
        "depth_tolerance": float(depth_tolerance),
        "visible_faces": int(np.count_nonzero(visible)),
        "visible_face_ratio": float(np.count_nonzero(visible) / max(1, faces.shape[0])),
        "occupied_pixels": occupied_pixels,
        "rasterized_face_pixel_samples": raster_samples,
        "fallback_centroid_samples": int(fallback_ids.size),
        "front_hit_face_pixel_samples": front_samples,
        "atlas": {
            "pixels": rows * cols,
            "occupied_pixels": occupied_pixels,
            "face_score_contribution": "one point for any first-hit pixel support in this view",
        },
    }


def raster_bounds(
    triangles: np.ndarray,
    rows: int,
    cols: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lower = np.floor(np.min(triangles, axis=1)).astype(np.int64)
    upper = np.ceil(np.max(triangles, axis=1)).astype(np.int64)
    lower[:, 0] = np.clip(lower[:, 0], 0, cols - 1)
    upper[:, 0] = np.clip(upper[:, 0], 0, cols - 1)
    lower[:, 1] = np.clip(lower[:, 1], 0, rows - 1)
    upper[:, 1] = np.clip(upper[:, 1], 0, rows - 1)
    widths = upper[:, 0] - lower[:, 0] + 1
    heights = upper[:, 1] - lower[:, 1] + 1
    return lower, upper, widths, heights, widths * heights


def unrasterized_face_samples(
    projected_triangles: np.ndarray,
    triangle_depth: np.ndarray,
    rasterized: np.ndarray,
    rows: int,
    cols: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    face_ids = np.flatnonzero(~rasterized)
    if face_ids.size == 0:
        return face_ids, np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    centroids = np.rint(np.mean(projected_triangles[face_ids], axis=1)).astype(np.int64)
    centroids[:, 0] = np.clip(centroids[:, 0], 0, cols - 1)
    centroids[:, 1] = np.clip(centroids[:, 1], 0, rows - 1)
    pixels = centroids[:, 1] * cols + centroids[:, 0]
    depth = np.mean(triangle_depth[face_ids], axis=1)
    return face_ids, pixels, depth


def fixed_projection_depth_buffer(
    points: np.ndarray,
    faces: np.ndarray,
    view: ViewSpec,
    *,
    rows: int,
    cols: int,
    mins: np.ndarray,
    spans: np.ndarray,
) -> np.ndarray:
    """Raster one mesh into a caller-owned projection so before/after pixels match exactly."""
    min_depth = np.full(rows * cols, np.inf, dtype=np.float64)
    if faces.size == 0:
        return min_depth.reshape(rows, cols)
    coords = points[:, view.project_axes]
    uv = np.clip((coords - mins) / spans, 0.0, 1.0)
    projected_points = uv * np.asarray([cols - 1, rows - 1], dtype=np.float64)
    projected_triangles = projected_points[faces]
    triangle_depth = (points @ np.asarray(view.depth_vector, dtype=np.float64))[faces]
    rasterized = np.zeros(faces.shape[0], dtype=bool)
    for face_ids, pixels, depth in triangle_raster_samples(
        projected_triangles,
        triangle_depth,
        rows,
        cols,
    ):
        rasterized[face_ids] = True
        np.minimum.at(min_depth, pixels, depth)
    fallback_ids, fallback_pixels, fallback_depth = unrasterized_face_samples(
        projected_triangles,
        triangle_depth,
        rasterized,
        rows,
        cols,
    )
    if fallback_ids.size:
        np.minimum.at(min_depth, fallback_pixels, fallback_depth)
    return min_depth.reshape(rows, cols)


def six_view_depth_regression(
    before_points: np.ndarray,
    before_faces: np.ndarray,
    after_points: np.ndarray,
    after_faces: np.ndarray,
    *,
    grid_size: int,
    depth_tolerance: float,
    pixel_tolerance: int = DEFAULT_REGRESSION_PIXEL_TOLERANCE,
) -> dict[str, Any]:
    """Reject material silhouette loss or depth recession in matched axis views.

    A closure or source-projection step normally uses a different triangulation
    than its source.  Sampling both meshes only at exact pixel centers therefore
    produces isolated one-pixel disagreements even when their projected surfaces
    overlap.  Compare each source pixel with the nearest candidate hit inside a
    small image-space neighborhood while retaining the exact-pixel counts as raw
    diagnostics.  Candidate surfaces that are closer than the source are expected
    when openings are sealed, so they are diagnostic and never depth regression.
    """
    if grid_size < 2:
        raise ValueError("visibility regression grid must be at least 2")
    if depth_tolerance < 0.0:
        raise ValueError("visibility regression depth tolerance must be non-negative")
    if pixel_tolerance < 0:
        raise ValueError("visibility regression pixel tolerance must be non-negative")
    numeric_floor = float(np.max(np.ptp(before_points, axis=0))) * 1.0e-12
    effective_tolerance = max(float(depth_tolerance), numeric_floor)
    view_rows: list[dict[str, Any]] = []
    for view in VIEWS:
        coords = np.concatenate(
            (before_points[:, view.project_axes], after_points[:, view.project_axes]),
            axis=0,
        )
        rows, cols, mins, spans = grid_shape(coords, grid_size)
        before_depth = fixed_projection_depth_buffer(
            before_points,
            before_faces,
            view,
            rows=rows,
            cols=cols,
            mins=mins,
            spans=spans,
        )
        after_depth = fixed_projection_depth_buffer(
            after_points,
            after_faces,
            view,
            rows=rows,
            cols=cols,
            mins=mins,
            spans=spans,
        )
        before_finite = np.isfinite(before_depth)
        after_finite = np.isfinite(after_depth)
        raw_new_background = before_finite & ~after_finite
        raw_deeper_surface = (
            before_finite
            & after_finite
            & (after_depth > before_depth + effective_tolerance)
        )
        closer_surface = (
            before_finite
            & after_finite
            & (after_depth < before_depth - effective_tolerance)
        )
        if pixel_tolerance:
            neighborhood_size = 2 * int(pixel_tolerance) + 1
            matched_after_depth = minimum_filter(
                after_depth,
                size=neighborhood_size,
                mode="constant",
                cval=np.inf,
            )
        else:
            matched_after_depth = after_depth
        matched_after_finite = np.isfinite(matched_after_depth)
        new_background = before_finite & ~matched_after_finite
        deeper_surface = (
            before_finite
            & matched_after_finite
            & (matched_after_depth > before_depth + effective_tolerance)
        )
        view_rows.append(
            {
                "view": view.name,
                "grid": {"rows": rows, "cols": cols},
                "visible_surface_pixels_before": int(np.count_nonzero(before_finite)),
                "visible_surface_pixels_after": int(np.count_nonzero(after_finite)),
                "raw_new_background_pixels": int(np.count_nonzero(raw_new_background)),
                "raw_deeper_surface_pixels": int(np.count_nonzero(raw_deeper_surface)),
                "new_background_pixels": int(np.count_nonzero(new_background)),
                "deeper_surface_pixels": int(np.count_nonzero(deeper_surface)),
                "closer_surface_pixels": int(np.count_nonzero(closer_surface)),
            }
        )
    regression_pixels = sum(
        row["new_background_pixels"]
        + row["deeper_surface_pixels"]
        for row in view_rows
    )
    new_background_pixels = sum(row["new_background_pixels"] for row in view_rows)
    deeper_surface_pixels = sum(row["deeper_surface_pixels"] for row in view_rows)
    raw_regression_pixels = sum(
        row["raw_new_background_pixels"] + row["raw_deeper_surface_pixels"]
        for row in view_rows
    )
    closer_surface_pixels = sum(row["closer_surface_pixels"] for row in view_rows)
    return {
        "method": "matched_projection_six_view_triangle_raster_depth_comparison",
        "contract": (
            "no new background and no newly exposed deeper surface after image-space "
            "registration tolerance; closer surfaces are closure diagnostics"
        ),
        "passed": regression_pixels == 0,
        "grid_size_longest_axis": int(grid_size),
        "depth_tolerance": effective_tolerance,
        "pixel_registration_tolerance": int(pixel_tolerance),
        "raw_regression_pixels": int(raw_regression_pixels),
        "new_background_pixels": int(new_background_pixels),
        "deeper_surface_pixels": int(deeper_surface_pixels),
        "regression_pixels": int(regression_pixels),
        "closer_surface_pixels": int(closer_surface_pixels),
        "views": view_rows,
    }


def resolve_visibility_depth_tolerance(
    points: np.ndarray,
    faces: np.ndarray,
    requested_absolute: float,
    *,
    bbox_ratio: float = DEFAULT_DEPTH_TOLERANCE_BBOX_RATIO,
    edge_ratio: float = DEFAULT_DEPTH_TOLERANCE_EDGE_RATIO,
) -> tuple[float, dict[str, Any]]:
    if requested_absolute < 0.0:
        raise ValueError("--depth-tolerance must be non-negative")
    if bbox_ratio < 0.0 or edge_ratio < 0.0:
        raise ValueError("depth-tolerance ratios must be non-negative")
    bbox_extents = np.ptp(points, axis=0)
    bbox_max_extent = float(np.max(bbox_extents))
    if not np.isfinite(bbox_max_extent) or bbox_max_extent <= 0.0:
        raise ValueError("cannot derive visibility tolerance from an empty or zero-size bbox")

    triangles = points[faces]
    edge_lengths = np.concatenate(
        (
            np.linalg.norm(triangles[:, 1] - triangles[:, 0], axis=1),
            np.linalg.norm(triangles[:, 2] - triangles[:, 1], axis=1),
            np.linalg.norm(triangles[:, 0] - triangles[:, 2], axis=1),
        )
    )
    valid_edges = edge_lengths[np.isfinite(edge_lengths) & (edge_lengths > 0.0)]
    median_edge_length = float(np.median(valid_edges)) if valid_edges.size else 0.0
    bbox_component = bbox_max_extent * bbox_ratio
    edge_component = median_edge_length * edge_ratio
    automatic = max(bbox_component, edge_component)
    scale_cap = bbox_max_extent * MAX_DEPTH_TOLERANCE_BBOX_RATIO
    automatic = min(automatic, scale_cap)
    effective = automatic if requested_absolute == 0.0 else min(requested_absolute, automatic)
    source = "automatic_bbox_and_local_edge"
    if requested_absolute > 0.0:
        source = "explicit_ceiling" if requested_absolute <= automatic else "explicit_oversize_clamped_to_mesh_scale"
    return effective, {
        "requested_absolute": float(requested_absolute),
        "effective": float(effective),
        "source": source,
        "bbox_max_extent": bbox_max_extent,
        "bbox_ratio": float(bbox_ratio),
        "bbox_component": float(bbox_component),
        "median_triangle_edge_length": median_edge_length,
        "edge_ratio": float(edge_ratio),
        "edge_component": float(edge_component),
        "max_bbox_ratio": MAX_DEPTH_TOLERANCE_BBOX_RATIO,
        "scale_cap": float(scale_cap),
    }


def exterior_visibility_mask(
    points: np.ndarray,
    faces: np.ndarray,
    *,
    grid_size: int,
    depth_tolerance: float,
    dilate_rings: int,
    min_visible_views: int = 1,
    include_external_directions: bool = False,
) -> (
    tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]
    | tuple[np.ndarray, np.ndarray, list[dict[str, Any]], np.ndarray]
):
    if not 1 <= min_visible_views <= len(VIEWS):
        raise ValueError(f"visibility-min-views must be between 1 and {len(VIEWS)}")
    visible_score = np.zeros(faces.shape[0], dtype=np.int16)
    external_direction_sum = np.zeros((faces.shape[0], 3), dtype=np.float64)
    view_reports = []
    for view in VIEWS:
        visible, view_report = visible_faces_from_view(
            points,
            faces,
            view,
            grid_size=grid_size,
            depth_tolerance=depth_tolerance,
        )
        visible_score[visible] += 1
        external_direction_sum[visible] -= np.asarray(
            view.depth_vector, dtype=np.float64
        )
        view_report["external_direction_contribution"] = (
            -np.asarray(view.depth_vector, dtype=np.float64)
        ).tolist()
        view_reports.append(view_report)

    # A single first-hit is enough to prove that a face contributes to the
    # observed exterior.  min_visible_views remains reportable corroboration
    # evidence, but is never allowed to turn a visible face into a deleteable one.
    selected = visible_score > 0
    for report in view_reports:
        report["hard_keep_contract"] = "any_first_hit_view"
        report["requested_corroboration_views"] = int(min_visible_views)
    for _ in range(max(0, dilate_rings)):
        selected = dilate_by_vertices(faces, selected, points.shape[0])
    if include_external_directions:
        lengths = np.linalg.norm(external_direction_sum, axis=1)
        external_directions = np.divide(
            external_direction_sum,
            lengths[:, None],
            out=np.zeros_like(external_direction_sum),
            where=lengths[:, None] > 0.0,
        )
        return selected, visible_score, view_reports, external_directions
    return selected, visible_score, view_reports


def preserve_whole_visible_components(
    faces: np.ndarray,
    directly_visible: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Keep every face of an edge-connected component once any face is directly visible."""
    directly_visible = np.asarray(directly_visible, dtype=bool)
    if directly_visible.shape != (faces.shape[0],):
        raise ValueError("directly_visible must contain one value per face")
    _, edge_faces = edge_topology(faces)
    component_ids = face_component_labels(faces.shape[0], edge_faces)
    visible_component_ids = np.unique(component_ids[directly_visible])
    selected = np.isin(component_ids, visible_component_ids)
    component_count = int(component_ids.max(initial=-1) + 1)
    return selected, {
        "method": "edge_connected_component_visibility_expansion",
        "contract": (
            "a component with any multi-view first-hit face is kept whole so visibility filtering "
            "cannot cut a new boundary into that component"
        ),
        "component_count": component_count,
        "visible_component_count": int(visible_component_ids.size),
        "fully_unseen_component_count": int(component_count - visible_component_ids.size),
        "directly_visible_faces": int(np.count_nonzero(directly_visible)),
        "component_hard_keep_faces": int(np.count_nonzero(selected)),
        "hidden_faces_restored_by_component_guard": int(
            np.count_nonzero(selected & ~directly_visible)
        ),
        "fully_unseen_faces": int(np.count_nonzero(~selected)),
    }


def outside_flood_face_mask(
    points: np.ndarray,
    faces: np.ndarray,
    *,
    grid_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Mark surface faces adjacent to empty voxels connected to the far field."""
    if grid_size < 16:
        raise ValueError("outside-flood-grid must be at least 16")
    max_extent = float(np.max(np.ptp(points, axis=0)))
    if not np.isfinite(max_extent) or max_extent <= 0.0:
        raise ValueError("cannot build outside flood atlas from a zero-size mesh")
    pitch = max_extent / float(grid_size)
    source = trimesh.Trimesh(vertices=points, faces=faces, process=False)
    voxel = source.voxelized(pitch=pitch)
    shell = np.asarray(voxel.matrix, dtype=bool)
    padded_shell = np.pad(shell, 1, mode="constant", constant_values=False)
    filled = binary_fill_holes(padded_shell)
    outside = ~filled

    triangles = points[faces]
    samples = np.concatenate([triangles.mean(axis=1)[:, None, :], triangles], axis=1)
    sample_indices = voxel.points_to_indices(samples.reshape(-1, 3)).reshape(faces.shape[0], 4, 3) + 1
    exposed = np.zeros(faces.shape[0], dtype=bool)
    shape = np.asarray(outside.shape, dtype=np.int64)
    for axis in range(3):
        for direction in (-1, 1):
            neighbors = sample_indices.copy()
            neighbors[:, :, axis] += direction
            valid = np.all((neighbors >= 0) & (neighbors < shape), axis=2)
            values = np.ones(valid.shape, dtype=bool)
            bounded = neighbors[valid]
            values[valid] = outside[bounded[:, 0], bounded[:, 1], bounded[:, 2]]
            exposed |= np.any(values, axis=1)
    return exposed, {
        "method": "voxel_shell_six_connected_outside_flood",
        "grid_size_longest_axis": int(grid_size),
        "pitch": pitch,
        "matrix_shape": [int(value) for value in shell.shape],
        "shell_voxels": int(np.count_nonzero(shell)),
        "enclosed_or_shell_voxels": int(np.count_nonzero(filled)),
        "outside_voxels": int(np.count_nonzero(outside)),
        "outside_adjacent_faces": int(np.count_nonzero(exposed)),
        "outside_adjacent_face_ratio": float(np.count_nonzero(exposed) / max(faces.shape[0], 1)),
        "role": "diagnostic_exterior_evidence_that_cannot_override_multi_view_first_hit_hard_keep",
    }


def dilate_by_vertices(faces: np.ndarray, selected: np.ndarray, point_count: int) -> np.ndarray:
    vertex_selected = np.zeros(point_count, dtype=bool)
    vertex_selected[faces[selected].ravel()] = True
    return np.any(vertex_selected[faces], axis=1)


def voxel_watertight_remesh(
    points: np.ndarray,
    faces: np.ndarray,
    pitch: float,
    *,
    seal_radius_voxels: int,
    max_projection_distance: float,
    sdf_band_voxels: float = 6.0,
    sdf_smoothing_sigma: float = 0.5,
    max_sdf_memory_gb: float = 4.0,
    implicit_field_path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    source = trimesh.Trimesh(vertices=points, faces=faces, process=False)
    remesh_points, remesh_faces, sdf_report = build_tsdf_closure(
        points,
        faces,
        config=SdfClosureConfig(
            pitch=float(pitch),
            seal_radius_voxels=int(seal_radius_voxels),
            band_voxels=float(sdf_band_voxels),
            smoothing_sigma_voxels=float(sdf_smoothing_sigma),
            max_memory_gb=float(max_sdf_memory_gb),
            max_projection_distance=float(max_projection_distance),
        ),
        artifact_path=implicit_field_path,
    )
    remeshed = trimesh.Trimesh(
        vertices=remesh_points,
        faces=remesh_faces,
        process=False,
    )
    remeshed.remove_unreferenced_vertices()
    try:
        remeshed.fix_normals()
    except Exception:
        pass
    report = {
        "method": "sealed_exterior_far_field_flood_then_marching_cubes",
        "implicit_method": "flood_signed_exact_narrow_band_tsdf_zero_surface",
        "pitch": pitch,
        "exterior_volume": sdf_report["exterior_volume"],
        "signed_distance_field": sdf_report,
        "source_trimesh_watertight": bool(source.is_watertight),
        "filled_voxels": int(sdf_report["solid_voxels"]),
        "output_trimesh_watertight": bool(remeshed.is_watertight),
    }
    return (
        np.asarray(remeshed.vertices, dtype=np.float64),
        np.asarray(remeshed.faces, dtype=np.int64),
        report,
    )


def voxel_count(voxel: Any) -> int | None:
    try:
        return int(voxel.encoding.sparse_indices.shape[0])
    except Exception:
        return None


def build_closure_proxy_artifact(
    args: argparse.Namespace,
    status: dict[str, Any],
    source_points: np.ndarray,
    source_faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], Path, dict[str, Any], np.ndarray, np.ndarray]:
    from source_shell_preservation import select_preserved_shells

    update_run_status(args, status, "closure_proxy_started", voxel_pitch=pitch_status(args))
    max_extent = float(np.max(np.ptp(source_points, axis=0)))
    atlas_seal_radius_world = max_extent / max(float(args.sealed_exterior_grid), 1.) * float(args.sealed_exterior_radius_voxels)
    seal_radius = max(1, int(np.ceil(atlas_seal_radius_world / float(args.voxel_pitch))))
    keep, preservation = select_preserved_shells(source_points, source_faces)
    kept_rows = np.flatnonzero(keep)
    kept_points, inverse = np.unique(source_faces[keep], return_inverse=True)
    kept_faces = inverse.reshape(-1, 3)
    if np.all(keep):
        rebuilt_points, rebuilt_faces = np.empty((0, 3)), np.empty((0, 3), dtype=np.int64)
        remesh_report = {"method": "certified_source_surface_retained", "pitch": float(args.voxel_pitch),
                         "voxel_reconstruction_applied": False}
    else:
        repair_points, repair_faces = compact_mesh(source_points, source_faces, np.flatnonzero(~keep))
        update_run_status(args, status, "closure_repair_regions_started", voxel_pitch=pitch_status(args))
        rebuilt_points, rebuilt_faces, remesh_report = voxel_watertight_remesh(
            repair_points, repair_faces, args.voxel_pitch, seal_radius_voxels=seal_radius,
            max_projection_distance=MAX_SOURCE_PROJECTION_DISTANCE_VOXELS * float(args.voxel_pitch),
            sdf_band_voxels=float(getattr(args, "sdf_band_voxels", 6.0)),
            sdf_smoothing_sigma=float(getattr(args, "sdf_smoothing_sigma", .5)),
            max_sdf_memory_gb=float(getattr(args, "max_sdf_memory_gb", 4.0)),
            implicit_field_path=args.output_dir / "implicit_field.npz")
        remesh_report["voxel_reconstruction_applied"] = True
        iterations = int(getattr(args, "scaffold_smoothing_iterations", 0))
        if iterations:
            rebuilt_points, smoothing = smooth_closure_scaffold(rebuilt_points, rebuilt_faces, float(args.voxel_pitch), iterations)
            remesh_report["scaffold_smoothing"] = smoothing
    proxy_points = np.vstack((source_points[kept_points], rebuilt_points))
    proxy_faces = np.vstack((kept_faces, rebuilt_faces + len(kept_points)))
    retained_points = np.r_[kept_points, np.full(len(rebuilt_points), -1, dtype=np.int64)]
    retained_faces = np.r_[kept_rows, np.full(len(rebuilt_faces), -1, dtype=np.int64)]
    remesh_report["source_preservation"] = {**preservation, "all_faces_preserved": bool(np.all(keep)),
        "repair_source_faces": int((~keep).sum()), "rebuilt_output_faces": len(rebuilt_faces)}
    remesh_report["seal_radius_resolution"] = {"atlas_grid_size": int(args.sealed_exterior_grid),
        "atlas_radius_voxels": int(args.sealed_exterior_radius_voxels), "atlas_radius_world": atlas_seal_radius_world,
        "closure_radius_voxels": seal_radius, "closure_radius_world": seal_radius * float(args.voxel_pitch)}
    proxy_points, intersection_repair = repair_closure_proxy_self_intersections(proxy_points, proxy_faces, float(args.voxel_pitch))
    remesh_report["self_intersection_repair"] = intersection_repair
    path = args.output_dir / "closure_proxy.vtp"
    write_vtp(path, proxy_points, proxy_faces, {"retained_source_face_id": retained_faces},
              point_data={"retained_source_point_id": retained_points})
    report = mesh_report(proxy_points, proxy_faces)
    update_run_status(args, status, "closure_proxy_written", outputs={"closure_proxy_vtp": str(path)},
                      metrics={"closure_proxy": topology_brief(report)}, voxel_pitch=pitch_status(args), remesh=remesh_report)
    return proxy_points, proxy_faces, remesh_report, path, report, retained_points, retained_faces


def resolve_voxel_pitch(args: argparse.Namespace, candidate_report: dict[str, Any]) -> None:
    requested_pitch = float(args.voxel_pitch)
    divisor = args.voxel_pitch_bbox_divisor
    args.requested_voxel_pitch = requested_pitch
    args.voxel_pitch_source = "explicit"
    args.voxel_pitch_bbox_max_extent = None
    if requested_pitch < 0.0:
        raise ValueError("--voxel-pitch must be positive, or zero for bbox-derived resolution")
    if requested_pitch > 0.0:
        return
    if divisor is None:
        raise ValueError("--voxel-pitch-bbox-divisor is required when --voxel-pitch is zero")
    divisor = float(divisor)
    if divisor <= 0.0:
        raise ValueError("--voxel-pitch-bbox-divisor must be positive")
    max_extent = max(float(value) for value in candidate_report["bounds"]["extents"])
    if max_extent <= 0.0:
        raise ValueError("cannot derive voxel pitch from an empty candidate bbox")
    args.voxel_pitch = max_extent / divisor
    args.voxel_pitch_source = "bbox_max_extent_divisor"
    args.voxel_pitch_bbox_divisor = divisor
    args.voxel_pitch_bbox_max_extent = max_extent


def prepare_group_candidate(
    args: argparse.Namespace,
    points: np.ndarray,
    faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any] | None]:
    candidate_indices = np.arange(faces.shape[0], dtype=np.int64)
    if not args.group_source_gltf:
        return candidate_indices, points, faces, None

    try:
        diagnostic_keep, group_filter = gltf_group_keep_mask(
            args.group_source_gltf,
            faces.shape[0],
            args.remove_name_regex,
        )
    except Exception as exc:
        group_filter = {
            "method": "gltf_geometry_name_diagnostic_unavailable",
            "source": str(args.group_source_gltf),
            "role": "diagnostic_only",
            "geometry_filter_applied": False,
            "input_geometry_truth": "mesh_input",
            "mapping_status": "unavailable",
            "diagnostic_error": {"type": type(exc).__name__, "message": str(exc)},
            "removed_triangles": 0,
            "removed_triangle_ratio": 0.0,
            "geometry_truth_preserved": True,
            "output_vtp": None,
        }
        return candidate_indices, points, faces, group_filter
    group_filter["diagnostic_keep_triangles"] = int(np.count_nonzero(diagnostic_keep))
    group_filter["geometry_truth_preserved"] = True
    group_filter["output_vtp"] = None
    return candidate_indices, points, faces, group_filter


def remap_face_values(
    source_face_ids: np.ndarray,
    source_values: np.ndarray,
    target_source_face_ids: np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    source_face_ids = np.asarray(source_face_ids, dtype=np.int64)
    source_values = np.asarray(source_values)
    target_source_face_ids = np.asarray(target_source_face_ids, dtype=np.int64)
    if source_values.ndim == 0 or source_face_ids.size != source_values.shape[0]:
        raise ValueError(f"cannot remap {name}: source ids and values have different lengths")
    if source_face_ids.size == 0:
        if target_source_face_ids.size:
            raise ValueError(f"cannot remap {name}: target ids exist but source map is empty")
        return source_values.copy()
    order = np.argsort(source_face_ids, kind="mergesort")
    sorted_ids = source_face_ids[order]
    positions = np.searchsorted(sorted_ids, target_source_face_ids)
    valid = positions < sorted_ids.size
    valid &= sorted_ids[np.minimum(positions, sorted_ids.size - 1)] == target_source_face_ids
    if not np.all(valid):
        missing = target_source_face_ids[~valid][:10].astype(int).tolist()
        raise ValueError(f"cannot remap {name}: target source ids are missing: {missing}")
    return source_values[order[positions]]


def apply_group_diagnostic_report_semantics(
    report: dict[str, Any],
    group_filter: dict[str, Any] | None,
) -> None:
    if not group_filter:
        return
    report["limitations"] = [
        row
        for row in report.get("limitations", [])
        if not row.startswith("GLTF group/name mapping")
        and row != "Group filtering assumes the GLTF geometry flatten order matches the input mesh triangle order."
    ]
    report["limitations"].append(
        "GLTF names and flattened face ranges are diagnostic metadata only; no source mesh triangles are removed from them."
    )
    repair_report = report.get("repair_report", {})
    for row in repair_report.get("geometry_to_mesh_trace", []):
        if row.get("stage") == "stage0_group_filter":
            row.update({
                "operation": "diagnose named non-target group candidates without changing mesh geometry",
                "removed_triangles": 0,
                "output": None,
                "status": "diagnostic_only",
            })
    removed = repair_report.get("change_summary", {}).get("removed", {})
    removed["group_name_removed_triangles"] = 0
    removed["largest_removed_groups"] = []
    repair_report.setdefault("change_summary", {})["diagnostic_only_group_name_candidates"] = {
        "triangles": int(group_filter.get("diagnostic_remove_candidate_triangles", 0)),
        "groups": int(group_filter.get("removed_groups", 0)),
        "geometry_filter_applied": False,
    }


def main() -> int:
    args = parse_args()
    args.component_filter_thresholds = component_filter_thresholds(args)
    if args.surface_strategy == "source-material" and not args._source_material_worker:
        from mesh_timeout import run_mesh_with_timeout
        command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--_source-material-worker"]
        return run_mesh_with_timeout(args, command, timeout_seconds=args.source_timeout_seconds)
    if args.surface_strategy == "projected-shell" and args.purpose != "print" and not args._projected_shell_worker:
        from mesh_timeout import run_mesh_with_timeout
        command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--_projected-shell-worker"]
        return run_mesh_with_timeout(args, command, timeout_seconds=args.repair_timeout_seconds)
    prepare_output_dir(args)
    try:
        status = initial_run_status(args)
        update_run_status(args, status, "starting")
        try:
            return run_pipeline(args, status)
        except Exception as exc:
            record_failure(args, status, exc)
            raise
    finally:
        release_output_lease(args)

def component_filter_thresholds(args: argparse.Namespace) -> dict[str, float]:
    values = {
        "exterior_support_threshold": float(args.exterior_component_support_threshold),
        "sealed_exterior_support_threshold": float(args.sealed_exterior_support_threshold),
        "internal_max_face_ratio": float(args.internal_component_max_face_ratio),
        "internal_max_diameter_ratio": float(args.internal_component_max_diameter_ratio),
        "internal_max_projected_bbox_area_ratio": float(
            args.internal_component_max_projected_bbox_area_ratio
        ),
        "floating_max_face_ratio": float(args.floating_fragment_max_face_ratio),
        "floating_max_diameter_ratio": float(args.floating_fragment_max_diameter_ratio),
        "shell_envelope_min_component_face_ratio": float(args.shell_envelope_min_component_face_ratio),
        "shell_envelope_margin_ratio": float(args.shell_envelope_margin_ratio),
    }
    invalid = {name: value for name, value in values.items() if not 0.0 <= value <= 1.0}
    if invalid:
        raise ValueError(f"component filter thresholds must be ratios in [0, 1]: {invalid}")
    return values


def run_pipeline(args: argparse.Namespace, status: dict[str, Any]) -> int:
    if args.purpose == "print":
        from print_mesh import run_print_pipeline
        return run_print_pipeline(args, status)
    if getattr(args, "surface_strategy", "projected-shell") == "source-material":
        from source_material_repair import repair_source_file
        update_run_status(args, status, "source_material_started")
        report = repair_source_file(args.projection_target or args.input, args.output_dir,
                                    max_error=args.source_max_error,
                                    boundary_mode=args.source_boundary_mode,
                                    policy_decisions=load_policy_decisions(args.policy_decisions),
                                    grid_size=args.visibility_grid,
                                    original_input=args.input,
                                    certify_reverse_fidelity=args.certify_reverse_fidelity,
                                    max_recovery_facets=getattr(args, "source_max_recovery_facets", None) or 1000,
                                    union_backend=getattr(args, "source_union_backend", None) or "cgal",
                                    assembly_context=getattr(args, "source_assembly_context", None),
                                    progress=lambda stage: update_run_status(args, status, stage))
        update_run_status(args, status, "completed", outputs=report["outputs"],
                          metrics={"decision": report["decision"]},
                          accepted_mesh_vtp=report["outputs"]["accepted_mesh_vtp"])
        return 0 if report["accepted"] else 2
    update_run_status(args, status, "read_input")
    source_mesh = read_surface(args.input)
    points = np.asarray(source_mesh.points, dtype=np.float64)
    faces = triangle_faces(source_mesh)
    original_report = mesh_report(points, faces)
    update_run_status(args, status, "input_metrics_computed", metrics={"original_input": topology_brief(original_report)})

    candidate_indices, group_points, group_faces, group_filter = prepare_group_candidate(args, points, faces)

    requested_depth_tolerance = float(args.depth_tolerance)
    args.requested_depth_tolerance = requested_depth_tolerance
    args.depth_tolerance, visibility_scale = resolve_visibility_depth_tolerance(
        group_points,
        group_faces,
        requested_depth_tolerance,
        bbox_ratio=args.depth_tolerance_bbox_ratio,
        edge_ratio=args.depth_tolerance_edge_ratio,
    )
    args.visibility_tolerance_report = visibility_scale
    update_run_status(
        args,
        status,
        "stage1_visibility_started",
        metrics={"stage1_visibility_scale": visibility_scale},
    )
    directly_visible, visible_score, view_reports, group_external_directions = cast(
        tuple[np.ndarray, np.ndarray, list[dict[str, Any]], np.ndarray],
        exterior_visibility_mask(
            group_points,
            group_faces,
            grid_size=args.visibility_grid,
            depth_tolerance=args.depth_tolerance,
            dilate_rings=0,
            min_visible_views=args.visibility_min_views,
            include_external_directions=True,
        ),
    )
    selected, component_visibility_report = preserve_whole_visible_components(
        group_faces,
        directly_visible,
    )
    outside_exposed, outside_flood_report = outside_flood_face_mask(
        group_points,
        group_faces,
        grid_size=args.outside_flood_grid,
    )
    first_hit_and_outside_exposed = directly_visible & outside_exposed
    selected_before_dilation = selected.copy()
    for _ in range(max(0, int(args.dilate_rings))):
        selected = dilate_by_vertices(group_faces, selected, group_points.shape[0])
    local_stage1_indices = np.flatnonzero(selected)
    if local_stage1_indices.size == 0:
        raise ValueError("stage1 exterior visibility atlas selected no source triangles")
    visibility_labels_path = write_visibility_labels(
        args,
        group_points,
        group_faces,
        candidate_indices,
        visible_score=visible_score,
        component_visible_hard_keep=selected_before_dilation,
        stage1_selected=selected,
        outside_exposed=outside_exposed,
    )
    stage1_indices = candidate_indices[local_stage1_indices]
    stage1_exterior_score = visible_score[local_stage1_indices]
    stage1_points, stage1_faces = compact_mesh(points, faces, stage1_indices)
    stage1_path = write_stage1(
        args,
        stage1_points,
        stage1_faces,
        stage1_indices,
        visible_score,
        local_stage1_indices,
        outside_exposed=outside_exposed,
    )
    stage1_report = mesh_report(stage1_points, stage1_faces)
    update_run_status(
        args,
        status,
        "stage1_exterior_candidate_written",
        outputs={
            "visibility_labeled_source_vtp": str(visibility_labels_path),
            "stage1_exterior_candidate_vtp": str(stage1_path),
        },
        metrics={"stage1_exterior_candidate": topology_brief(stage1_report)},
    )

    inventory_before = build_inventory(
        stage1_points,
        stage1_faces,
        stage1_indices,
        "stage1_exterior_candidate",
        max_items=args.policy_item_limit,
        exterior_face_score=stage1_exterior_score,
        component_thresholds=args.component_filter_thresholds,
    )
    update_run_status(args, status, "deterministic_repair_started")
    candidate_points, candidate_faces, candidate_sources, deterministic_passes = run_deterministic_repair(
        stage1_points,
        stage1_faces,
        stage1_indices,
    )
    seam_repair = {"status": "not_requested", "applied": False}
    if args.stitch_tolerance:
        from opening_repair import stitch_boundary_seams
        update_run_status(args, status, "boundary_seam_stitching")
        candidate_faces, seam_repair = stitch_boundary_seams(
            candidate_points, candidate_faces, tolerance=args.stitch_tolerance)
    candidate_exterior_score = remap_face_values(
        stage1_indices,
        stage1_exterior_score,
        candidate_sources,
        name="exterior view score",
    )
    update_run_status(args, status, "sealed_exterior_component_atlas_started")
    candidate_sealed_exterior_mask, sealed_exterior_report = build_sealed_exterior_atlas(
        candidate_points,
        candidate_faces,
        grid_size=args.sealed_exterior_grid,
        seal_radius_voxels=args.sealed_exterior_radius_voxels,
        surface_band_voxels=args.sealed_exterior_band_voxels,
    )
    pre_prune_points = candidate_points
    pre_prune_faces = candidate_faces
    pre_prune_sources = candidate_sources.copy()
    pre_prune_exterior_score = candidate_exterior_score.copy()
    pre_prune_inventory = build_inventory(
        candidate_points,
        candidate_faces,
        candidate_sources,
        "deterministic_cleanup_candidate",
        max_items=args.policy_item_limit,
        exterior_face_score=candidate_exterior_score,
        sealed_exterior_face_mask=candidate_sealed_exterior_mask,
        component_thresholds=args.component_filter_thresholds,
    )
    candidate_points, candidate_faces, candidate_sources, candidate_exterior_score, component_prune_pass = (
        prune_classified_internal_components(
            candidate_points,
            candidate_faces,
            candidate_sources,
            candidate_exterior_score,
            pre_prune_inventory,
        )
    )
    if component_prune_pass["status"] == "applied":
        visibility_regression = six_view_depth_regression(
            pre_prune_points,
            pre_prune_faces,
            candidate_points,
            candidate_faces,
            grid_size=min(int(args.visibility_grid), 720),
            depth_tolerance=float(args.depth_tolerance),
        )
        if not visibility_regression["passed"]:
            trial_after = component_prune_pass["after"]
            scope = component_prune_pass["scope"]
            thresholds = component_prune_pass["thresholds"]
            attempted_component_ids = thresholds.pop("removed_component_ids", [])
            attempted_component_reasons = thresholds.pop("removed_component_reasons", {})
            thresholds.update(
                {
                    "attempted_component_ids": attempted_component_ids,
                    "attempted_component_reasons": attempted_component_reasons,
                    "rollback_reason_codes": ["six_view_aperture_or_depth_regression"],
                }
            )
            scope["attempted_removed_source_triangle_count"] = scope[
                "removed_source_triangle_count"
            ]
            scope["attempted_removed_source_triangle_ids"] = scope[
                "removed_source_triangle_ids"
            ]
            scope["removed_source_triangle_count"] = 0
            scope["removed_source_triangle_ids"] = []
            scope["removed_component_count"] = 0
            component_prune_pass.update(
                {
                    "status": "rolled_back",
                    "after": component_prune_pass["before"],
                    "trial_after": trial_after,
                    "failure_reason": (
                        "component pruning exposed background or a deeper surface in the "
                        "fixed-projection six-view visibility guard"
                    ),
                }
            )
            candidate_points = pre_prune_points
            candidate_faces = pre_prune_faces
            candidate_sources = pre_prune_sources
            candidate_exterior_score = pre_prune_exterior_score
    else:
        visibility_regression = {
            "method": "matched_projection_six_view_triangle_raster_depth_comparison",
            "passed": True,
            "status": "not_run",
            "reason": "component prune did not propose a committed geometry change",
        }
    component_prune_pass["visibility_regression_guard"] = visibility_regression
    deterministic_passes.append(component_prune_pass)
    candidate_sealed_exterior_mask = remap_face_values(
        pre_prune_sources,
        candidate_sealed_exterior_mask,
        candidate_sources,
        name="sealed exterior face mask",
    ).astype(bool)
    candidate_external_directions = remap_face_values(
        stage1_indices,
        group_external_directions[local_stage1_indices],
        candidate_sources,
        name="external direction evidence",
    )
    candidate_path = write_source_preserving_candidate(
        args,
        candidate_points,
        candidate_faces,
        candidate_sources,
        exterior_face_score=candidate_exterior_score,
        sealed_exterior_face_mask=candidate_sealed_exterior_mask,
        face_external_directions=candidate_external_directions,
    )
    candidate_report = mesh_report(candidate_points, candidate_faces)
    resolve_voxel_pitch(args, candidate_report)
    update_run_status(
        args,
        status,
        "source_preserving_candidate_written",
        outputs={"source_preserving_candidate_vtp": str(candidate_path)},
        metrics={"deterministic_repair_candidate": topology_brief(candidate_report)},
        voxel_pitch=pitch_status(args),
    )

    inventory_after = (
        {**pre_prune_inventory, "stage": "deterministic_repair_candidate"}
        if component_prune_pass["status"] == "skipped"
        else build_inventory(
            candidate_points,
            candidate_faces,
            candidate_sources,
            "deterministic_repair_candidate",
            max_items=args.policy_item_limit,
            exterior_face_score=candidate_exterior_score,
            sealed_exterior_face_mask=candidate_sealed_exterior_mask,
            component_thresholds=args.component_filter_thresholds,
        )
    )
    policy_evidence = attach_policy_evidence_views(
        candidate_points,
        candidate_faces,
        inventory_after,
        args.output_dir,
        image_size=min(max(int(args.preview_size), 256), 900),
    )
    policy_packet = build_policy_packet(
        inventory_after,
        args.target_name,
        max_items=args.policy_item_limit,
        run_context=run_parameters(args),
    )
    policy_packet["evidence_report"] = policy_evidence
    policy_packet_path = args.output_dir / "ai_policy_packet.json"
    policy_decisions = resolve_policy_decisions(policy_packet, load_policy_decisions(args.policy_decisions))
    opening_transactions = []
    if args.policy_decisions:
        from opening_repair import apply_opening_decisions
        reviewed_faces, _generated, opening_transactions, applied = apply_opening_decisions(
            candidate_points, candidate_faces, inventory_after, policy_decisions, default_action="cap",
        )
        (args.output_dir / "opening_transactions.json").write_text(json.dumps(opening_transactions, indent=2), encoding="utf-8")
        policy_packet_path.write_text(json.dumps(policy_packet, indent=2), encoding="utf-8")
        if not applied:
            raise ValueError("opening decisions were not applied; unresolved or preserved free boundaries require wall geometry (use print thicken), and cannot enter global shell closure")
        added = len(reviewed_faces) - len(candidate_faces)
        candidate_faces = reviewed_faces
        candidate_sources = np.r_[candidate_sources, np.full(added, -1, dtype=np.int64)]
        candidate_exterior_score = np.r_[candidate_exterior_score, np.ones(added)]
        candidate_sealed_exterior_mask = np.r_[candidate_sealed_exterior_mask, np.ones(added, dtype=bool)]
        candidate_external_directions = np.vstack([candidate_external_directions, np.zeros((added, 3))])
        candidate_path = write_source_preserving_candidate(
            args, candidate_points, candidate_faces, candidate_sources,
            exterior_face_score=candidate_exterior_score,
            sealed_exterior_face_mask=candidate_sealed_exterior_mask,
            face_external_directions=candidate_external_directions,
        )
        candidate_report = mesh_report(candidate_points, candidate_faces)
        inventory_after = build_inventory(
            candidate_points, candidate_faces, candidate_sources, "after_reviewed_opening_caps",
            max_items=args.policy_item_limit, exterior_face_score=candidate_exterior_score,
            sealed_exterior_face_mask=candidate_sealed_exterior_mask,
            component_thresholds=args.component_filter_thresholds,
        )

    projection_target = getattr(args, "projection_target", None)
    projection_points, projection_faces = candidate_points, candidate_faces
    projection_reliable = (candidate_exterior_score > 0) | candidate_sealed_exterior_mask
    projection_normals = None
    if projection_target is not None:
        reviewed_surface = read_surface(projection_target, clean=False)
        projection_points = np.asarray(reviewed_surface.points, dtype=np.float64)
        projection_faces = triangle_faces(reviewed_surface)
        projection_reliable = None
        projection_normals = reviewed_surface.point_data.get("Normals")

    (
        stage2_points,
        stage2_faces,
        remesh_report,
        closure_proxy_path,
        closure_proxy_report,
        retained_source_points,
        retained_source_faces,
    ) = build_closure_proxy_artifact(
        args,
        status,
        candidate_points,
        candidate_faces,
    )
    proxy_intersection_repair = remesh_report["self_intersection_repair"]
    preserved_vertices = retained_source_points >= 0
    preserved_faces = retained_source_faces >= 0
    if projection_target is None and projection_reliable is not None:
        projection_reliable[retained_source_faces[preserved_faces]] = True

    update_run_status(args, status, "source_projection_started", voxel_pitch=pitch_status(args))
    projection_thresholds = ProjectionThresholds(
        max_projection_distance=(
            MAX_SOURCE_PROJECTION_DISTANCE_VOXELS * float(args.voxel_pitch)
        ),
        orientation_vote_min_margin=0.02,
        source_edge_barycentric_margin=0.0,
        collision_tolerance_bbox_ratio=1e-8,
        cross_component_jump_ratio=getattr(args, "cross_component_jump_ratio", None),
    )
    projected_points, projected_faces, projection_provenance, projection_report = project_closure_to_source(
        stage2_points,
        stage2_faces,
        projection_points,
        projection_faces,
        thresholds=projection_thresholds,
        reliable_source_face_mask=projection_reliable,
        preserved_vertex_mask=preserved_vertices,
    )
    projection_report["source_query"]["path"] = str(projection_target or candidate_path)
    projection_report["sealed_exterior_construction"] = remesh_report.get(
        "exterior_volume"
    )
    projection_report["face_origin_codes"] = {
        "1": "source_projected",
        "2": "mixed_source_sdf_transition",
        "3": "sdf_generated",
        "4": "retained_prepared_source_surface",
    }
    projection_report["resolution_context"] = {
        "closure_pitch": float(args.voxel_pitch),
        "closure_pitch_source": getattr(args, "voxel_pitch_source", "explicit"),
        "projection_distance_in_closure_pitches": MAX_SOURCE_PROJECTION_DISTANCE_VOXELS,
        "resolved_max_projection_distance": projection_report.get("thresholds", {}).get(
            "resolved_max_projection_distance"
        ),
    }
    projected_points, projection_provenance, self_intersection = repair_projected_self_intersections(
        projected_points,
        stage2_points,
        projected_faces,
        projection_provenance,
        certified_proxy_report=proxy_intersection_repair["final_self_intersection"],
    )
    projection_report["closure_proxy_self_intersection_repair"] = proxy_intersection_repair
    projection_report["source_preservation"] = {**remesh_report["source_preservation"],
        "geometry_unchanged": bool(np.array_equal(projected_points[preserved_vertices],
            candidate_points[retained_source_points[preserved_vertices]]))}
    if np.any(preserved_faces):
        orientation_ok = False
        if self_intersection.get("passed"):
            from print_geometry import orient_material
            oriented, _ = orient_material(trimesh.Trimesh(projected_points, projected_faces, process=False))
            orientation_ok = bool(np.all(np.logical_or.reduce([
                np.all(np.roll(np.asarray(oriented.faces), shift, axis=1) == projected_faces, axis=1)
                for shift in range(3)])))
        projection_report["source_preservation"]["assembled_material_orientation"] = orientation_ok
    if np.any(preserved_faces) and not np.all(preserved_faces):
        rebuilt_points, rebuilt_faces = compact_mesh(projected_points, projected_faces, np.flatnonzero(~preserved_faces))
        projection_report["reconstructed_region_metrics"] = mesh_report(rebuilt_points, rebuilt_faces)
    projection_provenance["point_data"]["retained_source_point_id"] = retained_source_points
    projection_provenance["cell_data"]["retained_source_face_id"] = retained_source_faces
    projection_provenance["cell_data"]["face_origin"][preserved_faces] = 4
    projection_provenance["point_data"]["Normals"] = interpolate_source_normals(
        projected_points, projected_faces, projection_points, projection_faces,
        projection_provenance, projection_normals,
    )
    source_corners = source_mesh.cell_data.get("corner_normals")
    if source_corners is not None and np.any(preserved_faces):
        corners = projection_provenance["point_data"]["Normals"][projected_faces].reshape(-1, 9)
        original_ids = candidate_sources[retained_source_faces[preserved_faces]]
        known = original_ids >= 0
        offset = source_mesh.n_verts + source_mesh.n_lines
        corners[np.flatnonzero(preserved_faces)[known]] = np.asarray(source_corners)[offset + original_ids[known]]
        projection_provenance["cell_data"]["corner_normals"] = corners
    projection_report["normal_transfer"] = {
        "method": "barycentric_source_point_normals_on_projected_vertices",
        "geometry_changed": False, "fallback_normals": "candidate_geometry",
    }
    projected_candidate_path = write_source_projected_watertight_candidate(
        args,
        projected_points,
        projected_faces,
        projection_provenance,
    )
    post_write_validation = roundtrip_validation(
        projected_candidate_path,
        projected_points,
        projected_faces,
    )
    projection_report = update_projection_report(
        projection_report,
        projected_points,
        projected_faces,
        projection_provenance,
        self_intersection,
        post_write_validation,
    )
    projection_report["six_view_depth_regression"] = {
        **six_view_depth_regression(
            candidate_points,
            candidate_faces,
            projected_points,
            projected_faces,
            grid_size=min(int(args.visibility_grid), 720),
            depth_tolerance=max(float(args.depth_tolerance), 3.0 * float(args.voxel_pitch)),
        ),
        "status": "computed",
    }
    projection_report_path = args.output_dir / "source_projection_report.json"
    projection_report_path.write_text(json.dumps(projection_report, indent=2), encoding="utf-8")
    projected_candidate_report = mesh_report(projected_points, projected_faces)
    update_run_status(
        args,
        status,
        "source_projected_watertight_candidate_written",
        outputs={
            "source_projected_watertight_candidate_vtp": str(projected_candidate_path),
            "source_projection_report_json": str(projection_report_path),
        },
        metrics={
            "source_projected_watertight_candidate": topology_brief(projected_candidate_report),
            "self_intersection": self_intersection,
            "post_write_validation": post_write_validation,
        },
        voxel_pitch=pitch_status(args),
    )
    projected_visual_drift = silhouette_drift_from_meshes(
        candidate_points,
        candidate_faces,
        projected_points,
        projected_faces,
        [VIEWS[0], VIEWS[2], VIEWS[4]],
        max_size=min(args.visibility_grid, 720),
    )
    previews = write_previews(
        args,
        candidate_points,
        candidate_faces,
        stage2_points,
        stage2_faces,
        VIEWS,
        projected_points=projected_points,
        projected_faces=projected_faces,
    )
    extraction = extraction_report(view_reports, visible_score, stage1_indices, faces)
    extraction.update({
        "method": "stl_truth_triangle_first_hit_hard_keep_with_outside_flood_diagnostic",
        "geometry_truth": "input_mesh",
        "gltf_role": "diagnostic_only" if group_filter is not None else "not_provided",
        "depth_tolerance": visibility_scale,
        "visibility_score": {
            "meaning": "number of independent view atlases with first-hit pixel support",
            "hard_keep_minimum_views": 1,
            "requested_corroboration_views": args.visibility_min_views,
            "corroboration_cannot_override_hard_keep": True,
            "histogram": np.bincount(visible_score, minlength=len(VIEWS) + 1).astype(int).tolist(),
        },
        "continuity": {
            "method": "shared_vertex_local_dilation",
            "rings": max(0, int(args.dilate_rings)),
            "direct_first_hit_triangles": int(np.count_nonzero(visible_score > 0)),
            "corroborated_first_hit_triangles": int(
                np.count_nonzero(visible_score >= args.visibility_min_views)
            ),
            "visible_hard_keep_triangles": int(np.count_nonzero(selected_before_dilation)),
            "first_hit_and_outside_flood_triangles": int(np.count_nonzero(first_hit_and_outside_exposed)),
            "selected_after_dilation_triangles": int(stage1_indices.size),
        },
        "component_visibility_guard": component_visibility_report,
        "outside_flood": outside_flood_report,
        "sealed_exterior_component_atlas": sealed_exterior_report,
    })
    outputs = outputs_report(
        args,
        visibility_labels_path,
        stage1_path,
        candidate_path,
        closure_proxy_path,
        policy_packet_path,
        previews,
        projected_candidate_path,
        projection_report_path,
    )
    report = build_report(
        args,
        {
            "original_input": original_report,
            "stage1_exterior_candidate": stage1_report,
            "deterministic_repair_candidate": candidate_report,
            "closure_proxy": closure_proxy_report,
            "source_projected_watertight_candidate": projected_candidate_report,
        },
        extraction,
        deterministic_passes,
        inventory_before,
        inventory_after,
        policy_packet,
        policy_decisions,
        remesh_report,
        outputs,
        group_filter,
        projection_report,
        post_write_validation,
        projected_visual_drift,
    )
    apply_group_diagnostic_report_semantics(report, group_filter)
    report["opening_transactions"] = opening_transactions
    report["boundary_stitch"] = seam_repair
    if args.purpose == "cfd":
        if report["decision"]["status"] == "accepted":
            from cfd_surface_refine import refine_cfd_surface

            update_run_status(args, status, "cfd_surface_refinement_started")
            refinement = refine_cfd_surface(
                Path(report["outputs"]["accepted_mesh_vtp"]), projection_target or candidate_path,
                args.output_dir / "cfd", max_edge_length=args.cfd_max_edge_length,
                max_output_faces=args.cfd_max_output_faces,
            )
            record_cfd_refinement(report, refinement)
        else:
            report["cfd_surface_refinement"] = {"status": "skipped", "reason": "watertight input was not accepted"}
    html_path = args.output_dir / "two_stage_report.html"
    report["outputs"]["html_report"] = str(html_path)
    report_path = args.output_dir / "two_stage_report.json"
    report["outputs"]["report_json"] = str(report_path)
    policy_packet_path.write_text(json.dumps(policy_packet, indent=2), encoding="utf-8")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    update_run_status(
        args,
        status,
        "json_report_written",
        outputs=report["outputs"],
        metrics={"decision": report["decision"]},
        voxel_pitch=pitch_status(args),
        accepted_mesh_vtp=report["outputs"].get("accepted_mesh_vtp"),
    )
    write_html_report(report, html_path, "Watertight Exterior Shell Audit")
    update_run_status(
        args,
        status,
        "completed",
        outputs=report["outputs"],
        metrics={"decision": report["decision"]},
        voxel_pitch=pitch_status(args),
        accepted_mesh_vtp=report["outputs"].get("accepted_mesh_vtp"),
    )
    print_summary(
        report_path,
        stage1_report,
        candidate_report,
        closure_proxy_report,
        report,
        projected_candidate_report,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
