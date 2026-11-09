from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import manifold3d as manifold
import numpy as np
import pyvista as pv

from mesh_io import triangle_faces, write_vtp
from mesh_metrics import edge_topology, face_component_labels, mesh_report
from mesh_topology_repair import collapse_short_manifold_edges
from source_projected_validation import exact_self_intersection_report


def collapse_nanoface_edges(
    points: np.ndarray,
    faces: np.ndarray,
    *,
    pinned_points: np.ndarray | None = None,
    max_edge_length_m: float = 10e-9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, object]]]:
    """Compatibility adapter for the earlier source-preserving trial API.

    ``max_edge_length_m`` remains in meters for old callers. New geometry
    code should call ``collapse_short_manifold_edges`` with input mesh units.
    """
    max_edge_length_m = float(max_edge_length_m)
    if not np.isfinite(max_edge_length_m) or max_edge_length_m <= 0:
        raise ValueError("max_edge_length_m must be finite and positive")
    points, faces, face_ids, actions = collapse_short_manifold_edges(
        points, faces, pinned_points=pinned_points, max_edge_length=max_edge_length_m
    )
    compatible_actions = [
        {**{key: value for key, value in action.items() if key != "length"}, "length_m": action["length"]}
        for action in actions
    ]
    return points, faces, face_ids, compatible_actions


def run_trial(
    source_vtp: Path,
    output_dir: Path,
    *,
    primitive_node_index: int,
    component_id: int,
    thickness_m: float,
    bottom_bevel_m: float,
    offset_mode: str = "face_native",
    build_mode: str = "triangle_prisms",
    batch_triangles: int = 32,
) -> dict[str, Any]:
    if thickness_m <= 0 or bottom_bevel_m < 0 or batch_triangles <= 0:
        raise ValueError("thickness must be positive; bevel and batch size must be nonnegative/positive")
    if offset_mode not in {"face_native", "component_mean_native"}:
        raise ValueError("offset_mode must be face_native or component_mean_native")
    if build_mode not in {"triangle_prisms", "component_shell"}:
        raise ValueError("build_mode must be triangle_prisms or component_shell")
    if build_mode == "component_shell" and component_id != -1:
        raise ValueError("component_shell operates on every edge-connected part of the native primitive; use component_id=-1")

    source = pv.read(source_vtp)
    points = np.asarray(source.points, dtype=np.float64)
    faces_all = triangle_faces(source)
    cell_offset = source.n_verts + source.n_lines
    node_index = np.asarray(source.cell_data["gltf_node_index"])[cell_offset : cell_offset + len(faces_all)]
    original_cell_ids = np.asarray(source.cell_data["vtkOriginalCellIds"])[
        cell_offset : cell_offset + len(faces_all)
    ]
    normals = np.asarray(source.point_data["Normals"], dtype=np.float64)
    primitive_mask = node_index == primitive_node_index
    primitive_faces = faces_all[primitive_mask]
    primitive_original_cell_ids = original_cell_ids[primitive_mask]
    triangles = points[primitive_faces]
    areas = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    ) * 0.5
    valid = areas > 1e-16
    primitive_faces = primitive_faces[valid]
    primitive_original_cell_ids = primitive_original_cell_ids[valid]
    source_points, inverse = np.unique(primitive_faces, return_inverse=True)
    local_points = points[source_points]
    local_faces = inverse.reshape(-1, 3)
    topology, edge_to_faces = edge_topology(local_faces)
    component_labels = face_component_labels(len(local_faces), edge_to_faces)
    selected_faces = (
        np.arange(len(local_faces), dtype=np.int64)
        if component_id < 0
        else np.flatnonzero(component_labels == component_id)
    )
    if selected_faces.size == 0:
        raise ValueError(f"component {component_id} is absent from node {primitive_node_index}")

    faces = local_faces[selected_faces]
    source_face_ids = primitive_original_cell_ids[selected_faces]
    weld_displacement_max = 0.0
    local_normals = normals[source_points]
    if build_mode == "component_shell":
        rounded_positions = np.round(local_points, decimals=12)
        _, first_indices, weld_inverse = np.unique(
            rounded_positions, axis=0, return_index=True, return_inverse=True
        )
        welded_points = local_points[first_indices]
        normal_sums = np.zeros((len(welded_points), 3), dtype=np.float64)
        np.add.at(normal_sums, weld_inverse, local_normals)
        normal_lengths = np.linalg.norm(normal_sums, axis=1)
        local_normals = normal_sums / np.maximum(normal_lengths[:, None], 1e-30)
        weld_displacement_max = float(
            np.max(np.linalg.norm(local_points - welded_points[weld_inverse], axis=1))
        )
        local_points = welded_points
        faces = weld_inverse[faces]
    component_topology, _ = edge_topology(faces)
    if build_mode == "component_shell" and (
        component_topology["non_manifold_edges"]
        or component_topology["non_manifold_vertices"]
    ):
        raise ValueError(
            "exact-coordinate-welded primitive still has non-manifold edges/vertices: "
            f"{component_topology['non_manifold_edges']}/{component_topology['non_manifold_vertices']}"
        )
    triangles = local_points[faces]
    geometric_normals = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    geometric_normals /= np.maximum(np.linalg.norm(geometric_normals, axis=1)[:, None], 1e-30)
    native_normals = normals[source_points[faces]].mean(axis=1)
    native_normals /= np.maximum(np.linalg.norm(native_normals, axis=1)[:, None], 1e-30)
    alignment = np.einsum("ij,ij->i", geometric_normals, native_normals)
    native_normals[alignment < 0] *= -1
    component_normal = native_normals.mean(axis=0)
    component_normal /= max(float(np.linalg.norm(component_normal)), 1e-30)
    extrusion_normals = (
        np.broadcast_to(component_normal, native_normals.shape)
        if offset_mode == "component_mean_native"
        else native_normals
    )
    extrusion_alignment = np.einsum("ij,ij->i", geometric_normals, extrusion_normals)

    started = time.monotonic()
    solids = []
    if build_mode == "component_shell":
        component_used_points, component_inverse = np.unique(faces, return_inverse=True)
        component_points = local_points[component_used_points]
        faces = component_inverse.reshape(-1, 3)
        component_normals = local_normals[component_used_points]
        component_faces, sidewall_faces, cell_data = extrude_source_sheet(
            faces,
            len(component_points),
            source_face_ids,
        )
        candidate_points = np.vstack((component_points, component_points - component_normals * thickness_m))
        candidate_faces = np.vstack((faces, component_faces, sidewall_faces))
        candidate_faces, flipped_components, normal_signs = orient_components_outward(
            candidate_points, candidate_faces
        )
        component_normals *= normal_signs[: len(component_points), None]
        united = None
    else:
        prism_faces = np.asarray(
            [
                [0, 1, 2], [3, 5, 4],
                [0, 3, 4], [0, 4, 1],
                [1, 4, 5], [1, 5, 2],
                [2, 5, 3], [2, 3, 0],
            ],
            dtype=np.uint64,
        )
        solids = []
        for start in range(0, len(faces), batch_triangles):
            batch = triangles[start : start + batch_triangles]
            batch_normals = extrusion_normals[start : start + batch_triangles]
            bottom = batch - batch_normals[:, None, :] * thickness_m
            if bottom_bevel_m:
                center = bottom.mean(axis=1, keepdims=True)
                radial = bottom - center
                radial /= np.maximum(np.linalg.norm(radial, axis=2, keepdims=True), 1e-30)
                bottom += radial * bottom_bevel_m
            vertices = np.concatenate((batch, bottom), axis=1).reshape(-1, 3)
            batch_faces = (
                np.arange(len(batch), dtype=np.uint64)[:, None, None] * 6 + prism_faces[None, :, :]
            ).reshape(-1, 3)
            solid = manifold.Manifold(
                manifold.Mesh64(
                    np.ascontiguousarray(vertices, dtype=np.float64),
                    np.ascontiguousarray(batch_faces, dtype=np.uint64),
                )
            )
            if solid.status() != manifold.Error.NoError:
                raise RuntimeError(f"invalid prism batch at source triangle {start}: {solid.status()}")
            solids.append(solid)
        united = manifold.Manifold.batch_boolean(solids, manifold.OpType.Add)
    report: dict[str, Any] = {
        "method": (
            "native source triangles plus a translated inner skin and boundary sidewalls"
            if build_mode == "component_shell"
            else "native-normal triangular prisms with Manifold3D union"
        ),
        "status": "diagnostic_only",
        "source_vtp": str(source_vtp),
        "primitive_node_index": int(primitive_node_index),
        "component_id": int(component_id),
        "selected_connected_component_count": int(len(np.unique(component_labels[selected_faces]))),
        "component_source_triangles": int(len(faces)),
        "component_boundary_edges_before": int(component_topology["boundary_edges"]),
        "component_non_manifold_edges_before": int(component_topology["non_manifold_edges"]),
        "component_non_manifold_vertices_before": int(component_topology["non_manifold_vertices"]),
        "exact_coordinate_weld_max_displacement_m": weld_displacement_max,
        "prism_thickness_m": float(thickness_m),
        "bottom_in_plane_bevel_m": float(bottom_bevel_m),
        "offset_mode": offset_mode,
        "build_mode": build_mode,
        "opening_semantics": (
            {
                "cap_faces_added_across_source_apertures": 0,
                "boundary_sidewalls_added": int(len(cell_data["face_origin_code"]) - 2 * len(faces)),
                "note": "Every source boundary is extruded into a wall that preserves its through-opening; this does not decide whether the opening is functionally correct.",
            }
            if build_mode == "component_shell"
            else None
        ),
        "flipped_components_to_positive_volume": flipped_components if build_mode == "component_shell" else [],
        "component_mean_native_normal": component_normal.tolist(),
        "face_normal_offset_alignment_p01_median_p99": [
            float(np.percentile(extrusion_alignment, q)) for q in (1, 50, 99)
        ],
        "native_normal_alignment_absolute_p01": float(np.percentile(np.abs(alignment), 1)),
        "batch_triangles": int(batch_triangles),
        "batch_count": int(len(solids)),
        "manifold_boolean_status": str(united.status()) if united else "not_used_for_component_shell",
        "wall_seconds": time.monotonic() - started,
        "printer_wall_thickness_qualified": False,
    }
    if united is None or united.status() == manifold.Error.NoError:
        if united is not None:
            mesh = united.to_mesh64()
            candidate_points = np.asarray(mesh.vert_properties, dtype=np.float64)[:, :3]
            candidate_faces = np.asarray(mesh.tri_verts, dtype=np.int64)
        metrics = mesh_report(candidate_points, candidate_faces)
        candidate_keys = face_keys(candidate_points, candidate_faces)
        source_geometry_points = component_points if build_mode == "component_shell" else local_points
        source_keys = set(face_keys(source_geometry_points, faces))
        retained = sum(key in source_keys for key in candidate_keys)
        output_dir.mkdir(parents=True, exist_ok=True)
        candidate_path = output_dir / "candidate.vtp"
        point_data = None
        output_cell_data = None
        if build_mode == "component_shell":
            point_data = {"Normals": np.vstack((component_normals, -component_normals))}
            output_cell_data = cell_data
        write_vtp(candidate_path, candidate_points, candidate_faces, output_cell_data, point_data=point_data)

        readback = pv.read(candidate_path)
        readback_points = np.asarray(readback.points, dtype=np.float64)
        readback_faces = triangle_faces(readback)
        readback_metrics = mesh_report(readback_points, readback_faces)
        exact_roundtrip = bool(
            np.array_equal(readback_points, candidate_points)
            and np.array_equal(readback_faces, candidate_faces)
        )
        report["candidate"] = {
            "path": str(candidate_path),
            "points": int(len(candidate_points)),
            "triangles": int(len(candidate_faces)),
            "boundary_edges": int(metrics["topology"]["boundary_edges"]),
            "non_manifold_edges": int(metrics["topology"]["non_manifold_edges"]),
            "non_manifold_vertices": int(metrics["topology"]["non_manifold_vertices"]),
            "components": int(metrics["topology"]["components"]["count"]),
            "degenerate_faces": int(metrics["quality"]["degenerate_faces"]),
            "volume_reliable": bool(metrics["volume"]["reliable"]),
            "exact_native_triangles_retained": int(retained),
            "source_triangle_retention_fraction": float(retained / len(faces)),
            "generated_or_remeshed_faces": int(len(candidate_faces) - retained),
        }
        report["serialization_roundtrip"] = {
            "geometry_exact": exact_roundtrip,
            "points": int(len(readback_points)),
            "triangles": int(len(readback_faces)),
            "boundary_edges": int(readback_metrics["topology"]["boundary_edges"]),
            "non_manifold_edges": int(readback_metrics["topology"]["non_manifold_edges"]),
            "non_manifold_vertices": int(readback_metrics["topology"]["non_manifold_vertices"]),
            "degenerate_faces": int(readback_metrics["quality"]["degenerate_faces"]),
        }
        zero_topology = (
            readback_metrics["topology"]["boundary_edges"] == 0
            and readback_metrics["topology"]["non_manifold_edges"] == 0
            and readback_metrics["topology"]["non_manifold_vertices"] == 0
            and readback_metrics["quality"]["degenerate_faces"] == 0
        )
        if exact_roundtrip and zero_topology:
            exact = exact_self_intersection_report(readback_points, readback_faces)
            report["exact_self_intersection"] = {
                key: value for key, value in exact.items() if key != "reported_pairs"
            }

    report_path = output_dir / "trial.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def extrude_source_sheet(
    faces: np.ndarray,
    point_count: int,
    source_face_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Close one manifold sheet by adding an offset inner skin and edge walls."""
    faces = np.asarray(faces, dtype=np.int64)
    source_face_ids = np.asarray(source_face_ids, dtype=np.int64)
    edge_directions: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for face_id, (a, b, c) in enumerate(faces):
        for left, right in ((a, b), (b, c), (c, a)):
            key = (min(int(left), int(right)), max(int(left), int(right)))
            edge_directions.setdefault(key, []).append((int(left), int(right)))
    if any(len(incidence) > 2 for incidence in edge_directions.values()):
        raise ValueError("component sheet contains non-manifold edges")

    bottom_faces = faces[:, ::-1] + point_count
    side_faces = []
    for incidence in edge_directions.values():
        if len(incidence) != 1:
            continue
        left, right = incidence[0]
        bottom_left = left + point_count
        bottom_right = right + point_count
        side_faces.extend(
            (
                (left, bottom_left, bottom_right),
                (left, bottom_right, right),
            )
        )
    side_array = np.asarray(side_faces, dtype=np.int64).reshape(-1, 3)
    source_rows = np.concatenate((source_face_ids, source_face_ids))
    cell_data = {
        "source_face_id": np.concatenate((source_rows, np.full(len(side_array), -1, dtype=np.int64))),
        "face_origin_code": np.concatenate(
            (
                np.zeros(len(faces), dtype=np.uint8),
                np.ones(len(faces), dtype=np.uint8),
                np.full(len(side_array), 2, dtype=np.uint8),
            )
        ),
    }
    return bottom_faces, side_array, cell_data


def orient_components_outward(
    points: np.ndarray, faces: np.ndarray
) -> tuple[np.ndarray, list[int], np.ndarray]:
    _, edge_to_faces = edge_topology(faces)
    labels = face_component_labels(len(faces), edge_to_faces)
    oriented = np.asarray(faces, dtype=np.int64).copy()
    normal_signs = np.ones(len(points), dtype=np.float64)
    flipped = []
    for component_id in range(int(labels.max()) + 1 if len(labels) else 0):
        face_ids = np.flatnonzero(labels == component_id)
        component_faces = oriented[face_ids]
        triangles = points[component_faces]
        volume = float(
            np.einsum(
                "ij,ij->i",
                triangles[:, 0],
                np.cross(triangles[:, 1], triangles[:, 2]),
            ).sum()
            / 6.0
        )
        if volume < 0:
            oriented[face_ids] = oriented[face_ids, ::-1]
            normal_signs[np.unique(component_faces)] *= -1
            flipped.append(component_id)
    return oriented, flipped, normal_signs


def face_keys(points: np.ndarray, faces: np.ndarray) -> list[bytes]:
    coordinates = np.round(points[faces], decimals=12)
    order = np.lexsort((coordinates[:, :, 2], coordinates[:, :, 1], coordinates[:, :, 0]), axis=1)
    coordinates = np.take_along_axis(coordinates, order[:, :, None], axis=1)
    return [triangle.tobytes() for triangle in coordinates]


def main() -> None:
    parser = argparse.ArgumentParser(description="Source-preserving native-triangle prism union trial")
    parser.add_argument("source_vtp", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--primitive-node-index", type=int, required=True)
    parser.add_argument("--component-id", type=int, required=True, help="Use -1 to process every connected surface component in the primitive.")
    parser.add_argument("--thickness-m", type=float, default=1e-5)
    parser.add_argument("--bottom-bevel-m", type=float, default=0.0)
    parser.add_argument(
        "--offset-mode",
        choices=("face_native", "component_mean_native"),
        default="face_native",
    )
    parser.add_argument(
        "--build-mode",
        choices=("triangle_prisms", "component_shell"),
        default="triangle_prisms",
    )
    parser.add_argument("--batch-triangles", type=int, default=32)
    args = parser.parse_args()
    report = run_trial(
        args.source_vtp,
        args.output_dir,
        primitive_node_index=args.primitive_node_index,
        component_id=args.component_id,
        thickness_m=args.thickness_m,
        bottom_bevel_m=args.bottom_bevel_m,
        offset_mode=args.offset_mode,
        build_mode=args.build_mode,
        batch_triangles=args.batch_triangles,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
