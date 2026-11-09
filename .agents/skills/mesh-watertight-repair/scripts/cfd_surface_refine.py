"""Optional CFD surface resolution stage, after watertight-shell acceptance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyvista as pv
from adaptive_depth_refine import adaptive_edge_refine
from html_report import write_html_report
from mesh_io import read_surface, triangle_faces, write_vtp
from mesh_metrics import mesh_report
from source_preserving_repair import build_decision, gate
from source_projected_closure import ProjectionThresholds, interpolate_source_normals, project_closure_to_source
from source_projected_validation import (
    exact_self_intersection_report,
    repair_projected_self_intersections,
    roundtrip_validation,
    watertight_metrics,
)


def source_target_sizes(
    source: pv.PolyData, shell_points: np.ndarray, shell_faces: np.ndarray, max_edge_length: float
) -> tuple[np.ndarray, np.ndarray]:
    """Map original-source crease/boundary evidence to shell faces in compiled batch queries."""
    faces = triangle_faces(source)
    triangles = np.asarray(source.points)[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    normals = np.divide(normals, lengths[:, None], out=np.zeros_like(normals), where=lengths[:, None] > 0)
    edges = np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1)
    owners = np.tile(np.arange(len(faces)), 3)
    _, inverse, counts = np.unique(edges, axis=0, return_inverse=True, return_counts=True)
    first = np.full(len(counts), len(faces), dtype=np.int64)
    last = np.zeros(len(counts), dtype=np.int64)
    np.minimum.at(first, inverse, owners)
    np.maximum.at(last, inverse, owners)
    crease = (counts != 2) | (np.einsum("ij,ij->i", normals[first], normals[last]) < np.cos(np.deg2rad(30)))
    feature_faces = np.zeros(len(faces), dtype=bool)
    np.logical_or.at(feature_faces, owners, crease[inverse])
    nearest = np.asarray(source.find_closest_cell(shell_points[shell_faces].mean(axis=1)), dtype=np.int64)
    return np.where(feature_faces[nearest], max_edge_length / 2, max_edge_length), nearest


def refinement_measurements(
    points: np.ndarray,
    faces: np.ndarray,
    parent_faces: np.ndarray,
    input_points: np.ndarray,
    before: dict,
    sizes: np.ndarray,
    projection_limit: float,
    scaffold: np.ndarray,
    exact: dict,
) -> tuple[dict, dict]:
    after = mesh_report(points, faces)
    lengths = np.linalg.norm(points[faces] - points[np.roll(faces, -1, axis=1)], axis=2)
    edge_ratio = float(np.max(lengths / sizes[parent_faces, None]))
    displacement = float(np.max(np.linalg.norm(points - scaffold, axis=1)))
    before_euler = before["points"] - before["topology"]["edges"] + before["triangles"]
    after_euler = after["points"] - after["topology"]["edges"] + after["triangles"]
    fixed = np.array_equal(points[: len(input_points)], input_points)
    gates = {
        "watertight": gate(watertight_metrics(after), after["topology"], "closed outward manifold"),
        "exact_self_intersection": gate(exact.get("passed") is True, exact.get("intersection_pairs"), 0),
        "components_preserved": gate(
            after["topology"]["components"]["count"] == before["topology"]["components"]["count"],
            after["topology"]["components"]["count"],
            before["topology"]["components"]["count"],
        ),
        "euler_preserved": gate(after_euler == before_euler, after_euler, before_euler),
        "input_vertices_fixed": gate(fixed, fixed, True),
        "source_projection_bounded": gate(
            displacement <= projection_limit * (1 + 1e-10), displacement, projection_limit
        ),
        "target_edge_length": gate(edge_ratio <= 1 + 1e-10, edge_ratio, 1.0),
        "aspect_ratio_max": gate(
            after["quality"]["aspect_ratio"]["max"] <= 40, after["quality"]["aspect_ratio"]["max"], 40
        ),
    }
    return after, gates


# @entry: source-based sizing, conforming subdivision, bounded new-vertex projection, and certification.
def refine_cfd_surface(
    mesh_path: Path, source_path: Path, output_dir: Path, *, max_edge_length: float, max_output_faces: int = 2_000_000
) -> dict:
    if not np.isfinite(max_edge_length) or max_edge_length <= 0 or max_output_faces <= 0:
        raise ValueError("CFD max edge length and face budget must be positive and finite")
    shell, source = read_surface(mesh_path), read_surface(source_path, clean=False)
    input_points, input_faces = np.asarray(shell.points), triangle_faces(shell)
    before = mesh_report(input_points, input_faces)
    if not watertight_metrics(before):
        raise ValueError("CFD refinement requires an already watertight, outward, nondegenerate surface")
    input_exact = exact_self_intersection_report(input_points, input_faces)
    if input_exact.get("passed") is not True:
        raise ValueError("CFD refinement input must pass exact self-intersection certification")
    output_dir.mkdir(parents=True, exist_ok=False)
    sizes, source_faces = source_target_sizes(source, input_points, input_faces, max_edge_length)
    scaffold, faces, inherited, iterations = adaptive_edge_refine(
        input_points,
        input_faces,
        sizes * 0.8,
        (sizes < max_edge_length).astype(float),
        np.arange(len(input_faces)),
        max_iterations=12,
        max_output_faces=max_output_faces,
    )
    scaffold_exact = exact_self_intersection_report(scaffold, faces)
    if scaffold_exact.get("passed") is not True:
        raise ValueError("Subdivided scaffold failed exact self-intersection certification")
    movable = np.arange(len(scaffold)) >= len(input_points)
    projection_limit = max_edge_length * 0.05
    points, faces, provenance, projection = project_closure_to_source(
        scaffold,
        faces,
        np.asarray(source.points),
        triangle_faces(source),
        thresholds=ProjectionThresholds(max_projection_distance=projection_limit, source_edge_barycentric_margin=0.0),
        movable_vertex_mask=movable,
    )
    points, provenance, exact = repair_projected_self_intersections(
        points,
        scaffold,
        faces,
        provenance,
        certified_proxy_report=scaffold_exact,
    )
    normals = interpolate_source_normals(
        points, faces, np.asarray(source.points), triangle_faces(source), provenance,
        source.point_data.get("Normals"),
    )
    if "Normals" in shell.point_data:
        normals[: len(input_points)] = np.asarray(shell.point_data["Normals"])
    parents = inherited["source_triangle_index"]
    cells = {name: np.asarray(values)[parents] for name, values in shell.cell_data.items()}
    cells.update(
        watertight_parent_face_id=parents,
        sizing_source_face_id=source_faces[parents],
        target_edge_length=sizes[parents],
    )
    candidate = output_dir / "cfd_surface_candidate.vtp"
    write_vtp(
        candidate,
        points,
        faces,
        cells,
        point_data={"Normals": normals, **{f"cfd_{name}": values for name, values in provenance["point_data"].items()}},
    )
    after, gates = refinement_measurements(
        points, faces, parents, input_points, before, sizes, projection_limit, scaffold, exact
    )
    readback = roundtrip_validation(candidate, points, faces)
    saved = pv.read(candidate)
    saved_exact = exact_self_intersection_report(np.asarray(saved.points), triangle_faces(saved))
    gates["serialized_readback"] = gate(
        readback.get("passed") is True and saved_exact.get("passed") is True,
        readback,
        "identical watertight readback and exact zero intersections",
    )
    gates["face_budget"] = gate(len(faces) <= max_output_faces, len(faces), max_output_faces)
    decision = build_decision(gates, str(candidate))
    report = {
        "decision": decision,
        "input": {"path": str(mesh_path), "source_geometry": str(source_path), "units": "same as input coordinates"},
        "method": "source_feature_size_field_conforming_edge_split_bounded_new_vertex_projection",
        "parameters": {
            "purpose": "cfd",
            "max_edge_length": max_edge_length,
            "feature_edge_length": max_edge_length / 2,
            "feature_angle_degrees": 30,
            "max_output_faces": max_output_faces,
        },
        "stages": {
            "watertight_input": {"path": str(mesh_path), "metrics": before},
            "cfd_refined_surface": {
                "path": str(candidate),
                "metrics": after,
                "accepted_final_geometry": decision["status"] == "accepted",
            },
        },
        "gates": gates,
        "iterations": iterations,
        "source_projection": projection,
        "self_intersection": exact,
        "post_write_validation": readback,
        "outputs": {
            "watertight_mesh_vtp": str(mesh_path),
            "cfd_surface_candidate_vtp": str(candidate),
            "accepted_mesh_vtp": decision["final_output_path"],
        },
        "limitations": [
            "Surface resolution and geometric quality only; boundary tags, passage semantics, volume mesh and boundary layers require separate acceptance.",
            "Subdivision does not repair inherited sliver triangles or recover features lost during closure.",
            "All original shell vertices are fixed; only newly inserted vertices can project to original source geometry.",
        ],
    }
    (output_dir / "cfd_refinement_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_html_report(report, output_dir / "cfd_refinement_report.html", "CFD Surface Refinement")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mesh", type=Path)
    parser.add_argument("--source-mesh", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--max-edge-length", type=float, required=True, help="Maximum edge length in input coordinate units."
    )
    parser.add_argument("--max-output-faces", type=int, default=2_000_000)
    args = parser.parse_args()
    report = refine_cfd_surface(
        args.mesh,
        args.source_mesh,
        args.output_dir,
        max_edge_length=args.max_edge_length,
        max_output_faces=args.max_output_faces,
    )
    print(json.dumps(report["decision"], indent=2))


if __name__ == "__main__":
    main()
