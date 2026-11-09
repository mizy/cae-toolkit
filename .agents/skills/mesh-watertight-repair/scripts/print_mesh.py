"""Print-purpose worker: explicit material intent, millimetre checks, and audited exports."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pyvista as pv
import trimesh
from html_report import write_html_report
from mesh_io import triangle_faces, write_vtp
from mesh_metrics import mesh_report
from opening_repair import apply_opening_decisions
from print_checks import fidelity_check, layer_check, wall_check
from print_export import export_print_files, run_slicer
from print_geometry import construct_wall, material_valid, orient_material, union_parts
from source_projected_closure import interpolate_source_corner_normals
from repair_inventory import build_inventory
from repair_policy import (
    build_policy_packet,
    load_policy_decisions,
    resolve_policy_decisions,
)
from run_status import update_run_status
from source_preserving_repair import build_decision, gate, run_deterministic_repair
from source_projected_validation import exact_self_intersection_report


def add_print_arguments(parser) -> None:
    group = parser.add_argument_group("3D printing (all print lengths are millimetres)")
    group.add_argument("--print-units", choices=("mm", "cm", "m", "inch"))
    group.add_argument(
        "--print-size-mm",
        type=float,
        help="Final longest dimension; alternatively supply input units",
    )
    group.add_argument(
        "--print-mode",
        choices=("preserve", "solid", "hollow", "thicken"),
        default="preserve",
    )
    group.add_argument(
        "--print-parts", choices=("preserve", "merge"), default="preserve"
    )
    group.add_argument(
        "--print-openings", choices=("review", "cap", "preserve"), default="review"
    )
    group.add_argument(
        "--print-wall-mm",
        type=float,
        default=0.8,
        help="Explicit wall for hollow/thicken modes",
    )
    group.add_argument(
        "--print-min-wall-mm",
        type=float,
        default=0.8,
        help="Sampled wall threshold; configure for printer/material",
    )
    group.add_argument("--print-max-error-mm", type=float, default=0.1)
    group.add_argument("--print-layer-height-mm", type=float, default=0.2)
    group.add_argument("--print-max-voxels", type=int, default=8_000_000)
    group.add_argument("--print-process", choices=("fff", "sla"), default="fff")
    group.add_argument("--print-up-axis", choices=("x", "y", "z"), default="z")
    group.add_argument(
        "--print-slicer",
        type=Path,
        help="Optional PrusaSlicer-compatible CLI executable",
    )
    group.add_argument("--print-slicer-profile", type=Path)


def validate_print_arguments(args) -> None:
    for key in (
        "print_wall_mm",
        "print_min_wall_mm",
        "print_max_error_mm",
        "print_layer_height_mm",
    ):
        if not np.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if args.print_size_mm is not None and (
        not np.isfinite(args.print_size_mm) or args.print_size_mm <= 0
    ):
        raise ValueError("print_size_mm must be finite and positive")
    if args.print_max_voxels <= 0:
        raise ValueError("print_max_voxels must be positive")
    if (
        not args.print_units
        and not args.print_size_mm
        and args.input.suffix.lower() not in (".glb", ".gltf")
    ):
        raise ValueError("printing requires explicit --print-units or --print-size-mm")
    if bool(args.print_slicer) != bool(args.print_slicer_profile):
        raise ValueError("print slicer and printer profile must be supplied together")
    if args.print_slicer and args.print_process != "fff":
        raise ValueError(
            "the external G-code interface supports FFF; SLA preparation exports geometry for a resin slicer"
        )


def print_source(args) -> tuple[trimesh.Trimesh, float, list[dict]]:
    if args.input.suffix.lower() in (".glb", ".gltf"):
        from gltf_surface import read_gltf_surface

        surface = read_gltf_surface(args.input)
    else:
        surface = pv.read(args.input)
    if isinstance(surface, pv.MultiBlock):
        surface = surface.combine()
    if not isinstance(surface, pv.PolyData):
        surface = surface.extract_surface()
    surface = surface.triangulate()
    native_corner_normals = (
        np.asarray(surface.cell_data["corner_normals"], dtype=np.float64).copy()
        if "corner_normals" in surface.cell_data
        else None
    )
    native_point_normals = (
        np.asarray(surface.point_data["Normals"], dtype=np.float64).copy()
        if "Normals" in surface.point_data
        else None
    )
    if native_corner_normals is not None or native_point_normals is not None:
        native_points = np.asarray(surface.points, dtype=np.float64).copy()
        native_faces = triangle_faces(surface).copy()
        if native_corner_normals is not None and native_corner_normals.shape != (
            len(native_faces),
            9,
        ):
            raise ValueError("corner_normals must contain three normals per source triangle")
    surface.cell_data["print_source_face_id"] = np.arange(surface.n_cells, dtype=np.int64)
    surface = surface.clean(
        lines_to_points=False, polys_to_lines=False, strips_to_polys=False
    )
    points = np.asarray(surface.points, dtype=np.float64)
    clean_face_ids = np.asarray(surface.cell_data["print_source_face_id"], dtype=np.int64)
    unit = args.print_units or (
        "m" if args.input.suffix.lower() in (".glb", ".gltf") else "mm"
    )
    factor = {"mm": 1.0, "cm": 10.0, "m": 1000.0, "inch": 25.4}[unit]
    if args.print_size_mm:
        factor = args.print_size_mm / float(np.ptp(points, axis=0).max())
    axis = "xyz".index(args.print_up_axis)

    def to_print_axes(values: np.ndarray) -> np.ndarray:
        result = np.asarray(values, dtype=np.float64).copy()
        if axis != 2:
            result[:, [axis, 2]] = result[:, [2, axis]]
            result[:, axis] *= -1
        return result

    points = to_print_axes(points * factor)
    if native_corner_normals is not None:
        normal_points = native_points[native_faces].reshape(-1, 3)
        normal_faces = np.arange(len(normal_points), dtype=np.int64).reshape(-1, 3)
        normal_values = native_corner_normals.reshape(-1, 3)
    elif native_point_normals is not None:
        normal_points = native_points
        normal_faces = native_faces
        normal_values = native_point_normals
    else:
        normal_points = normal_faces = normal_values = None
    if normal_points is not None:
        normal_points = to_print_axes(normal_points * factor)
        normal_values = to_print_axes(normal_values)
    points, faces, source_face_ids, passes = run_deterministic_repair(
        points, triangle_faces(surface), clean_face_ids
    )
    mesh = trimesh.Trimesh(points, faces, process=False)
    mesh, _ = orient_material(mesh)
    if normal_values is not None:
        mesh.face_attributes["corner_normals"] = interpolate_source_corner_normals(
            points,
            mesh.faces,
            normal_points,
            normal_faces,
            normal_values,
            source_face_ids,
        ).reshape(-1, 3, 3)
    return mesh, factor, passes


def _source_face_matches(
    points: np.ndarray,
    faces: np.ndarray,
    source: trimesh.Trimesh,
    source_face_ids: np.ndarray,
) -> np.ndarray:
    """Keep normal provenance only when all candidate corners lie on the source face."""
    result = np.asarray(source_face_ids, dtype=np.int64).copy()
    valid = result >= 0
    if not np.any(valid):
        return result
    candidate = points[faces[valid]]
    source_triangles = source.vertices[source.faces[result[valid]]]
    edge_a = source_triangles[:, 1] - source_triangles[:, 0]
    edge_b = source_triangles[:, 2] - source_triangles[:, 0]
    normal = np.cross(edge_a, edge_b)
    normal_length = np.linalg.norm(normal, axis=1)
    offset = candidate - source_triangles[:, :1]
    plane_distance = np.abs(np.einsum("fci,fi->fc", offset, normal)) / np.maximum(
        normal_length[:, None], 1e-30
    )
    d00 = np.einsum("fi,fi->f", edge_a, edge_a)
    d01 = np.einsum("fi,fi->f", edge_a, edge_b)
    d11 = np.einsum("fi,fi->f", edge_b, edge_b)
    d20 = np.einsum("fci,fi->fc", offset, edge_a)
    d21 = np.einsum("fci,fi->fc", offset, edge_b)
    denominator = d00 * d11 - d01 * d01
    v = (d11[:, None] * d20 - d01[:, None] * d21) / np.maximum(denominator[:, None], 1e-30)
    w = (d00[:, None] * d21 - d01[:, None] * d20) / np.maximum(denominator[:, None], 1e-30)
    barycentric = np.stack((1.0 - v - w, v, w), axis=2)
    scale = float(np.ptp(source.vertices, axis=0).max())
    position_tolerance = max(scale * 1e-9, 1e-12)
    edge_scale = np.maximum(
        np.minimum(np.linalg.norm(edge_a, axis=1), np.linalg.norm(edge_b, axis=1)),
        position_tolerance,
    )
    barycentric_tolerance = position_tolerance / edge_scale
    matches = (
        (np.max(plane_distance, axis=1) <= position_tolerance)
        & (np.min(barycentric, axis=(1, 2)) >= -barycentric_tolerance)
        & (np.max(barycentric, axis=(1, 2)) <= 1.0 + barycentric_tolerance)
    )
    result[np.flatnonzero(valid)[~matches]] = -1
    return result


def _set_print_corner_normals(
    candidate: trimesh.Trimesh,
    source: trimesh.Trimesh,
    source_face_ids: np.ndarray,
) -> None:
    if "corner_normals" not in source.face_attributes:
        return
    source_corner_normals = np.asarray(source.face_attributes["corner_normals"], dtype=np.float64)
    source_points = source.vertices[source.faces].reshape(-1, 3)
    source_faces = np.arange(len(source_points), dtype=np.int64).reshape(-1, 3)
    candidate.face_attributes["corner_normals"] = interpolate_source_corner_normals(
        candidate.vertices,
        candidate.faces,
        source_points,
        source_faces,
        source_corner_normals.reshape(-1, 3),
        source_face_ids,
    ).reshape(-1, 3, 3)


def print_candidate(source, args, inventory, decisions):
    default = args.print_openings
    if args.print_mode == "thicken" and default == "review":
        default = "preserve"
    faces, generated, transactions, ready = apply_opening_decisions(
        source.vertices,
        source.faces,
        inventory,
        decisions,
        default_action=default,
        preserve_as_wall=args.print_mode == "thicken",
    )
    candidate = trimesh.Trimesh(source.vertices.copy(), faces, process=False)
    normal_source_face_ids = None
    if "corner_normals" in source.face_attributes:
        normal_source_face_ids = np.full(len(candidate.faces), -1, dtype=np.int64)
        normal_source_face_ids[: len(source.faces)] = np.arange(len(source.faces), dtype=np.int64)
    candidate, _ = orient_material(candidate)
    before = candidate.copy()
    before_normal_source_face_ids = (
        normal_source_face_ids.copy() if normal_source_face_ids is not None else None
    )
    before_generated = generated.copy()
    operation = {"mode": args.print_mode, "status": "unchanged"}
    try:
        if ready and (
            args.print_mode == "solid"
            or (args.print_parts == "merge" and args.print_mode != "thicken")
        ):
            candidate = union_parts(
                candidate,
                fill_cavities=args.print_mode == "solid",
                merge=args.print_parts == "merge",
            )
            operation["status"] = "material_boolean_operations"
        if ready and args.print_mode in ("hollow", "thicken"):
            candidate, operation = construct_wall(
                candidate,
                mode=args.print_mode,
                thickness=args.print_wall_mm,
                error=args.print_max_error_mm,
                max_voxels=args.print_max_voxels,
            )
            if args.print_mode == "thicken" and args.print_parts == "merge":
                candidate = union_parts(candidate)
        if len(candidate.faces) != len(before.faces) or not np.array_equal(
            candidate.faces, before.faces
        ):
            nearest = pv.wrap(before).find_closest_cell(candidate.triangles_center)
            nearest = np.asarray(nearest, dtype=int)
            generated = generated[nearest]
            if before_normal_source_face_ids is not None:
                normal_source_face_ids = before_normal_source_face_ids[nearest]
                normal_source_face_ids[generated > 0] = -1
                normal_source_face_ids = _source_face_matches(
                    candidate.vertices, candidate.faces, source, normal_source_face_ids
                )
    except (ValueError, RuntimeError, MemoryError) as exc:
        candidate, generated = before, before_generated
        normal_source_face_ids = before_normal_source_face_ids
        operation = {"mode": args.print_mode, "status": "failed", "reason": str(exc)}
    return candidate, generated, transactions, ready, operation, normal_source_face_ids


def certify_print_candidate(candidate, source, generated, args, ready, normal_source_face_ids):
    candidate, shells = orient_material(candidate)
    _set_print_corner_normals(candidate, source, normal_source_face_ids)
    metrics = mesh_report(candidate.vertices, candidate.faces)
    closed = material_valid(metrics, shells)
    exact = exact_self_intersection_report(candidate.vertices, candidate.faces)
    distances, fidelity = fidelity_check(
        source,
        candidate,
        generated,
        error=args.print_max_error_mm,
        mode=args.print_mode,
        thickness=args.print_wall_mm,
        merged=args.print_parts == "merge" or args.print_mode == "solid",
        shells=shells,
    )
    thickness, wall = (
        wall_check(candidate, args.print_min_wall_mm)
        if closed and exact.get("passed")
        else (
            np.full(len(candidate.faces), -1.0),
            {"passed": False, "reason": "closed nonintersecting material required"},
        )
    )
    before_euler = int(source.euler_number)
    before_components = len(
        trimesh.graph.connected_components(
            source.face_adjacency, nodes=np.arange(len(source.faces))
        )
    )
    topology_preserved = (
        args.print_mode not in ("preserve", "thicken")
        or args.print_parts == "merge"
        or (
            (not source.is_watertight or int(candidate.euler_number) == before_euler)
            and len(shells) == before_components
        )
    )
    gates = {
        "opening_decisions_applied": gate(ready, ready, True),
        "closed_material": gate(
            closed,
            metrics["topology"],
            "closed manifold with nesting-aware material orientation",
        ),
        "exact_self_intersection": gate(
            exact.get("passed") is True, exact.get("intersection_pairs"), 0
        ),
        "source_fidelity": gate(fidelity["passed"], fidelity, args.print_max_error_mm),
        "sampled_minimum_wall": gate(wall["passed"], wall, args.print_min_wall_mm),
        "protected_topology": gate(
            topology_preserved,
            {"euler": int(candidate.euler_number), "shells": len(shells)},
            {"euler": before_euler, "shells": before_components},
        ),
        "sla_closed_cavities": gate(
            args.print_process != "sla" or not any(row["depth"] % 2 for row in shells),
            sum(row["depth"] % 2 for row in shells),
            "SLA requires no sealed cavity; add verified drainage or use solid mode",
        ),
    }
    return (
        candidate,
        shells,
        metrics,
        exact,
        distances,
        thickness,
        fidelity,
        wall,
        gates,
    )


# @entry: called directly by the existing purpose switch before exterior-only filtering.
def run_print_pipeline(args, status) -> int:
    update_run_status(args, status, "print_read_input")
    source, factor, passes = print_source(args)
    source_path = args.output_dir / "print_source_mm.vtp"
    source_cell_data = {}
    if "corner_normals" in source.face_attributes:
        source_cell_data["corner_normals"] = np.asarray(
            source.face_attributes["corner_normals"], dtype=np.float32
        ).reshape(-1, 9)
    write_vtp(source_path, source.vertices, source.faces, source_cell_data)
    inventory = build_inventory(
        source.vertices,
        source.faces,
        np.arange(len(source.faces)),
        "print_input_mm",
        max_items=args.policy_item_limit,
    )
    for region in inventory["boundary_regions"]["items"]:
        region["requires_policy"] = True
    parameters = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if key.startswith("print_")
    }
    packet = build_policy_packet(
        inventory,
        "print-material",
        max_items=args.policy_item_limit,
        run_context={
            "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
            "scale_to_mm": factor,
            **{
                key: parameters[key]
                for key in (
                    "print_mode",
                    "print_parts",
                    "print_openings",
                    "print_wall_mm",
                    "print_max_error_mm",
                    "print_up_axis",
                )
            },
        },
    )
    supplied = load_policy_decisions(args.policy_decisions)
    # A declared whole-model cap/thicken operation supplies defaults; stale supplied decisions never inherit them.
    decisions = resolve_policy_decisions(packet, supplied)
    if args.print_openings != "review" or args.print_mode == "thicken":
        supplied_ids = {row.get("item_id") or row.get("id") for row in supplied}
        decisions = [row for row in decisions if row["item_id"] in supplied_ids]
    candidate, generated, transactions, ready, operation, normal_source_face_ids = (
        print_candidate(source, args, inventory, decisions)
    )
    update_run_status(args, status, "print_validate_material")
    candidate, shells, metrics, exact, distances, thickness, fidelity, wall, gates = (
        certify_print_candidate(
            candidate, source, generated, args, ready, normal_source_face_ids
        )
    )
    gates["geometry_operation_completed"] = gate(
        operation.get("status") != "failed",
        operation,
        "requested material operation completed",
    )
    candidate_path = args.output_dir / "print_candidate.vtp"
    candidate_cell_data = {
        "print_generated_face": generated,
        "print_source_distance_mm": distances,
        "print_thickness_mm": thickness,
    }
    if "corner_normals" in candidate.face_attributes:
        candidate_cell_data["corner_normals"] = np.asarray(
            candidate.face_attributes["corner_normals"], dtype=np.float32
        ).reshape(-1, 9)
    write_vtp(candidate_path, candidate.vertices, candidate.faces, candidate_cell_data)
    # Identical serialized arrays transfer the in-memory topology/CGAL certificate.
    saved = pv.read(candidate_path)
    identical = np.array_equal(saved.points, candidate.vertices) and np.array_equal(
        triangle_faces(saved), candidate.faces
    )
    gates["vtp_readback"] = gate(
        identical, identical, "identical certified points and connectivity"
    )
    layers = (
        layer_check(candidate, args.print_layer_height_mm)
        if gates["closed_material"]["passed"] and exact.get("passed")
        else {"passed": False, "reason": "closed nonintersecting material required"}
    )
    (args.output_dir / "print_layers.json").write_text(
        json.dumps(layers), encoding="utf-8"
    )
    gates["layer_contours"] = gate(
        layers["passed"],
        {key: value for key, value in layers.items() if key != "layers"},
        "closed finite contours",
    )
    exports = (
        export_print_files(candidate, args.output_dir, args.print_max_error_mm)
        if gates["closed_material"]["passed"] and exact.get("passed")
        else []
    )
    gates["export_readback"] = gate(
        bool(exports) and all(row["passed"] for row in exports),
        exports,
        "STL and 3MF preserve certified material geometry",
    )
    decision = build_decision(gates, str(candidate_path))
    slicer = (
        run_slicer(
            args.print_slicer,
            args.print_slicer_profile,
            args.output_dir / "print_candidate.3mf",
            args.output_dir,
        )
        if decision["status"] == "accepted"
        else {
            "status": "geometry_rejected",
            "passed": False,
            "machine_toolpath_validated": False,
        }
    )
    if args.print_slicer:
        gates["external_slicer"] = gate(
            slicer["passed"],
            slicer,
            "profile-based CLI produces nonempty extrusion toolpath",
        )
        decision = build_decision(gates, str(candidate_path))
    accepted = decision["status"] == "accepted"
    report = {
        "decision": decision,
        "input": {
            "path": str(args.input),
            "kind": "mesh",
            "units": args.print_units
            or (
                "m (glTF convention)"
                if args.input.suffix.lower() in (".glb", ".gltf")
                else "unknown; explicit final size"
            ),
            "scale_to_mm": factor,
        },
        "target": {
            "purpose": "print",
            "coordinate_units": "mm",
            "material_mode": args.print_mode,
            "parts": args.print_parts,
            "up_axis": args.print_up_axis,
        },
        "parameters": parameters,
        "deterministic_passes": passes,
        "unresolved_policy_packet": packet,
        "policy_decisions": decisions,
        "opening_transactions": transactions,
        "printing": {
            "operation": operation,
            "shells": shells,
            "wall_check": wall,
            "fidelity": fidelity,
            "layer_check": {
                key: value for key, value in layers.items() if key != "layers"
            },
            "slicer": slicer,
            "geometry_accepted": accepted,
            "printer_ready": accepted and slicer["passed"],
        },
        "stages": {
            "print_source": {"path": str(source_path)},
            "print_candidate": {
                "path": str(candidate_path),
                "metrics": metrics,
                "accepted_final_geometry": accepted,
            },
        },
        "gates": gates,
        "self_intersection": exact,
        "outputs": {
            "report_json": str(args.output_dir / "print_report.json"),
            "html_report": str(args.output_dir / "print_report.html"),
            "accepted_mesh_vtp": str(candidate_path) if accepted else None,
            "accepted_mesh_available": accepted,
            "print_candidate_vtp": str(candidate_path),
            "print_files": exports,
            "accepted_print_files": [row["path"] for row in exports]
            if accepted
            else [],
            "print_layers_json": str(args.output_dir / "print_layers.json"),
            "mesh_result": {
                "accepted": accepted,
                "engineering_ready": accepted and slicer["passed"],
                "role": "print_geometry" if accepted else "diagnostic_print_candidate",
            },
            "mesh_previews": [
                {
                    "label": "Print input at target scale / 打印输入（按目标尺寸换算）",
                    "path": str(source_path),
                },
                {
                    "label": "Accepted print geometry / 打印几何通过"
                    if accepted
                    else "Rejected print candidate / 打印候选未通过",
                    "path": str(candidate_path),
                },
            ],
        },
        "limitations": [
            "Wall thickness is a sampled normal-chord estimate, not a certified minimum.",
            "Distances use seven samples per triangle, not a certified Hausdorff bound.",
            "Layer contour checks do not replace machine-specific slicing or a physical print.",
            "Hollowing creates sealed cavities; SLA needs separately verified drain paths or solid mode.",
        ],
    }
    for name, data in (
        ("print_report.json", report),
        ("ai_policy_packet.json", packet),
    ):
        (args.output_dir / name).write_text(
            json.dumps(data, indent=2), encoding="utf-8"
        )
    write_html_report(
        report,
        args.output_dir / "print_report.html",
        "3D Print Material Repair / 打印修复报告",
    )
    update_run_status(
        args,
        status,
        "completed",
        outputs=report["outputs"],
        metrics={"decision": decision},
        accepted_mesh_vtp=report["outputs"]["accepted_mesh_vtp"],
    )
    print(json.dumps(decision))
    return 0
