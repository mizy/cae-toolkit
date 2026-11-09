"""Geometry-only source-preserving material repair, with explicit boundary intent.

Lengths use input coordinates. This worker does not infer CFD fluid domains or
printing wall requirements. AI reviews semantic boundary choices upstream.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import igl
import numpy as np
import pyvista as pv

from mesh_io import read_surface, same_oriented_geometry, triangle_faces
from material_cleanup import _surface, _cleanup, _compact, _geometry_gate, _candidate_distance
from opening_repair import apply_opening_decisions, build_preserved_opening_rays, check_preserved_opening_rays
from repair_inventory import build_inventory
from repair_policy import build_policy_packet, load_policy_decisions, resolve_policy_decisions
from source_distance_bound import candidate_to_source_bound
from source_fidelity_recovery import recover_source_fidelity, select_missing_source_facets
from source_material_closure import build_source_material
from source_preserving_repair import run_deterministic_repair
from mesh_topology_repair import collapse_independent_short_edges
from source_projected_closure import interpolate_source_corner_normals


def _normal_corners(source, faces):
    if "corner_normals" in source.cell_data:
        values = np.asarray(source.cell_data["corner_normals"], dtype=float)
        if values.shape != (len(faces), 9) or not np.isfinite(values).all():
            raise ValueError("corner_normals must contain three finite normals per triangle")
        return values
    if "Normals" in source.point_data:
        values = np.asarray(source.point_data["Normals"], dtype=float)
        if values.shape != (source.n_points, 3) or not np.isfinite(values).all():
            raise ValueError("Normals must contain a finite vector per source point")
        return values[faces].reshape(-1, 9)
    return None


def _prepare(source, center, scale, boundary_mode, max_error, supplied):
    raw_faces = triangle_faces(source)
    raw_points = (np.asarray(source.points, dtype=float) - center) / scale
    points, inverse = np.unique(raw_points, axis=0, return_inverse=True)
    faces = inverse[raw_faces]
    corners = _normal_corners(source, raw_faces)
    reference = _surface(points, faces)
    if corners is not None:
        reference.cell_data["corner_normals"] = corners
    points, kept_faces, rows, passes = run_deterministic_repair(
        points, faces, np.arange(len(faces), dtype=np.int64)
    )
    if not len(kept_faces):
        raise ValueError("input contains no nondegenerate source triangles")
    # Deterministic cleanup compacts points; recover row geometry without changing the reference frame.
    inventory = build_inventory(points * scale + center, kept_faces, rows, "input_source_material", max_items=len(kept_faces))
    packet = build_policy_packet(
        inventory, "watertight-material", max_items=len(kept_faces),
        run_context={"source_geometry_sha256": hashlib.sha256(
            np.asarray(source.points).tobytes() + raw_faces.tobytes()
        ).hexdigest(), "max_error": max_error},
    )
    decisions = resolve_policy_decisions(packet, supplied or [])
    if boundary_mode != "review":
        supplied_ids = {item.get("item_id") or item.get("id") for item in supplied or []}
        decisions = [item for item in decisions if item["item_id"] in supplied_ids]
    rays = build_preserved_opening_rays(points, kept_faces, inventory["boundary_regions"]["items"], decisions)
    rays["coordinate_frame"] = "normalized"
    action = {"wall": "preserve", "cap": "cap", "review": "review"}[boundary_mode]
    prepared_faces, _, transactions, ready = apply_opening_decisions(
        points, kept_faces, inventory, decisions, default_action=action,
        preserve_as_wall=True,
    )
    caps = prepared_faces[len(kept_faces):]
    if len(caps):
        reference_points = np.vstack((reference.points, points))
        reference_faces = np.vstack((faces, caps + reference.n_points))
        expanded = _surface(reference_points, reference_faces)
        if corners is not None:
            triangles = points[caps]
            normals = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
            normals /= np.linalg.norm(normals, axis=1)[:, None]
            expanded.cell_data["corner_normals"] = np.vstack((corners, np.tile(normals, (1, 3))))
        reference = expanded
        rows = np.r_[rows, np.arange(len(faces), len(faces)+len(caps))]
    return reference, points, prepared_faces, rows, {
        "ready": ready and rays["status"] == "ready", "passes": passes, "opening_transactions": transactions,
        "policy_packet": packet, "policy_decisions": decisions, "coordinate_frame": "input",
        "preserved_opening_rays": rays,
        "source_faces": len(raw_faces), "declared_caps": len(caps),
    }


def _attach_normals(candidate, source):
    if "corner_normals" not in source.cell_data:
        return
    source_points = np.asarray(source.points)[triangle_faces(source)].reshape(-1, 3)
    source_faces = np.arange(len(source_points)).reshape(-1, 3)
    normals = np.asarray(source.cell_data["corner_normals"]).reshape(-1, 3)
    rows = np.asarray(candidate.cell_data["source_facet_id"])
    # Side walls have geometric normals; original and parallel inner faces retain source normal islands.
    mapped = rows.copy()
    valid = rows >= 0
    source_tri = source_points[source_faces[rows[valid]]]
    triangles = np.asarray(candidate.points)[triangle_faces(candidate)[valid]]
    a = np.cross(source_tri[:, 1]-source_tri[:, 0], source_tri[:, 2]-source_tri[:, 0])
    b = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    cosine = np.abs(np.einsum("ij,ij->i", a, b)) / np.maximum(np.linalg.norm(a, axis=1)*np.linalg.norm(b, axis=1), np.finfo(float).tiny)
    mapped[np.flatnonzero(valid)[cosine < .99]] = -1
    candidate.cell_data["corner_normals"] = interpolate_source_corner_normals(
        np.asarray(candidate.points), triangle_faces(candidate), source_points, source_faces, normals, mapped
    )


# @entry: generic material candidate workflow; no names, model IDs, or fixed facet lists.
def repair_source_mesh(source, *, max_error, boundary_mode="review", grid_size=128, policy_decisions=None, max_recovery_rounds=2, progress=None, certify_reverse_fidelity=False, max_recovery_facets=1000, union_backend="cgal"):
    if not np.isfinite(max_error) or max_error <= 0:
        raise ValueError("max_error must be finite and positive in input coordinates")
    if boundary_mode not in ("wall", "cap", "review"):
        raise ValueError("boundary_mode must be wall, cap or review")
    if union_backend not in ("cgal", "manifold", "local", "planar"):
        raise ValueError("union_backend must be cgal, manifold, local or planar")
    if not isinstance(max_recovery_rounds, int) or max_recovery_rounds < 0:
        raise ValueError("max_recovery_rounds must be a nonnegative integer")
    if not isinstance(max_recovery_facets, int) or isinstance(max_recovery_facets, bool) or max_recovery_facets < 1:
        raise ValueError("max_recovery_facets must be a positive integer")
    if not isinstance(source, pv.PolyData):
        source = source.extract_surface(algorithm="dataset_surface")
    source = source.triangulate()
    points = np.asarray(source.points, dtype=float)
    faces = triangle_faces(source)
    if not len(faces) or not np.isfinite(points).all():
        raise ValueError("input must contain finite triangle geometry")
    triangles = points[faces]
    if not np.any(np.linalg.norm(np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0]), axis=1) > 0):
        raise ValueError("input contains no nondegenerate source triangles")
    center = (points.min(axis=0) + points.max(axis=0)) / 2
    scale = float(np.linalg.norm(np.ptp(points, axis=0)))
    coordinate_resolution = abs(float(np.spacing(np.max(np.abs(points)))))
    if scale <= 0 or max_error / scale <= 128*np.finfo(float).eps or max_error <= coordinate_resolution * 128:
        raise ValueError("error budget is below the coordinate precision")
    error = float(max_error / scale)
    if progress:
        progress("source_material_prepare")
    reference, p, f, source_rows, preparation = _prepare(source, center, scale, boundary_mode, max_error, policy_decisions)
    report = {"accepted": False, "target": {"name": "watertight-material", "purpose": "watertight"}, "engineering_ready": False,
              "max_error": float(max_error), "normalization": {"center": center.tolist(), "scale": scale},
              "boundary_mode": boundary_mode, "union_backend": union_backend, "preparation": preparation}
    report["limitations"] = [
        "Opening decisions express inferred or supplied intent; geometry acceptance does not certify their functional correctness.",
        "Explicit wall/cap modes are global geometry policies, not per-region AI review.",
        "Functional-opening ray checks sample straight passages only; they do not certify volumetric connectivity.",
    ]
    report["unresolved_policy_packet"] = preparation.pop("policy_packet")
    report["policy_decisions"] = preparation.pop("policy_decisions")
    if not preparation["ready"]:
        report["failure_reason"] = "boundary semantic review or a supported cap is required"
        report["candidate_role"] = "unmodified_reference_after_failure"
        return source.copy(), report
    try:
        depth = error / 2
        ray_origins = np.asarray(preparation["preserved_opening_rays"]["origins"]).reshape(-1, 3)
        if len(ray_origins):
            squared, _, _ = igl.point_mesh_squared_distance(ray_origins, np.asarray(reference.points), triangle_faces(reference))
            depth = min(depth, float(np.sqrt(np.maximum(squared, 0)).min()) / 8)
            if depth <= np.finfo(float).eps * 128:
                raise ValueError("cap and functional-opening decisions leave no resolvable material-wall clearance")
        if progress:
            progress("source_material_construct")
        if union_backend == "planar":
            from planar_sheet_material import build_planar_sheet_material
            p, f, row_ids, construction = build_planar_sheet_material(p, f, wall_depth=depth)
            construction["coordinate_frame"] = "normalized"
            construction["projection_max_displacement_input_units"] = construction["projection_max_displacement"] * scale
            construction["wall_depth_input_units"] = depth * scale
        elif union_backend == "local":
            from source_material_local import build_local_material
            p, f, row_ids, construction = build_local_material(
                p, f, wall_depth=depth, max_error=error, progress=progress)
        else:
            p, f, row_ids, construction = build_source_material(p, f, max_error=depth, progress=progress, union_backend=union_backend)
        report["construction"] = construction
        if union_backend == "local" and not construction["components_passed"]:
            raise ValueError(construction["failure_reason"])
        mapped_rows = np.full(len(row_ids), -1, dtype=np.int64)
        mapped = row_ids >= 0
        mapped_rows[mapped] = source_rows[row_ids[mapped]]
        candidate = _surface(p, f, mapped_rows)
        if union_backend in ("manifold", "local", "planar"):
            if progress:
                progress("source_material_local_short_edges")
            from scipy.spatial import cKDTree
            reference_points = np.asarray(reference.points)
            _, nearest = cKDTree(reference_points).query(p)
            pinned = np.all(p == reference_points[nearest], axis=1)
            p, f, kept, actions, local = collapse_independent_short_edges(
                p, f, pinned_points=pinned, max_edge_length=error / 8
            )
            p, f = _compact(p, f)
            candidate = _surface(p, f, mapped_rows[kept])
            report["cleanup"] = actions
            report["local_short_edge_repair"] = local
            report["fast_backend_policy"] = "bounded post-assembly short-edge repair; no global cleanup or patch recovery"
        else:
            if progress:
                progress("source_material_cleanup")
            candidate, cleanup = _cleanup(candidate, np.asarray(reference.points), error, progress)
            report["cleanup"] = cleanup
        if progress:
            progress("source_material_pre_recovery_geometry")
        report["pre_recovery_geometry"] = _geometry_gate(candidate)
        if not report["pre_recovery_geometry"]["passed"]:
            raise ValueError("initial material geometry failed before fidelity recovery")
        current_geometry = report["pre_recovery_geometry"]
        identical = same_oriented_geometry(reference, candidate)
        drift = 0.0
        if identical:
            restored = ((points - center) / scale) * scale + center
            drift = float(np.linalg.norm(restored - points, axis=1).max(initial=0.0))
            if not np.isfinite(drift):
                raise ValueError("normalization round-trip error could not be bounded")
            if drift:
                drift = float(np.nextafter(drift * (1 + 8 * np.finfo(float).eps), np.inf))
        missing = np.empty(0, dtype=np.int64)
        if identical:
            if progress:
                progress("source_material_exact_identity")
            report["source_fidelity_final"] = {
                "status": "exact_geometry_match", "method": "oriented_triangle_identity",
                "coordinate_frame": "normalized", "sample_count": 0,
                "source_facets_selected": 0, "normalization_drift_bound_input_units": drift,
            }
        else:
            if progress:
                progress("source_material_fidelity_recovery")
            for _ in range(0 if union_backend in ("manifold", "local", "planar") else max_recovery_rounds):
                recovered, fidelity = recover_source_fidelity(reference, candidate, max_error=error, grid_size=grid_size, progress=progress, max_recovery_facets=max_recovery_facets)
                report["source_fidelity"] = {**fidelity, "coordinate_frame": "normalized",
                    "sample_distance_original_units": {k: float(v) * scale for k, v in fidelity.get("sample_distance_input_units", {}).items()},
                    "raw_sample_distance_original_units": {k: float(v) * scale for k, v in fidelity.get("raw_sample_distance_input_units", {}).items()}}
                if fidelity["status"] == "within_budget":
                    break
                if fidelity["status"] != "union_proposal_created":
                    raise ValueError(fidelity.get("failure_reason", "source recovery failed"))
                candidate, extra = _cleanup(recovered, np.asarray(reference.points), error, progress)
                report["cleanup"].extend(extra)
                current_geometry = _geometry_gate(candidate)
                report["recovery_geometry"] = current_geometry
                if not current_geometry["passed"]:
                    raise ValueError("material geometry failed after fidelity recovery")
            if progress:
                progress("source_material_validate")
            missing, measured = select_missing_source_facets(
                np.asarray(reference.points), triangle_faces(reference),
                np.asarray(candidate.points), triangle_faces(candidate), max_error=error, grid_size=grid_size,
                progress=progress, prefer_soup=union_backend in ("manifold", "local", "planar"),
            )
            report["source_fidelity_final"] = {**measured, "coordinate_frame": "normalized",
                "sample_distance_original_units": {k: float(v) * scale for k, v in measured["sample_distance_input_units"].items()},
                "raw_sample_distance_original_units": {k: float(v) * scale for k, v in measured["raw_sample_distance_input_units"].items()}}
        report["geometry"] = current_geometry
        normalized = candidate.clean(lines_to_points=False, polys_to_lines=False, strips_to_polys=False)
        report["normalized_read"] = _geometry_gate(normalized)
        counts = same_oriented_geometry(candidate, normalized)
        report["normalized_geometry_preserved"] = counts
        report["functional_openings"] = check_preserved_opening_rays(
            np.asarray(candidate.points), triangle_faces(candidate), preparation["preserved_opening_rays"]
        )
        distance = 0.0 if identical else _candidate_distance(reference, candidate)
        bound_details = None
        if identical:
            bound = drift / scale
        elif union_backend == "planar":
            from continuous_surface_bound import source_to_candidate_bound
            bound_details = source_to_candidate_bound(
                np.asarray(candidate.points), triangle_faces(candidate),
                np.asarray(reference.points), triangle_faces(reference), max_error=error,
                max_subtriangles=max(1_000_000, candidate.n_cells),
            )
            bound = bound_details["max_certified_upper_bound"] if bound_details["passed"] else None
        else:
            bound = candidate_to_source_bound(reference, candidate, max_error=error)
        report["candidate_surface_distance"] = {"sampled_max": None if identical else distance * scale,
            "samples_per_face": 0 if identical else 7, "normalized_geometry_match": identical,
            "exact_geometry_match": identical and drift == 0,
            "continuous_one_sided_bound": drift if identical else None if bound is None else bound * scale,
            "reverse_continuous_certificate": identical and drift <= max_error}
        if bound_details is not None:
            bound_details["coordinate_frame"] = "normalized"
            bound_details["numerics"] = "floating-point queries and numerical guard; not an exact-arithmetic certificate"
            report["candidate_surface_distance"]["numerical_continuous_bound"] = bound_details
        if certify_reverse_fidelity and identical:
            report["reverse_surface_bound"] = {"status": "within_budget" if drift <= max_error else "exceeded", "passed": drift <= max_error,
                "method": "oriented_triangle_identity", "coordinate_frame": "normalized",
                "normalization_drift_bound_input_units": drift,
                "scope": "entire reference, including internal material surfaces"}
        elif certify_reverse_fidelity:
            from continuous_surface_bound import source_to_candidate_bound
            if progress:
                progress("source_material_reverse_surface_bound")
            source_faces = triangle_faces(reference)
            reverse = source_to_candidate_bound(
                np.asarray(reference.points), source_faces, np.asarray(candidate.points), triangle_faces(candidate),
                max_error=error, max_subtriangles=max(1_000_000, len(source_faces)),
            )
            reverse["coordinate_frame"] = "normalized"
            reverse["scope"] = "entire reference, including internal material surfaces"
            reverse["numerics"] = "floating-point distance queries; not an exact-arithmetic proof"
            report["reverse_surface_bound"] = reverse
            report["candidate_surface_distance"]["reverse_continuous_certificate"] = reverse["passed"]
        distance_ok = drift <= max_error if identical else bound is not None and bound <= error
        fidelity_ok = not len(missing) and distance_ok
        report["acceptance_checks"] = {
            "geometry": bool(report["geometry"]["passed"]),
            "normalized_geometry": bool(report["normalized_read"]["passed"]),
            "normalized_geometry_preserved": bool(counts),
            "functional_openings": bool(report["functional_openings"]["passed"]),
            "source_witnesses_within_budget": not len(missing),
            "source_facets_over_budget": int(len(missing)),
            "candidate_distance_within_budget": bool(distance_ok),
        }
        report["accepted"] = bool(report["geometry"]["passed"] and report["normalized_read"]["passed"]
                                  and report["functional_openings"]["passed"] and counts and fidelity_ok)
        if not report["accepted"]:
            report["failure_reason"] = "geometry, normalized readback, or sampled fidelity gate failed"
        _attach_normals(candidate, reference)
        candidate.points = np.asarray(candidate.points) * scale + center
        for name in source.field_data:
            candidate.field_data[name] = np.asarray(source.field_data[name]).copy()
        report["output_frame_geometry"] = _geometry_gate(candidate)
        world_clean = candidate.clean(lines_to_points=False, polys_to_lines=False, strips_to_polys=False)
        report["output_frame_normalized_read"] = _geometry_gate(world_clean)
        report["output_frame_normalized_geometry_preserved"] = same_oriented_geometry(candidate, world_clean)
        world_ok = (report["output_frame_geometry"]["passed"]
                    and report["output_frame_normalized_read"]["passed"]
                    and report["output_frame_normalized_geometry_preserved"])
        report["acceptance_checks"].update({
            "output_frame_geometry": bool(report["output_frame_geometry"]["passed"]),
            "output_frame_normalized_geometry": bool(report["output_frame_normalized_read"]["passed"]),
            "output_frame_normalized_geometry_preserved": bool(report["output_frame_normalized_geometry_preserved"]),
        })
        report["accepted"] &= world_ok
        if not world_ok and not report.get("failure_reason"):
            report["failure_reason"] = "output-frame geometry or normalized readback gate failed"
        return candidate, report
    except Exception as exc:
        report["accepted"] = False
        report["failure_reason"] = f"{type(exc).__name__}: {exc}"
        report["candidate_role"] = "unmodified_reference_after_failure"
        return source.copy(), report


def repair_source_file(input_path, output_dir, *, max_error, boundary_mode="review", policy_decisions=None, grid_size=128, original_input=None, progress=None, certify_reverse_fidelity=False, max_recovery_facets=1000, union_backend="cgal", assembly_context=None):
    """Own saved candidate, semantic packet, full readback and 3D report artifacts."""
    input_path, output_dir = Path(input_path), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate, report = repair_source_mesh(read_surface(input_path, clean=False), max_error=max_error, boundary_mode=boundary_mode, policy_decisions=policy_decisions, grid_size=grid_size, progress=progress, certify_reverse_fidelity=certify_reverse_fidelity, max_recovery_facets=max_recovery_facets, union_backend=union_backend)
    if progress:
        progress("source_material_serialize_validate")
    output = output_dir / "source_material_candidate.vtp"
    candidate.save(output)
    saved = pv.read(output)
    normalized = read_surface(output)
    raw_exact = (np.array_equal(saved.points, candidate.points)
                 and np.array_equal(triangle_faces(saved), triangle_faces(candidate)))
    if report["accepted"]:
        raw_geometry, normalized_geometry = _geometry_gate(saved), _geometry_gate(normalized)
    else:
        raw_geometry = normalized_geometry = {"passed": None, "status": "not_checked", "reason": "candidate already rejected"}
    report["serialized_readback"] = {
        "raw_geometry_exact": raw_exact,
        "normalized_geometry_preserved": same_oriented_geometry(saved, normalized),
        "raw": raw_geometry, "normalized": normalized_geometry,
    }
    report["accepted"] &= all((raw_exact, report["serialized_readback"]["normalized_geometry_preserved"],
                               report["serialized_readback"]["raw"]["passed"],
                               report["serialized_readback"]["normalized"]["passed"]))
    if assembly_context is None:
        report["assembly_context_validation"] = {"status": "not_checked", "passed": None,
            "scope": "part geometry only; no surrounding assembly supplied"}
    elif report["accepted"]:
        from assembly_intersections import compare_assembly_intersections
        try:
            context = read_surface(Path(assembly_context), clean=False).remove_unused_points()
            original = read_surface(input_path, clean=False).remove_unused_points()
            assembly = compare_assembly_intersections(original, saved, context)
            report["assembly_context_validation"] = {**assembly, "context_path": str(Path(assembly_context).resolve())}
            report["accepted"] &= assembly["passed"]
            if not assembly["passed"]:
                report["failure_reason"] = "assembly-context intersection gate failed or was incomplete"
        except (ValueError, RuntimeError, OSError, IndexError) as exc:
            report["assembly_context_validation"] = {"status": "failed", "passed": False, "reason": str(exc)}
            report["accepted"] = False
            report["failure_reason"] = "assembly-context validation failed"
    else:
        report["assembly_context_validation"] = {"status": "not_checked", "passed": None,
            "reason": "part geometry or readback failed before assembly validation"}
    report["input"] = {"path": str(input_path.resolve()), "units": "input coordinates"}
    report["fidelity_reference"] = {"path": str(input_path.resolve()), "geometry": "reference input with only explicitly declared caps"}
    if original_input is not None:
        report["original_pipeline_input"] = str(Path(original_input).resolve())
    if not report["accepted"] and not report.get("failure_reason"):
        report["failure_reason"] = "serialized material-geometry gate failed"
    report["outputs"] = {"candidate": str(output), "accepted_mesh_vtp": str(output) if report["accepted"] else None}
    report["outputs"]["mesh_previews"] = [
        {"label": "Fidelity reference input", "path": str(input_path.resolve())},
        {"label": "Accepted material geometry" if report["accepted"] else ("Unmodified reference after failure" if report.get("candidate_role") else "Unaccepted diagnostic candidate"), "path": str(output.resolve())},
    ]
    report["decision"] = {"status": "accepted" if report["accepted"] else "rejected",
                          "reason": report.get("failure_reason", "all material-geometry gates passed")}
    (output_dir / "source_material_report.json").write_text(json.dumps(report, indent=2))
    from html_report import write_html_report
    write_html_report(report, output_dir / "source_material_report.html", "Source-preserving material geometry")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-error", type=float, required=True, help="Geometric error budget in input units")
    parser.add_argument("--boundary-mode", choices=("wall", "cap", "review"), default="review")
    parser.add_argument("--policy-decisions", type=Path)
    parser.add_argument("--certify-reverse-fidelity", action="store_true", help="Attempt a bounded continuous source-to-candidate surface check")
    parser.add_argument("--max-recovery-facets", type=int, default=1000, help="Maximum source facets allowed in one recovery union")
    parser.add_argument("--union-backend", choices=("cgal", "manifold", "local", "planar"), default="cgal")
    parser.add_argument("--assembly-context", type=Path, help="Unchanged assembly excluding input part, in the same frame and units")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--grid-size", type=int, default=128, help="Supplemental first-hit sampling; seven geometric witnesses cover every source triangle")
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        parser.error("output directory is not empty; use a fresh directory or --overwrite")
    report = repair_source_file(args.input, args.output_dir, max_error=args.max_error,
                                boundary_mode=args.boundary_mode,
                                policy_decisions=load_policy_decisions(args.policy_decisions), grid_size=args.grid_size,
                                certify_reverse_fidelity=args.certify_reverse_fidelity,
                                max_recovery_facets=args.max_recovery_facets, union_backend=args.union_backend,
                                assembly_context=args.assembly_context)
    return 0 if report["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
