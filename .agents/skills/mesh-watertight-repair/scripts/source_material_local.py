"""Repair individual material components before assembling positive solids.

This experimental route keeps healthy components unchanged, bounds local Boolean
inputs, and rejects unresolved parts before a global union. Its caller supplies
a subprocess deadline; component size limits alone cannot interrupt native code.
"""
from __future__ import annotations

import time

import numpy as np

from material_cleanup import _candidate_distance, _cleanup, _geometry_gate, _surface, _union, geometry_gate_arrays
from mesh_io import triangle_faces
from mesh_topology_repair import round_degenerate_vertices
from source_distance_bound import candidate_to_source_bound
from source_material_closure import _build_components, _manifold_union_blocks, _volume


def repair_material_component(block, source, *, max_error, sheet_faces=0):
    """Return a locally checked proposal, or the unchanged block on failure."""
    points, faces, rows = block
    started = time.monotonic()
    initial = geometry_gate_arrays(points, faces)
    report = {"passed": initial["passed"], "changed": False, "attempts": [],
              "input_faces": len(faces), "initial_geometry": initial}
    if initial["passed"]:
        return block, report
    if len(faces) > 10000:
        report["failure_reason"] = "local_component_face_budget_exceeded"
        return block, report
    source_faces = triangle_faces(source)[np.unique(rows)]
    source_points = np.asarray(source.points)[np.unique(source_faces)]
    source_part = _surface(np.asarray(source.points), source_faces)
    variants = [("local_material_union", points)]
    if sheet_faces:
        half = len(points) // 2
        if len(points) % 2 or faces[:sheet_faces].max() >= half:
            raise ValueError("sheet shell no longer has paired source and inner vertices")
        thin = points.copy()
        thin[half:] = points[:half] + (points[half:] - points[:half]) / 8
        variants.insert(0, ("reduced_generated_wall", thin))
    for method, proposal in variants:
        attempt = {"method": method, "passed": False}
        report["attempts"].append(attempt)
        try:
            p, f, r = _union(proposal, faces, rows)
            candidate, actions = _cleanup(_surface(p, f, r), source_points, max_error)
            geometry = _geometry_gate(candidate)
            if not geometry["passed"] and geometry["metrics"]["quality"]["degenerate_faces"]:
                p, f, kept, rounding = round_degenerate_vertices(
                    candidate.points, triangle_faces(candidate), max_displacement=max_error / 1024)
                attempt["rounding"] = rounding
                if rounding["applied"]:
                    p, f, r = _union(p, f, np.asarray(candidate.cell_data["source_facet_id"])[kept])
                    candidate, extra = _cleanup(_surface(p, f, r), source_points, max_error)
                    actions.extend(extra)
                    geometry = _geometry_gate(candidate)
            attempt.update(geometry=geometry, cleanup_actions=actions)
            if not geometry["passed"]:
                continue
            bound = candidate_to_source_bound(source, candidate, max_error=max_error)
            reverse = _candidate_distance(candidate, source_part)
            attempt.update(candidate_bound=bound, source_sample_max=reverse,
                           source_samples_per_face=7, passed=bool(bound <= max_error and reverse <= max_error))
            if attempt["passed"]:
                report.update(passed=True, changed=True, seconds=time.monotonic() - started)
                # Component storage outlives the temporary VTK mesh and must own its buffers.
                return (np.array(candidate.points, copy=True), triangle_faces(candidate).copy(),
                        np.array(candidate.cell_data["source_facet_id"], copy=True)), report
        except (ValueError, RuntimeError) as exc:
            attempt["failure_reason"] = f"{type(exc).__name__}: {exc}"
    report.update(seconds=time.monotonic() - started, failure_reason="local_geometry_or_fidelity_failed")
    return block, report


# @entry: source-material worker calls this only for the explicit local backend.
def build_local_material(points, faces, *, wall_depth, max_error, progress=None):
    oriented, blocks, components, splits, nested, interactions = _build_components(
        points, faces, wall_depth, progress=progress)
    source = _surface(points, faces)
    report = {"method": "component repair before positive material union", "union_backend": "local",
              "components": components, "fan_split_points": splits,
              "nested_closed_component_relations": nested,
              "closed_component_aabb_interactions": interactions, "component_repairs": [],
              "retained_components": 0, "repaired_components": 0, "components_passed": False,
              "max_component_faces": 10000, "max_repaired_components": 64,
              "acceptance": "local proposals only; full geometry and source gates remain required"}
    for index, (block, detail) in enumerate(zip(blocks, components)):
        if progress:
            progress(f"source_material_local_component_{index + 1}_of_{len(blocks)}")
        if _volume(block[0], block[1]) <= 0:
            report.update(failed_component=index, failure_reason="local backend requires positive material shells")
            return points, oriented, np.arange(len(faces)), report
        if report["repaired_components"] >= 64 and not geometry_gate_arrays(block[0], block[1])["passed"]:
            report["failure_reason"] = "local_component_count_budget_exceeded"
            return points, oriented, np.arange(len(faces)), report
        candidate, checked = repair_material_component(
            block, source, max_error=max_error,
            sheet_faces=detail["input_faces"] if detail["mode"] == "inward_offset_sheet_shell" else 0)
        if not checked["passed"] or checked["changed"]:
            report["component_repairs"].append({"component": index, **checked})
        if not checked["passed"]:
            report.update(failed_component=index, failure_reason=checked["failure_reason"])
            return points, oriented, np.arange(len(faces)), report
        blocks[index] = candidate
        report["repaired_components"] += int(checked["changed"])
        report["retained_components"] += int(not checked["changed"])
    report["components_passed"] = True
    if not splits and not interactions and all(row["mode"] == "preserved_closed_component" for row in components) and not report["repaired_components"]:
        return points, oriented, np.arange(len(faces)), report
    if progress:
        progress("source_material_local_assembly")
    p, f, rows, count = _manifold_union_blocks(blocks)
    report.update(boolean_applied=True, input_faces_to_union=count, output_faces=len(f))
    return p, f, rows, report
