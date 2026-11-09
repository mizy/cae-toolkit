"""Select source shells that can bypass voxel reconstruction unchanged."""
from __future__ import annotations

import numpy as np

from material_cleanup import geometry_gate_arrays
from mesh_metrics import component_face_rows, edge_topology, face_component_labels


# @entry: the projected-shell closure builder owns the returned source-face mask.
def select_preserved_shells(points, faces):
    """Keep a valid complete source, or certified nonintersecting components.

    Triangle bounds narrow cross-component queries; exact predicates decide
    intersections. The final combined proxy must be certified again because a
    voxel repair can extend into a retained component's neighborhood.
    """
    points, faces = np.asarray(points), np.asarray(faces, dtype=np.int64)
    whole = geometry_gate_arrays(points, faces)
    keep = np.zeros(len(faces), dtype=bool)
    report = {"method": "certified_source_identity_or_nonintersecting_closed_components",
              "whole_source_valid": whole["passed"],
              "source_faces": len(faces), "preserved_faces": 0, "preserved_components": 0,
              "interaction_deferred_components": 0, "intersecting_closed_components": 0,
              "invalid_closed_components": 0,
              "closed_components_checked": 0}
    if whole["passed"]:
        keep[:] = True
        report.update(preserved_faces=len(faces),
                      preserved_components=whole["metrics"]["topology"]["components"]["count"])
        return keep, report
    _, edges = edge_topology(faces)
    labels = face_component_labels(len(faces), edges)
    groups = component_face_rows(labels)
    open_faces = np.zeros(len(faces), dtype=bool)
    for incident in edges.values():
        if len(incident) != 2:
            open_faces[incident] = True
    triangles = points[faces]
    lower, upper = triangles.min(axis=1), triangles.max(axis=1)
    for index, ids in enumerate(groups):
        if np.any(open_faces[ids]):
            continue
        if report["closed_components_checked"] >= 1024:
            report["check_budget_exhausted"] = True
            break
        used, inverse = np.unique(faces[ids], return_inverse=True)
        checked = geometry_gate_arrays(points[used], inverse.reshape(-1, 3))
        report["closed_components_checked"] += 1
        if not checked["passed"]:
            report["invalid_closed_components"] += 1
            continue
        overlap = np.all(lower <= upper[ids].max(axis=0), axis=1)
        overlap &= np.all(upper >= lower[ids].min(axis=0), axis=1)
        overlap[ids] = False
        nearby = np.flatnonzero(overlap)
        if len(nearby) + len(ids) > 100000:
            report["interaction_deferred_components"] += 1
            continue
        if len(nearby):
            from igl.copyleft import cgal
            other_points, other_inverse = np.unique(faces[nearby], return_inverse=True)
            pairs = cgal.intersect_other(
                np.ascontiguousarray(points[used], dtype=np.float64), inverse.reshape(-1, 3),
                np.ascontiguousarray(points[other_points], dtype=np.float64), other_inverse.reshape(-1, 3),
                detect_only=True, first_only=True, slow_and_more_precise_rounding=True, cutoff=1)[0]
            if len(pairs):
                report["intersecting_closed_components"] += 1
                continue
        keep[ids] = True
        report["preserved_components"] += 1
    report["preserved_faces"] = int(keep.sum())
    return keep, report
