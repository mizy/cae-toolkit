"""Geometry checks shared by local voxel reconstruction and exact intersection cutting."""

from __future__ import annotations

import numpy as np
from material_cleanup import geometry_gate_arrays
from mesh_metrics import edge_topology
from source_projected_validation import exact_self_intersection_report


def _seam_continuity(points, faces, output_points, output_faces, selected):
    _, old_edges = edge_topology(faces)
    _, new_edges = edge_topology(output_faces)
    selected = set(map(int, selected))
    old_tri, new_tri = points[faces], output_points[output_faces]
    old_normals = np.cross(old_tri[:, 1] - old_tri[:, 0], old_tri[:, 2] - old_tri[:, 0])
    new_normals = np.cross(new_tri[:, 1] - new_tri[:, 0], new_tri[:, 2] - new_tri[:, 0])
    angle = lambda a, b: float(
        np.degrees(np.arccos(np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1.0, 1.0)))
    )
    before, after = [], []
    for edge, incident in old_edges.items():
        inside = [i for i in incident if i in selected]
        outside = [i for i in incident if i not in selected]
        if not inside or not outside:
            continue
        connected = new_edges.get(edge, [])
        if len(incident) != 2 or len(connected) != 2:
            return {"passed": False, "reason": "fixed seam is not shared by exactly two faces"}
        before.append(angle(old_normals[inside[0]], old_normals[outside[0]]))
        after.append(angle(new_normals[connected[0]], new_normals[connected[1]]))
    if not before:
        return {"passed": False, "reason": "no verifiable fixed seam"}
    return {
        "passed": True,
        "edges": len(before),
        "before_max_deg": max(before),
        "after_max_deg": max(after),
        "max_increase_deg": float(np.max(np.asarray(after) - before)),
    }


def _local_geometry(out_points, out_faces, lineage):
    geometry = geometry_gate_arrays(out_points, out_faces)
    if geometry["passed"]:
        return geometry, {"passed": True, "remaining_outside_intersection_pairs": 0}
    metrics = geometry["metrics"]
    topology = metrics["topology"]
    basic = (
        all(
            topology[key] == 0
            for key in ("boundary_edges", "non_manifold_edges", "non_manifold_vertices", "inconsistent_winding_edges")
        )
        and metrics["quality"]["degenerate_faces"] == 0
        and metrics["volume"]["reliable"]
        and topology["components"]["count"] == 1
        and metrics["volume"]["signed_sum"] > 0
    )
    if not basic or geometry["exact"].get("passed") is not False:
        return geometry, {"passed": False, "reason": "assembled topology, volume or orientation failed"}
    exact = exact_self_intersection_report(out_points, out_faces, max_reported_pairs=10000)
    pairs = np.asarray(exact.get("reported_pairs", []), dtype=np.int64).reshape(-1, 2)
    if exact.get("status") != "computed" or len(pairs) >= 10000:
        return geometry, {"passed": False, "reason": "local intersection proof exceeded pair budget"}
    new_pairs = pairs[np.any(lineage[pairs] < 0, axis=1)]
    return geometry, {
        "passed": len(new_pairs) == 0,
        "generated_intersection_pairs": len(new_pairs),
        "remaining_outside_intersection_pairs": len(pairs) - len(new_pairs),
        "method": "exact_all_face_scan_filtered_to_generated_faces",
    }


def _euler(faces):
    edges = np.unique(np.sort(np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1), axis=0)
    return int(len(np.unique(faces)) - len(edges) + len(faces))
