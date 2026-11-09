"""Check a replacement against the unchanged surrounding assembly."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mesh_io import read_surface, triangle_faces
from source_projected_validation import exact_self_intersection_report


def _cross_intersections(part, context_points, context_faces, context_rows, max_pairs):
    part_points = np.asarray(part.points, dtype=np.float64)
    part_faces = triangle_faces(part)
    used, inverse = np.unique(context_faces.ravel(), return_inverse=True)
    local_faces = inverse.reshape(-1, 3) + len(part_points)
    points = np.vstack((part_points, context_points[used]))
    exact = exact_self_intersection_report(points, np.vstack((part_faces, local_faces)),
                                           max_reported_pairs=max_pairs)
    pairs = np.asarray(exact.get("reported_pairs", []), dtype=np.int64).reshape(-1, 2)
    cross = pairs[(pairs[:, 0] < len(part_faces)) != (pairs[:, 1] < len(part_faces))]
    neighbor_rows = set()
    for a, b in cross:
        neighbor_rows.add(int(context_rows[(b if a < len(part_faces) else a) - len(part_faces)]))
    complete = exact.get("status") == "computed" and not exact.get("reported_pairs_truncated", True)
    # libigl's cutoff can stop discovery at the requested limit without reporting truncation.
    complete = bool(complete and exact.get("intersection_pairs", max_pairs) < max_pairs)
    return {"status": "computed" if complete else "incomplete", "cross_pairs": len(cross),
            "neighbor_face_rows": sorted(neighbor_rows), "exact": exact}


# @entry: original and replacement occupy the same frame; context excludes original faces.
def compare_assembly_intersections(original, replacement, context, *, max_context_faces=20000, max_pairs=100000):
    if max_context_faces <= 0 or max_pairs <= 0:
        raise ValueError("face and pair limits must be positive")
    original_points = np.asarray(original.points, dtype=np.float64)
    replacement_points = np.asarray(replacement.points, dtype=np.float64)
    context_points = np.asarray(context.points, dtype=np.float64)
    context_faces = triangle_faces(context)
    if (not len(original_points) or not len(replacement_points) or not len(context_faces)
            or not np.isfinite(original_points).all() or not np.isfinite(replacement_points).all()
            or not np.isfinite(context_points).all()):
        raise ValueError("original, replacement and context must contain finite nonempty geometry")
    lower = np.minimum(original_points.min(axis=0), replacement_points.min(axis=0))
    upper = np.maximum(original_points.max(axis=0), replacement_points.max(axis=0))
    triangles = context_points[context_faces]
    rows = np.flatnonzero(np.all(triangles.min(axis=1) <= upper, axis=1)
                          & np.all(triangles.max(axis=1) >= lower, axis=1))
    report = {"method": "union_aabb_broad_phase_and_cgal_exact_cross_intersections",
              "scope": "replacement versus provided unchanged context; does not certify the complete assembly",
              "context_faces_total": len(context_faces), "context_faces_in_bounds": len(rows),
              "bounds": [lower.tolist(), upper.tolist()], "max_context_faces": max_context_faces,
              "max_pairs": max_pairs,
              "limitations": ["Context must exclude the original part and use the same coordinate frame and units.",
                              "Contacts along shared seams are conservatively counted as intersections.",
                              "Changes within a context face intersected both before and after are not classified."]}
    if len(rows) > max_context_faces:
        return {**report, "status": "context_budget_exceeded", "passed": False}
    if not len(rows):
        return {**report, "status": "computed", "passed": True,
                "before": {"cross_pairs": 0, "neighbor_face_rows": []},
                "after": {"cross_pairs": 0, "neighbor_face_rows": []},
                "newly_intersected_neighbor_face_rows": []}
    before = _cross_intersections(original, context_points, context_faces[rows], rows, max_pairs)
    after = _cross_intersections(replacement, context_points, context_faces[rows], rows, max_pairs)
    complete = before["status"] == after["status"] == "computed"
    new_faces = sorted(set(after["neighbor_face_rows"]) - set(before["neighbor_face_rows"]))
    return {**report, "status": "computed" if complete else "incomplete",
            "passed": bool(complete and after["cross_pairs"] == 0), "before": before, "after": after,
            "newly_intersected_neighbor_face_rows": new_faces}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original", type=Path, help="Original part before repair")
    parser.add_argument("replacement", type=Path, help="Candidate part in the same frame and units")
    parser.add_argument("context", type=Path, help="Unchanged assembly excluding the original part")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-context-faces", type=int, default=20000)
    parser.add_argument("--max-pairs", type=int, default=100000)
    args = parser.parse_args()
    meshes = [read_surface(path, clean=False).remove_unused_points()
              for path in (args.original, args.replacement, args.context)]
    report = compare_assembly_intersections(*meshes, max_context_faces=args.max_context_faces,
                                            max_pairs=args.max_pairs)
    report["inputs"] = {"original": str(args.original.resolve()), "replacement": str(args.replacement.resolve()),
                        "context": str(args.context.resolve())}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
