"""Conservative source-triangle to candidate-surface distance bounds."""
from __future__ import annotations

import igl
import numpy as np


def source_to_candidate_bound(
    source_points: np.ndarray,
    source_faces: np.ndarray,
    candidate_points: np.ndarray,
    candidate_faces: np.ndarray,
    *,
    max_error: float,
    max_depth: int = 10,
    max_subtriangles: int = 1_000_000,
) -> dict[str, object]:
    """Bound every source point's distance to candidate triangles, or return unproven.

    For a source subtriangle whose vertices project to one candidate triangle,
    the projections' convex combinations stay in that candidate triangle. The
    largest vertex distance therefore bounds the whole subtriangle. Otherwise,
    the distance-to-surface function is 1-Lipschitz, so the largest vertex
    distance plus the subtriangle diameter is a conservative fallback bound.
    Unresolved leaves at either budget limit are never reported as passing.
    """
    source_points = np.asarray(source_points, dtype=np.float64)
    candidate_points = np.asarray(candidate_points, dtype=np.float64)
    source_faces = np.asarray(source_faces, dtype=np.int64)
    candidate_faces = np.asarray(candidate_faces, dtype=np.int64)
    if (source_points.ndim != 2 or source_points.shape[1] != 3 or
            candidate_points.ndim != 2 or candidate_points.shape[1] != 3 or
            not np.isfinite(source_points).all() or not np.isfinite(candidate_points).all()):
        raise ValueError("point arrays must be finite (N, 3) coordinates")
    for name, faces, points in (("source_faces", source_faces, source_points),
                                ("candidate_faces", candidate_faces, candidate_points)):
        if (faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0 or
                np.any(faces < 0) or np.any(faces >= len(points))):
            raise ValueError(f"{name} must be a nonempty triangle array with valid indices")
    if not np.isfinite(max_error) or max_error < 0:
        raise ValueError("max_error must be finite and nonnegative")
    if not isinstance(max_depth, int) or max_depth < 0:
        raise ValueError("max_depth must be a nonnegative integer")
    if not isinstance(max_subtriangles, int) or max_subtriangles < len(source_faces):
        raise ValueError("max_subtriangles must cover the input source triangles")

    leaves = source_points[source_faces].copy()
    depths = np.zeros(len(leaves), dtype=np.int16)
    certified = 0
    max_bound = 0.0
    unresolved = 0
    visited = 0
    exceeded_witness = 0.0
    deepest = 0
    scale = max(float(np.linalg.norm(np.ptp(np.vstack((source_points, candidate_points)), axis=0))),
                float(np.max(np.abs(np.vstack((source_points, candidate_points))))), 1.0)
    guard = scale * np.finfo(np.float64).eps * 128
    tolerance = float(max_error)

    while len(leaves):
        visited += len(leaves)
        flat = leaves.reshape(-1, 3)
        distance2 = np.empty(len(flat), dtype=np.float64)
        nearest = np.empty(len(flat), dtype=np.int64)
        for start in range(0, len(flat), 50_000):
            stop = min(start + 50_000, len(flat))
            d2, face_ids, _ = igl.point_mesh_squared_distance(
                np.ascontiguousarray(flat[start:stop]), candidate_points, candidate_faces
            )
            distance2[start:stop] = np.asarray(d2).reshape(-1)
            nearest[start:stop] = np.asarray(face_ids, dtype=np.int64).reshape(-1)
        if (not np.isfinite(distance2).all() or np.any(distance2 < 0)
                or np.any((nearest < 0) | (nearest >= len(candidate_faces)))):
            return {
                "status": "unproven",
                "passed": False,
                "max_error": tolerance,
                "reason": "candidate distance query returned invalid data",
                "certified_subtriangles": certified,
                "unproven_subtriangles": unresolved + len(leaves),
                "max_certified_upper_bound": max_bound,
                "subdivision_depth": deepest,
                "method": "common_candidate_triangle_with_lipschitz_fallback",
                "numerical_guard": guard,
            }
        vertex_distance = np.sqrt(np.maximum(distance2, 0.0)).reshape(-1, 3)
        nearest = nearest.reshape(-1, 3)
        diameters = np.maximum.reduce([
            np.linalg.norm(leaves[:, 0] - leaves[:, 1], axis=1),
            np.linalg.norm(leaves[:, 1] - leaves[:, 2], axis=1),
            np.linalg.norm(leaves[:, 2] - leaves[:, 0], axis=1),
        ])
        local_upper = vertex_distance.max(axis=1) + guard
        same_face = (nearest[:, 0] == nearest[:, 1]) & (nearest[:, 1] == nearest[:, 2])
        local_upper[~same_face] += diameters[~same_face]
        accepted = local_upper <= tolerance
        witnessed = np.flatnonzero(vertex_distance.max(axis=1) - guard > tolerance)
        if len(witnessed):
            exceeded_witness = float(vertex_distance[witnessed[0]].max())
            return {
                "status": "exceeded",
                "passed": False,
                "max_error": tolerance,
                "witness_distance": exceeded_witness,
                "certified_subtriangles": certified,
                "unproven_subtriangles": int(np.count_nonzero(~accepted) - len(witnessed)),
                "max_certified_upper_bound": max_bound,
                "subdivision_depth": deepest,
                "method": "common_candidate_triangle_with_lipschitz_fallback",
                "numerical_guard": guard,
            }
        if np.any(accepted):
            certified += int(np.count_nonzero(accepted))
            max_bound = max(max_bound, float(local_upper[accepted].max(initial=0.0)))
        pending = ~accepted
        if not np.any(pending):
            break
        can_split = pending & (depths < max_depth)
        cannot_split = pending & ~can_split
        unresolved += int(np.count_nonzero(cannot_split))
        if not np.any(can_split):
            break
        parent = leaves[can_split]
        if visited + 4 * len(parent) > max_subtriangles:
            unresolved += len(parent)
            break
        ab = (parent[:, 0] + parent[:, 1]) * 0.5
        bc = (parent[:, 1] + parent[:, 2]) * 0.5
        ca = (parent[:, 2] + parent[:, 0]) * 0.5
        children = np.stack((
            np.stack((parent[:, 0], ab, ca), axis=1),
            np.stack((ab, parent[:, 1], bc), axis=1),
            np.stack((ca, bc, parent[:, 2]), axis=1),
            np.stack((ab, bc, ca), axis=1),
        ), axis=1).reshape(-1, 3, 3)
        new_depths = np.repeat(depths[can_split] + 1, 4)
        deepest = max(deepest, int(new_depths.max(initial=0)))
        leaves = children
        depths = new_depths

    return {
        "status": "within_budget" if unresolved == 0 else "unproven",
        "passed": unresolved == 0,
        "max_error": tolerance,
        "certified_subtriangles": certified,
        "unproven_subtriangles": unresolved,
        "max_certified_upper_bound": max_bound,
        "subdivision_depth": deepest,
        "visited_subtriangles": visited,
        "method": "common_candidate_triangle_with_lipschitz_fallback",
        "numerical_guard": guard,
    }
