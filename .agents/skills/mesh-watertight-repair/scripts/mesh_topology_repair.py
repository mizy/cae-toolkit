"""Local topology operations that preserve the source triangle lineage."""
from __future__ import annotations

from fractions import Fraction

import numpy as np

from mesh_metrics import edge_topology, triangle_area_error_bound


def round_degenerate_vertices(points, faces, *, max_displacement):
    """Propose bounded rounding at degenerate faces; callers must recertify the solid.

    Only vertices incident to numerically degenerate faces may move. A grid is
    used to make near-coincident Boolean vertices agree, not to rebuild a surface.
    This is a repair proposal, not CGAL's iterative snap-rounding algorithm.
    """
    points = np.asarray(points, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if not np.isfinite(max_displacement) or max_displacement <= 0:
        raise ValueError("rounding displacement must be finite and positive")
    tri = points[faces]
    areas = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) * .5
    bad = areas <= triangle_area_error_bound(points, faces)
    if not np.any(bad) or np.count_nonzero(bad) > 1000:
        return points, faces, np.arange(len(faces)), {"applied": False}
    vertices = np.unique(faces[bad])
    changed = points.copy()
    changed[vertices] = np.rint(points[vertices] / max_displacement) * max_displacement
    displacement = np.linalg.norm(changed[vertices] - points[vertices], axis=1)
    if not np.isfinite(displacement).all() or displacement.max(initial=0.) > max_displacement:
        raise ValueError("rounding exceeded its displacement budget")
    changed, inverse = np.unique(changed, axis=0, return_inverse=True)
    result = inverse[faces]
    tri = changed[result]
    keep = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) > 0
    if not np.any(keep):
        raise ValueError("rounding removed the entire component")
    return changed, result[keep], np.flatnonzero(keep), {
        "applied": True, "moved_vertices": int(np.count_nonzero(displacement)),
        "max_displacement": float(displacement.max(initial=0.)),
        "removed_zero_area_faces": int(np.count_nonzero(~keep)),
    }


def _try_collapse_edge(
    points: np.ndarray,
    faces: np.ndarray,
    face_ids: np.ndarray,
    incident: list[int],
    keep: int,
    victim: int,
    area_epsilon: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    retained = np.ones(len(faces), dtype=bool)
    retained[incident] = False
    next_faces = faces[retained].copy()
    affected = np.any(next_faces == victim, axis=1)
    before = points[next_faces[affected]]
    next_faces[affected] = np.where(next_faces[affected] == victim, keep, next_faces[affected])
    if np.any(
        (next_faces[affected, 0] == next_faces[affected, 1])
        | (next_faces[affected, 1] == next_faces[affected, 2])
        | (next_faces[affected, 2] == next_faces[affected, 0])
    ):
        return None
    after = points[next_faces[affected]]
    old_normal = np.cross(before[:, 1] - before[:, 0], before[:, 2] - before[:, 0])
    new_normal = np.cross(after[:, 1] - after[:, 0], after[:, 2] - after[:, 0])
    new_area = np.linalg.norm(new_normal, axis=1) * 0.5
    if np.any(new_area <= area_epsilon) or np.any(np.einsum("ij,ij->i", old_normal, new_normal) <= 0):
        return None
    keep_star = np.sort(next_faces[np.any(next_faces == keep, axis=1)], axis=1)
    if len(np.unique(keep_star, axis=0)) != len(keep_star):
        return None
    return next_faces, face_ids[retained]


def collapse_independent_short_edges(points, faces, *, pinned_points, max_edge_length,
                                     max_bad_faces=1000, max_star_faces=50000):
    """Collapse disjoint valid one-ring patches without repeatedly scanning the full mesh."""
    points = np.asarray(points, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    pinned = np.asarray(pinned_points, dtype=bool)
    if pinned.shape != (len(points),) or not np.isfinite(max_edge_length) or max_edge_length <= 0:
        raise ValueError("valid source pins and a positive edge limit are required")
    if max_bad_faces < 1 or max_star_faces < 1:
        raise ValueError("local repair budgets must be positive")
    identity = np.arange(len(faces), dtype=np.int64)
    tri = points[faces]
    areas = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) * .5
    bad = np.flatnonzero(areas <= triangle_area_error_bound(points, faces))
    summary = {"status": "completed", "degenerate_faces_before": len(bad), "collapsed_edges": 0}
    if not len(bad):
        return points, faces, identity, [], summary
    if len(bad) > max_bad_faces:
        summary["status"] = "bad_face_budget_exceeded"
        return points, faces, identity, [], summary
    if np.any((faces[:, 0] == faces[:, 1]) | (faces[:, 1] == faces[:, 2]) | (faces[:, 2] == faces[:, 0])):
        summary["status"] = "indexed_degeneracy_requires_other_repair"
        return points, faces, identity, [], summary
    topology, edge_faces = edge_topology(faces)
    if any(topology[key] for key in ("boundary_edges", "non_manifold_edges", "non_manifold_vertices")):
        summary["status"] = "input_not_closed_manifold"
        return points, faces, identity, [], summary
    vertices = np.unique(faces[bad])
    star_ids = np.flatnonzero(np.any(np.isin(faces, vertices), axis=1))
    summary["star_faces"] = len(star_ids)
    if len(star_ids) > max_star_faces:
        summary["status"] = "star_face_budget_exceeded"
        return points, faces, identity, [], summary
    neighbors = {int(vertex): set() for vertex in vertices}
    incident_faces = {int(vertex): set() for vertex in vertices}
    for face_id in star_ids:
        face = faces[face_id]
        for vertex in face:
            if int(vertex) in neighbors:
                neighbors[int(vertex)].update(int(other) for other in face if other != vertex)
                incident_faces[int(vertex)].add(int(face_id))
    candidates, seen = [], set()
    for face_id in bad:
        for a, b in zip(faces[face_id], np.roll(faces[face_id], -1)):
            edge = (min(int(a), int(b)), max(int(a), int(b)))
            if edge in seen:
                continue
            seen.add(edge)
            incident = edge_faces[edge]
            length = float(np.linalg.norm(points[edge[0]] - points[edge[1]]))
            if length > max_edge_length or len(incident) != 2 or (pinned[edge[0]] and pinned[edge[1]]):
                continue
            opposite = {int(next(v for v in faces[i] if int(v) not in edge)) for i in incident}
            if neighbors[edge[0]] & neighbors[edge[1]] == opposite:
                candidates.append((length, edge, incident))
    scale = float(np.linalg.norm(np.ptp(points, axis=0)))
    area_epsilon = max(float(np.median(areas)) * 1e-12, scale * scale * 1e-18)
    changed = faces.copy()
    mapped_rows = identity.copy()
    removed = np.zeros(len(faces), dtype=bool)
    used_vertices, actions = set(), []
    for length, edge, incident in sorted(candidates):
        local_ids = np.asarray(sorted(incident_faces[edge[0]] | incident_faces[edge[1]]), dtype=np.int64)
        local_vertices = set(faces[local_ids].ravel().tolist())
        if local_vertices & used_vertices:
            continue
        local_incident = np.flatnonzero(np.isin(local_ids, incident)).tolist()
        for keep, victim in sorted((edge, edge[::-1]), key=lambda pair: (not pinned[pair[0]], pair[0])):
            if pinned[victim]:
                continue
            result = _try_collapse_edge(points, faces[local_ids], local_ids, local_incident, keep, victim, area_epsilon)
            if result is None:
                continue
            local_faces, kept = result
            changed[kept] = local_faces
            removed[np.setdiff1d(local_ids, kept)] = True
            used_vertices.update(local_vertices)
            actions.append({"operator": "independent_short_edge_collapse", "edge": list(edge),
                            "length": length, "kept_point_id": int(keep), "collapsed_point_id": int(victim),
                            "pinned_source_endpoint": bool(pinned[keep]),
                            "removed_input_face_ids": sorted(int(i) for i in incident)})
            break
    flips = 0
    bounded_flips = 0
    for face_id in bad:
        if removed[face_id] or any(int(vertex) in used_vertices for vertex in faces[face_id]):
            continue
        face = faces[face_id]
        local = points[face]
        lengths = np.linalg.norm(local[[1, 2, 0]] - local[[0, 1, 2]], axis=1)
        edge_index = int(np.argmax(lengths))
        a, c, b = (int(face[edge_index]), int(face[(edge_index + 1) % 3]), int(face[(edge_index + 2) % 3]))
        rational = [[Fraction.from_float(float(x)) for x in points[index]] for index in (a, b, c)]
        ab = [rational[1][i] - rational[0][i] for i in range(3)]
        ac = [rational[2][i] - rational[0][i] for i in range(3)]
        cross = (ab[1] * ac[2] - ab[2] * ac[1], ab[2] * ac[0] - ab[0] * ac[2], ab[0] * ac[1] - ab[1] * ac[0])
        length2 = sum(x * x for x in ac)
        projection = sum(x * y for x, y in zip(ab, ac))
        if length2 == 0 or not 0 < projection < length2:
            continue
        exact_collinear = not any(cross)
        deviation_bound = 0.0
        if not exact_collinear:
            if np.any(pinned[face]):
                continue
            from source_distance_bound import _sqrt_upper
            distance_squared = (sum(x * x for x in ab) * length2 - projection * projection) / length2
            if distance_squared < 0:
                continue
            # The original sliver is at most d from its long edge and the new
            # triangulation is at most d from the old neighbor; hence 2d bounds both directions.
            deviation_bound = _sqrt_upper(4 * distance_squared)
            if deviation_bound > max_edge_length:
                continue
        incident = edge_faces[(min(a, c), max(a, c))]
        if len(incident) != 2:
            continue
        other = next(i for i in incident if i != face_id)
        if removed[other] or any(int(vertex) in used_vertices for vertex in faces[other]):
            continue
        neighbor = faces[other]
        if not any(neighbor[i] == c and neighbor[(i + 1) % 3] == a for i in range(3)):
            continue
        d = int(next(vertex for vertex in neighbor if vertex not in (a, c)))
        if (min(b, d), max(b, d)) in edge_faces:
            continue
        new_faces = np.asarray([[c, b, d], [b, a, d]], dtype=np.int64)
        new_tri = points[new_faces]
        normals = np.cross(new_tri[:, 1] - new_tri[:, 0], new_tri[:, 2] - new_tri[:, 0])
        old_normal = np.cross(points[neighbor[1]] - points[neighbor[0]], points[neighbor[2]] - points[neighbor[0]])
        if np.any(normals @ old_normal <= 0) or np.any(np.linalg.norm(normals, axis=1) * .5 <= triangle_area_error_bound(points, new_faces)):
            continue
        changed[face_id], changed[other] = new_faces
        mapped_rows[face_id] = mapped_rows[other]
        used_vertices.update(int(vertex) for vertex in np.r_[face, neighbor])
        actions.append({"operator": "exact_collinear_edge_flip" if exact_collinear else "bounded_near_collinear_edge_flip",
                        "edge": [a, c], "new_edge": [b, d], "bidirectional_bound": deviation_bound,
                        "source_face_id": int(other), "degenerate_face_id": int(face_id)})
        flips += 1
        bounded_flips += int(not exact_collinear)
    summary.update(candidate_edges=len(candidates), collapsed_edges=len(actions))
    summary["collinear_flips"] = flips
    summary["bounded_near_collinear_flips"] = bounded_flips
    summary["collapsed_edges"] = len(actions) - flips
    kept = identity[~removed]
    return points, changed[kept], mapped_rows[kept], actions, summary


def collapse_short_manifold_edges(
    points: np.ndarray,
    faces: np.ndarray,
    *,
    pinned_points: np.ndarray | None = None,
    max_edge_length: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, object]]]:
    """Collapse short edges around degenerate faces when the link condition holds.

    The length limit uses the same units as ``points``. Input points are
    retained, while returned face ids map each output face to its input row.
    """
    points = np.asarray(points, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64).copy()
    max_edge_length = float(max_edge_length)
    if not np.isfinite(max_edge_length) or max_edge_length <= 0:
        raise ValueError("max_edge_length must be finite and positive")
    pinned = np.zeros(len(points), dtype=bool) if pinned_points is None else np.asarray(pinned_points, dtype=bool)
    if pinned.shape != (len(points),):
        raise ValueError("pinned_points must contain one flag per input point")
    scale = float(np.linalg.norm(np.ptp(points, axis=0))) if len(points) else 0.0
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("points must span a finite, nonzero extent")
    triangles = points[faces]
    areas = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1) * 0.5
    area_epsilon = max(float(np.median(areas)) * 1e-12, scale * scale * 1e-18)
    length_epsilon = max(scale * 1e-15, np.max(np.abs(points)) * np.finfo(float).eps * 4)
    face_ids = np.arange(len(faces), dtype=np.int64)
    actions: list[dict[str, object]] = []

    while True:
        triangles = points[faces]
        areas = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1) * 0.5
        lengths = np.stack(
            [np.linalg.norm(triangles[:, (i + 1) % 3] - triangles[:, i], axis=1) for i in range(3)],
            axis=1,
        )
        bad_faces = np.flatnonzero((areas <= area_epsilon) | (lengths.min(axis=1) <= length_epsilon))
        if not len(bad_faces):
            break
        topology, edge_faces = edge_topology(faces)
        if topology["boundary_edges"] or topology["non_manifold_edges"] or topology["non_manifold_vertices"]:
            break
        links: dict[int, set[int]] = {}
        for left, right in edge_faces:
            links.setdefault(left, set()).add(right)
            links.setdefault(right, set()).add(left)
        collapsed = False
        for face_id in bad_faces:
            face = faces[face_id]
            edges = []
            for left, right in zip(face, np.roll(face, -1)):
                edge = (min(int(left), int(right)), max(int(left), int(right)))
                incident = edge_faces[edge]
                length = float(np.linalg.norm(points[edge[0]] - points[edge[1]]))
                opposite = {int(next(v for v in faces[i] if int(v) not in edge)) for i in incident}
                if length <= max_edge_length and len(incident) == 2 and links[edge[0]] & links[edge[1]] == opposite:
                    edges.append((length, edge, incident))
            for length, edge, incident in sorted(edges):
                endpoints = sorted(((edge[0], edge[1]), (edge[1], edge[0])), key=lambda pair: (not pinned[pair[0]], pair[0]))
                for keep, victim in endpoints:
                    if pinned[victim]:
                        continue
                    result = _try_collapse_edge(points, faces, face_ids, incident, keep, victim, area_epsilon)
                    if result is None:
                        continue
                    actions.append(
                        {
                            "edge": list(edge),
                            "length": length,
                            "kept_point_id": int(keep),
                            "collapsed_point_id": int(victim),
                            "pinned_source_endpoint": bool(pinned[keep]),
                            "removed_input_face_ids": sorted(int(face_ids[i]) for i in incident),
                        }
                    )
                    faces, face_ids = result
                    collapsed = True
                    break
                if collapsed:
                    break
            if collapsed:
                break
        if not collapsed:
            break
    return points, faces, face_ids, actions
