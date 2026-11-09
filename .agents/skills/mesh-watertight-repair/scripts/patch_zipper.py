"""Join a retained source boundary loop to a local reconstruction loop."""

from __future__ import annotations

import numpy as np


def _point_segment_distance(point, start, end):
    edge = end - start
    length2 = float(edge @ edge)
    if length2 == 0.0:
        return float(np.linalg.norm(point - start))
    t = np.clip(float((point - start) @ edge) / length2, 0.0, 1.0)
    return float(np.linalg.norm(point - (start + t * edge)))


def _segment_distance(a, b, c, d):
    u, v, w = b - a, d - c, a - c
    aa, bb, cc = float(u @ u), float(u @ v), float(v @ v)
    dd, ee = float(u @ w), float(v @ w)
    distances = [
        _point_segment_distance(a, c, d),
        _point_segment_distance(b, c, d),
        _point_segment_distance(c, a, b),
        _point_segment_distance(d, a, b),
    ]
    denominator = aa * cc - bb * bb
    if denominator > np.finfo(float).eps * aa * cc:
        s, t = (bb * ee - cc * dd) / denominator, (aa * ee - bb * dd) / denominator
        if 0.0 <= s <= 1.0 and 0.0 <= t <= 1.0:
            distances.append(float(np.linalg.norm(w + s * u - t * v)))
    return min(distances)


def _validate_loop(points, loop, name, tolerance, max_vertices):
    ids = np.asarray(loop, dtype=np.int64)
    if ids.ndim != 1 or len(ids) < 4:
        raise ValueError(f"{name} must be a closed loop with at least three unique vertices")
    if ids[0] == ids[-1]:
        ids = ids[:-1]
    if len(ids) < 3 or len(ids) > max_vertices or len(np.unique(ids)) != len(ids):
        raise ValueError(f"{name} is degenerate, repeated, or over the vertex budget")
    if ids.min() < 0 or ids.max() >= len(points):
        raise ValueError(f"{name} references an invalid point")
    loop_points = points[ids]
    lengths = np.linalg.norm(np.roll(loop_points, -1, axis=0) - loop_points, axis=1)
    if np.any(lengths <= tolerance):
        raise ValueError(f"{name} contains a zero-length edge")
    return ids, loop_points, lengths


def _arc_sample(coords, cumulative, fraction):
    fraction = np.mod(fraction, 1.0)
    edge = np.searchsorted(cumulative, fraction, side="right") - 1
    edge = np.clip(edge, 0, len(coords) - 1)
    span = cumulative[edge + 1] - cumulative[edge]
    t = np.divide(fraction - cumulative[edge], span, out=np.zeros_like(fraction, dtype=float), where=span > 0)
    return coords[edge] + t[:, None] * (coords[(edge + 1) % len(coords)] - coords[edge])


def _check_simple(loops, tolerance):
    segments = []
    for which, (_, coords, _) in enumerate(loops):
        ends = np.roll(coords, -1, axis=0)
        lower, upper = np.minimum(coords, ends), np.maximum(coords, ends)
        segments.append((coords, ends, lower, upper))
        for left in range(len(coords)):
            # Disjoint expanded boxes cannot contain segments within tolerance.
            overlap = np.all(lower[left] <= upper + tolerance, axis=1)
            overlap &= np.all(lower <= upper[left] + tolerance, axis=1)
            for right in np.flatnonzero(overlap):
                if right <= left + 1 or (left == 0 and right == len(coords) - 1):
                    continue
                if _segment_distance(coords[left], ends[left], coords[right], ends[right]) <= tolerance:
                    raise ValueError(f"boundary loop {which} is not simple")
    outer, outer_ends, outer_lower, outer_upper = segments[0]
    inner, inner_ends, inner_lower, inner_upper = segments[1]
    for a in range(len(outer)):
        overlap = np.all(outer_lower[a] <= inner_upper + tolerance, axis=1)
        overlap &= np.all(inner_lower <= outer_upper[a] + tolerance, axis=1)
        for b in np.flatnonzero(overlap):
            if _segment_distance(outer[a], outer_ends[a], inner[b], inner_ends[b]) <= tolerance:
                raise ValueError("boundary loops intersect or are numerically coincident")


def _candidate(points, outer_ids, inner_ids, rotation, max_bridge_length):
    n, m = len(outer_ids), len(inner_ids)
    outer, inner = points[outer_ids], points[inner_ids]
    outer_lengths = np.linalg.norm(np.roll(outer, -1, axis=0) - outer, axis=1)
    inner_lengths = np.linalg.norm(np.roll(inner, -1, axis=0) - inner, axis=1)
    outer_fraction = np.r_[0.0, np.cumsum(outer_lengths) / outer_lengths.sum()]
    inner_fraction = np.r_[0.0, np.cumsum(inner_lengths) / inner_lengths.sum()]
    offset = inner_fraction[rotation]
    outer_to_inner = _arc_sample(inner, inner_fraction, outer_fraction[:-1] + offset)
    inner_to_outer = _arc_sample(outer, outer_fraction, inner_fraction[:-1] - offset)
    bridges = np.r_[np.linalg.norm(outer - outer_to_inner, axis=1), np.linalg.norm(inner - inner_to_outer, axis=1)]
    if bridges.max(initial=0.0) > max_bridge_length:
        return None
    bridge_score = float(bridges @ bridges)
    faces, i, j, offset = [], 0, 0, 0
    while i < n or j < m:
        next_outer = outer_fraction[i + 1] if i < n else np.inf
        next_inner = inner_fraction[j + 1] if j < m else np.inf
        oi, oj = i % n, (i + 1) % n
        ij, inn = (j + rotation) % m, (j + 1 + rotation) % m
        if next_outer <= next_inner:
            faces.append((outer_ids[oi], outer_ids[oj], inner_ids[ij]))
            i += 1
        else:
            faces.append((outer_ids[oi], inner_ids[inn], inner_ids[ij]))
            j += 1
    faces = np.asarray(faces, dtype=np.int64)
    memberships = np.isin(faces, outer_ids)
    cross_edges = memberships != np.roll(memberships, -1, axis=1)
    lengths = np.linalg.norm(points[faces] - points[np.roll(faces, -1, axis=1)], axis=2)
    if lengths[cross_edges].max(initial=0.0) > max_bridge_length:
        return None
    tri = points[faces]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    scale = max(float(np.linalg.norm(np.ptp(points[np.unique(faces)], axis=0))), 1e-150)
    if np.any(area <= max(scale * scale * np.finfo(float).eps * 128, np.finfo(float).tiny)):
        return None
    return faces, bridge_score


def _normal_candidate(
    points, outer_ids, inner_ids, rotation, max_bridge_length, outer_normals, inner_normals, cosine_limit, state_budget
):
    n, m = len(outer_ids), len(inner_ids)
    if (n + 1) * (m + 1) > state_budget:
        raise ValueError("normal-constrained zipper state budget exceeded")
    outer, inner = points[outer_ids], points[inner_ids]
    if np.linalg.norm(outer[0] - inner[rotation]) > max_bridge_length:
        return None
    scale = max(float(np.linalg.norm(np.ptp(np.vstack((outer, inner)), axis=0))), 1e-150)
    costs = np.full((n + 1, m + 1), np.inf)
    choices = np.zeros((n + 1, m + 1), dtype=np.uint8)
    costs[0, 0] = 0.0

    def add(i, j, ni, nj, face, boundary_normal, step, bridge):
        start_bridge = float(np.linalg.norm(outer[i % n] - inner[(j + rotation) % m]))
        if max(start_bridge, bridge) > max_bridge_length:
            return
        tri = points[np.asarray(face, dtype=np.int64)]
        normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        length = float(np.linalg.norm(normal))
        if length <= scale * scale * np.finfo(float).eps * 128:
            return
        dot = float((normal / length) @ boundary_normal)
        if dot < cosine_limit:
            return
        cost = bridge * bridge + scale * scale * (1.0 - dot) ** 2
        value = costs[i, j] + cost
        tolerance = max(abs(value), abs(cost), scale * scale) * 1e-13
        if value < costs[ni, nj] - tolerance:
            costs[ni, nj], choices[ni, nj] = value, step

    for i in range(n + 1):
        for j in range(m + 1):
            if not np.isfinite(costs[i, j]):
                continue
            oi, ij = i % n, (j + rotation) % m
            if i < n:
                oj = (i + 1) % n
                bridge = float(np.linalg.norm(outer[oj] - inner[ij]))
                add(i, j, i + 1, j, (outer_ids[oi], outer_ids[oj], inner_ids[ij]), outer_normals[oi], 1, bridge)
            if j < m:
                inn = (j + 1 + rotation) % m
                bridge = float(np.linalg.norm(outer[oi] - inner[inn]))
                add(i, j, i, j + 1, (outer_ids[oi], inner_ids[inn], inner_ids[ij]), inner_normals[ij], 2, bridge)
    if not np.isfinite(costs[n, m]):
        return None
    faces, i, j = [], n, m
    while i or j:
        step = choices[i, j]
        if step == 1:
            oi = (i - 1) % n
            ij = (j + rotation) % m
            faces.append((outer_ids[oi], outer_ids[i % n], inner_ids[ij]))
            i -= 1
        elif step == 2:
            oi = i % n
            ij = (j - 1 + rotation) % m
            inn = (ij + 1) % m
            faces.append((outer_ids[oi], inner_ids[inn], inner_ids[ij]))
            j -= 1
        else:
            return None
    faces.reverse()
    return np.asarray(faces, dtype=np.int64), float(costs[n, m])


# @entry: returns a stitch proposal only; the whole mesh must still pass exact gates.
def stitch_boundary_loops(
    points,
    outer_loop,
    inner_loop,
    *,
    max_bridge_length,
    outer_edge_normals=None,
    inner_edge_normals=None,
    max_normal_angle=20.0,
    max_vertices=512,
    max_dp_states=100000,
):
    """Make an oriented strip without moving either loop or retained source geometry.

    `outer_loop` edges must be directed as the missing strip should traverse the
    fixed source rim. `inner_loop` edges are traversed in reverse by the strip.
    Loop cyclic starts may rotate, but neither input loop is reversed.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("points must be a finite N by 3 array")
    if not np.isfinite(max_bridge_length) or max_bridge_length <= 0 or max_vertices < 3:
        raise ValueError("a positive bridge budget and vertex budget are required")
    coordinate_scale = max(
        float(np.abs(points).max(initial=0.0)), float(np.linalg.norm(np.ptp(points, axis=0))), 1e-150
    )
    tolerance = coordinate_scale * np.finfo(np.float64).eps * 64
    outer = _validate_loop(points, outer_loop, "outer_loop", tolerance, max_vertices)
    inner = _validate_loop(points, inner_loop, "inner_loop", tolerance, max_vertices)
    if np.intersect1d(outer[0], inner[0]).size:
        raise ValueError("boundary loops must not share point IDs")
    _check_simple((outer, inner), tolerance)
    normal_enabled = outer_edge_normals is not None or inner_edge_normals is not None
    if normal_enabled:
        if outer_edge_normals is None or inner_edge_normals is None:
            raise ValueError("both source and patch boundary normals are required")
        outer_normals = np.array(outer_edge_normals, dtype=float, copy=True)
        inner_normals = np.array(inner_edge_normals, dtype=float, copy=True)
        if (
            outer_normals.shape != (len(outer[0]), 3)
            or inner_normals.shape != (len(inner[0]), 3)
            or not np.isfinite(outer_normals).all()
            or not np.isfinite(inner_normals).all()
            or not np.isfinite(max_normal_angle)
            or not 0.0 <= max_normal_angle < 90.0
        ):
            raise ValueError("boundary normals or maximum angle are invalid")
        for normals in (outer_normals, inner_normals):
            lengths = np.linalg.norm(normals, axis=1)
            if np.any(lengths <= 0.0):
                raise ValueError("boundary normals must be nonzero")
            normals /= lengths[:, None]
        limit = float(np.cos(np.deg2rad(max_normal_angle)))
        distances = np.linalg.norm(outer[1][:, None, :] - inner[1][None, :, :], axis=2)
        anchors = np.argsort(distances[0], kind="stable")[: min(8, len(inner[0]))]
        if (
            len(outer[0]) + len(inner[0]) > max_dp_states
            or len(outer[0]) * len(inner[0]) * len(anchors) > max_dp_states * 8
        ):
            raise ValueError("normal-constrained zipper exceeds its state budget")
        candidates = [
            _normal_candidate(
                points,
                outer[0],
                inner[0],
                int(k),
                max_bridge_length,
                outer_normals,
                inner_normals,
                limit,
                max_dp_states,
            )
            for k in anchors
        ]
    else:
        candidates = [_candidate(points, outer[0], inner[0], k, max_bridge_length) for k in range(len(inner[0]))]
    valid = [(score, faces) for candidate in candidates if candidate is not None for faces, score in [candidate]]
    if not valid:
        budgets = "boundary-normal, bridge and triangle-area" if normal_enabled else "bridge and triangle-area"
        raise ValueError(f"bounded oriented zipper search found no strip satisfying the {budgets} budgets")
    valid.sort(key=lambda item: item[0])
    scale = float(np.linalg.norm(np.ptp(points[np.unique(np.r_[outer[0], inner[0]])], axis=0)))
    alignment_tolerance = max(valid[0][0] * 1e-10, scale * scale * np.finfo(float).eps * 128)
    tied = [candidate for score, candidate in valid if score - valid[0][0] <= alignment_tolerance]

    def oriented_key(candidate):
        rotations = (candidate.argmin(axis=1)[:, None] + np.arange(3)) % 3
        canonical = np.take_along_axis(candidate, rotations, axis=1)
        ordered = canonical[np.lexsort((canonical[:, 2], canonical[:, 1], canonical[:, 0]))]
        return tuple(ordered.ravel())

    # Symmetric rings may have equally good triangulations of the same specified
    # annulus. Pick an oriented canonical triangulation without changing the pairing.
    faces = min(tied, key=oriented_key)
    for left, right in zip(outer[0], np.roll(outer[0], -1)):
        if not any(
            (int(left), int(right)) == (int(a), int(b)) for face in faces for a, b in zip(face, np.roll(face, -1))
        ):
            raise ValueError("zipper does not preserve the directed source rim")
    for left, right in zip(inner[0], np.roll(inner[0], -1)):
        if not any(
            (int(right), int(left)) == (int(a), int(b)) for face in faces for a, b in zip(face, np.roll(face, -1))
        ):
            raise ValueError("zipper does not oppose the patch boundary direction")
    return faces
