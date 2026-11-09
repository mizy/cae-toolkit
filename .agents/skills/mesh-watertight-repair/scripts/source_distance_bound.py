"""Conservative candidate-to-source bounds using face lineage."""
from __future__ import annotations

import math
import sys
from fractions import Fraction

import igl
import numpy as np
import pyvista as pv
import trimesh

from mesh_io import triangle_faces
from mesh_metrics import triangle_area_error_bound


def _point_segment_distance2(point, start, end):
    edge = tuple(right - left for left, right in zip(start, end))
    offset = tuple(value - left for value, left in zip(point, start))
    length2 = sum(value * value for value in edge)
    if length2 == 0:
        return sum(value * value for value in offset)
    projection = sum(value * direction for value, direction in zip(offset, edge))
    if projection <= 0:
        return sum(value * value for value in offset)
    if projection >= length2:
        tail = tuple(value - right for value, right in zip(point, end))
        return sum(value * value for value in tail)
    return sum(value * value for value in offset) - projection * projection / length2


def _point_triangle_distance2(point, triangle):
    """Return exact squared distance for the input binary64 coordinates."""
    point = tuple(Fraction.from_float(float(value)) for value in point)
    a, b, c = (
        tuple(Fraction.from_float(float(value)) for value in vertex)
        for vertex in triangle
    )
    ab = tuple(right - left for left, right in zip(a, b))
    ac = tuple(right - left for left, right in zip(a, c))
    ap = tuple(value - left for value, left in zip(point, a))
    dot = lambda left, right: sum(x * y for x, y in zip(left, right))
    d00, d01, d11 = dot(ab, ab), dot(ab, ac), dot(ac, ac)
    denominator = d00 * d11 - d01 * d01
    if denominator == 0:
        raise ValueError("source birth triangle is exactly degenerate")
    d20, d21 = dot(ap, ab), dot(ap, ac)
    u = d11 * d20 - d01 * d21
    v = d00 * d21 - d01 * d20
    if u >= 0 and v >= 0 and u + v <= denominator:
        nx = ab[1] * ac[2] - ab[2] * ac[1]
        ny = ab[2] * ac[0] - ab[0] * ac[2]
        nz = ab[0] * ac[1] - ab[1] * ac[0]
        signed = ap[0] * nx + ap[1] * ny + ap[2] * nz
        return signed * signed / (nx * nx + ny * ny + nz * nz)
    return min(
        _point_segment_distance2(point, a, b),
        _point_segment_distance2(point, b, c),
        _point_segment_distance2(point, c, a),
    )


def _sqrt_upper(value):
    """Round an exact rational square root upward to binary64."""
    if value < 0:
        raise ValueError("exact squared distance must be nonnegative")
    if value == 0:
        return 0.0
    numerator, denominator = value.numerator, value.denominator
    exponent = numerator.bit_length() - denominator.bit_length()
    if exponent >= 0:
        if numerator < denominator << exponent:
            exponent -= 1
    elif numerator << -exponent < denominator:
        exponent -= 1
    mantissa = value / (1 << exponent) if exponent >= 0 else value * (1 << -exponent)
    odd = exponent % 2
    if odd:
        mantissa *= 2
    try:
        result = math.ldexp(math.sqrt(float(mantissa)), exponent // 2)
    except OverflowError:
        return math.inf
    if not math.isfinite(result):
        return math.inf
    if result == 0.0:
        result = math.nextafter(0.0, math.inf)
    while Fraction.from_float(result) ** 2 < value:
        result = math.nextafter(result, math.inf)
    return result


def candidate_to_source_bound(source: pv.PolyData, candidate: pv.PolyData, *, max_error: float) -> float:
    """Bound output faces by verified support points in their source triangles.

    The quick path uses trimesh only to propose a point. Exact rational sign
    checks make its barycentric weights a convex combination of the source
    triangle vertices, so the support itself is proven to lie on that triangle.
    The distance estimate uses a 64-epsilon allowance under IEEE binary64
    arithmetic. Subnormal scales, unstable triangles, invalid support points,
    and faces whose upper bound exceeds the normalized error use exact rational
    point-to-triangle distances.
    """
    if not np.isfinite(max_error) or max_error <= 0:
        raise ValueError("max_error must be finite and positive in normalized coordinates")
    points = np.asarray(candidate.points, dtype=np.float64)
    faces = triangle_faces(candidate)
    if not len(faces):
        return 0.0
    triangles = points[faces]
    source_points = np.asarray(source.points, dtype=np.float64)
    source_faces = triangle_faces(source)
    rows = np.asarray(candidate.cell_data["source_facet_id"], dtype=np.int64).copy()
    if rows.shape != (len(faces),) or np.any((rows < -1) | (rows >= len(source_faces))):
        raise ValueError("candidate source-face lineage is invalid")
    unknown = rows < 0
    if np.any(unknown):
        source_triangles = source_points[source_faces]
        source_areas = np.linalg.norm(np.cross(source_triangles[:, 1] - source_triangles[:, 0],
                                               source_triangles[:, 2] - source_triangles[:, 0]), axis=1) * .5
        valid_rows = np.flatnonzero(source_areas > triangle_area_error_bound(source_points, source_faces))
        if not len(valid_rows):
            raise ValueError("source has no reliable triangles for distance lookup")
        _, nearest, _ = igl.point_mesh_squared_distance(
            triangles[unknown].mean(axis=1), source_points, source_faces[valid_rows]
        )
        nearest = np.asarray(nearest, dtype=np.int64).reshape(-1)
        if len(nearest) != np.count_nonzero(unknown) or np.any((nearest < 0) | (nearest >= len(valid_rows))):
            raise ValueError("candidate source-face distance lookup is invalid")
        rows[unknown] = valid_rows[nearest]
    if np.any((rows < 0) | (rows >= len(source_faces))):
        raise ValueError("candidate source-face distance lookup is invalid")

    supports = source_points[source_faces[rows]]
    queries = triangles.reshape(-1, 3)
    support_vertices = np.repeat(supports, 3, axis=0)
    quick_upper = np.full(len(queries), np.inf, dtype=np.float64)
    coordinate_scale = max(
        float(np.linalg.norm(np.ptp(np.vstack((source_points, points)), axis=0))),
        float(np.abs(source_points).max(initial=0.0)),
        float(np.abs(points).max(initial=0.0)),
    )
    epsilon = np.finfo(np.float64).eps
    roundoff = coordinate_scale * epsilon * 64
    if coordinate_scale >= sys.float_info.min * 1024 and roundoff > 0:
        try:
            nearest = trimesh.triangles.closest_point(support_vertices, queries)
            if np.isfinite(nearest).all():
                a, b, c = (support_vertices[:, index] for index in range(3))
                ab, ac = b - a, c - a
                d00 = np.einsum("ij,ij->i", ab, ab)
                d01 = np.einsum("ij,ij->i", ab, ac)
                d11 = np.einsum("ij,ij->i", ac, ac)
                denominator = d00 * d11 - d01 * d01
                ratio = denominator / np.maximum(d00 * d11, np.finfo(float).tiny)
                stable = np.isfinite(ratio) & (ratio > 1e-10)
                offset = nearest - a
                d20 = np.einsum("ij,ij->i", offset, ab)
                d21 = np.einsum("ij,ij->i", offset, ac)
                u = np.zeros(len(queries), dtype=np.float64)
                v = np.zeros(len(queries), dtype=np.float64)
                u[stable] = (d11[stable] * d20[stable] - d01[stable] * d21[stable]) / denominator[stable]
                v[stable] = (d00[stable] * d21[stable] - d01[stable] * d20[stable]) / denominator[stable]
                for index in np.flatnonzero(stable):
                    uf = Fraction.from_float(float(u[index]))
                    vf = Fraction.from_float(float(v[index]))
                    if uf < 0 or vf < 0 or uf + vf > 1:
                        stable[index] = False
                if np.any(stable):
                    support_point = a[stable] + u[stable, None] * ab[stable] + v[stable, None] * ac[stable]
                    distance = np.linalg.norm(queries[stable] - support_point, axis=1)
                    quick_upper[stable] = distance + roundoff
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            pass

    quick_faces = quick_upper.reshape(-1, 3).max(axis=1)
    exact_faces = np.flatnonzero(~np.isfinite(quick_faces) | (quick_faces > max_error))
    fast_faces = np.ones(len(quick_faces), dtype=bool)
    fast_faces[exact_faces] = False
    bound = float(quick_faces[fast_faces].max(initial=0.0))
    for face_id in exact_faces:
        squared = [
            _point_triangle_distance2(triangles[face_id, corner], supports[face_id])
            for corner in range(3)
        ]
        bound = max(bound, _sqrt_upper(max(squared)))
    return bound
