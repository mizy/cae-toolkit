"""Cut an exact source-surface collar for a fixed-boundary local replacement."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
from local_defect_regions import _boundary
from mesh_metrics import edge_topology
from patch_zipper import stitch_boundary_loops


# @entry: local SDF construction uses this collar before connecting its inset zero surface.
def build_source_collar(points, faces, region_face_rows, *, fraction=0.25):
    """Keep the source geometry at the cut seam and expose a new inner cut loop.

    A piecewise-linear scalar is zero at the source cut rim and one at interior
    source vertices. Clipping source triangles at `fraction` retains a collar
    whose faces remain in their original triangle planes. Edge intersections
    are shared by source edge ID. No original vertex is moved.
    """
    if not np.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError("collar fraction must lie strictly between zero and one")
    points, faces = np.asarray(points, dtype=float), np.asarray(faces, dtype=np.int64)
    rows = np.asarray(region_face_rows, dtype=np.int64)
    patch = faces[rows]
    _, patch_edges = edge_topology(patch)
    _, loops, boundary = _boundary(patch, np.arange(len(patch)), patch_edges)
    if boundary["status"] != "closed_loops" or len(loops) != 1:
        raise ValueError("source collar requires one simple source cut loop")
    rim = set(loops[0][:-1])
    values = np.ones(len(points))
    values[list(rim)] = 0.0
    new_points, intersections, collar, births = [], {}, [], []

    def crossing(left, right):
        edge = tuple(sorted((int(left), int(right))))
        if edge not in intersections:
            weight = (fraction - values[left]) / (values[right] - values[left])
            intersections[edge] = len(points) + len(new_points)
            new_points.append(points[left] + weight * (points[right] - points[left]))
        return intersections[edge]

    for row, triangle in zip(rows, patch):
        polygon = []
        for left, right in zip(triangle, np.roll(triangle, -1)):
            inside_left, inside_right = (
                values[left] <= fraction,
                values[right] <= fraction,
            )
            if inside_left:
                polygon.append(int(left))
            if inside_left != inside_right:
                polygon.append(crossing(left, right))
        for index in range(1, len(polygon) - 1):
            collar.append([polygon[0], polygon[index], polygon[index + 1]])
            births.append(int(row))
    if not new_points or not collar:
        raise ValueError("source region has no interior collar contour")
    new_points, collar = np.asarray(new_points), np.asarray(collar, dtype=np.int64)
    _, edges = edge_topology(collar)
    _, collar_loops, boundary = _boundary(collar, np.arange(len(collar)), edges)
    inner = [loop for loop in collar_loops if all(v >= len(points) for v in loop)]
    if boundary["status"] != "closed_loops" or len(collar_loops) != 2 or len(inner) != 1:
        raise ValueError("source collar does not have one unambiguous inner loop")
    combined = np.vstack((points, new_points))
    tri = combined[collar]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    if np.any(np.linalg.norm(normals, axis=1) == 0):
        raise ValueError("source collar contains a degenerate triangle")
    inner_loop = list(reversed(inner[0]))
    boundary_normals = []
    for left, right in pairwise(inner_loop):
        row = edges[tuple(sorted((left, right)))][0]
        boundary_normals.append(normals[row] / np.linalg.norm(normals[row]))
    return (
        new_points,
        collar,
        inner_loop,
        np.asarray(boundary_normals),
        np.asarray(births),
    )


def _patch_boundary_normals(points, faces, loop):
    _, edges = edge_topology(faces)
    scale = max(float(np.linalg.norm(np.ptp(points, axis=0))), 1e-150)
    min_area = max(scale * scale * np.finfo(float).eps * 128, np.finfo(float).tiny)
    normals, face_rows = [], []
    for left, right in pairwise(loop):
        adjacent = edges.get(tuple(sorted((int(left), int(right)))), [])
        if len(adjacent) != 1:
            raise ValueError("patch boundary edge requires one adjacent face")
        row = adjacent[0]
        face = faces[row]
        if (int(left), int(right)) not in {(int(a), int(b)) for a, b in zip(face, np.roll(face, -1))}:
            raise ValueError("patch boundary edge direction does not match its face")
        triangle = points[face]
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        length = float(np.linalg.norm(normal))
        if not np.isfinite(length) or length <= min_area:
            raise ValueError("patch boundary face normal is unreliable")
        unit = normal / length
        if not np.isfinite(unit).all() or not np.isclose(np.linalg.norm(unit), 1.0, atol=1e-8):
            raise ValueError("patch boundary face normal is unreliable")
        normals.append(unit)
        face_rows.append(int(row))
    return np.asarray(normals, dtype=np.float64), face_rows


# @entry: preserve the source rim, then connect its exact collar to the SDF patch.
def connect_source_collar(
    points,
    faces,
    source_rows,
    source_loop,
    patch_points,
    patch_faces,
    patch_loop,
    *,
    max_bridge_length,
    fraction=0.25,
):
    source_points = np.asarray(points, dtype=np.float64)
    source_faces = np.asarray(faces, dtype=np.int64)
    patch_points = np.asarray(patch_points, dtype=np.float64)
    patch_faces = np.asarray(patch_faces, dtype=np.int64)
    collar_points, collar_faces, inner_loop, outer_normals, births = build_source_collar(
        source_points, source_faces, source_rows, fraction=fraction
    )
    _, collar_edges = edge_topology(collar_faces)
    source_cut = {tuple(sorted((int(a), int(b)))) for a, b in pairwise(source_loop)}
    collar_cut = {
        edge for edge, incident in collar_edges.items() if len(incident) == 1 and max(edge) < len(source_points)
    }
    if collar_cut != source_cut:
        raise ValueError("source collar does not preserve the source cut loop")
    outer_normals = np.asarray(outer_normals, dtype=np.float64)
    if (
        outer_normals.shape != (len(inner_loop) - 1, 3)
        or not np.isfinite(outer_normals).all()
        or not np.allclose(np.linalg.norm(outer_normals, axis=1), 1.0, atol=1e-8)
    ):
        raise ValueError("source collar inner-loop normals are unreliable")
    inner_normals, inner_face_rows = _patch_boundary_normals(patch_points, patch_faces, patch_loop)
    if len(inner_loop) - 1 > 512 or len(patch_loop) - 1 > 512:
        raise ValueError("zipper boundary vertex budget exceeded")

    combined = np.vstack((source_points, collar_points, patch_points))
    patch_loop_global = np.asarray(patch_loop, dtype=np.int64) + len(source_points) + len(collar_points)
    zipper = stitch_boundary_loops(
        combined,
        inner_loop,
        patch_loop_global,
        max_bridge_length=max_bridge_length,
        outer_edge_normals=outer_normals,
        inner_edge_normals=inner_normals,
        max_normal_angle=20.0,
    )
    return (
        collar_points,
        collar_faces,
        {
            "collar_faces_count": len(collar_faces),
            "collar_source_face_rows": births.astype(int).tolist(),
            "collar_inner_loop": inner_loop,
            "collar_inner_edge_normals": outer_normals.tolist(),
            "patch_boundary_edge_normals": inner_normals.tolist(),
            "patch_boundary_edge_faces": inner_face_rows,
            "zipper_faces": np.vstack((collar_faces, zipper)).tolist(),
            "zipper": {
                "status": "ready",
                "max_bridge_length": float(max_bridge_length),
                "max_normal_angle": 20.0,
                "outer_loop": "source_collar_inner_loop",
                "faces": len(zipper),
            },
        },
    )
