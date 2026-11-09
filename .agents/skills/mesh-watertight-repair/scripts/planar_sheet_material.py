"""Construct one material wall from explicitly reviewed near-coplanar open sheets."""
from __future__ import annotations

import numpy as np

from mesh_metrics import edge_topology, face_component_labels


# @entry: constructs a candidate; caller owns semantic, fidelity and geometry gates.
def build_planar_sheet_material(points, faces, *, wall_depth):
    import manifold3d

    points = np.asarray(points, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if not 0 < len(faces) <= 10000 or not np.isfinite(wall_depth) or wall_depth <= 0:
        raise ValueError("planar wall requires a positive depth and 1 to 10000 faces")
    topology, edges = edge_topology(faces)
    if topology["non_manifold_edges"] or topology["non_manifold_vertices"]:
        raise ValueError("planar wall input must contain manifold sheets")
    labels = face_component_labels(len(faces), edges)
    open_labels = {int(labels[incident[0]]) for incident in edges.values() if len(incident) == 1}
    if open_labels != set(map(int, np.unique(labels))):
        raise ValueError("planar wall requires only open sheet components")

    vertices = points[np.unique(faces)]
    origin = vertices.mean(axis=0)
    _, singular, basis = np.linalg.svd(vertices - origin, full_matrices=False)
    if len(singular) < 3 or singular[1] <= singular[0] * 1e-10:
        raise ValueError("sheet plane is degenerate")
    local = (points - origin) @ basis.T
    deviation = float(np.abs((vertices - origin) @ basis[2]).max())
    if deviation > wall_depth / 4:
        raise ValueError("sheet plane deviation exceeds one quarter of wall depth")
    tri = points[faces]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    if np.any(lengths == 0) or np.any(np.abs(normals @ basis[2]) / lengths < np.cos(np.radians(3))):
        raise ValueError("sheet face normals do not agree with the common plane")
    # Match the canonical material-side policy used by the source-sheet worker.
    normal = basis[2].copy()
    if normal[np.argmax(np.abs(normal))] < 0:
        normal *= -1
    if np.dot(np.cross(basis[0], basis[1]), normal) < 0:
        basis[1] *= -1
    basis[2] = normal
    local = (points - origin) @ basis.T
    contours = []
    for face in faces:
        xy = local[face, :2]
        a, b = xy[1] - xy[0], xy[2] - xy[0]
        signed_area = a[0] * b[1] - a[1] * b[0]
        if signed_area == 0:
            raise ValueError("sheet face becomes degenerate in the common plane")
        contours.append(xy if signed_area > 0 else xy[::-1])
    section = manifold3d.CrossSection(contours, manifold3d.FillRule.Positive)
    if section.is_empty():
        raise ValueError("planar union did not produce a cross section")
    solid = section.extrude(wall_depth)
    mesh = solid.to_mesh64()
    result_points = np.asarray(mesh.vert_properties, dtype=np.float64)[:, :3].copy()
    result_faces = np.asarray(mesh.tri_verts, dtype=np.int64)
    result_points[:, 2] -= wall_depth
    result_points = result_points @ basis + origin
    if not len(result_faces):
        raise ValueError("planar wall extrusion produced no faces")
    return result_points, result_faces, np.full(len(result_faces), -1, dtype=np.int64), {
        "method": "project_near_coplanar_open_sheets_union_2d_then_extrude",
        "union_backend": "planar", "source_faces": len(faces),
        "source_components": len(open_labels), "projection_max_displacement": deviation,
        "projection_displacement_limit": float(wall_depth / 4), "wall_depth": float(wall_depth),
        "output_faces": len(result_faces), "source_face_map_complete": False,
        "source_lineage": "generated geometry; no exact per-face source mapping",
        "acceptance": "not performed by this geometry builder",
    }
