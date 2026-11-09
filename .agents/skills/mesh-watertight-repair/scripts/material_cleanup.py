"""Material Boolean cleanup and geometry gates shared by whole and local repairs."""
from __future__ import annotations

import igl
import numpy as np
import pyvista as pv
import trimesh
from scipy.spatial import ConvexHull

from mesh_io import triangle_faces
from mesh_metrics import edge_topology, mesh_report
from mesh_topology_repair import collapse_short_manifold_edges
from print_geometry import orient_material
from source_projected_validation import exact_self_intersection_report


def _surface(points, faces, source_rows=None):
    mesh = pv.PolyData(points, np.column_stack((np.full(len(faces), 3), faces)).ravel())
    if source_rows is not None:
        mesh.cell_data["source_facet_id"] = np.asarray(source_rows, dtype=np.int64)
    return mesh

def _union(points, faces, rows, extra=None):
    from igl.copyleft import cgal
    if extra is None:
        other_points, other_faces = np.empty((0, 3)), np.empty((0, 3), dtype=np.int64)
        births = rows
    else:
        other_points, other_faces = extra
        births = np.r_[rows, np.full(len(other_faces), -1, dtype=np.int64)]
    p, f, j = cgal.mesh_boolean(points, faces, other_points, other_faces, "union")
    p, f, j = np.asarray(p), np.asarray(f, dtype=np.int64), np.asarray(j, dtype=np.int64)
    if j.shape != (len(f),) or np.any((j < 0) | (j >= len(births))):
        raise ValueError("CGAL returned invalid source-face lineage")
    return p, f, births[j]

def _contact_neighborhood(points, faces, radius):
    from mesh_vertex_topology import non_manifold_vertex_ids
    _, edges = edge_topology(faces)
    bad_edges = [edge for edge, incident in edges.items() if len(incident) > 2]
    supports = [points[list(edge)] for edge in bad_edges]
    _, first, counts = np.unique(points, axis=0, return_index=True, return_counts=True)
    covered = {int(vertex) for edge in bad_edges for vertex in edge}
    vertices = set(non_manifold_vertex_ids(faces, edges)) - covered
    contact_points = np.vstack((points[sorted(vertices)], points[first[counts > 1]]))
    supports.extend(point.reshape(1, 3) for point in np.unique(contact_points, axis=0))
    if not supports:
        return None
    blocks, triangles, offset = [], [], 0
    kernel = np.vstack((np.eye(3), -np.eye(3))) * radius
    for support in supports:
        local = np.unique((support[:, None, :] + kernel[None, :, :]).reshape(-1, 3), axis=0)
        hull = ConvexHull(local)
        f = np.asarray(hull.simplices).copy()
        n = np.cross(local[f[:, 1]]-local[f[:, 0]], local[f[:, 2]]-local[f[:, 0]])
        flip = np.einsum("ij,ij->i", n, hull.equations[:, :3]) < 0
        f[flip] = f[flip, ::-1]
        blocks.append(local)
        triangles.append(f + offset)
        offset += len(local)
    return np.vstack(blocks), np.vstack(triangles)

def _compact(points, faces):
    used, inverse = np.unique(faces, return_inverse=True)
    return points[used], inverse.reshape(-1, 3)

def _cleanup(candidate, source_points, error, progress=None):
    points = np.asarray(candidate.points, dtype=float)
    faces = triangle_faces(candidate)
    rows = np.asarray(candidate.cell_data["source_facet_id"], dtype=np.int64)
    actions = []
    for iteration in range(3):
        if progress:
            progress("source_material_cleanup_exact_check")
        exact = exact_self_intersection_report(points, faces, first_only=True)
        if progress:
            progress("source_material_cleanup_contacts")
        contacts = _contact_neighborhood(points, faces, error / 32)
        if contacts is not None:
            if progress:
                progress("source_material_cleanup_contact_union")
            points, faces, rows = _union(points, faces, rows, contacts)
            actions.append({"round": iteration, "operator": "bounded_contact_neighborhood", "radius": error / 32})
        elif not exact.get("passed"):
            if progress:
                progress("source_material_cleanup_intersection_union")
            points, faces, rows = _union(points, faces, rows)
            actions.append({"round": iteration, "operator": "exact_material_union"})
        topology, _ = edge_topology(faces)
        if all(topology[key] == 0 for key in ("boundary_edges", "non_manifold_edges", "non_manifold_vertices")):
            if progress:
                progress("source_material_cleanup_short_edges")
            # Pin true source coordinates, not previously generated Boolean vertices.
            from scipy.spatial import cKDTree
            _, nearest = cKDTree(source_points).query(points)
            pinned = np.all(points == source_points[nearest], axis=1)
            points, faces, kept, collapsed = collapse_short_manifold_edges(
                points, faces, pinned_points=pinned, max_edge_length=error / 8,
            )
            rows = rows[kept]
            actions.extend(collapsed)
        points, faces = _compact(points, faces)
        checked = mesh_report(points, faces)
        if (checked["topology"]["closed_manifold"] and checked["quality"]["degenerate_faces"] == 0
                and exact_self_intersection_report(points, faces, first_only=True).get("passed")):
            break
    material, _ = orient_material(trimesh.Trimesh(points, faces, process=False))
    return _surface(np.asarray(material.vertices), np.asarray(material.faces), rows), actions

def _geometry_gate(mesh):
    return geometry_gate_arrays(np.asarray(mesh.points), triangle_faces(mesh))


def geometry_gate_arrays(points, faces):
    """Validate native arrays without allocating a temporary VTK object per component."""
    metrics = mesh_report(points, faces)
    topology = metrics["topology"]
    closed = all(topology[key] == 0 for key in (
        "boundary_edges", "non_manifold_edges", "non_manifold_vertices", "inconsistent_winding_edges"
    ))
    basic = closed and metrics["quality"]["degenerate_faces"] == 0 and metrics["volume"]["reliable"]
    if not basic:
        return {"passed": False, "metrics": metrics, "material_orientation": False,
                "material_orientation_status": "not_checked",
                "exact": {"status": "not_checked", "passed": None, "failure_reason": "basic geometry failed"}}
    exact = exact_self_intersection_report(points, faces, first_only=True)
    exact["first_only"] = True
    exact["intersection_pairs_is_lower_bound"] = exact.get("passed") is False and exact.get("status") == "computed"
    if exact.get("passed") is not True:
        return {"passed": False, "metrics": metrics, "material_orientation": False,
                "material_orientation_status": "not_checked",
                "exact": {k:v for k,v in exact.items() if k != "reported_pairs"}}
    oriented_mesh, _ = orient_material(trimesh.Trimesh(points, faces, process=False))
    oriented_faces = np.asarray(oriented_mesh.faces)
    oriented = oriented_faces.shape == faces.shape and bool(np.all(np.logical_or.reduce([
        np.all(np.roll(oriented_faces, shift, axis=1) == faces, axis=1) for shift in range(3)
    ])))
    return {"passed": bool(oriented), "metrics": metrics, "material_orientation": bool(oriented),
            "material_orientation_status": "checked", "exact": {k:v for k,v in exact.items() if k != "reported_pairs"}}

def _candidate_distance(source, candidate):
    points, faces = np.asarray(candidate.points), triangle_faces(candidate)
    weights = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1],
                          [.5, .5, 0], [.5, 0, .5], [0, .5, .5], [1/3, 1/3, 1/3]])
    samples = np.einsum("sj,fjk->fsk", weights, points[faces]).reshape(-1, 3)
    squared, _, _ = igl.point_mesh_squared_distance(samples, np.asarray(source.points), triangle_faces(source))
    if np.asarray(squared).size != len(samples) or not np.isfinite(squared).all() or np.any(squared < 0):
        raise ValueError("candidate distance query returned invalid values")
    return float(np.sqrt(np.maximum(squared, 0)).max(initial=0.0))
