"""Fit a reviewed boundary's neighboring surface and make a candidate patch."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyvista as pv
from scipy.interpolate import RBFInterpolator

from mesh_io import read_surface, triangle_faces
from mesh_metrics import edge_topology
from patch_zipper import _segment_distance
from repair_inventory import build_inventory


def fit_vertices(points, faces, region, rings):
    loop = np.asarray(region["ordered_vertex_ids"], dtype=np.int64)
    if not region.get("simple_closed_loop") or len(loop) < 3 or len(loop) > 512:
        raise ValueError("one simple boundary loop with at most 512 vertices is required")
    if rings < 1:
        raise ValueError("fit rings must be positive")
    _, edges = edge_topology(faces)
    selected = set(map(int, region["face_ids"]))
    frontier = selected.copy()
    for _ in range(rings - 1):
        neighbors = set()
        for row in frontier:
            face = faces[row]
            for a, b in zip(face, np.roll(face, -1)):
                neighbors.update(edges[tuple(sorted((int(a), int(b))))])
        frontier = neighbors - selected
        selected.update(frontier)
    return loop, np.unique(faces[sorted(selected)]), len(selected)


def build_patch(points, faces, region, *, rings, max_fit_error, max_offset, subdivisions,
                mesh_method="gmsh", mesh_size=None):
    """Build a quadratic height-field patch; fixed boundary remains source geometry."""
    points = np.asarray(points, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if (subdivisions < 0 or subdivisions > 4 or not np.isfinite(max_fit_error) or max_fit_error <= 0
            or not np.isfinite(max_offset) or max_offset <= 0):
        raise ValueError("subdivisions, fit-error or patch-offset budget are invalid")
    loop, fit_ids, fit_faces = fit_vertices(points, faces, region, rings)
    boundary = points[loop]
    center = boundary.mean(axis=0)
    _, singular, basis = np.linalg.svd(boundary - center, full_matrices=False)
    if singular[1] <= singular[0] * 1e-10:
        raise ValueError("boundary projection is degenerate")
    scale = float(np.linalg.norm(np.ptp(boundary, axis=0)))
    local = (points - center) @ basis.T / scale
    uv = local[:, :2]

    def features(x):
        u, v = x[:, 0], x[:, 1]
        return np.column_stack((np.ones(len(x)), u, v, u*u, u*v, v*v))

    coeff, _, rank, _ = np.linalg.lstsq(features(uv[fit_ids]), local[fit_ids, 2], rcond=None)
    if rank < 6:
        raise ValueError("neighboring surface cannot determine a quadratic model")
    residual = (features(uv[fit_ids]) @ coeff - local[fit_ids, 2]) * scale
    max_residual = float(np.max(np.abs(residual)))
    if not np.isfinite(max_residual) or max_residual > max_fit_error:
        raise ValueError(f"surface fit residual {max_residual:g} exceeds budget {max_fit_error:g}")
    boundary_uv = uv[loop]
    projected = np.column_stack((boundary_uv, np.zeros(len(loop))))
    for i in range(len(loop)):
        for j in range(i + 2, len(loop)):
            if i == 0 and j == len(loop) - 1:
                continue
            if _segment_distance(projected[i], projected[(i+1) % len(loop)],
                                 projected[j], projected[(j+1) % len(loop)]) <= 1e-10:
                raise ValueError("boundary projection is self-crossing or ambiguous")
    # Interpolate the mismatch at the rim, keeping all original boundary coordinates exactly.
    delta = local[loop, 2] - features(boundary_uv) @ coeff
    try:
        correction = RBFInterpolator(boundary_uv, delta, kernel="thin_plate_spline", degree=1)
    except np.linalg.LinAlgError as exc:
        raise ValueError("boundary projection cannot support interpolation") from exc
    if mesh_method == "gmsh":
        if mesh_size is None:
            size = float(np.median(np.linalg.norm(np.roll(boundary_uv, -1, axis=0) - boundary_uv, axis=1)))
        elif np.isfinite(mesh_size) and mesh_size > 0:
            size = float(mesh_size / scale)
        else:
            raise ValueError("mesh size must be finite and positive in source units")
        local_uv, triangles = mesh_projected_boundary(boundary_uv, size)
        sample = local_uv[len(loop):]
        z = features(sample) @ coeff + correction(sample)
        interior = center + scale * (sample[:, :1] * basis[0] + sample[:, 1:] * basis[1] + z[:, None] * basis[2])
        vertices = np.vstack((boundary, interior))
    elif mesh_method == "centroid":
        polygon = pv.PolyData(projected, np.r_[len(loop), np.arange(len(loop))]).triangulate()
        triangles = triangle_faces(polygon)
        if len(triangles) != len(loop) - 2:
            raise ValueError("boundary projection could not be triangulated as a simple polygon")
        vertices = [p.copy() for p in boundary]
        local_uv = [p.copy() for p in boundary_uv]
        for _ in range(subdivisions):
            refined = []
            for a, b, c in triangles:
                q = (local_uv[a] + local_uv[b] + local_uv[c]) / 3
                z = (features(q[None, :]) @ coeff + correction(q[None, :])).item()
                index = len(vertices)
                vertices.append(center + scale * (q[0] * basis[0] + q[1] * basis[1] + z * basis[2]))
                local_uv.append(q)
                refined.extend(((a, b, index), (b, c, index), (c, a, index)))
            triangles = np.asarray(refined, dtype=np.int64)
        vertices = np.asarray(vertices)
    else:
        raise ValueError("mesh method must be gmsh or centroid")
    offset = float(np.max(np.abs((vertices - center) @ basis[2])))
    if not np.isfinite(offset) or offset > max_offset:
        raise ValueError(f"modeled patch offset {offset:g} exceeds budget {max_offset:g}")
    return vertices, triangles, {
        "method": "quadratic_height_field_with_boundary_residual_interpolation",
        "boundary_vertices": len(loop), "fit_faces": fit_faces, "fit_vertices": len(fit_ids),
        "fit_max_residual": max_residual, "fit_rms_residual": float(np.sqrt(np.mean(residual**2))),
        "fit_error_budget": float(max_fit_error), "mesh_method": mesh_method,
        "mesh_size": float(size * scale) if mesh_method == "gmsh" else None,
        "subdivisions": subdivisions if mesh_method == "centroid" else None,
        "max_patch_offset_from_boundary_plane": offset, "max_patch_offset_budget": float(max_offset),
        "source_boundary_vertex_ids": loop.tolist(),
        "patch_bounds": [vertices.min(axis=0).tolist(), vertices.max(axis=0).tolist()],
        "status": "candidate_only",
        "limitations": ["Single-valued height field in the boundary projection; no overhang or multi-sheet reconstruction.",
                        "Fit residual is measured only on selected source vertices; missing shape and functional intent are unproven."],
    }


def mesh_projected_boundary(boundary, size):
    """Mesh the projected polygon without splitting or moving its boundary edges."""
    import gmsh

    extents = np.ptp(boundary, axis=0)
    minimum_size = np.sqrt(2 * float(np.prod(extents)) / 100000)
    if not np.isfinite(size) or size <= 0 or size < minimum_size:
        raise ValueError("Gmsh target size exceeds the estimated triangle budget")
    if gmsh.isInitialized():
        raise RuntimeError("Gmsh is already initialized; run patch meshing in a separate process")
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Mesh.MeshSizeMax", size)
        gmsh.option.setNumber("Mesh.MeshSizeMin", size * 0.5)
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.option.setNumber("Mesh.ElementOrder", 1)
        gmsh.model.add("modeled_patch")
        points = [gmsh.model.geo.addPoint(float(q[0]), float(q[1]), 0, size) for q in boundary]
        lines = [gmsh.model.geo.addLine(points[i], points[(i + 1) % len(points)]) for i in range(len(points))]
        surface = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop(lines)])
        gmsh.model.geo.synchronize()
        for line in lines:
            gmsh.model.mesh.setTransfiniteCurve(line, 2)
        gmsh.model.mesh.generate(2)
        tags, xyz, _ = gmsh.model.mesh.getNodes()
        nodes = np.asarray(xyz).reshape(-1, 3)[:, :2]
        tags = np.asarray(tags, dtype=np.int64)
        boundary_tags = []
        for point in points:
            point_tags, _, _ = gmsh.model.mesh.getNodes(0, point)
            if len(point_tags) != 1:
                raise ValueError("Gmsh did not preserve one node for each boundary vertex")
            boundary_tags.append(int(point_tags[0]))
        indices = {int(tag): i for i, tag in enumerate(tags)}
        if len(set(boundary_tags)) != len(boundary) or any(tag not in indices for tag in boundary_tags):
            raise ValueError("Gmsh boundary nodes are missing or duplicated")
        boundary_set = set(boundary_tags)
        interior_tags = [int(tag) for tag in tags if int(tag) not in boundary_set]
        order = boundary_tags + interior_tags
        mapping = {tag: i for i, tag in enumerate(order)}
        _, elements = gmsh.model.mesh.getElementsByType(2, surface)
        triangles = np.asarray([mapping[int(tag)] for tag in elements], dtype=np.int64).reshape(-1, 3)
        if not len(triangles) or len(triangles) > 100000:
            raise ValueError("Gmsh patch exceeds the triangle budget or is empty")
        output = np.vstack((boundary, nodes[[indices[tag] for tag in interior_tags]]))
        _, edge_faces = edge_topology(triangles)
        actual = {edge for edge, incident in edge_faces.items() if len(incident) == 1}
        expected = {tuple(sorted((i, (i + 1) % len(boundary)))) for i in range(len(boundary))}
        if actual != expected:
            raise ValueError("Gmsh changed the fixed boundary or introduced extra boundary edges")
        return output, triangles
    finally:
        gmsh.finalize()


def build_sphere_patch(points, faces, region, *, rings, max_fit_error, patch_rings):
    """Fit a reviewed spherical cap and retain the exact source boundary."""
    points = np.asarray(points, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if not np.isfinite(max_fit_error) or max_fit_error <= 0 or not 2 <= patch_rings <= 128:
        raise ValueError("a positive fit budget and 2 to 128 patch rings are required")
    loop, fit_ids, fit_faces = fit_vertices(points, faces, region, rings)
    source = points[fit_ids]
    origin = source.mean(axis=0)
    shifted = source - origin
    matrix = np.column_stack((2 * shifted, np.ones(len(shifted))))
    solution, _, rank, _ = np.linalg.lstsq(matrix, np.sum(shifted * shifted, axis=1), rcond=None)
    if rank < 4 or solution[3] + solution[:3] @ solution[:3] <= 0:
        raise ValueError("neighboring surface does not determine a sphere")
    center = origin + solution[:3]
    radius = float(np.sqrt(solution[3] + solution[:3] @ solution[:3]))
    residual = np.abs(np.linalg.norm(source - center, axis=1) - radius)
    if not np.isfinite(residual).all() or residual.max() > max_fit_error:
        raise ValueError("sphere fit residual exceeds budget")
    vectors = points[loop] - center
    axis = vectors.mean(axis=0)
    axis_length = np.linalg.norm(axis)
    if axis_length < radius * 1e-8:
        raise ValueError("sphere cap direction is ambiguous")
    axis /= axis_length
    axial = vectors @ axis
    radial = vectors - axial[:, None] * axis
    lengths = np.linalg.norm(radial, axis=1)
    if np.any(lengths < radius * 1e-8) or np.any(axial <= 0):
        raise ValueError("sphere boundary must lie on one cap and avoid the pole")
    radial /= lengths[:, None]
    angles = np.arctan2(lengths, axial)
    vertices = list(points[loop].copy())
    loops = [np.arange(len(loop))]
    for i in range(1, patch_rings):
        angle = angles * (1 - i / patch_rings)
        generated = center + radius * (np.cos(angle)[:, None] * axis + np.sin(angle)[:, None] * radial)
        loops.append(np.arange(len(vertices), len(vertices) + len(loop)))
        vertices.extend(generated)
    pole = len(vertices)
    vertices.append(center + radius * axis)
    patch_faces = []
    for outer, inner in zip(loops[:-1], loops[1:]):
        for i in range(len(loop)):
            j = (i + 1) % len(loop)
            patch_faces.extend(((outer[i], outer[j], inner[i]), (outer[j], inner[j], inner[i])))
    for i in range(len(loop)):
        patch_faces.append((loops[-1][i], loops[-1][(i+1) % len(loop)], pole))
    return np.asarray(vertices), np.asarray(patch_faces), {
        "method": "sphere_fitted_to_reviewed_source_neighborhood", "center": center.tolist(),
        "radius": radius, "boundary_vertices": len(loop), "fit_faces": fit_faces,
        "fit_vertices": len(fit_ids), "fit_max_residual": float(residual.max()),
        "fit_rms_residual": float(np.sqrt(np.mean(residual**2))), "fit_error_budget": float(max_fit_error),
        "source_boundary_vertex_ids": loop.tolist(), "patch_rings": patch_rings, "status": "candidate_only",
        "limitations": ["Sphere model and missing cap direction require review; fit cannot prove the absent shape.",
                        "Only a single cap boundary on one side of the fitted sphere is supported."],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--region", required=True, help="Reviewed boundary_loop ID from the source inventory")
    parser.add_argument("--fit-rings", type=int, default=4)
    parser.add_argument("--model", choices=("quadratic", "sphere"), required=True)
    parser.add_argument("--max-fit-error", type=float, required=True, help="Source coordinate units")
    parser.add_argument("--max-patch-offset", type=float, help="Quadratic patch distance from boundary plane, in source units")
    parser.add_argument("--subdivisions", type=int, default=2)
    parser.add_argument("--mesh-method", choices=("gmsh", "centroid"), default="gmsh",
                        help="Quadratic patch triangulation (default gmsh)")
    parser.add_argument("--mesh-size", type=float, help="Gmsh interior target size in source units; default median boundary edge")
    parser.add_argument("--patch-rings", type=int, default=12)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source = read_surface(args.input, clean=False).remove_unused_points()
    points = np.asarray(source.points, dtype=np.float64)
    faces = triangle_faces(source)
    inventory = build_inventory(points, faces, np.arange(len(faces)), "model_patch", max_items=len(faces))
    items = [row for row in inventory["boundary_regions"]["items"] if row["id"] == args.region]
    if len(items) != 1:
        raise ValueError("reviewed region ID not found on this source")
    if args.model == "sphere":
        vertices, patch_faces, report = build_sphere_patch(points, faces, items[0], rings=args.fit_rings,
                                                         max_fit_error=args.max_fit_error, patch_rings=args.patch_rings)
    else:
        if args.max_patch_offset is None:
            parser.error("--model quadratic requires --max-patch-offset")
        vertices, patch_faces, report = build_patch(points, faces, items[0], rings=args.fit_rings,
                                                   max_fit_error=args.max_fit_error, max_offset=args.max_patch_offset,
                                                   subdivisions=args.subdivisions, mesh_method=args.mesh_method,
                                                   mesh_size=args.mesh_size)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    patch = pv.PolyData(vertices, np.column_stack((np.full(len(patch_faces), 3), patch_faces)).ravel())
    patch.save(args.output_dir / "modeled_patch.vtp")
    report.update(input=str(args.input.resolve()), region=args.region,
                  source_geometry_sha256=hashlib.sha256(points.tobytes() + faces.tobytes()).hexdigest(),
                  fit_rings=args.fit_rings, patch_triangles=len(patch_faces))
    (args.output_dir / "modeled_patch.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
