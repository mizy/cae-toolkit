"""Material shells, explicit wall construction, and Boolean part union in millimetres."""

from __future__ import annotations

import numpy as np
import pyvista as pv
import trimesh
from mesh_io import triangle_faces
from mesh_metrics import signed_volume, signed_volume_error_bound


def orient_material(mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, list[dict]]:
    mesh = mesh.copy()
    trimesh.repair.fix_winding(mesh)
    components = trimesh.graph.connected_components(
        mesh.face_adjacency, nodes=np.arange(len(mesh.faces))
    )
    pieces = [mesh.submesh([ids], append=True, repair=False) for ids in components]
    volumes = [signed_volume(part.vertices, part.faces) for part in pieces]
    closed = [
        part.is_watertight
        and abs(volume) > signed_volume_error_bound(part.vertices, part.faces)
        for part, volume in zip(pieces, volumes)
    ]
    for ids, volume, is_closed in zip(components, volumes, closed):
        if is_closed and volume < 0:
            mesh.faces[ids] = mesh.faces[ids, ::-1]
    containers = [[] for _ in pieces]
    for i, part in enumerate(pieces):
        if not closed[i]:
            continue
        for j, outer in enumerate(pieces):
            if (
                i == j
                or not closed[j]
                or not np.all(part.bounds[0] >= outer.bounds[0])
                or not np.all(part.bounds[1] <= outer.bounds[1])
            ):
                continue
            inside = pv.PolyData(part.vertices).select_enclosed_points(
                pv.wrap(outer), tolerance=1e-9
            )
            if np.all(inside["SelectedPoints"]):
                containers[i].append(j)
    shells = []
    for i, ids in enumerate(components):
        depth = len(containers[i])
        parent = (
            min(containers[i], key=lambda j: abs(volumes[j])) if containers[i] else None
        )
        if depth % 2:
            mesh.faces[ids] = mesh.faces[ids, ::-1]
        shells.append(
            {
                "depth": depth,
                "parent": parent,
                "closed": bool(closed[i]),
                "faces": len(ids),
                "signed_volume_mm3": float((-1) ** depth * abs(volumes[i])),
            }
        )
    return mesh, shells


def union_parts(
    mesh: trimesh.Trimesh, *, fill_cavities: bool = False, merge: bool = True
) -> trimesh.Trimesh:
    import manifold3d as manifold

    oriented, shells = orient_material(mesh)
    components = trimesh.graph.connected_components(
        oriented.face_adjacency, nodes=np.arange(len(oriented.faces))
    )
    pieces = [oriented.submesh([ids], append=True, repair=False) for ids in components]
    for part in pieces:
        part.fix_normals()
    if not all(part.is_volume for part in pieces):
        raise ValueError(
            "Boolean union requires closed nonzero-volume parts; repair open boundaries first"
        )
    material_parts = [
        manifold.Manifold(
            manifold.Mesh64(
                np.asarray(part.vertices), np.asarray(part.faces, dtype=np.uint64)
            )
        )
        for part in pieces
    ]
    if any(part.status() != manifold.Error.NoError for part in material_parts):
        raise ValueError("Boolean input failed Manifold material validation")
    solids = []
    for i, part in enumerate(pieces):
        if fill_cavities and shells[i]["depth"] > 0:
            continue
        if shells[i]["depth"] % 2 and not fill_cavities:
            continue
        solid = material_parts[i]
        if not fill_cavities:
            for j, row in enumerate(shells):
                if row["parent"] == i:
                    solid = solid - material_parts[j]
        solids.append(solid)
    if merge:
        solids = [manifold.Manifold.batch_boolean(solids, manifold.OpType.Add)]
    outputs = []
    for solid in solids:
        if solid.status() != manifold.Error.NoError:
            raise ValueError(f"material Boolean failed: {solid.status()}")
        result = solid.to_mesh64()
        outputs.append(
            trimesh.Trimesh(
                np.asarray(result.vert_properties)[:, :3],
                np.asarray(result.tri_verts),
                process=False,
            )
        )
    return trimesh.util.concatenate(outputs)


def construct_wall(
    mesh: trimesh.Trimesh,
    *,
    mode: str,
    thickness: float,
    error: float,
    max_voxels: int,
) -> tuple[trimesh.Trimesh, dict]:
    pitch = min(error / 2, thickness / 5)
    padding = thickness + 3 * pitch
    origin = mesh.bounds[0] - padding
    shape = np.ceil((mesh.extents + 2 * padding) / pitch).astype(int) + 1
    count = int(np.prod(shape.astype(object)))
    if count > max_voxels:
        raise MemoryError(
            f"wall grid needs {count} voxels; budget {max_voxels}; split the model or change the explicit error/thickness budget"
        )
    grid = pv.ImageData(dimensions=shape, spacing=(pitch,) * 3, origin=origin)
    surface = pv.wrap(mesh)
    if mode == "thicken" and not mesh.is_watertight:
        _, closest = surface.find_closest_cell(grid.points, return_closest_point=True)
        phi = np.linalg.norm(grid.points - closest, axis=1) - thickness / 2
        field = "unsigned_source_distance_minus_half_wall"
    else:
        phi = np.asarray(grid.compute_implicit_distance(surface)["implicit_distance"])
        if mode == "hollow":
            if float(np.max(-phi)) <= thickness:
                raise ValueError("requested wall thickness leaves no internal cavity")
            phi = np.maximum(phi, -phi - thickness)
            field = "signed_material_distance_hollow_wall"
        else:
            phi = phi - thickness / 2
            field = "signed_material_distance_outward_thickening"
    grid["wall_sdf"] = phi
    result = grid.contour([0.0], scalars="wall_sdf").triangulate().clean()
    if result.n_cells == 0:
        raise ValueError("wall construction produced no surface")
    output = trimesh.Trimesh(result.points, triangle_faces(result), process=False)
    # Exact-zero grid samples can create collapsed contour slivers. Rebuild the
    # closed solid with Manifold, then let independent CGAL/fidelity gates certify it.
    import manifold3d as manifold

    solid = manifold.Manifold(
        manifold.Mesh64(
            np.asarray(output.vertices), np.asarray(output.faces, dtype=np.uint64)
        )
    )
    if solid.status() != manifold.Error.NoError:
        raise ValueError(
            f"wall contour is not a valid material solid: {solid.status()}"
        )
    cleaned = solid.to_mesh64()
    output = trimesh.Trimesh(
        np.asarray(cleaned.vert_properties)[:, :3],
        np.asarray(cleaned.tri_verts),
        process=False,
    )
    output, _ = orient_material(output)
    return output, {
        "mode": mode,
        "field": field,
        "pitch_mm": pitch,
        "wall_mm": thickness,
        "voxels": count,
        "contour_cleanup": "Manifold collapsed-sliver cleanup; independently recertified downstream",
    }


def material_valid(metrics: dict, shells: list[dict]) -> bool:
    topology = metrics["topology"]
    return bool(
        all(
            topology[key] == 0
            for key in (
                "boundary_edges",
                "non_manifold_edges",
                "non_manifold_vertices",
                "inconsistent_winding_edges",
            )
        )
        and metrics["quality"]["degenerate_faces"] == 0
        and metrics["volume"]["reliable"]
        and shells
        and all(
            row["closed"] and abs(row["signed_volume_mm3"]) > 1e-15 for row in shells
        )
    )
