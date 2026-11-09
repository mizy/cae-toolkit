from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from typing import Any

import numpy as np

from mesh_metrics import edge_topology


def _edge_directions(faces: np.ndarray) -> dict[tuple[int, int], list[tuple[int, int]]]:
    edges: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for face_id, face in enumerate(faces):
        for left, right in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            a, b = int(left), int(right)
            key = (min(a, b), max(a, b))
            edges[key].append((face_id, 1 if (a, b) == key else -1))
    return dict(edges)


def split_surface_fans(
    points: np.ndarray, faces: np.ndarray, *, cut_vertices=()
) -> tuple[np.ndarray, np.ndarray, int]:
    """Separate disconnected face fans without moving input vertices."""
    points = np.asarray(points, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    cuts = set(int(vertex) for vertex in cut_vertices)
    edges = _edge_directions(faces)
    if not cuts and not any(len(incident) > 2 for incident in edges.values()):
        topology, _ = edge_topology(faces)
        if topology["non_manifold_vertices"] == 0:
            return points, faces, 0

    corner_count = len(faces) * 3
    parent = np.arange(corner_count, dtype=np.int64)

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return value

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for edge, incident in edges.items():
        if len(incident) != 2 or any(vertex in cuts for vertex in edge):
            continue
        left_face, right_face = incident[0][0], incident[1][0]
        for vertex in edge:
            left_corner = left_face * 3 + int(np.flatnonzero(faces[left_face] == vertex)[0])
            right_corner = right_face * 3 + int(np.flatnonzero(faces[right_face] == vertex)[0])
            union(left_corner, right_corner)

    roots = np.asarray([find(corner) for corner in range(corner_count)], dtype=np.int64)
    corners_by_vertex: dict[int, set[int]] = defaultdict(set)
    for corner, vertex in enumerate(faces.ravel()):
        corners_by_vertex[int(vertex)].add(int(roots[corner]))
    root_to_point: dict[int, int] = {}
    output_points = list(points)
    for vertex, fan_roots in corners_by_vertex.items():
        ordered_roots = sorted(fan_roots)
        root_to_point[ordered_roots[0]] = vertex
        for root in ordered_roots[1:]:
            root_to_point[root] = len(output_points)
            output_points.append(points[vertex].copy())
    output_faces = np.fromiter(
        (root_to_point[int(root)] for root in roots), dtype=np.int64, count=corner_count
    ).reshape(-1, 3)
    return np.asarray(output_points, dtype=np.float64), output_faces, len(output_points) - len(points)


def _orient_components(
    faces: np.ndarray,
) -> tuple[np.ndarray, list[np.ndarray], list[bool]]:
    edges = _edge_directions(faces)
    if any(len(incident) > 2 for incident in edges.values()):
        raise ValueError("fan split left an edge with more than two incident faces")
    adjacency: list[list[tuple[int, int]]] = [[] for _ in faces]
    for incident in edges.values():
        if len(incident) != 2:
            continue
        (left, dir_left), (right, dir_right) = incident
        relative = -dir_left * dir_right
        adjacency[left].append((right, relative))
        adjacency[right].append((left, relative))

    signs = np.zeros(len(faces), dtype=np.int8)
    components: list[np.ndarray] = []
    orientable: list[bool] = []
    for seed in range(len(faces)):
        if signs[seed]:
            continue
        signs[seed] = 1
        stack = [seed]
        component = []
        consistent = True
        while stack:
            current = stack.pop()
            component.append(current)
            for other, relative in adjacency[current]:
                expected = signs[current] * relative
                if not signs[other]:
                    signs[other] = expected
                    stack.append(other)
                elif signs[other] != expected:
                    consistent = False
        components.append(np.asarray(component, dtype=np.int64))
        orientable.append(consistent)
    result = faces.copy()
    result[signs < 0] = result[signs < 0, ::-1]
    return result, components, orientable


def _volume(points: np.ndarray, faces: np.ndarray) -> float:
    reference = points.mean(axis=0)
    triangles = points[faces] - reference
    signed = np.einsum("ij,ij->i", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2]))
    return float(signed.sum() / 6.0)


def _canonical_open_orientation(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    triangles = points[faces]
    area_vectors = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    total = area_vectors.sum(axis=0)
    if np.linalg.norm(total) <= np.finfo(np.float64).eps * np.linalg.norm(area_vectors, axis=1).sum():
        total = area_vectors[np.argmax(np.linalg.norm(area_vectors, axis=1))]
    axis = int(np.argmax(np.abs(total)))
    return faces[:, ::-1].copy() if total[axis] < 0.0 else faces.copy()


def _component_points(
    points: np.ndarray, faces: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    used, inverse = np.unique(faces, return_inverse=True)
    return points[used], inverse.reshape(-1, 3).astype(np.int64, copy=False)


def _correct_offset_directions(faces, face_normals, directions):
    """Find a shared inward direction for vertices whose averaged normal crosses a face."""
    if not np.isfinite(face_normals).all() or not np.isfinite(directions).all():
        raise ValueError("inward offset directions must be finite")
    alignment = np.einsum("fij,fj->fi", directions[faces], face_normals)
    affected = np.unique(faces[alignment <= 1e-8])
    if not len(affected):
        return directions, 0, []
    from scipy.optimize import linprog

    result_directions = directions.copy()
    unresolved = []
    for vertex in affected:
        incident = np.flatnonzero(np.any(faces == vertex, axis=1))
        normals = face_normals[incident]
        constraints = np.column_stack((-normals, np.ones(len(normals))))
        result = linprog([0., 0., 0., -1.], A_ub=constraints, b_ub=np.zeros(len(normals)),
                         bounds=[(-1., 1.)] * 3 + [(0., 1.)], method="highs")
        if not result.success or result.x[3] <= 1e-8:
            unresolved.append(int(vertex))
            continue
        direction = result.x[:3]
        length = np.linalg.norm(direction)
        if not np.isfinite(length) or length == 0:
            unresolved.append(int(vertex))
            continue
        direction /= length
        if np.any(normals @ direction <= 1e-8):
            unresolved.append(int(vertex))
            continue
        result_directions[vertex] = direction
    return result_directions, len(affected) - len(unresolved), unresolved


def _sheet_shell(
    points: np.ndarray, faces: np.ndarray, source_ids: np.ndarray, max_error: float, *, split_round=0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    faces = _canonical_open_orientation(points, faces)
    component_points, local_faces = _component_points(points, faces)
    triangles = component_points[local_faces]
    area_vectors = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(area_vectors, axis=1)
    vertex_normals = np.zeros_like(component_points)
    for corner in range(3):
        np.add.at(vertex_normals, local_faces[:, corner], area_vectors)
    normal_lengths = np.linalg.norm(vertex_normals, axis=1)
    for vertex in np.flatnonzero(normal_lengths == 0.0):
        incident = np.flatnonzero(np.any(local_faces == vertex, axis=1))
        vertex_normals[vertex] = area_vectors[incident[np.argmax(lengths[incident])]]
    vertex_normals /= np.linalg.norm(vertex_normals, axis=1)[:, None]
    vertex_normals, corrected, unresolved = _correct_offset_directions(local_faces, area_vectors / lengths[:, None], vertex_normals)
    if unresolved:
        if split_round >= 3:
            raise ValueError("feature splitting could not resolve inward offset directions")
        split_points, split_faces, count = split_surface_fans(component_points, local_faces, cut_vertices=unresolved)
        if not count:
            raise ValueError("feature splitting could not separate unresolved face fans")
        out_points, out_faces, out_rows, corrected, extra = _sheet_shell(
            split_points, split_faces, source_ids, max_error, split_round=split_round + 1
        )
        return out_points, out_faces, out_rows, corrected, count + extra
    inner_points = component_points - vertex_normals * max_error
    if np.any(np.all(inner_points == component_points, axis=1)):
        raise ValueError("max_error is too small to move some inner-shell vertices at input precision")

    edges = _edge_directions(local_faces)
    if any(len(incident) > 2 for incident in edges.values()):
        raise ValueError("open sheet still has a non-manifold edge")
    walls: list[tuple[int, int, int]] = []
    wall_source: list[int] = []
    for edge, incident in edges.items():
        if len(incident) != 1:
            continue
        face_id = incident[0][0]
        face = local_faces[face_id]
        left, right = next(
            (int(a), int(b))
            for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0]))
            if (min(int(a), int(b)), max(int(a), int(b))) == edge
        )
        walls.extend(
            (
                (left, left + len(component_points), right + len(component_points)),
                (left, right + len(component_points), right),
            )
        )
        wall_source.extend((int(source_ids[face_id]), int(source_ids[face_id])))
    inner_faces = local_faces[:, ::-1] + len(component_points)
    shell_faces = np.vstack((local_faces, inner_faces, np.asarray(walls, dtype=np.int64).reshape(-1, 3)))
    shell_source = np.concatenate((source_ids, source_ids, np.asarray(wall_source, dtype=np.int64)))
    return np.vstack((component_points, inner_points)), shell_faces, shell_source, corrected, 0


def _triangle_prisms(
    points: np.ndarray, faces: np.ndarray, source_ids: np.ndarray, max_error: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    point_blocks, face_blocks = [], []
    lineage = []
    side_faces = np.asarray(
        ((0, 3, 4), (0, 4, 1), (1, 4, 5), (1, 5, 2), (2, 5, 3), (2, 3, 0)), dtype=np.int64
    )
    for face_id, face in enumerate(faces):
        triangle = points[face]
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        length = np.linalg.norm(normal)
        if length == 0.0:
            raise ValueError(f"cannot build a prism around degenerate input face {face_id}")
        normal /= length
        block_points = np.vstack((triangle, triangle - normal * max_error))
        unmoved = np.any(np.all(block_points[:3] == block_points[3:], axis=1))
        if not np.isfinite(block_points).all() or unmoved:
            raise ValueError(f"max_error cannot form a finite prism around input face {face_id}")
        block_faces = np.vstack((np.asarray(((0, 1, 2), (3, 5, 4))), side_faces))
        if _volume(block_points, block_faces) < 0.0:
            block_faces = block_faces[:, ::-1]
        face_blocks.append(block_faces + 6 * face_id)
        point_blocks.append(block_points)
        lineage.extend([int(source_ids[face_id])] * len(block_faces))
    return np.vstack(point_blocks), np.vstack(face_blocks), np.asarray(lineage, dtype=np.int64)


def _union(
    points: np.ndarray, faces: np.ndarray, soup_source_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from igl.copyleft import cgal

    empty_points = np.empty((0, 3), dtype=np.float64)
    empty_faces = np.empty((0, 3), dtype=np.int64)
    output_points, output_faces, birth_faces = cgal.mesh_boolean(
        np.ascontiguousarray(points, dtype=np.float64),
        np.ascontiguousarray(faces, dtype=np.int64),
        empty_points,
        empty_faces,
        "union",
    )
    output_points = np.asarray(output_points, dtype=np.float64)
    output_faces = np.asarray(output_faces, dtype=np.int64)
    birth_faces = np.asarray(birth_faces, dtype=np.int64)
    if birth_faces.shape != (len(output_faces),):
        raise ValueError("CGAL face map length does not match output faces")
    if np.any((birth_faces < 0) | (birth_faces >= len(soup_source_ids))):
        raise ValueError("CGAL returned an unmapped output face")
    return output_points, output_faces, soup_source_ids[birth_faces]


def _build_component_block(
    component_id: int, face_ids: np.ndarray, split_points: np.ndarray,
    oriented_faces: np.ndarray, source_ids: np.ndarray, orientable: bool, max_error: float,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], dict[str, Any]]:
    faces = oriented_faces[face_ids]
    rows = source_ids[face_ids]
    local_points, local_faces = _component_points(split_points, faces)
    _, edges = edge_topology(local_faces)
    closed = bool(edges and all(len(incident) == 2 for incident in edges.values()))
    volume = _volume(local_points, local_faces) if closed and orientable else None
    corrected = 0
    feature_splits = 0
    if orientable and closed and np.isfinite(volume) and volume != 0.0:
        block = (local_points, local_faces, rows)
        mode, side, reason = (
            "preserved_closed_component",
            "input component winding; nested cavity signs retained",
            None,
        )
    elif orientable and not closed:
        block_points, block_faces, block_rows, corrected, feature_splits = _sheet_shell(split_points, faces, rows, max_error)
        block = (block_points, block_faces, block_rows)
        mode, side, reason = (
            "inward_offset_sheet_shell",
            "BFS orientation with canonical area-weighted geometric-normal sign",
            None,
        )
    else:
        block = _triangle_prisms(split_points, faces, rows, max_error)
        mode = "bounded_triangle_prism_fallback"
        side = "per-face geometric normal; material side is ambiguous"
        reason = "closed_component_has_zero_or_unreliable_signed_volume" if closed else "component_could_not_be_consistently_oriented"
    return block, {
        "component_id": component_id,
        "input_faces": len(face_ids),
        "closed": closed,
        "orientable": orientable,
        "mode": mode,
        "side_selection": side,
        "fallback_reason": reason,
        "input_signed_volume": volume,
        "offset_direction_corrected_vertices": corrected,
        "offset_feature_split_vertices": feature_splits,
    }


def _build_components(
    points: np.ndarray, faces: np.ndarray, max_error: float, progress=None
) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray, np.ndarray]], list[dict[str, Any]], int, list[dict[str, Any]], int]:
    source_ids = np.arange(len(faces), dtype=np.int64)
    if progress:
        progress("source_material_split_fans")
    split_points, split_faces, split_count = split_surface_fans(points, faces)
    if progress:
        progress("source_material_orient_components")
    oriented, components, orientable = _orient_components(split_faces)
    if progress:
        progress("source_material_build_component_shells")
    rows = [
        _build_component_block(i, face_ids, split_points, oriented, source_ids, orientable[i], max_error)
        for i, face_ids in enumerate(components)
    ]
    blocks = [row[0] for row in rows]
    details = [row[1] for row in rows]
    if progress:
        progress("source_material_closed_component_relations")
    parent, nested, aabb_interactions = _closed_component_relations(blocks, details)
    for root in (
        i for i, row in enumerate(details)
        if row["mode"] == "preserved_closed_component" and i not in parent
    ):
        stack, tree = [root], []
        while stack:
            component_id = stack.pop()
            tree.append(component_id)
            stack.extend(child for child, ancestor in parent.items() if ancestor == component_id)
        reversed_tree = details[root]["input_signed_volume"] < 0.0
        if reversed_tree:
            for component_id in tree:
                block_points, block_faces, source_ids = blocks[component_id]
                blocks[component_id] = (block_points, block_faces[:, ::-1].copy(), source_ids)
                oriented[source_ids] = oriented[source_ids, ::-1]
        for component_id in tree:
            details[component_id]["whole_tree_orientation_reversed"] = reversed_tree
    return oriented, blocks, details, split_count, nested, aabb_interactions


def _union_blocks(
    blocks: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    point_blocks, face_blocks, source_blocks = [], [], []
    point_offset = 0
    for block_points, block_faces, source_ids in blocks:
        point_blocks.append(block_points)
        face_blocks.append(block_faces + point_offset)
        source_blocks.append(source_ids)
        point_offset += len(block_points)
    points = np.vstack(point_blocks)
    faces = np.vstack(face_blocks)
    source_ids = np.concatenate(source_blocks)
    result_points, result_faces, result_source_ids = _union(points, faces, source_ids)
    return result_points, result_faces, result_source_ids, len(faces)


def _manifold_union_blocks(blocks):
    """Union positive closed blocks while retaining their source-face IDs."""
    import manifold3d as manifold

    solids = []
    source_ids = np.concatenate([block[2] for block in blocks])
    for points, faces, rows in blocks:
        if np.any(rows < -1):
            raise ValueError("Manifold source-face IDs must be mapped or generated (-1)")
        solid = manifold.Manifold(manifold.Mesh64(
            np.ascontiguousarray(points, dtype=np.float64),
            np.ascontiguousarray(faces, dtype=np.uint64),
            face_id=np.ascontiguousarray(rows + 1, dtype=np.uint64),
        ))
        if solid.status() != manifold.Error.NoError or not np.isfinite(solid.volume()) or solid.volume() <= 0:
            raise ValueError("Manifold union requires individually valid positive material shells")
        solids.append(solid)
    output = manifold.Manifold.batch_boolean(solids, manifold.OpType.Add)
    mesh = output.to_mesh64()
    if output.status() != manifold.Error.NoError:
        raise ValueError(f"Manifold material union failed: {output.status()}")
    points = np.asarray(mesh.vert_properties, dtype=np.float64)[:, :3].copy()
    faces = np.array(mesh.tri_verts, dtype=np.int64, copy=True)
    rows = np.asarray(mesh.face_id, dtype=np.int64) - 1
    if (not np.isfinite(points).all() or rows.shape != (len(faces),)
            or not np.isin(rows, source_ids).all()):
        raise ValueError("Manifold returned invalid geometry or source-face IDs")
    return points, faces, rows, sum(len(block[1]) for block in blocks)


def _closed_bbox_intersections(
    bounds: dict[int, tuple[np.ndarray, np.ndarray]],
) -> Iterator[tuple[int, int]]:
    order = sorted(bounds, key=lambda i: bounds[i][0][0])
    active: list[int] = []
    for current in order:
        low, high = bounds[current]
        active = [i for i in active if bounds[i][1][0] >= low[0]]
        for other in active:
            other_low, other_high = bounds[other]
            if np.all(np.minimum(high, other_high) >= np.maximum(low, other_low)):
                yield other, current
        active.append(current)


def _closed_component_relations(
    blocks: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    details: list[dict[str, Any]],
) -> tuple[dict[int, int], list[dict[str, Any]], int]:
    """Find closed-shell nesting with bbox pruning and component winding."""
    closed = [i for i, row in enumerate(details) if row["mode"] == "preserved_closed_component"]
    bounds = {i: (blocks[i][0].min(axis=0), blocks[i][0].max(axis=0)) for i in closed}
    overlaps = _closed_bbox_intersections(bounds) if len(bounds) > 1 else iter(())
    parents: dict[int, list[tuple[float, int, float]]] = defaultdict(list)
    strict_nesting: set[tuple[int, int]] = set()
    aabb_interactions = 0
    if len(closed) > 1:
        import igl

        for left, right in overlaps:
            aabb_interactions += 1
            for outer, inner in ((left, right), (right, left)):
                outer_low, outer_high = bounds[outer]
                inner_low, inner_high = bounds[inner]
                contained = np.all(outer_low <= inner_low) and np.all(
                    outer_high >= inner_high
                ) and (np.any(outer_low < inner_low) or np.any(outer_high > inner_high))
                if not contained:
                    continue
                outer_points, outer_faces, _ = blocks[outer]
                inner_points, _, _ = blocks[inner]
                winding = float(
                    np.asarray(igl.winding_number(
                        outer_points, outer_faces.astype(np.int32, copy=False), inner_points[:1]
                    )).reshape(-1)[0]
                )
                if np.isfinite(winding) and abs(winding) > 0.5:
                    if not _nested_surfaces_disjoint(blocks[outer], blocks[inner]):
                        continue
                    strict_nesting.add((outer, inner))
                    box_volume = float(np.prod(outer_high - outer_low))
                    parents[inner].append((box_volume, outer, winding))
        for left, right in _closed_bbox_intersections(bounds):
            if details[left]["input_signed_volume"] * details[right]["input_signed_volume"] >= 0.0:
                continue
            if (left, right) in strict_nesting or (right, left) in strict_nesting:
                continue
            intersects = _closed_component_surfaces_intersect(blocks[left], blocks[right])
            if intersects:
                raise ValueError(
                    "ambiguous material semantics: opposite-oriented closed components "
                    f"{left} and {right} intersect without strict nesting"
                )
    direct_parents: dict[int, int] = {}
    relations = []
    for child, candidates in parents.items():
        _, parent, winding = min(candidates, key=lambda item: (item[0], item[1]))
        direct_parents[child] = parent
        parent_volume = float(details[parent]["input_signed_volume"])
        child_volume = float(details[child]["input_signed_volume"])
        relations.append(
            {
                "outer_component_id": parent,
                "inner_component_id": child,
                "child_point_winding_number": winding,
                "input_orientation_relation": "cavity_boundary"
                if parent_volume * child_volume < 0.0
                else "same_material_orientation",
            }
        )

    return direct_parents, relations, aabb_interactions if closed else 0


def _closed_component_surfaces_intersect(left, right) -> bool:
    """Use exact CGAL checks to distinguish crossing shells from bbox overlap."""
    from source_projected_validation import exact_self_intersection_report

    left_points, left_faces, _ = left
    right_points, right_faces, _ = right
    for points, faces in ((left_points, left_faces), (right_points, right_faces)):
        if exact_self_intersection_report(points, faces, first_only=True).get("passed") is not True:
            raise ValueError(
                "cannot classify opposite-oriented closed components because a component "
                "failed its exact self-intersection check"
            )
    combined = np.vstack((left_points, right_points))
    joined = np.vstack((left_faces, right_faces + len(left_points)))
    result = exact_self_intersection_report(combined, joined, first_only=True)
    if result.get("status") != "computed":
        raise ValueError("exact CGAL intersection detection is required to classify closed components")
    return result.get("passed") is False


def _nested_surfaces_disjoint(outer, inner) -> bool:
    """An inside point proves containment only if the connected surfaces cannot cross."""
    from source_projected_validation import exact_self_intersection_report
    outer_points, outer_faces, _ = outer
    inner_points, inner_faces, _ = inner
    triangles = outer_points[outer_faces]
    low, high = inner_points.min(axis=0), inner_points.max(axis=0)
    overlaps = np.all(triangles.min(axis=1) <= high, axis=1) & np.all(triangles.max(axis=1) >= low, axis=1)
    if not np.any(overlaps):
        return True
    points = np.vstack((outer_points, inner_points))
    faces = np.vstack((outer_faces[overlaps], inner_faces + len(outer_points)))
    return exact_self_intersection_report(points, faces, first_only=True).get("passed") is True


# @entry
def build_source_material(
    points: np.ndarray, faces: np.ndarray, *, max_error: float, progress=None, union_backend="cgal"
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Build a candidate; max_error sets the inward offset and prism depth in input units."""
    points = np.asarray(points, dtype=np.float64)
    raw_faces = np.asarray(faces)
    if not np.issubdtype(raw_faces.dtype, np.integer):
        raise ValueError("faces must use an integer index type")
    faces = np.asarray(faces, dtype=np.int64)
    max_error = float(max_error)
    if union_backend not in ("cgal", "manifold"):
        raise ValueError("union_backend must be cgal or manifold")
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("points must be a finite (N, 3) array")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise ValueError("faces must be a non-empty (M, 3) triangle array")
    if np.any((faces < 0) | (faces >= len(points))):
        raise ValueError("face indices must reference input points")
    if not np.isfinite(max_error) or max_error <= 0.0:
        raise ValueError("max_error must be finite and positive in input coordinate units")
    if np.any((faces[:, 0] == faces[:, 1]) | (faces[:, 1] == faces[:, 2]) | (faces[:, 2] == faces[:, 0])):
        raise ValueError("input faces must contain three distinct point indices")
    triangles = points[faces]
    area_lengths = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1
    )
    if np.any(area_lengths == 0.0):
        raise ValueError("input faces must have non-zero geometric area")

    if progress:
        progress("source_material_build_components")
    (
        oriented,
        blocks,
        component_details,
        split_count,
        nested_relations,
        closed_component_interactions,
    ) = _build_components(points, faces, max_error, progress=progress)
    needs_union = split_count > 0 or any(
        item["mode"] != "preserved_closed_component" for item in component_details
    ) or closed_component_interactions > 0
    details = {
        "nested_closed_component_relations": nested_relations,
        "closed_component_aabb_interactions": closed_component_interactions,
        "aabb_interactions_are_broad_phase_only": True,
    }
    if not needs_union:
        return points, oriented, np.arange(len(faces), dtype=np.int64), {
            "method": "closed source components retained without Boolean",
            "max_error": max_error,
            "fan_split_points": split_count,
            "components": component_details,
            **details,
            "boolean_applied": False,
            "union_backend": "none",
            "acceptance": "not performed by this geometry builder",
        }

    if progress:
        progress("source_material_boolean_union")
    if union_backend == "manifold":
        result_points, result_faces, result_source_ids, soup_face_count = _manifold_union_blocks(blocks)
    else:
        result_points, result_faces, result_source_ids, soup_face_count = _union_blocks(blocks)
    return result_points, result_faces, result_source_ids, {
        "method": (
            "fan-split source sheets, bounded inward material shells, "
            f"and {union_backend} material union"
        ),
        "max_error": max_error,
        "fan_split_points": split_count,
        "components": component_details,
        **details,
        "boolean_applied": True,
        "union_backend": union_backend,
        "input_faces_to_union": soup_face_count,
        "output_faces": len(result_faces),
        "source_face_map_complete": len(result_source_ids) == len(result_faces),
        "acceptance": "not performed by this geometry builder",
    }
