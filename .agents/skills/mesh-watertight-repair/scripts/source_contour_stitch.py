"""Clip and weld a local surface patch to an exact source contour."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pyvista as pv
from local_defect_regions import _boundary
from mesh_io import triangle_faces
from mesh_metrics import edge_topology
from source_patch_collar import _patch_boundary_normals, build_source_collar
from source_projected_closure import _barycentric, _closest_source_points

_MAX_CONTOUR_NODES = 4096
_MAX_NEW_TRIANGLES = 100_000


def _closed_loop(loop, point_count: int, name: str) -> np.ndarray:
    ids = np.asarray(loop, dtype=np.int64).reshape(-1)
    if (
        len(ids) < 4
        or ids[0] != ids[-1]
        or len(np.unique(ids[:-1])) != len(ids) - 1
        or ids.min() < 0
        or ids.max() >= point_count
    ):
        raise ValueError(f"{name} must be one simple closed loop")
    if len(ids) - 1 > _MAX_CONTOUR_NODES:
        raise ValueError("common contour exceeds the 4096-node budget")
    return ids


def _single_boundary_loop(faces: np.ndarray) -> list[int]:
    _, edges = edge_topology(faces)
    _, loops, boundary = _boundary(faces, np.arange(len(faces)), edges)
    if boundary["status"] != "closed_loops" or len(loops) != 1:
        raise ValueError("clipped patch must have one simple boundary loop")
    return loops[0]


# @entry: trim one voxel surface at the same source scalar level as its collar.
def clip_source_contour(
    zero_p,
    zero_f,
    source_p,
    source_f,
    rows,
    source_loop,
    *,
    fraction=0.25,
):
    """Keep the SDF side whose interpolated source scalar is at least ``fraction``.

    The source scalar is zero outside the selected ROI and on its cut rim. It
    reaches one only at selected ROI interior vertices. Closest source triangles
    and barycentric interpolation give every SDF vertex its clipping value.
    """
    zero_p = np.asarray(zero_p, dtype=np.float64)
    zero_f = np.asarray(zero_f, dtype=np.int64)
    source_p = np.asarray(source_p, dtype=np.float64)
    source_f = np.asarray(source_f, dtype=np.int64)
    rows = np.asarray(rows, dtype=np.int64).reshape(-1)
    if (
        zero_p.ndim != 2
        or zero_p.shape[1] != 3
        or not len(zero_p)
        or zero_f.ndim != 2
        or zero_f.shape[1] != 3
        or not len(zero_f)
        or source_p.ndim != 2
        or source_p.shape[1] != 3
        or not len(source_p)
        or source_f.ndim != 2
        or source_f.shape[1] != 3
        or not len(source_f)
        or not np.isfinite(zero_p).all()
        or not np.isfinite(source_p).all()
        or zero_f.min() < 0
        or zero_f.max() >= len(zero_p)
        or source_f.min() < 0
        or source_f.max() >= len(source_p)
        or rows.size == 0
        or rows.min() < 0
        or rows.max() >= len(source_f)
        or len(np.unique(rows)) != len(rows)
        or not np.isfinite(fraction)
        or not 0.0 < fraction < 1.0
    ):
        raise ValueError("invalid source-contour clipping inputs")
    source_loop = _closed_loop(source_loop, len(source_p), "source loop")
    selected_vertices = np.unique(source_f[rows])
    rim = np.unique(source_loop[:-1])
    interior = np.setdiff1d(selected_vertices, rim, assume_unique=True)
    if not len(interior):
        raise ValueError("source ROI has no interior vertices for scalar clipping")

    closest, owner_rows, distances = _closest_source_points(source_p, source_f, zero_p)
    if (
        owner_rows.shape != (len(zero_p),)
        or closest.shape != zero_p.shape
        or not np.isfinite(closest).all()
        or not np.isfinite(distances).all()
        or np.any(owner_rows < 0)
        or np.any(owner_rows >= len(source_f))
    ):
        raise ValueError("nearest source-triangle query returned invalid values")
    weights = _barycentric(closest, source_p[source_f[owner_rows]])
    weights = np.clip(weights, 0.0, 1.0)
    weight_sums = weights.sum(axis=1)
    if not np.isfinite(weights).all() or np.any(weight_sums <= 1e-20):
        raise ValueError("source-contour barycentric interpolation is invalid")
    weights /= weight_sums[:, None]

    vertex_values = np.zeros(len(source_p), dtype=np.float64)
    vertex_values[interior] = 1.0
    corner_values = np.zeros((len(source_f), 3), dtype=np.float64)
    corner_values[rows] = vertex_values[source_f[rows]]
    scalar = np.einsum("ij,ij->i", weights, corner_values[owner_rows])
    selected = np.zeros(len(source_f), dtype=bool)
    selected[rows] = True
    outside = ~selected[owner_rows]
    scalar[outside] = 0.0
    if not np.isfinite(scalar).all():
        raise ValueError("source-contour interpolation produced non-finite values")

    packed = np.column_stack((np.full(len(zero_f), 3), zero_f)).ravel()
    surface = pv.PolyData(zero_p, packed)
    surface.point_data["source_scalar"] = scalar
    surface.GetPointData().SetActiveScalars("source_scalar")
    clipped = surface.clip_scalar(scalars="source_scalar", value=float(fraction), invert=False)
    clipped = clipped.triangulate()
    points = np.asarray(clipped.points, dtype=np.float64)
    faces = triangle_faces(clipped)
    if not len(points) or not np.isfinite(points).all():
        raise ValueError("source-contour clipping produced invalid points")
    clipped_scalar = np.asarray(clipped.point_data["source_scalar"], dtype=np.float64)
    scalar_tolerance = 8.0 * np.finfo(np.float32).eps
    if not np.isfinite(clipped_scalar).all() or np.any(clipped_scalar < fraction - scalar_tolerance):
        raise ValueError("source-contour clipping retained points below the scalar level")

    triangles = points[faces]
    area2 = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1)
    if not np.isfinite(area2).all():
        raise ValueError("source-contour clipping produced non-finite triangles")
    zero_area = int(np.count_nonzero(area2 == 0.0))
    faces = faces[area2 > 0.0]
    if not len(faces):
        raise ValueError("source-contour clipping produced no non-degenerate triangles")
    used, inverse = np.unique(faces.ravel(), return_inverse=True)
    points, faces = points[used], inverse.reshape(-1, 3)
    boundary_loop = _single_boundary_loop(faces)
    owners = np.unique(owner_rows)
    return (
        points,
        faces,
        {
            "status": "ready",
            "method": "closest_source_barycentric_scalar_clip",
            "fraction": float(fraction),
            "source_owner_rows": owners.astype(int).tolist(),
            "source_owner_rows_selected": int(np.count_nonzero(selected[owner_rows])),
            "source_owner_rows_outside": int(np.count_nonzero(outside)),
            "finite_source_queries": len(distances),
            "clipped_scalar_min": float(clipped_scalar.min()),
            "clipped_scalar_tolerance": float(scalar_tolerance),
            "boundary_edges": len(boundary_loop) - 1,
            "boundary_loop_count": 1,
            "zero_area_facets_removed": zero_area,
            "points": len(points),
            "triangles": len(faces),
        },
    )


def _pava(values: np.ndarray) -> np.ndarray:
    starts, ends, means, weights = [], [], [], []
    for index, value in enumerate(values):
        starts.append(index)
        ends.append(index + 1)
        means.append(float(value))
        weights.append(1)
        while len(means) > 1 and means[-2] > means[-1]:
            right_mean, right_weight = means.pop(), weights.pop()
            right_start, right_end = starts.pop(), ends.pop()
            left_mean, left_weight = means.pop(), weights.pop()
            left_start = starts.pop()
            left_end = ends.pop()
            if left_end != right_start:
                raise RuntimeError("isotonic blocks are not adjacent")
            weight = left_weight + right_weight
            starts.append(left_start)
            ends.append(right_end)
            means.append((left_mean * left_weight + right_mean * right_weight) / weight)
            weights.append(weight)
    result = np.empty(len(values), dtype=np.float64)
    for start, end, mean in zip(starts, ends, means):
        result[start:end] = mean
    return result


def _project_to_contour(queries: np.ndarray, contour: np.ndarray):
    starts, edges = contour[:-1], contour[1:] - contour[:-1]
    lengths = np.linalg.norm(edges, axis=1)
    if np.any(lengths <= 0.0):
        raise ValueError("source contour contains a zero-length edge")
    cumulative = np.r_[0.0, np.cumsum(lengths)]
    best_distance = np.full(len(queries), np.inf)
    best_arc = np.zeros(len(queries), dtype=np.float64)
    best_point = np.zeros_like(queries)
    for index, (start, edge, length) in enumerate(zip(starts, edges, lengths)):
        parameter = np.clip(((queries - start) @ edge) / (length * length), 0.0, 1.0)
        projected = start + parameter[:, None] * edge
        distance = np.linalg.norm(queries - projected, axis=1)
        better = distance < best_distance
        best_distance[better] = distance[better]
        best_arc[better] = cumulative[index] + parameter[better] * length
        best_point[better] = projected[better]
    return best_point, best_arc, best_distance, cumulative


def _split_faces_on_edges(points, faces: np.ndarray, insertions: dict[tuple[int, int], list[int]]):
    points = [np.asarray(point, dtype=np.float64) for point in points]
    output, births = [], []
    for row, face in enumerate(faces):
        split_edges = [
            (int(left), int(right), int(opposite), insertions[(int(left), int(right))])
            for left, right, opposite in zip(face, np.roll(face, -1), np.roll(face, -2))
            if (int(left), int(right)) in insertions
        ]
        if not split_edges:
            output.append(face.tolist())
            births.append(row)
        elif len(split_edges) == 1:
            left, right, opposite, inserted = split_edges[0]
            chain = [left, *inserted, right]
            for a, b in pairwise(chain):
                output.append([int(a), int(b), opposite])
                births.append(row)
        else:
            polygon = []
            for left, right in zip(face, np.roll(face, -1)):
                left, right = int(left), int(right)
                polygon.append(left)
                polygon.extend(insertions.get((left, right), ()))
            if len(set(polygon)) != len(polygon):
                raise ValueError("contour split creates a repeated triangle boundary node")
            center_id = len(points)
            points.append(np.mean(np.asarray(points)[polygon], axis=0))
            for index in range(len(polygon)):
                output.append([center_id, polygon[index], polygon[(index + 1) % len(polygon)]])
                births.append(row)
    return np.asarray(output, dtype=np.int64), np.asarray(births, dtype=np.int64), np.asarray(points)


def _area_normals(points: np.ndarray, faces: np.ndarray):
    triangles = points[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    scale = max(float(np.linalg.norm(np.ptp(points, axis=0))), 1e-150)
    minimum = max(scale * scale * np.finfo(float).eps * 128, np.finfo(float).tiny)
    if not np.isfinite(normals).all() or np.any(lengths <= minimum):
        raise ValueError("conformal contour split creates a degenerate triangle")
    return normals / lengths[:, None]


def _validate_patch_inputs(points, faces, loop):
    points = np.asarray(points, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or not len(points)
        or faces.ndim != 2
        or faces.shape[1] != 3
        or not len(faces)
        or not np.isfinite(points).all()
        or faces.min() < 0
        or faces.max() >= len(points)
    ):
        raise ValueError("invalid patch arrays")
    loop = _closed_loop(loop, len(points), "patch loop")
    _, loops, boundary = _boundary(faces, np.arange(len(faces)), edge_topology(faces)[1])
    if boundary["status"] != "closed_loops" or len(loops) != 1:
        raise ValueError("patch must have one simple boundary loop")
    expected = {tuple(sorted(pair)) for pair in pairwise(map(int, loop))}
    actual = {tuple(sorted(pair)) for pair in pairwise(map(int, loops[0]))}
    if actual != expected:
        raise ValueError("patch loop does not match the patch boundary")
    _patch_boundary_normals(points, faces, loop)
    return points, faces, loop


# @entry: weld an SDF boundary to the exact piecewise-linear source collar contour.
def stitch_source_contour(
    source_p,
    source_f,
    rows,
    source_loop,
    patch_p,
    patch_f,
    patch_loop,
    *,
    max_displacement,
    fraction=0.25,
):
    """Return collar-prefixed patch points and body faces sharing one contour.

    The original source cut loop is retained as the collar's outer boundary.
    Patch boundary vertices move only within ``max_displacement``. The 20-degree
    seam gate applies to the final collar and patch faces sharing each contour
    edge; unsupported or oversized contours fail closed.
    """
    source_p = np.asarray(source_p, dtype=np.float64)
    source_f = np.asarray(source_f, dtype=np.int64)
    rows = np.asarray(rows, dtype=np.int64).reshape(-1)
    patch_p, patch_f, patch_loop = _validate_patch_inputs(patch_p, patch_f, patch_loop)
    if (
        source_p.ndim != 2
        or source_p.shape[1] != 3
        or not len(source_p)
        or source_f.ndim != 2
        or source_f.shape[1] != 3
        or not len(source_f)
        or not np.isfinite(source_p).all()
        or source_f.min() < 0
        or source_f.max() >= len(source_p)
        or rows.size == 0
        or rows.min() < 0
        or rows.max() >= len(source_f)
        or len(np.unique(rows)) != len(rows)
        or not np.isfinite(max_displacement)
        or max_displacement < 0.0
        or not np.isfinite(fraction)
        or not 0.0 < fraction < 1.0
    ):
        raise ValueError("invalid source-contour stitching inputs")
    source_loop = _closed_loop(source_loop, len(source_p), "source loop")
    patch_loop = _closed_loop(patch_loop, len(patch_p), "patch loop")
    collar_p, collar_f, inner_loop, _, births = build_source_collar(source_p, source_f, rows, fraction=fraction)
    inner_loop = np.asarray(inner_loop, dtype=np.int64)
    source_collar_points = np.vstack((source_p, collar_p))
    contour = source_collar_points[inner_loop]
    inner_lengths = np.linalg.norm(np.diff(contour, axis=0), axis=1)
    cumulative = np.r_[0.0, np.cumsum(inner_lengths)]
    circumference = float(cumulative[-1])
    if not np.isfinite(circumference) or circumference <= 0.0:
        raise ValueError("source collar contour has invalid length")

    source_cut = {tuple(sorted(edge)) for edge in pairwise(map(int, source_loop))}
    _, collar_edges = edge_topology(collar_f)
    collar_cut = {edge for edge, incident in collar_edges.items() if len(incident) == 1 and max(edge) < len(source_p)}
    if collar_cut != source_cut:
        raise ValueError("source collar does not preserve the original cut loop")

    patch_ids = patch_loop[:-1].copy()
    patch_boundary_points = patch_p[patch_ids]
    _, arcs, residuals, cumulative = _project_to_contour(patch_boundary_points, contour)
    if np.max(residuals, initial=0.0) > max_displacement:
        raise ValueError("patch boundary exceeds the source-contour displacement budget")
    deltas = np.diff(np.r_[arcs, arcs[0]])
    forward_wraps = np.flatnonzero(deltas < -0.5 * circumference)
    reverse_wraps = np.flatnonzero(deltas > 0.5 * circumference)
    if len(forward_wraps) != 1 or len(reverse_wraps):
        raise ValueError("patch boundary does not follow the source contour direction")
    start = (int(forward_wraps[0]) + 1) % len(patch_ids)
    order = np.r_[np.arange(start, len(patch_ids)), np.arange(0, start)]
    ordered_ids = patch_ids[order]
    ordered_arcs = arcs[order].copy()
    for index in range(1, len(ordered_arcs)):
        if ordered_arcs[index] < ordered_arcs[index - 1] - 0.5 * circumference:
            ordered_arcs[index:] += circumference
    if np.any(np.diff(ordered_arcs) < -0.25 * circumference):
        raise ValueError("patch boundary has a large local contour reversal")

    coordinate_scale = max(float(np.linalg.norm(np.ptp(np.vstack((source_p, patch_p)), axis=0))), 1e-150)
    min_step = max(circumference * 1e-9, coordinate_scale * np.finfo(float).eps * 128)
    indexes = np.arange(len(ordered_arcs), dtype=np.float64)
    fitted = _pava(ordered_arcs - indexes * min_step) + indexes * min_step
    if fitted[-1] - fitted[0] >= circumference - min_step:
        raise ValueError("patch boundary contour order is not a single traversal")
    target_arcs = np.mod(fitted, circumference)
    target_points = np.empty((len(target_arcs), 3), dtype=np.float64)
    for index, arc in enumerate(target_arcs):
        segment = min(int(np.searchsorted(cumulative, arc, side="right") - 1), len(inner_lengths) - 1)
        parameter = (arc - cumulative[segment]) / inner_lengths[segment]
        target_points[index] = contour[segment] + parameter * (contour[segment + 1] - contour[segment])
    moves = np.linalg.norm(patch_p[ordered_ids] - target_points, axis=1)
    if np.max(moves, initial=0.0) > max_displacement:
        raise ValueError("monotone source-contour fit exceeds the displacement budget")

    snap_tolerance = max(circumference * 1e-9, coordinate_scale * np.finfo(float).eps * 256)
    corner_arcs = cumulative[:-1]
    corner_local_ids = inner_loop[:-1] - len(source_p)
    snapped_ids = np.full(len(ordered_ids), -1, dtype=np.int64)
    snapped_arcs = fitted.copy()
    for index, arc in enumerate(fitted):
        wrapped = arc % circumference
        delta = np.abs(corner_arcs - wrapped)
        delta = np.minimum(delta, circumference - delta)
        corner = int(np.argmin(delta))
        if delta[corner] <= snap_tolerance:
            snapped_ids[index] = corner_local_ids[corner]
            snapped_arcs[index] = arc + (
                (corner_arcs[corner] - wrapped + circumference / 2) % circumference - circumference / 2
            )
            target_points[index] = contour[corner]
    moves = np.linalg.norm(patch_p[ordered_ids] - target_points, axis=1)
    if np.max(moves, initial=0.0) > max_displacement:
        raise ValueError("snapping to a source-contour corner exceeds the displacement budget")
    if np.any(np.diff(snapped_arcs) < -snap_tolerance):
        raise ValueError("source-corner snapping reverses the contour order")

    collar_count = len(collar_p)
    local_ids = np.where(snapped_ids >= 0, snapped_ids, collar_count + ordered_ids)
    remap = collar_count + np.arange(len(patch_p), dtype=np.int64)
    remap[ordered_ids] = local_ids
    patched_points = patch_p.copy()
    patched_points[ordered_ids] = target_points
    combined_patch_points = np.vstack((collar_p, patched_points))
    mapped_faces = collar_count + patch_f
    mapped_faces = remap[mapped_faces - collar_count]
    valid_faces = np.asarray([len(set(map(int, face))) == 3 for face in mapped_faces])
    collapsed_faces = int(np.count_nonzero(~valid_faces))
    mapped_faces = mapped_faces[valid_faces]
    if not len(mapped_faces):
        raise ValueError("contour collapse removed every patch face")

    common_ids, common_arcs = [], []
    for index in range(len(ordered_ids)):
        point_id, arc = int(local_ids[index]), float(snapped_arcs[index])
        if common_ids and common_ids[-1] == point_id:
            continue
        common_ids.append(point_id)
        common_arcs.append(arc)
        next_arc = float(snapped_arcs[(index + 1) % len(snapped_arcs)])
        if index == len(snapped_arcs) - 1 or next_arc <= arc:
            next_arc += circumference
        corner_candidates = []
        for corner, base_arc in enumerate(corner_arcs):
            lift = base_arc + np.ceil((arc - base_arc) / circumference) * circumference
            if arc + snap_tolerance < lift < next_arc - snap_tolerance:
                corner_candidates.append((float(lift), int(corner_local_ids[corner])))
        for lift, corner_id in sorted(corner_candidates):
            if common_ids[-1] != corner_id:
                common_ids.append(corner_id)
                common_arcs.append(lift)
    if len(common_ids) > _MAX_CONTOUR_NODES:
        raise ValueError("common contour exceeds the 4096-node budget")
    if len(common_ids) > 1 and common_ids[-1] == common_ids[0]:
        common_ids.pop()
        common_arcs.pop()
    if len(set(common_ids)) != len(common_ids) or len(common_ids) < 3:
        raise ValueError("common source contour is not a simple loop")

    patch_insertions = {}
    for index in range(len(ordered_ids)):
        left = int(local_ids[index])
        right_index = (index + 1) % len(ordered_ids)
        right = int(local_ids[right_index])
        if left == right:
            continue
        arc = float(snapped_arcs[index])
        next_arc = float(snapped_arcs[right_index])
        if right_index == 0 or next_arc <= arc:
            next_arc += circumference
        inserts = []
        for corner, base_arc in enumerate(corner_arcs):
            lift = base_arc + np.ceil((arc - base_arc) / circumference) * circumference
            if arc + snap_tolerance < lift < next_arc - snap_tolerance:
                inserts.append((float(lift), int(corner_local_ids[corner])))
        if inserts:
            patch_insertions[(left, right)] = [point for _, point in sorted(inserts)]
    body_faces, _, combined_patch_points = _split_faces_on_edges(combined_patch_points, mapped_faces, patch_insertions)
    combined_global = np.vstack((source_p, combined_patch_points))
    common_global = np.asarray(common_ids, dtype=np.int64) + len(source_p)
    collar_insertions = {}
    for segment, (left, right) in enumerate(pairwise(map(int, inner_loop))):
        lo, hi = float(cumulative[segment]), float(cumulative[segment + 1])
        nodes = []
        for local_id, arc in zip(common_ids, common_arcs):
            lift = arc + np.ceil((lo - arc) / circumference) * circumference
            if lo + snap_tolerance < lift < hi - snap_tolerance:
                nodes.append((float(lift), int(local_id) + len(source_p)))
        if nodes:
            collar_insertions[(right, left)] = [node for _, node in sorted(nodes, reverse=True)]
    split_collar, collar_birth_rows, combined_global = _split_faces_on_edges(
        combined_global, collar_f, collar_insertions
    )
    combined_patch_points = combined_global[len(source_p) :]
    births_by_face = np.asarray(births, dtype=np.int64)
    split_births = births_by_face[collar_birth_rows]

    if len(body_faces) + len(split_collar) > _MAX_NEW_TRIANGLES:
        raise ValueError("conformal contour exceeds the 100000-triangle budget")
    generated_faces = np.vstack((body_faces + len(source_p), split_collar))
    normals = _area_normals(combined_global, generated_faces)
    topology, generated_edges = edge_topology(generated_faces)
    expected_boundary = source_cut
    actual_boundary = {edge for edge, incident in generated_edges.items() if len(incident) == 1}
    if topology["non_manifold_edges"] or topology["non_manifold_vertices"] or actual_boundary != expected_boundary:
        raise ValueError("conformal contour patch does not preserve the original outer seam topology")

    seam_angles, seam_edge_count = [], 0
    body_face_count = len(body_faces)
    for left, right in pairwise(np.r_[common_global, common_global[0]]):
        edge = tuple(sorted((int(left), int(right))))
        incident = generated_edges.get(edge, [])
        if (
            len(incident) != 2
            or not any(row < body_face_count for row in incident)
            or not any(row >= body_face_count for row in incident)
        ):
            raise ValueError("shared contour edge is not owned by one collar and one patch face")
        oriented = []
        for row in incident:
            face = generated_faces[row]
            direction = next(
                ((int(a), int(b)) for a, b in zip(face, np.roll(face, -1)) if tuple(sorted((int(a), int(b)))) == edge),
                None,
            )
            oriented.append(direction)
        if oriented[0] != oriented[1][::-1]:
            raise ValueError("shared contour faces have inconsistent edge direction")
        seam_angles.append(
            float(np.degrees(np.arccos(np.clip(normals[incident[0]] @ normals[incident[1]], -1.0, 1.0))))
        )
        seam_edge_count += 1
    max_seam_angle = max(seam_angles, default=float("inf"))
    if max_seam_angle > 20.0 + 1e-8:
        raise ValueError(f"conformal contour seam exceeds the 20 degree dihedral limit ({max_seam_angle:.6g} deg)")

    return (
        combined_patch_points,
        body_faces,
        {
            "collar_faces_count": len(split_collar),
            "collar_source_face_rows": split_births.astype(int).tolist(),
            "collar_inner_loop": inner_loop.astype(int).tolist(),
            "zipper_faces": split_collar.astype(int).tolist(),
            "zipper": {
                "status": "ready",
                "method": "shared_source_contour",
                "max_displacement": float(max_displacement),
                "max_normal_angle": 20.0,
                "max_seam_dihedral_deg": float(max_seam_angle),
                "seam_edges": seam_edge_count,
                "contour_nodes": len(common_ids),
                "collapsed_patch_faces": collapsed_faces,
                "faces": 0,
            },
            "patch": {"points": len(combined_patch_points), "triangles": len(body_faces)},
            "topology": topology,
        },
    )
