"""Replace one reviewed local surface region while retaining the surrounding mesh."""

from __future__ import annotations

import igl
import numpy as np
from continuous_surface_bound import source_to_candidate_bound
from local_defect_regions import extract_defect_regions
from local_patch_checks import _euler, _local_geometry, _seam_continuity
from local_sdf_patch import build_local_sdf_patch
from mesh_metrics import edge_topology
from source_projected_closure import _barycentric, _closest_source_points


def _assemble_patch(points, faces, selected, patch_points, patch_faces, patch_report):
    combined = np.vstack((points, patch_points))
    strip = np.asarray(patch_report["zipper_faces"], dtype=np.int64)
    keep = np.ones(len(faces), dtype=bool)
    keep[selected] = False
    generated = np.vstack((patch_faces + len(points), strip))
    output = np.vstack((faces[keep], generated))
    lineage = np.r_[np.flatnonzero(keep), np.full(len(generated), -1, dtype=np.int64)]
    return combined, output, lineage


def _cut_exposure(points, faces, selected, boundary_edges, pitch):
    """Verify that the fixed cut lies on the selected solid's exterior material boundary."""
    if len(boundary_edges) > 512:
        return {"passed": False, "reason": "source cut exceeds seam vertex budget"}
    selected = set(map(int, selected))
    _, edges = edge_topology(faces)
    probes, normals, steps = [], [], []
    for left, right in boundary_edges:
        incident = [row for row in edges[tuple(sorted((left, right)))] if row in selected]
        if len(incident) != 1:
            return {"passed": False, "reason": "source cut has ambiguous supporting faces"}
        triangle = points[faces[incident[0]]]
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        length = float(np.linalg.norm(normal))
        if length == 0:
            return {"passed": False, "reason": "source cut has degenerate supporting faces"}
        probes.append((points[left] + points[right]) * 0.4 + triangle.mean(axis=0) * 0.2)
        normals.append(normal / length)
        steps.append(min(pitch * 0.001, float(np.linalg.norm(points[left] - points[right])) * 0.001))
    probes, normals, steps = np.asarray(probes), np.asarray(normals), np.asarray(steps)
    resolution = max(float(np.max(np.abs(points))), 1.0) * np.finfo(float).eps * 256
    if not len(probes) or np.any(steps <= resolution):
        return {"passed": False, "reason": "source cut side probes are below coordinate precision"}
    offsets = normals * steps[:, None]
    winding = np.asarray(igl.winding_number(points, faces, np.vstack((probes + offsets, probes - offsets))))
    count = len(probes)
    valid = np.isfinite(winding[:count]) & np.isfinite(winding[count:])
    valid &= (np.abs(winding[:count]) < 0.25) & (winding[count:] > 0.75)
    invalid = np.flatnonzero(~valid)
    return {
        "passed": not len(invalid),
        "probes": count,
        "unexposed_source_edges": [list(boundary_edges[i]) for i in invalid],
        "reason": None if not len(invalid) else "source cut crosses buried or ambiguous material faces",
        "numerics": "floating-point winding side probes; global exact intersection gates remain required",
    }


def _reconstruct(points, faces, selected, pitch, max_error, max_voxels, trusted_rows):
    tri = points[faces[selected]]
    samples = tri.reshape(-1, 3)
    center = samples.mean(axis=0)
    _, _, axes = np.linalg.svd(samples - center, full_matrices=False)
    frame = axes.T
    if np.linalg.det(frame) < 0:
        frame[:, -1] *= -1
    local = (points - center) @ frame
    region_points = local[faces[selected]].reshape(-1, 3)
    bounds = np.array([region_points.min(axis=0) - 3 * pitch, region_points.max(axis=0) + 3 * pitch])
    area = float(np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum() / 2)
    edges = np.linalg.norm(tri[:, [1, 2, 0]] - tri[:, [0, 1, 2]], axis=2)
    _, patch_edges = edge_topology(faces[selected])
    rim_lengths = [
        float(np.linalg.norm(points[a] - points[b])) for (a, b), incident in patch_edges.items() if len(incident) == 1
    ]
    bridge = max(6 * pitch, float(np.median(edges)) * 3, max(rim_lengths) * 1.25)
    p, f, report = build_local_sdf_patch(
        local,
        faces,
        defect_face_rows=selected,
        roi_bounds=bounds,
        pitch=pitch,
        max_voxels=max_voxels,
        max_patch_area=area * 3,
        max_error=max_error,
        max_bridge_length=bridge,
        reliable_source_face_rows=trusted_rows,
    )
    report["coordinate_frame"] = {
        "kind": "local_pca",
        "origin_in_input": center.tolist(),
        "axes_in_input": frame.tolist(),
    }
    return (None if p is None else p @ frame.T + center), f, report, bridge


def _transport_generated_corner_normals(
    points, faces, candidate_rows, source_points, source_faces, source_corner_normals, trusted_rows, max_distance
):
    """Interpolate the native source corner field onto nearby generated faces."""
    if not len(candidate_rows):
        return np.empty((0, 9), dtype=source_corner_normals.dtype), np.zeros((len(candidate_rows), 3), dtype=bool)
    triangles = points[faces[candidate_rows]]
    queries = triangles.reshape(-1, 3)
    mapped_all = np.zeros((len(queries), 3), dtype=source_corner_normals.dtype)
    supported_all = np.zeros(len(queries), dtype=bool)
    if not len(trusted_rows):
        return mapped_all.reshape(-1, 9), supported_all.reshape(-1, 3)
    query_rows = np.flatnonzero(np.isfinite(queries).all(axis=1))
    if not len(query_rows):
        return mapped_all.reshape(-1, 9), supported_all.reshape(-1, 3)
    source_triangles = source_faces[trusted_rows]
    closest, local_face_ids, distances = _closest_source_points(source_points, source_triangles, queries[query_rows])
    valid_face_ids = (local_face_ids >= 0) & (local_face_ids < len(trusted_rows))
    source_rows = trusted_rows[np.clip(local_face_ids, 0, len(trusted_rows) - 1)]
    support_triangles = source_points[source_faces[source_rows]]
    weights = np.clip(_barycentric(closest, support_triangles), 0.0, 1.0)
    weight_sum = weights.sum(axis=1)
    weights /= np.maximum(weight_sum[:, None], 1e-20)

    face_rows = candidate_rows[query_rows // 3]
    generated_triangles = points[faces[face_rows]]
    generated_normals = np.cross(
        generated_triangles[:, 1] - generated_triangles[:, 0], generated_triangles[:, 2] - generated_triangles[:, 0]
    )
    generated_lengths = np.linalg.norm(generated_normals, axis=1)
    generated_normals /= np.maximum(generated_lengths[:, None], np.finfo(float).tiny)
    source_normals = np.cross(
        support_triangles[:, 1] - support_triangles[:, 0], support_triangles[:, 2] - support_triangles[:, 0]
    )
    source_lengths = np.linalg.norm(source_normals, axis=1)
    source_normals /= np.maximum(source_lengths[:, None], np.finfo(float).tiny)
    alignment = np.einsum("ij,ij->i", generated_normals, source_normals)
    supported = valid_face_ids & np.isfinite(closest).all(axis=1) & np.isfinite(distances)
    supported &= distances <= max_distance
    supported &= generated_lengths > 0
    supported &= source_lengths > 0
    supported &= np.isfinite(alignment) & (alignment >= np.cos(np.deg2rad(20.0)))
    supported &= np.isfinite(weights).all(axis=1) & (weight_sum > 1e-20)

    native = source_corner_normals[source_rows]
    mapped = np.einsum("ij,ijk->ik", weights, native)
    mapped_lengths = np.linalg.norm(mapped, axis=1)
    supported &= np.isfinite(mapped).all(axis=1) & np.isfinite(mapped_lengths) & (mapped_lengths > 1e-15)
    mapped_all[query_rows[supported]] = mapped[supported]
    supported_all[query_rows[supported]] = True
    return mapped_all.reshape(-1, 9), supported_all.reshape(-1, 3)


def _fill_generated_corner_normals(
    corners,
    points,
    faces,
    lineage,
    reconstruction,
    source_points,
    source_faces,
    source_corner_normals,
    trusted_rows,
    max_distance,
):
    generated_rows = np.flatnonzero(lineage < 0)
    collar_births = np.asarray(reconstruction.get("collar_source_face_rows", []), dtype=np.int64)
    collar_rows = np.empty(0, dtype=np.int64)
    if len(collar_births):
        start = int(np.count_nonzero(lineage >= 0)) + reconstruction["patch"]["triangles"]
        collar_rows = np.arange(start, start + len(collar_births))

    transport_rows = np.setdiff1d(generated_rows, collar_rows, assume_unique=True)
    mapped, supported = _transport_generated_corner_normals(
        points,
        faces,
        transport_rows,
        source_points,
        source_faces,
        source_corner_normals,
        trusted_rows,
        max_distance,
    )
    transported = corners[transport_rows].reshape(-1, 3, 3)
    transported[supported] = mapped.reshape(-1, 3, 3)[supported]
    corners[transport_rows] = transported.reshape(-1, 9)

    collar_corner_count = 0
    if len(collar_births):
        from source_projected_closure import interpolate_source_corner_normals

        source_flat = source_points[source_faces].reshape(-1, 3)
        source_triangles = np.arange(len(source_flat)).reshape(-1, 3)
        corners[collar_rows] = interpolate_source_corner_normals(
            points,
            faces[collar_rows],
            source_flat,
            source_triangles,
            source_corner_normals.reshape(-1, 3),
            collar_births,
        )
        collar_corner_count = 3 * len(collar_rows)
    return corners, {
        "trusted_source_faces": len(trusted_rows),
        "generated_corners_supported_by_trusted_faces": int(supported.sum()),
        "generated_corners_falling_back_to_geometry": int(supported.size - supported.sum()),
        "collar_birth_corners_transported": int(collar_corner_count),
    }


# @entry: a deterministic geometry operator; caller owns intent and subprocess deadline.
def repair_local_voxel_patch(
    points,
    faces,
    region_face_rows,
    *,
    pitch,
    max_error,
    max_voxels=2_000_000,
    max_region_faces=2000,
    max_seam_angle_increase=20.0,
    unreliable_source_face_rows=None,
):
    """Return a checked local replacement or unchanged source arrays on failure.

    The current SDF signer supports one closed positive source component and one
    simple cut loop. It does not invent thickness for open sheets. All source
    vertices and all triangles outside the selected region remain exact.
    """
    points, faces = np.asarray(points, dtype=np.float64), np.asarray(faces, dtype=np.int64)
    identity = np.arange(len(faces), dtype=np.int64)
    report = {
        "accepted": False,
        "local_repair_accepted": False,
        "engineering_ready": False,
        "method": "local_sdf_fixed_seam_replacement",
        "pitch": float(pitch),
        "max_error": float(max_error),
        "coordinate_units": "input units",
    }
    try:
        if (
            not np.isfinite(max_error)
            or max_error <= 0
            or not np.isfinite(pitch)
            or pitch <= 0
            or not np.isfinite(max_seam_angle_increase)
            or max_seam_angle_increase < 0
        ):
            raise ValueError("positive finite pitch and source-error budget are required")
        if len(faces) > 20000:
            raise ValueError("closed source context exceeds the local face budget")
        extracted = extract_defect_regions(
            points,
            faces,
            seed_faces=region_face_rows,
            halo_rings=0,
            max_region_faces=max_region_faces,
            max_total_faces=max_region_faces,
        )
        if len(extracted["regions"]) != 1 or extracted["regions"][0]["status"] != "ready":
            raise ValueError("selected source region is disconnected, branched or over budget")
        region = extracted["regions"][0]
        selected = np.asarray(region["face_rows"], dtype=np.int64)
        report.update(
            source_faces=len(faces),
            replaced_source_faces=len(selected),
            retained_source_faces=len(faces) - len(selected),
            boundary=region["boundary"],
        )
        report["cut_exposure"] = _cut_exposure(points, faces, selected, region["boundary_edges"], pitch)
        if not report["cut_exposure"]["passed"]:
            raise ValueError(report["cut_exposure"]["reason"])
        trusted = None
        if unreliable_source_face_rows is not None:
            bad = np.asarray(unreliable_source_face_rows, dtype=np.int64).reshape(-1)
            if np.any(bad < 0) or np.any(bad >= len(faces)):
                raise ValueError("unreliable source evidence contains invalid face rows")
            trusted = np.setdiff1d(selected, bad)
        patch_p, patch_f, sdf, bridge_limit = _reconstruct(
            points, faces, selected, pitch, max_error, max_voxels, trusted
        )
        report["reconstruction"] = sdf
        if patch_p is None:
            raise ValueError(sdf["reason"])
        if len(patch_f) > 100000:
            raise ValueError("local patch triangle budget exceeded")
        out_p, out_f, lineage = _assemble_patch(points, faces, selected, patch_p, patch_f, sdf)
        report["seam"] = _seam_continuity(points, faces, out_p, out_f, selected)
        if not report["seam"]["passed"] or report["seam"]["max_increase_deg"] > max_seam_angle_increase:
            raise ValueError("fixed seam continuity exceeded its angle budget")
        geometry, local = _local_geometry(out_p, out_f, lineage)
        report["geometry"] = geometry
        report["local_geometry"] = local
        if not local["passed"]:
            raise ValueError("local replacement introduced intersections or invalid geometry")
        generated = out_f[lineage < 0]
        report["patch_topology"] = {"before_euler": _euler(faces[selected]), "after_euler": _euler(generated)}
        if report["patch_topology"]["before_euler"] != report["patch_topology"]["after_euler"]:
            raise ValueError("local replacement changes source patch topology without reviewed intent")
        forward = source_to_candidate_bound(
            out_p, generated, points, faces, max_error=max_error, max_depth=9, max_subtriangles=200000
        )
        reverse = source_to_candidate_bound(
            points, faces[selected], out_p, out_f, max_error=max_error, max_depth=9, max_subtriangles=200000
        )
        report["source_distance"] = {
            "candidate_to_source": forward,
            "source_to_candidate": reverse,
            "numerics": "bounded floating-point queries; not an exact-arithmetic Hausdorff proof",
        }
        if not forward["passed"] or not reverse["passed"]:
            raise ValueError("local surface distance exceeded its budget or remains unproven")
        retained = lineage >= 0
        unchanged = np.array_equal(out_p[: len(points)], points) and np.array_equal(
            out_f[retained], faces[lineage[retained]]
        )
        if not unchanged:
            raise ValueError("local reconstruction modified protected source geometry")
        report.update(
            accepted=bool(unchanged and geometry["passed"]),
            local_repair_accepted=bool(unchanged),
            retained_geometry_unchanged=bool(unchanged),
            output_faces=len(out_f),
            output_points=len(out_p),
            bridge_limit=bridge_limit,
        )
        return out_p, out_f, lineage, report
    except (ValueError, RuntimeError) as exc:
        report.update(failure_reason=str(exc), rolled_back=True)
        return points, faces, identity, report


def repair_mesh_patch(source, region_face_rows, *, method="voxel", **settings):
    """Preserve retained source attributes and normals for the file-worker caller."""
    import pyvista as pv
    from mesh_io import triangle_faces

    original_faces = triangle_faces(source)
    if method == "intersection-cut":
        from local_intersection_cut import repair_local_intersection_cut

        p, f, lineage, report = repair_local_intersection_cut(
            source.points,
            original_faces,
            region_face_rows,
            max_error=settings["max_error"],
            max_seam_angle_increase=settings.get("max_seam_angle_increase", 20.0),
        )
    elif method == "voxel":
        p, f, lineage, report = repair_local_voxel_patch(source.points, original_faces, region_face_rows, **settings)
    elif method == "voxel-band":
        from voxel_band_patch import reconstruct_voxel_band

        p, f, lineage, report = reconstruct_voxel_band(source.points, original_faces, region_face_rows, **settings)
    else:
        raise ValueError("local repair method must be voxel, voxel-band or intersection-cut")
    if not report["local_repair_accepted"]:
        return source.copy(), report
    candidate = pv.PolyData(p, np.column_stack((np.full(len(f), 3), f)).ravel())
    point_lineage = np.r_[np.arange(source.n_points), np.full(len(p) - source.n_points, -1)]
    offset = source.n_verts + source.n_lines
    cell_lineage = (
        np.asarray(report["reconstruction"]["normal_source_face_rows"], dtype=np.int64)
        if method == "intersection-cut"
        else lineage
    )
    generated_attributes = []
    for original, destination, ids, cell_offset in (
        (source.point_data, candidate.point_data, point_lineage, 0),
        (source.cell_data, candidate.cell_data, cell_lineage, offset),
    ):
        for name, values in original.items():
            if name in ("local_patch_source_face_row", "local_patch_source_point_row"):
                continue
            values = np.asarray(values)
            if values.dtype.kind not in "biuf":
                continue
            fill = np.nan if values.dtype.kind == "f" else -1 if values.dtype.kind == "i" else 0
            result = np.full((len(ids), *values.shape[1:]), fill, dtype=values.dtype)
            known = ids >= 0
            result[known] = values[ids[known] + cell_offset]
            destination[name] = result
            generated_attributes.append(name)
    geometric = candidate.compute_normals(cell_normals=False, split_vertices=False)
    old_normals = source.point_data.get("Normals")
    if old_normals is None:
        old_normals = source.compute_normals(cell_normals=False, split_vertices=False).point_data["Normals"]
    normals = np.array(
        geometric.point_data["Normals"],
        dtype=np.result_type(geometric.point_data["Normals"].dtype, old_normals.dtype),
        copy=True,
    )
    normals[: source.n_points] = np.asarray(old_normals)
    candidate.point_data["Normals"] = normals
    kept = lineage >= 0
    old_corners = source.cell_data.get("corner_normals")
    corners = np.array(
        np.asarray(geometric.point_data["Normals"])[f].reshape(-1, 9),
        dtype=np.result_type(normals.dtype, old_corners.dtype if old_corners is not None else normals.dtype),
    )
    corners[kept] = (
        np.asarray(old_corners)[lineage[kept] + offset]
        if old_corners is not None
        else np.asarray(old_normals)[original_faces[lineage[kept]]].reshape(-1, 9)
    )
    normal_support_rows = np.setdiff1d(np.arange(len(original_faces)), lineage[kept])
    unreliable = settings.get("unreliable_source_face_rows")
    if unreliable is None:
        normal_support_rows = np.empty(0, dtype=np.int64)
    else:
        normal_support_rows = np.setdiff1d(normal_support_rows, np.asarray(unreliable, dtype=np.int64).reshape(-1))
    native_corner_normals = (
        np.asarray(old_corners)[offset : offset + len(original_faces)].reshape(-1, 3, 3)
        if old_corners is not None
        else np.asarray(old_normals)[original_faces]
    )
    if method == "intersection-cut":
        from source_projected_closure import interpolate_source_corner_normals

        source_flat = np.asarray(source.points)[original_faces].reshape(-1, 3)
        corners[~kept] = interpolate_source_corner_normals(
            p,
            f[~kept],
            source_flat,
            np.arange(len(source_flat)).reshape(-1, 3),
            native_corner_normals.reshape(-1, 3),
            cell_lineage[~kept],
        )
        normal_transport = {"generated_corners_from_exact_source_birth": int(np.count_nonzero(~kept) * 3)}
    else:
        corners, normal_transport = _fill_generated_corner_normals(
            corners,
            p,
            f,
            lineage,
            report["reconstruction"],
            np.asarray(source.points),
            original_faces,
            native_corner_normals,
            normal_support_rows,
            min(2 * float(settings["pitch"]), float(settings["max_error"])),
        )
    candidate.cell_data["corner_normals"] = corners
    candidate.cell_data["local_patch_source_face_row"] = lineage
    candidate.point_data["local_patch_source_point_row"] = point_lineage
    for name, values in source.field_data.items():
        candidate.field_data[name] = np.array(values, copy=True)
    candidate = candidate.clean(point_merging=False, lines_to_points=False, polys_to_lines=False, strips_to_polys=False)
    report["attributes"] = {
        "copied_retained_attributes": generated_attributes,
        "generated_values_missing_except_normals": (
            list(source.point_data) if method == "intersection-cut" else generated_attributes
        ),
        "generated_cell_values_inherit_exact_source_birth": method == "intersection-cut",
        "native_normal_transport": normal_transport,
    }
    report["operator_point_count_before_compaction"] = report["output_points"]
    report["output_points"] = candidate.n_points
    report["retained_original_points"] = int(
        np.count_nonzero(candidate.point_data["local_patch_source_point_row"] >= 0)
    )
    return candidate, report
