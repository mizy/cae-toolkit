"""Reconstruct one source-conforming voxel band around a local surface defect."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pyvista as pv
import trimesh
from local_defect_regions import _boundary, extract_defect_regions
from local_patch_checks import _euler, _local_geometry, _seam_continuity
from local_voxel_field import build_local_zero_surface
from mesh_metrics import (
    edge_topology,
    face_component_labels,
    inconsistent_winding_edges,
)
from source_projected_validation import exact_self_intersection_report


def _rows(values, face_count: int, name: str) -> np.ndarray:
    rows = np.asarray(values, dtype=np.int64).reshape(-1)
    if (
        not len(rows)
        or rows.min() < 0
        or rows.max() >= face_count
        or len(np.unique(rows)) != len(rows)
    ):
        raise ValueError(f"{name} must contain unique valid face rows")
    return np.sort(rows)


def _region_rings(
    faces: np.ndarray, selected: np.ndarray, seeds: np.ndarray
) -> dict[str, Any]:
    selected_set = set(map(int, selected))
    seed_set = set(map(int, seeds))
    _, edges = edge_topology(faces[selected])
    local_rows = np.asarray(selected, dtype=np.int64)
    neighbors: dict[int, set[int]] = {int(row): set() for row in local_rows}
    for incident in edges.values():
        if len(incident) == 2:
            left, right = int(local_rows[incident[0]]), int(local_rows[incident[1]])
            neighbors[left].add(right)
            neighbors[right].add(left)
    distance = {row: 0 for row in seed_set}
    frontier = sorted(seed_set)
    for row in frontier:
        if row not in selected_set:
            raise ValueError(
                "unreliable source evidence must lie inside the selected band"
            )
    while frontier:
        current = frontier.pop(0)
        for other in neighbors[current]:
            if other not in distance:
                distance[other] = distance[current] + 1
                frontier.append(other)
    counts: dict[int, int] = {}
    for row, ring in distance.items():
        counts[ring] = counts.get(ring, 0) + 1
    return {
        "seed_faces": len(seed_set),
        "ring_face_counts": [
            counts[index] for index in range(max(counts, default=-1) + 1)
        ],
        "reconstructed_faces": len(selected),
        "unreached_faces": len(selected_set - set(distance)),
    }


def _feature_edges(
    points, faces, source_edges, selected, defect, *, min_angle_deg=30.0
):
    selected_set, defect_set = set(map(int, selected)), set(map(int, defect))
    triangles = points[faces]
    normals = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    lengths = np.linalg.norm(normals, axis=1)
    normals /= np.maximum(lengths[:, None], np.finfo(float).tiny)
    features = set()
    feature_faces = set()
    cosine = float(np.cos(np.deg2rad(min_angle_deg)))
    for edge, incident in source_edges.items():
        if len(incident) != 2 or not all(row in selected_set for row in incident):
            continue
        if (
            any(row in defect_set for row in incident)
            or min(lengths[incident]) <= np.finfo(float).tiny
        ):
            continue
        if float(np.dot(normals[incident[0]], normals[incident[1]])) < cosine:
            features.add(edge)
            feature_faces.update(incident)
    vertices = (
        np.unique(faces[np.asarray(sorted(feature_faces), dtype=np.int64)].reshape(-1))
        if feature_faces
        else np.empty(0, dtype=np.int64)
    )
    return features, vertices, np.asarray(sorted(feature_faces), dtype=np.int64)


def _midpoint_subdivide(
    points: np.ndarray,
    patch: np.ndarray,
    boundary_edges: set[tuple[int, int]],
    target: float,
    max_faces: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Refine only interior edges; source cut edges and their vertices stay fixed."""
    coordinates = np.asarray(points, dtype=np.float64).tolist()
    faces = np.asarray(patch, dtype=np.int64).copy()
    rounds = 0
    for rounds in range(1, 25):
        _, edges = edge_topology(faces)
        boundary_faces = {
            incident[0]
            for edge, incident in edges.items()
            if edge in boundary_edges and len(incident) == 1
        }
        current_points = np.asarray(coordinates, dtype=np.float64)
        split = {
            edge
            for edge, incident in edges.items()
            if len(incident) == 2
            and edge not in boundary_edges
            and not any(row in boundary_faces for row in incident)
            and np.linalg.norm(current_points[edge[0]] - current_points[edge[1]])
            > target
        }
        if not split:
            return np.asarray(coordinates, dtype=np.float64), faces, rounds - 1
        if len(faces) + 3 * len(split) > max_faces:
            raise ValueError("voxel band triangle budget exceeded")
        midpoint = {}
        for edge in sorted(split):
            midpoint[edge] = len(coordinates)
            coordinates.append(
                ((current_points[edge[0]] + current_points[edge[1]]) * 0.5).tolist()
            )
        output = []
        for a, b, c in faces:
            e0, e1, e2 = (int(a), int(b)), (int(b), int(c)), (int(c), int(a))
            m0 = midpoint.get(tuple(sorted(e0)))
            m1 = midpoint.get(tuple(sorted(e1)))
            m2 = midpoint.get(tuple(sorted(e2)))
            count = sum(value is not None for value in (m0, m1, m2))
            if count == 0:
                output.append([a, b, c])
            elif count == 1:
                if m0 is not None:
                    output.extend(([a, m0, c], [m0, b, c]))
                elif m1 is not None:
                    output.extend(([b, m1, a], [m1, c, a]))
                else:
                    output.extend(([c, m2, b], [m2, a, b]))
            elif count == 2:
                if m0 is not None and m1 is not None:
                    output.extend(([b, m1, m0], [a, m0, m1], [a, m1, c]))
                elif m1 is not None and m2 is not None:
                    output.extend(([c, m2, m1], [b, m1, m2], [b, m2, a]))
                else:
                    output.extend(([a, m0, m2], [c, m2, m0], [c, m0, b]))
            else:
                output.extend(([a, m0, m2], [m0, b, m1], [m2, m1, c], [m0, m1, m2]))
        faces = np.asarray(output, dtype=np.int64)
    return np.asarray(coordinates, dtype=np.float64), faces, rounds


def _closest_points(surface: pv.PolyData, queries: np.ndarray) -> np.ndarray:
    points = np.asarray(queries, dtype=np.float64)
    cell_ids, closest = surface.find_closest_cell(points, return_closest_point=True)
    cell_ids = np.asarray(cell_ids).reshape(-1)
    closest = np.asarray(closest, dtype=np.float64)
    if (
        cell_ids.shape != (len(points),)
        or np.any(cell_ids < 0)
        or closest.shape != points.shape
        or not np.isfinite(closest).all()
    ):
        raise ValueError("voxel surface projection returned invalid cells or points")
    return closest


def _zero_surface_normals(surface: pv.PolyData) -> np.ndarray:
    packed = np.asarray(surface.faces, dtype=np.int64).reshape(-1, 4)
    triangles = np.asarray(surface.points, dtype=np.float64)[packed[:, 1:]]
    normals = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    lengths = np.linalg.norm(normals, axis=1)
    if not np.isfinite(lengths).all() or np.any(lengths <= np.finfo(float).tiny):
        raise ValueError("voxel surface contains degenerate triangles")
    return normals / lengths[:, None]


def _closest_points_and_normals(
    surface: pv.PolyData, queries: np.ndarray, surface_normals: np.ndarray
):
    points = np.asarray(queries, dtype=np.float64)
    cell_ids, closest = surface.find_closest_cell(points, return_closest_point=True)
    cell_ids = np.asarray(cell_ids).reshape(-1)
    closest = np.asarray(closest, dtype=np.float64)
    if (
        cell_ids.shape != (len(points),)
        or np.any(cell_ids < 0)
        or np.any(cell_ids >= len(surface_normals))
        or closest.shape != points.shape
        or not np.isfinite(closest).all()
        or not np.isfinite(surface_normals[cell_ids]).all()
    ):
        raise ValueError("voxel surface projection returned invalid cells or normals")
    return closest, surface_normals[cell_ids]


def _tangential_relax(points, faces, fixed_vertices, surface, *, passes=32, weight=0.2):
    """Fair the whole band tangentially, then reproject it to the voxel zero set."""
    points = np.asarray(points, dtype=np.float64).copy()
    fixed = np.zeros(len(points), dtype=bool)
    fixed[np.asarray(fixed_vertices, dtype=np.int64)] = True
    faces = np.asarray(faces, dtype=np.int64)
    edge_rows = np.unique(
        np.sort(
            np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1
        ),
        axis=0,
    )
    directed = np.vstack((edge_rows, edge_rows[:, ::-1]))
    adjacency_counts = np.bincount(directed[:, 0], minlength=len(points))
    movable = np.flatnonzero(~fixed & (adjacency_counts > 0))
    surface_normals = _zero_surface_normals(surface)
    before = points.copy()
    for _ in range(passes):
        sums = np.zeros_like(points)
        np.add.at(sums, directed[:, 0], points[directed[:, 1]])
        active = movable[adjacency_counts[movable] > 0]
        if not len(active):
            break
        closest, normals = _closest_points_and_normals(
            surface, points[active], surface_normals
        )
        tangent = sums[active] / adjacency_counts[active, None] - points[active]
        tangent -= np.einsum("ij,ij->i", tangent, normals)[:, None] * normals
        proposal = closest + weight * tangent
        snapped = _closest_points(surface, proposal)
        points[active] = snapped
        points[fixed] = before[fixed]
    return points


def _distance_gate(points, faces, candidate_points, patch_faces, selected, max_error):
    from continuous_surface_bound import source_to_candidate_bound

    candidate_to_source = source_to_candidate_bound(
        candidate_points,
        patch_faces,
        points,
        faces,
        max_error=max_error,
        max_depth=9,
        max_subtriangles=200_000,
    )
    source_to_candidate = source_to_candidate_bound(
        points,
        faces[selected],
        candidate_points,
        patch_faces,
        max_error=max_error,
        max_depth=9,
        max_subtriangles=200_000,
    )
    return {
        "passed": bool(
            candidate_to_source.get("passed") and source_to_candidate.get("passed")
        ),
        "limit": float(max_error),
        "candidate_to_full_source": candidate_to_source,
        "full_selected_source_to_candidate": source_to_candidate,
        "method": "bounded_source_to_candidate_triangle_distance",
    }


def reconstruct_voxel_band(
    points,
    faces,
    region_face_rows,
    *,
    pitch,
    max_error,
    max_voxels=2_000_000,
    max_region_faces=2000,
    smoothing_sigma_voxels=0.5,
    max_seam_angle_increase=20.0,
    unreliable_source_face_rows=None,
):
    """Rebuild a defect and both adjacent seam strips through one local voxel field.

    ``region_face_rows`` must include the defect and its seam-side strips. The
    unreliable rows identify defect seeds; geodesic rings inside the selected
    region describe the reconstructed transition band. Its source patch is
    conformingly refined, projected onto one locally signed, smoothed voxel
    zero surface, and relaxed tangentially while the outer source loop stays
    fixed. No collar or zipper surface is created.

    Returns ``(points, faces, lineage, report)``. A rejected proposal returns
    the exact input arrays and identity lineage. A caller must supervise this synchronous operator with a hard
    process timeout because a native winding, locator, or CGAL call cannot be
    interrupted safely from inside Python.
    """
    started = time.monotonic()
    source_points = np.asarray(points, dtype=np.float64)
    source_faces = np.asarray(faces, dtype=np.int64)
    identity = np.arange(len(source_faces), dtype=np.int64)
    report: dict[str, Any] = {
        "accepted": False,
        "local_repair_accepted": False,
        "engineering_ready": False,
        "rolled_back": False,
        "method": "source_conforming_voxel_band",
        "band": {},
        "grid": {},
        "boundary": {},
        "elapsed_seconds": 0.0,
        "hard_timeout_owner": "supervised caller process",
    }

    def reject(reason):
        if isinstance(report.get("reconstruction"), dict):
            report["reconstruction"].update(status="rejected", reason=str(reason))
        report.update(
            failure_reason=str(reason),
            rolled_back=True,
            elapsed_seconds=time.monotonic() - started,
        )
        return source_points.copy(), source_faces.copy(), identity, report

    try:
        if (
            source_points.ndim != 2
            or source_points.shape[1] != 3
            or not len(source_points)
            or source_faces.ndim != 2
            or source_faces.shape[1] != 3
            or not len(source_faces)
            or not np.isfinite(source_points).all()
            or source_faces.min() < 0
            or source_faces.max() >= len(source_points)
        ):
            raise ValueError("expected finite points and valid nonempty triangle faces")
        if (
            not np.isfinite(pitch)
            or pitch <= 0
            or not np.isfinite(max_error)
            or max_error <= 0
            or not np.isfinite(smoothing_sigma_voxels)
            or smoothing_sigma_voxels < 0
            or not np.isfinite(max_seam_angle_increase)
            or max_seam_angle_increase < 0
            or max_voxels <= 0
            or max_region_faces <= 0
        ):
            raise ValueError(
                "positive finite geometry limits and voxel budgets are required"
            )
        selected = _rows(region_face_rows, len(source_faces), "region_face_rows")
        unreliable_values = np.asarray(
            [] if unreliable_source_face_rows is None else unreliable_source_face_rows,
            dtype=np.int64,
        ).reshape(-1)
        unreliable = (
            np.empty(0, dtype=np.int64)
            if not len(unreliable_values)
            else _rows(
                unreliable_values, len(source_faces), "unreliable_source_face_rows"
            )
        )
        source_intersections = exact_self_intersection_report(
            source_points, source_faces
        )
        if source_intersections.get("status") != "computed":
            raise ValueError("exact source intersection check is unavailable")
        pairs = np.asarray(
            source_intersections.get("reported_pairs", []), dtype=np.int64
        ).reshape(-1, 2)
        if source_intersections.get("reported_pairs_truncated"):
            raise ValueError(
                "source defect pair report exceeded its exact-predicate budget"
            )
        if unreliable_source_face_rows is None:
            defect = np.unique(pairs).astype(np.int64)
        else:
            defect = unreliable
        if not len(defect) and len(pairs):
            defect = np.unique(pairs).astype(np.int64)
        if np.any(~np.isin(defect, selected)):
            raise ValueError(
                "unreliable source evidence must lie inside the selected band"
            )
        if len(selected) > max_region_faces or len(source_faces) > 20_000:
            raise ValueError(
                "voxel band or closed source context exceeds its face budget"
            )

        topology, source_edges = edge_topology(source_faces)
        component_ids = np.unique(
            face_component_labels(len(source_faces), source_edges)
        )
        if (
            topology["boundary_edges"]
            or topology["non_manifold_edges"]
            or topology["non_manifold_vertices"]
            or inconsistent_winding_edges(source_faces)
            or len(component_ids) != 1
        ):
            raise ValueError(
                "source must be one closed, consistently wound manifold component"
            )
        source_volume = trimesh.Trimesh(
            vertices=source_points, faces=source_faces, process=False
        )
        if not np.isfinite(source_volume.volume) or float(source_volume.volume) <= 0:
            raise ValueError("source component must have positive enclosed volume")

        region = extract_defect_regions(
            source_points,
            source_faces,
            seed_faces=selected,
            halo_rings=0,
            max_region_faces=max_region_faces,
            max_total_faces=max_region_faces,
        )
        if (
            len(region["regions"]) != 1
            or region["regions"][0]["status"] != "ready"
            or not np.array_equal(
                np.asarray(region["regions"][0]["face_rows"], dtype=np.int64), selected
            )
        ):
            raise ValueError(
                "defect and both strips must form one simple connected source patch"
            )
        _, loops, boundary = _boundary(source_faces, selected, source_edges)
        if boundary["status"] != "closed_loops" or len(loops) != 1:
            raise ValueError("voxel band requires one simple fixed source boundary")
        boundary_edges = {
            tuple(sorted((int(left), int(right))))
            for left, right in zip(loops[0][:-1], loops[0][1:])
        }
        feature_edges, feature_vertices, feature_faces = _feature_edges(
            source_points,
            source_faces,
            source_edges,
            selected,
            defect,
        )
        report["band"] = _region_rings(source_faces, selected, defect)
        report["boundary"] = {
            "edges": len(boundary_edges),
            "vertices": len(loops[0]) - 1,
            "max_displacement": 0.0,
            "fixed": True,
        }
        report["source_features"] = {
            "angle_threshold_deg": 30.0,
            "protected_edges": len(feature_edges),
            "protected_faces": len(feature_faces),
            "defect_incident_edges_excluded": True,
        }

        selected_mask = np.zeros(len(source_faces), dtype=bool)
        selected_mask[selected] = True
        uncovered = (
            pairs[~np.all(selected_mask[pairs], axis=1)]
            if len(pairs)
            else np.empty((0, 2), dtype=np.int64)
        )
        if len(uncovered):
            raise ValueError("exact source defects extend outside the selected band")
        report["source_intersections"] = {
            "status": source_intersections["status"],
            "pairs": len(pairs),
            "all_pairs_inside_band": True,
        }

        region_points = source_points[source_faces[selected]].reshape(-1, 3)
        center = region_points.mean(axis=0)
        _, _, axes = np.linalg.svd(region_points - center, full_matrices=False)
        frame = axes.T
        if np.linalg.det(frame) < 0:
            frame[:, -1] *= -1
        local_points = (source_points - center) @ frame
        local_region_points = local_points[source_faces[selected]].reshape(-1, 3)
        bounds = np.asarray(
            [local_region_points.min(axis=0), local_region_points.max(axis=0)]
        )
        source_surface = pv.PolyData(
            local_points,
            np.column_stack((np.full(len(source_faces), 3), source_faces)).ravel(),
        )
        zero_points, zero_faces, grid, failure, status = build_local_zero_surface(
            local_points,
            source_faces,
            source_surface,
            bounds,
            pitch=float(pitch),
            max_voxels=int(max_voxels),
            smoothing_sigma_voxels=float(smoothing_sigma_voxels),
        )
        report["grid"] = grid
        if failure:
            raise ValueError(failure)
        if (
            status != "ready"
            or zero_points is None
            or zero_faces is None
            or not len(zero_faces)
        ):
            raise ValueError("local voxel field did not produce a zero surface")
        global_zero_points = zero_points @ frame.T + center
        zero_surface = pv.PolyData(
            global_zero_points,
            np.column_stack((np.full(len(zero_faces), 3), zero_faces)).ravel(),
        )

        source_patch = source_faces[selected]
        boundary_vertices = np.unique(
            np.asarray(
                [vertex for edge in boundary_edges for vertex in edge], dtype=np.int64
            )
        )
        fixed_vertices = np.union1d(boundary_vertices, feature_vertices)
        patch_vertices = np.unique(source_patch)
        copied_vertices = np.setdiff1d(patch_vertices, fixed_vertices)
        copy_ids = np.arange(
            len(source_points),
            len(source_points) + len(copied_vertices),
            dtype=np.int64,
        )
        source_to_scaffold = np.arange(len(source_points), dtype=np.int64)
        source_to_scaffold[copied_vertices] = copy_ids
        scaffold_points = np.vstack((source_points, source_points[copied_vertices]))
        patch = source_to_scaffold[source_patch]
        feature_face_edges = {
            tuple(sorted((int(left), int(right))))
            for face in source_faces[feature_faces]
            for left, right in zip(face, np.roll(face, -1))
        }
        fixed_edges = boundary_edges | feature_face_edges
        refined_points, refined_faces, rounds = _midpoint_subdivide(
            scaffold_points,
            patch,
            fixed_edges,
            target=4.0 * pitch,
            max_faces=max_region_faces * 20,
        )
        interior_vertices = np.setdiff1d(np.unique(refined_faces), fixed_vertices)
        projected = refined_points.copy()
        projected[interior_vertices] = _closest_points(
            zero_surface, refined_points[interior_vertices]
        )
        projected = _tangential_relax(
            projected, refined_faces, fixed_vertices, zero_surface
        )
        displacement = np.linalg.norm(projected - refined_points, axis=1)
        if not np.isfinite(displacement).all():
            raise ValueError("voxel-band projection returned non-finite coordinates")
        report["reconstruction"] = {
            "method": "source_conforming_voxel_band",
            "status": "ready",
            "reason": "one voxel zero field controls geometry across the selected defect and transition rings",
            "field": "one signed-distance volume over defect and both seam-side strips",
            "surface": "source-conforming refined patch projected onto that voxel zero set",
            "subdivision_rounds": rounds,
            "tangential_fairing_passes": 32,
            "tangential_fairing_weight": 0.2,
            "source_patch_faces": len(patch),
            "output_patch_faces": len(refined_faces),
            "projected_vertices": len(interior_vertices),
            "seed_vertices": len(np.unique(source_faces[defect])) if len(defect) else 0,
            "max_scaffold_displacement": float(
                displacement[interior_vertices].max(initial=0.0)
            ),
            "copied_source_vertices": len(copied_vertices),
            "fixed_boundary_edges": len(boundary_edges),
            "fixed_source_feature_edges": len(feature_edges),
            "separate_collar_or_zipper_faces": 0,
            "patch": {
                "points": len(np.unique(refined_faces)),
                "triangles": len(refined_faces),
            },
            "collar_source_face_rows": [],
        }

        outside_mask = np.ones(len(source_faces), dtype=bool)
        outside_mask[selected] = False
        outside = np.flatnonzero(outside_mask)
        candidate_points = projected
        candidate_faces = np.vstack((source_faces[outside], refined_faces))
        lineage = np.r_[outside, np.full(len(refined_faces), -1, dtype=np.int64)]
        report["reconstruction"]["output_points"] = len(candidate_points)
        report["reconstruction"]["output_faces"] = len(candidate_faces)
        if not np.array_equal(candidate_points[: len(source_points)], source_points):
            raise ValueError("voxel band moved a point used by retained source faces")
        if not np.array_equal(
            candidate_points[boundary_vertices], source_points[boundary_vertices]
        ):
            raise ValueError("fixed source boundary vertices moved")
        report["outside_source"] = {"faces": len(outside), "exactly_preserved": True}
        report["seam"] = _seam_continuity(
            source_points,
            source_faces,
            candidate_points,
            candidate_faces,
            selected,
        )
        if (
            not report["seam"]["passed"]
            or report["seam"]["max_increase_deg"] > max_seam_angle_increase
        ):
            raise ValueError("fixed seam continuity exceeded its angle budget")

        report["source_deviation"] = _distance_gate(
            source_points,
            source_faces,
            candidate_points,
            refined_faces,
            selected,
            max_error,
        )
        if not report["source_deviation"]["passed"]:
            raise ValueError("source deviation exceeded its limit or remains unproven")

        patch_topology = {
            "before_euler": _euler(source_faces[selected]),
            "after_euler": _euler(refined_faces),
        }
        report["patch_topology"] = patch_topology
        if patch_topology["before_euler"] != patch_topology["after_euler"]:
            raise ValueError("voxel band changed the selected surface topology")
        geometry, local_geometry = _local_geometry(
            candidate_points, candidate_faces, lineage
        )
        report["geometry"] = geometry
        report["local_geometry"] = local_geometry
        if not local_geometry["passed"] or not geometry.get("passed"):
            raise ValueError(
                "voxel-band candidate failed exact topology, volume, or intersection gates"
            )
        report.update(
            accepted=True,
            local_repair_accepted=True,
            retained_geometry_unchanged=True,
            output_faces=len(candidate_faces),
            output_points=len(candidate_points),
        )
        report["elapsed_seconds"] = time.monotonic() - started
        return candidate_points, candidate_faces, lineage, report
    except (ValueError, RuntimeError, TypeError, MemoryError) as exc:
        return reject(exc)
