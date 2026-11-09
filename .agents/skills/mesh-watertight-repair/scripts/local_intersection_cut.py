"""Cut local self-intersections with an exact native-surface Boolean union."""

from __future__ import annotations

from numbers import Integral

import numpy as np
import trimesh
from continuous_surface_bound import source_to_candidate_bound
from local_defect_regions import extract_defect_regions
from local_patch_checks import _euler, _local_geometry, _seam_continuity
from material_cleanup import _union
from mesh_metrics import (
    edge_topology,
    face_component_labels,
    inconsistent_winding_edges,
)


def _cyclic_oriented_match(source_triangle: np.ndarray, candidate_triangle: np.ndarray) -> bool:
    return any(np.array_equal(source_triangle, np.roll(candidate_triangle, shift, axis=0)) for shift in range(3))


def _source_face_matches(points, faces, candidate_points, candidate_faces, birth):
    output_rows = [[] for _ in range(len(faces))]
    exact = np.zeros(len(candidate_faces), dtype=bool)
    for output_row, source_row in enumerate(birth):
        source_row = int(source_row)
        output_rows[source_row].append(output_row)
        exact[output_row] = _cyclic_oriented_match(
            points[faces[source_row]], candidate_points[candidate_faces[output_row]]
        )
    counts = np.asarray([len(rows) for rows in output_rows], dtype=np.int64)
    exact_counts = np.asarray([int(exact[rows].sum()) for rows in output_rows], dtype=np.int64)
    unchanged = (counts == 1) & (exact_counts == 1)
    return unchanged, exact, counts, exact_counts


def _append_new_points(points, candidate_points, candidate_faces):
    output_points = [row.copy() for row in points]
    coordinate_rows = {}
    for index, row in enumerate(points):
        coordinate_rows.setdefault(tuple(map(float, row)), index)
    candidate_to_output = np.full(len(candidate_points), -1, dtype=np.int64)
    used = np.unique(candidate_faces)
    for candidate_row in used:
        coordinate = candidate_points[candidate_row]
        key = tuple(map(float, coordinate))
        output_row = coordinate_rows.get(key)
        if output_row is None:
            output_row = len(output_points)
            output_points.append(coordinate.copy())
            coordinate_rows[key] = output_row
        elif not np.array_equal(output_points[output_row], coordinate):
            raise ValueError("exact coordinate lookup returned a nonmatching point")
        candidate_to_output[candidate_row] = output_row
    return np.asarray(output_points, dtype=np.float64), candidate_to_output


# @entry: propose a topology-preserving local cut using exact native-surface geometry.
def repair_local_intersection_cut(
    points,
    faces,
    region_face_rows,
    *,
    max_error,
    max_region_faces=2000,
    max_seam_angle_increase=20.0,
):
    """Return an exact local intersection cut, or unchanged source arrays on failure.

    The CGAL union keeps the source material surface and adds no thickness.
    Every source face outside the reviewed region must survive exactly once with
    its original oriented coordinates. Changed output faces keep their source
    birth row in ``report['reconstruction']['normal_source_face_rows']`` while
    their local repair lineage is ``-1``.
    """
    points = np.asarray(points, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    identity = np.arange(len(faces), dtype=np.int64)
    report = {
        "accepted": False,
        "local_repair_accepted": False,
        "engineering_ready": False,
        "method": "local_exact_intersection_cut",
        "max_error": float(max_error) if np.ndim(max_error) == 0 else None,
        "coordinate_units": "input units",
    }

    try:
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
            raise ValueError("expected finite points and valid nonempty triangle faces")
        rows = np.asarray(region_face_rows, dtype=np.int64).reshape(-1)
        if not len(rows) or rows.min() < 0 or rows.max() >= len(faces) or len(np.unique(rows)) != len(rows):
            raise ValueError("invalid selected source face rows")
        if (
            not np.isfinite(max_error)
            or max_error <= 0
            or not isinstance(max_region_faces, Integral)
            or max_region_faces <= 0
            or not np.isfinite(max_seam_angle_increase)
            or max_seam_angle_increase < 0
        ):
            raise ValueError("positive finite source error and seam budgets are required")
        if len(faces) > 20000:
            raise ValueError("closed source context exceeds the local face budget")
        if len(rows) > max_region_faces:
            raise ValueError("selected source region exceeds its face budget")

        topology, edge_faces = edge_topology(faces)
        components = np.unique(face_component_labels(len(faces), edge_faces))
        if (
            topology["boundary_edges"]
            or topology["non_manifold_edges"]
            or topology["non_manifold_vertices"]
            or inconsistent_winding_edges(faces)
            or len(components) != 1
        ):
            raise ValueError("source must be one closed consistently wound manifold component")
        source_mesh = trimesh.Trimesh(vertices=points, faces=faces, process=False)
        if not source_mesh.is_volume or float(source_mesh.volume) <= 0.0:
            raise ValueError("source component must have positive enclosed volume")

        extracted = extract_defect_regions(
            points,
            faces,
            seed_faces=rows,
            halo_rings=0,
            max_region_faces=int(max_region_faces),
            max_total_faces=int(max_region_faces),
        )
        if len(extracted["regions"]) != 1 or extracted["regions"][0]["status"] != "ready":
            raise ValueError("selected source region is disconnected, branched or over budget")
        region = extracted["regions"][0]
        selected = np.asarray(region["face_rows"], dtype=np.int64)
        if not np.array_equal(selected, np.sort(rows)):
            raise ValueError("selected source region is not one simple connected patch")

        report.update(
            source_faces=len(faces),
            selected_source_faces=len(selected),
            boundary=region["boundary"],
        )
        candidate_points, candidate_faces, birth = _union(
            points,
            faces,
            np.arange(len(faces), dtype=np.int64),
        )
        candidate_points = np.asarray(candidate_points, dtype=np.float64)
        candidate_faces = np.asarray(candidate_faces, dtype=np.int64)
        birth = np.asarray(birth, dtype=np.int64)
        if (
            candidate_points.ndim != 2
            or candidate_points.shape[1] != 3
            or candidate_faces.ndim != 2
            or candidate_faces.shape[1] != 3
            or not len(candidate_faces)
            or not np.isfinite(candidate_points).all()
            or candidate_faces.min() < 0
            or candidate_faces.max() >= len(candidate_points)
            or birth.shape != (len(candidate_faces),)
            or np.any(birth < 0)
            or np.any(birth >= len(faces))
        ):
            raise ValueError("CGAL union returned invalid geometry or source-face births")

        unchanged, exact_faces, output_counts, exact_counts = _source_face_matches(
            points, faces, candidate_points, candidate_faces, birth
        )
        changed_source_rows = np.flatnonzero(~unchanged)
        report.update(
            replaced_source_faces=len(changed_source_rows),
            retained_source_faces=int(len(faces) - len(changed_source_rows)),
        )
        outside = np.ones(len(faces), dtype=bool)
        outside[selected] = False
        outside_rows = np.flatnonzero(outside)
        changed_outside = outside_rows[~unchanged[outside_rows]]
        generated_candidate_rows = np.flatnonzero(~unchanged[birth])
        retained_rows = np.flatnonzero(unchanged)
        report["face_preservation"] = {
            "outside_roi": {
                "source_faces": len(outside_rows),
                "exact_oriented_faces_preserved_once": int(unchanged[outside_rows].sum()),
                "failed_source_rows": changed_outside.astype(int).tolist(),
                "all_preserved": bool(len(changed_outside) == 0),
            },
            "inside_roi": {
                "source_faces": len(selected),
                "unchanged_source_rows": np.intersect1d(retained_rows, selected).astype(int).tolist(),
                "changed_source_rows": np.intersect1d(changed_source_rows, selected).astype(int).tolist(),
            },
            "changed_output_faces": len(generated_candidate_rows),
            "changed_output_face_rows": generated_candidate_rows.astype(int).tolist(),
            "source_output_face_counts": output_counts.astype(int).tolist(),
            "source_exact_match_counts": exact_counts.astype(int).tolist(),
            "candidate_faces_exactly_matching_source": int(exact_faces.sum()),
        }
        if len(changed_outside):
            raise ValueError("exact union changed, split or removed a source face outside the selected region")
        if not len(changed_source_rows):
            raise ValueError("exact union made no source-face changes in the selected region")
        if np.any(outside[birth[~unchanged[birth]]]):
            raise ValueError("exact union created a face born outside the selected region")

        if not len(generated_candidate_rows):
            raise ValueError("exact union returned no changed faces for the selected region")
        assembled_points, candidate_to_output = _append_new_points(
            points, candidate_points, candidate_faces[generated_candidate_rows]
        )
        generated_faces = candidate_to_output[candidate_faces[generated_candidate_rows]]
        if np.any(generated_faces < 0):
            raise ValueError("generated faces reference an unmapped point")
        output_faces = np.vstack((faces[retained_rows], generated_faces))
        lineage = np.r_[retained_rows, np.full(len(generated_faces), -1, dtype=np.int64)]
        normal_source_face_rows = np.r_[retained_rows, birth[generated_candidate_rows]]
        generated = output_faces[lineage < 0]
        report["reconstruction"] = {
            "schema": "local_intersection_cut/v1",
            "status": "ready",
            "reason": "exact native-surface intersections removed inside the selected region",
            "method": "exact_source_intersection_cut",
            "source_roi": {
                "face_rows": selected.astype(int).tolist(),
                "changed_source_face_rows": changed_source_rows.astype(int).tolist(),
            },
            "patch": {"points": int(len(assembled_points) - len(points)), "triangles": len(generated)},
            "normal_source_face_rows": normal_source_face_rows.astype(int).tolist(),
            "collar_source_face_rows": [],
        }

        report["seam"] = _seam_continuity(points, faces, assembled_points, output_faces, selected)
        if not report["seam"]["passed"] or report["seam"]["max_increase_deg"] > max_seam_angle_increase:
            raise ValueError("fixed seam continuity exceeded its angle budget")

        geometry, local_geometry = _local_geometry(assembled_points, output_faces, lineage)
        report["geometry"] = geometry
        report["local_geometry"] = local_geometry
        if not geometry["passed"] or not local_geometry["passed"]:
            raise ValueError("exact union output failed local geometry or intersection gates")

        roi_output_rows = np.flatnonzero(np.isin(normal_source_face_rows, selected))
        patch_topology = {
            "before_euler": _euler(faces[selected]),
            "after_euler": _euler(output_faces[roi_output_rows]),
        }
        report["patch_topology"] = patch_topology
        if patch_topology["before_euler"] != patch_topology["after_euler"]:
            raise ValueError("exact cut changes selected-region topology without reviewed intent")

        forward = source_to_candidate_bound(
            points,
            faces[changed_source_rows],
            assembled_points,
            output_faces,
            max_error=max_error,
            max_depth=9,
            max_subtriangles=200000,
        )
        reverse = source_to_candidate_bound(
            assembled_points,
            generated,
            points,
            faces,
            max_error=max_error,
            max_depth=9,
            max_subtriangles=200000,
        )
        report["source_distance"] = {
            "candidate_to_source": reverse,
            "source_to_candidate": forward,
            "numerics": "bounded floating-point queries; not an exact-arithmetic Hausdorff proof",
        }
        if not forward["passed"] or not reverse["passed"]:
            raise ValueError("local source distance exceeded its budget or remains unproven")

        retained_geometry_unchanged = np.array_equal(assembled_points[: len(points)], points)
        retained_geometry_unchanged &= np.array_equal(output_faces[lineage >= 0], faces[lineage[lineage >= 0]])
        if not retained_geometry_unchanged:
            raise ValueError("retained source vertices or oriented faces were modified")
        report.update(
            retained_geometry_unchanged=bool(retained_geometry_unchanged),
            output_faces=len(output_faces),
            output_points=len(assembled_points),
            accepted=True,
            local_repair_accepted=True,
            rolled_back=False,
        )
        return assembled_points, output_faces, lineage, report
    except (TypeError, ValueError, RuntimeError, OverflowError) as exc:
        report.update(
            failure_reason=str(exc),
            rolled_back=True,
            local_repair_accepted=False,
            accepted=False,
        )
        return points, faces, identity, report
