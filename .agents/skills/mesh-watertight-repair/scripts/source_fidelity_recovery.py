"""Recover source facets whose visible first-hit geometry exceeds an error budget."""
from __future__ import annotations

from typing import Any
from collections.abc import Callable

import igl
import manifold3d as manifold
import numpy as np
import pyvista as pv
from scipy.spatial import ConvexHull

from mesh_io import triangle_faces
from source_fidelity_sampling import select_missing_source_facets
from source_projected_closure import interpolate_source_corner_normals


def _normal_support(
    mesh: pv.PolyData, points: np.ndarray, faces: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    corner = mesh.cell_data.get("corner_normals")
    if corner is not None:
        values = np.asarray(corner, dtype=np.float64)
        if values.shape != (len(faces), 9) or not np.isfinite(values).all():
            raise ValueError("corner_normals must contain three finite normals per triangle")
        support_points = points[faces].reshape(-1, 3)
        support_faces = np.arange(len(support_points), dtype=np.int64).reshape(-1, 3)
        return support_points, support_faces, values.reshape(-1, 3)
    point_normals = mesh.point_data.get("Normals")
    if point_normals is None:
        return None
    values = np.asarray(point_normals, dtype=np.float64)
    if values.shape != points.shape or not np.isfinite(values).all():
        raise ValueError("point Normals must contain one finite normal per point")
    return points, faces, values


def _minkowski_facets(
    points: np.ndarray, faces: np.ndarray, selected_rows: np.ndarray, radius: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    offsets = np.vstack((np.eye(3), -np.eye(3))) * radius
    solids: list[manifold.Manifold] = []
    for row in selected_rows:
        triangle = points[faces[int(row)]]
        if np.linalg.norm(np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])) <= 0.0:
            raise ValueError(f"source face row {int(row)} is degenerate")
        hull_points = np.unique(
            (triangle[:, None, :] + offsets[None, :, :]).reshape(-1, 3), axis=0
        )
        hull = ConvexHull(hull_points)
        hull_faces = hull.simplices.astype(np.int64, copy=True)
        normals = np.cross(
            hull_points[hull_faces[:, 1]] - hull_points[hull_faces[:, 0]],
            hull_points[hull_faces[:, 2]] - hull_points[hull_faces[:, 0]],
        )
        reverse = np.einsum("ij,ij->i", normals, hull.equations[:, :3]) < 0.0
        hull_faces[reverse] = hull_faces[reverse, ::-1]
        if np.einsum(
            "ij,ij->i",
            hull_points[hull_faces[:, 0]],
            np.cross(hull_points[hull_faces[:, 1]], hull_points[hull_faces[:, 2]]),
        ).sum() < 0.0:
            hull_faces = hull_faces[:, ::-1]
        encoded = (int(row) * 4 + np.arange(len(hull_faces)) % 4).astype(np.uint64)
        solid = manifold.Manifold(
            manifold.Mesh64(hull_points, hull_faces.astype(np.uint64), face_id=encoded)
        )
        if solid.status() != manifold.Error.NoError:
            raise ValueError(f"source facet Minkowski solid failed: {solid.status()}")
        solids.append(solid)
    patch = manifold.Manifold.batch_boolean(solids, manifold.OpType.Add)
    if patch.status() != manifold.Error.NoError:
        raise ValueError(f"source facet patch union failed: {patch.status()}")
    mesh = patch.to_mesh64()
    patch_points = np.asarray(mesh.vert_properties, dtype=np.float64)[:, :3]
    patch_faces = np.asarray(mesh.tri_verts, dtype=np.int64)
    patch_rows = np.asarray(mesh.face_id, dtype=np.uint64).reshape(-1) // 4
    if np.any(patch_rows >= len(faces)):
        raise ValueError("patch facet lineage is outside the source mesh")
    return patch_points, patch_faces, patch_rows.astype(np.int64)


def _fill_value(values: np.ndarray) -> Any:
    if np.issubdtype(values.dtype, np.floating):
        return np.nan
    if np.issubdtype(values.dtype, np.signedinteger):
        return -1
    if np.issubdtype(values.dtype, np.unsignedinteger):
        return np.iinfo(values.dtype).max
    if np.issubdtype(values.dtype, np.bool_):
        return False
    if values.dtype.kind in ("S", "U"):
        return ""
    raise ValueError(f"unsupported candidate cell-data dtype: {values.dtype}")


def _corner_normals(
    points: np.ndarray,
    faces: np.ndarray,
    birth: np.ndarray,
    receiver_face_count: int,
    patch_source_rows: np.ndarray,
    receiver_support: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    source_support: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
) -> np.ndarray:
    triangles = points[faces]
    geometric = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(geometric, axis=1)
    geometric /= np.maximum(lengths[:, None], 1e-30)
    result = np.repeat(geometric[:, None, :], 3, axis=1)
    receiver = birth < receiver_face_count
    patch = ~receiver
    if receiver_support is not None and np.any(receiver):
        output_ids = np.flatnonzero(receiver)
        support_points, support_faces, support_normals = receiver_support
        result[output_ids] = interpolate_source_corner_normals(
            points,
            faces[output_ids],
            support_points,
            support_faces,
            support_normals,
            birth[output_ids],
        ).reshape(-1, 3, 3)
    if source_support is not None and np.any(patch):
        output_ids = np.flatnonzero(patch)
        patch_ids = birth[output_ids] - receiver_face_count
        support_points, support_faces, support_normals = source_support
        result[output_ids] = interpolate_source_corner_normals(
            points,
            faces[output_ids],
            support_points,
            support_faces,
            support_normals,
            patch_source_rows[patch_ids],
        ).reshape(-1, 3, 3)
    return result.reshape(-1, 9).astype(np.float32)


def _patch_radius(
    points: np.ndarray, faces: np.ndarray, selected: np.ndarray, max_error: float
) -> tuple[float, float]:
    selected_edges = np.linalg.norm(
        points[faces[selected][:, [1, 2, 0]]] - points[faces[selected][:, [0, 1, 2]]],
        axis=2,
    )
    local_edge = float(np.median(selected_edges))
    radius = min(max_error * 0.02, local_edge * 0.001)
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("could not derive a positive patch radius from the error budget")
    return radius, local_edge


def _cgal_union(
    candidate_points: np.ndarray,
    candidate_faces: np.ndarray,
    patch_points: np.ndarray,
    patch_faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from igl.copyleft import cgal

    output_points, output_faces, birth = cgal.mesh_boolean(
        candidate_points, candidate_faces, patch_points, patch_faces, "union"
    )
    points = np.ascontiguousarray(output_points, dtype=np.float64)
    faces = np.ascontiguousarray(output_faces, dtype=np.int64)
    birth = np.asarray(birth, dtype=np.int64).reshape(-1)
    input_face_count = len(candidate_faces) + len(patch_faces)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or faces.ndim != 2
        or faces.shape[1] != 3
        or len(faces) != len(birth)
        or not np.isfinite(points).all()
        or np.any(birth < 0)
        or np.any(birth >= input_face_count)
        or np.any(faces < 0)
        or np.any(faces >= len(points))
    ):
        raise ValueError("CGAL union returned invalid geometry or birth-face lineage")
    return points, faces, birth


def _map_union_lineage(
    candidate: pv.PolyData,
    source: pv.PolyData,
    candidate_points: np.ndarray,
    candidate_faces: np.ndarray,
    source_points: np.ndarray,
    source_faces: np.ndarray,
    output_points: np.ndarray,
    output_faces: np.ndarray,
    birth: np.ndarray,
    patch_source_rows: np.ndarray,
) -> pv.PolyData:
    receiver_count = len(candidate_faces)
    generated = birth >= receiver_count
    output = pv.PolyData(
        output_points,
        np.column_stack((np.full(len(output_faces), 3, dtype=np.int64), output_faces)).ravel(),
    )
    for name in candidate.field_data.keys():
        output.field_data[name] = np.asarray(candidate.field_data[name]).copy()
    for name in candidate.cell_data.keys():
        values = np.asarray(candidate.cell_data[name])
        fill = np.full((len(patch_source_rows), *values.shape[1:]), _fill_value(values), dtype=values.dtype)
        output.cell_data[name] = np.concatenate((values, fill), axis=0)[birth]
    facet_ids = np.full(len(birth), -1, dtype=np.int64)
    receiver_facets = candidate.cell_data.get("source_facet_id")
    if receiver_facets is not None:
        receiver_facets = np.asarray(receiver_facets, dtype=np.int64)
        if receiver_facets.shape != (receiver_count,):
            raise ValueError("candidate source_facet_id must contain one value per triangle")
        facet_ids[~generated] = receiver_facets[birth[~generated]]
    facet_ids[generated] = patch_source_rows[birth[generated] - receiver_count]
    output.cell_data["source_facet_id"] = facet_ids
    output.cell_data["source_fidelity_recovery_generated"] = generated
    output.cell_data["source_fidelity_recovery_cgal_birth_face_index"] = birth.copy()
    source_rows = np.full(len(birth), -1, dtype=np.int64)
    source_rows[generated] = patch_source_rows[birth[generated] - receiver_count]
    output.cell_data["source_fidelity_recovery_source_face_row"] = source_rows

    source_support = _normal_support(source, source_points, source_faces)
    receiver_support = _normal_support(candidate, candidate_points, candidate_faces)
    if source_support is not None or receiver_support is not None:
        output.cell_data["corner_normals"] = _corner_normals(
            output_points,
            output_faces,
            birth,
            receiver_count,
            patch_source_rows,
            receiver_support,
            source_support,
        )
    return output


def _recover_proposal(
    source: pv.PolyData,
    candidate: pv.PolyData,
    *,
    max_error: float,
    grid_size: int,
    report: dict[str, Any],
    max_recovery_facets: int,
    progress: Callable[[str], None] | None = None,
) -> pv.PolyData:
    source_points = np.ascontiguousarray(source.points, dtype=np.float64)
    candidate_points = np.ascontiguousarray(candidate.points, dtype=np.float64)
    source_faces = triangle_faces(source)
    candidate_faces = triangle_faces(candidate)
    if not len(source_faces) or not len(candidate_faces):
        raise ValueError("source and candidate must contain triangles")
    if not np.isfinite(source_points).all() or not np.isfinite(candidate_points).all():
        raise ValueError("source and candidate points must be finite")
    selected, selection = select_missing_source_facets(
        source_points,
        source_faces,
        candidate_points,
        candidate_faces,
        max_error=max_error,
        grid_size=grid_size,
        progress=progress,
    )
    report.update(selection)
    if not len(selected):
        report.update(status="within_budget", proposal_faces=len(candidate_faces))
        return candidate
    if len(selected) > max_recovery_facets:
        report.update(status="recovery_facet_budget_exceeded", acceptance_ready=False,
                      failure_reason=f"{len(selected)} source facets require recovery; budget is {max_recovery_facets}",
                      proposal_faces=len(candidate_faces))
        return candidate
    unsupported = sorted(name for name in candidate.point_data.keys() if name != "Normals")
    if unsupported:
        raise ValueError(
            "candidate point data needs an explicit field-transfer policy: "
            + ", ".join(unsupported)
        )

    radius, local_edge = _patch_radius(source_points, source_faces, selected, max_error)
    if progress:
        progress("source_material_recovery_patch_construction")
    patch_points, patch_faces, patch_source_rows = _minkowski_facets(
        source_points, source_faces, selected, radius
    )
    report.update(
        {
            "patch_radius_input_units": float(radius),
            "patch_radius_fraction_of_max_error": float(radius / max_error),
            "patch_radius_fraction_of_median_selected_edge": float(radius / local_edge),
            "patch_triangles": int(len(patch_faces)),
        }
    )
    if progress:
        progress("source_material_recovery_patch_union")
    output_points, output_faces, birth = _cgal_union(
        candidate_points, candidate_faces, patch_points, patch_faces
    )
    patch_births = birth >= len(candidate_faces)
    if not np.any(patch_births):
        report.update(
            {
                "status": "failed_original_candidate_returned",
                "acceptance_ready": False,
                "proposal_points": int(len(candidate_points)),
                "proposal_faces": int(len(candidate_faces)),
                "patch_birth_faces": 0,
                "source_material_interior_witnesses_filtered": selection.get(
                    "source_material_interior_witnesses_filtered", 0
                ),
                "failure_reason": (
                    "all additive source patches were consumed by receiver material; "
                    "restoring a filled void requires a semantic Boolean difference, "
                    "not source-fidelity patch recovery"
                ),
            }
        )
        return candidate
    output = _map_union_lineage(
        candidate,
        source,
        candidate_points,
        candidate_faces,
        source_points,
        source_faces,
        output_points,
        output_faces,
        birth,
        patch_source_rows,
    )
    generated = patch_births
    report.update(
        {
            "status": "union_proposal_created",
            "proposal_points": int(len(output_points)),
            "proposal_faces": int(len(output_faces)),
            "candidate_birth_faces": int(np.count_nonzero(~generated)),
            "patch_birth_faces": int(np.count_nonzero(generated)),
            "acceptance_ready": False,
        }
    )
    return output


def recover_source_fidelity(
    source: pv.PolyData,
    candidate: pv.PolyData,
    *,
    max_error: float,
    grid_size: int = 720,
    progress: Callable[[str], None] | None = None,
    max_recovery_facets: int = 1000,
) -> tuple[pv.PolyData, dict[str, Any]]:
    """Build a union proposal from only source facets exceeding ``max_error``.

    Sampling and the Minkowski patch are expressed in the mesh's input units.
    A successful return is a union proposal only; the caller still runs all
    required global topology, exact-intersection, fidelity and readback gates.
    Any operation failure returns the original candidate unchanged.
    """
    if not np.isfinite(max_error) or max_error <= 0.0:
        raise ValueError("max_error must be finite and positive in input mesh units")
    if grid_size < 32:
        raise ValueError("grid_size must be at least 32")
    if not isinstance(max_recovery_facets, int) or isinstance(max_recovery_facets, bool) or max_recovery_facets < 1:
        raise ValueError("max_recovery_facets must be a positive integer")
    report: dict[str, Any] = {
        "method": "multi_view_first_hit_source_facet_error_and_minkowski_union",
        "status": "failed",
        "acceptance_ready": False,
        "max_error_input_units": float(max_error),
        "grid_size": int(grid_size),
        "max_recovery_facets": max_recovery_facets,
    }
    try:
        output = _recover_proposal(
            source,
            candidate,
            max_error=max_error,
            grid_size=grid_size,
            report=report,
            max_recovery_facets=max_recovery_facets,
            progress=progress,
        )
        return output, report
    except Exception as exc:
        report.update(
            {
                "status": "failed_original_candidate_returned",
                "failure_reason": f"{type(exc).__name__}: {exc}",
                "proposal_faces": int(candidate.n_cells),
            }
        )
        return candidate, report
