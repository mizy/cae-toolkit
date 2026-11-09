"""First-hit and geometric-witness source-facet error selection."""
from __future__ import annotations

from typing import Any, Callable

import igl
import numpy as np
import pyvista as pv

from mesh_io import triangle_faces
from solid_triangle_raster import ORTHOGRAPHIC_FRAMES, ViewSpec, projection_frame, rasterize_solid_view


def _checked_squared_distances(values, count: int, field: str) -> np.ndarray:
    distances = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(distances) != count or not np.isfinite(distances).all() or np.any(distances < 0):
        raise ValueError(f"{field} returned invalid squared distances")
    return distances


def _pixel_samples(
    points: np.ndarray, faces: np.ndarray, grid_size: int
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    sample_points: list[np.ndarray] = []
    sample_faces: list[np.ndarray] = []
    for index, (eye, up) in enumerate(ORTHOGRAPHIC_FRAMES):
        coords, basis = projection_frame(points, eye, up)
        lo = coords[:, :2].min(axis=0)
        hi = coords[:, :2].max(axis=0)
        span = hi - lo
        padding = np.maximum(span * 0.01, span.max() * 1.0e-6)
        bounds = np.asarray([lo - padding, hi + padding])
        raster = rasterize_solid_view(
            coords,
            faces,
            ViewSpec(f"orthographic_{index}", (0, 1), (0.0, 0.0, 1.0)),
            grid_size=grid_size,
            depth_tolerance=max(float(np.ptp(points, axis=0).max()) * 1.0e-12, 0.0),
            collect_face_evidence=True,
            projection_bounds=bounds,
            sample_centers=True,
        )
        y, x = np.nonzero(raster.silhouette)
        face_ids = raster.first_hit_face[y, x]
        valid = face_ids >= 0
        if not np.any(valid):
            continue
        y, x, face_ids = y[valid], x[valid], face_ids[valid]
        rows, cols = raster.depth.shape
        xy = bounds[0] + np.column_stack(
            (x / (cols - 1), y / (rows - 1))
        ) * (bounds[1] - bounds[0])
        projected = np.column_stack((xy, raster.depth[y, x]))
        sample_points.append(projected @ basis.T)
        sample_faces.append(face_ids)
    return sample_points, sample_faces


def _triangle_witnesses(
    points: np.ndarray, faces: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Sample seven deterministic barycentric locations on every source triangle."""
    barycentric = np.asarray(
        (
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.5, 0.5, 0.0),
            (0.0, 0.5, 0.5),
            (0.5, 0.0, 0.5),
            (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
        ),
        dtype=np.float64,
    )
    triangles = points[faces]
    samples = np.einsum("sc,fcd->fsd", barycentric, triangles).reshape(-1, 3)
    rows = np.repeat(np.arange(len(faces), dtype=np.int64), len(barycentric))
    return samples, rows


def _interior_witness_mask(
    points: np.ndarray, faces: np.ndarray, query: np.ndarray,
    progress: Callable[[str], None] | None = None,
    interior_threshold: float = 0.5,
    integer_margin: float | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Use approximate winding only to skip direct tests for likely exterior points.

    Approximate values never exclude a witness: every likely interior point is
    checked again with libigl's direct floating-point winding calculation. A
    false negative therefore stays visible and can only cause conservative
    recovery or rejection; the classification is not an exact predicate proof.
    """
    likely_inside = np.zeros(len(query), dtype=bool)
    try:
        if progress:
            progress("source_material_fast_winding")
        fast = np.asarray(
            igl.fast_winding_number(
                np.ascontiguousarray(points, dtype=np.float64),
                np.ascontiguousarray(faces, dtype=np.int64),
                np.ascontiguousarray(query, dtype=np.float64),
            ),
            dtype=np.float64,
        ).reshape(-1)
        if len(fast) != len(query) or not np.isfinite(fast).all():
            raise ValueError("fast winding returned invalid values")
        likely_inside = fast > 0.0
    except Exception:
        likely_inside[:] = True
    direct_inside = np.zeros(len(query), dtype=bool)
    candidates = np.flatnonzero(likely_inside)
    if len(candidates):
        if progress:
            progress("source_material_direct_winding")
        winding = np.asarray(
            igl.winding_number(
                np.ascontiguousarray(points, dtype=np.float64),
                np.ascontiguousarray(faces, dtype=np.int64),
                np.ascontiguousarray(query[candidates], dtype=np.float64),
            ),
            dtype=np.float64,
        ).reshape(-1)
        inside = np.isfinite(winding) & (winding > interior_threshold)
        if integer_margin is not None:
            inside &= np.abs(winding - np.rint(winding)) <= integer_margin
        direct_inside[candidates] = inside
    return direct_inside, {
        "winding_queries": int(len(query)),
        "direct_winding_queries": int(len(candidates)),
        "fast_winding_negative_queries_kept": int(len(query) - len(candidates)),
        "winding_numeric_method": "libigl floating-point winding number; not an exact point-side proof",
    }


def _closed_material_reference(
    points: np.ndarray, faces: np.ndarray, max_error: float,
    progress: Callable[[str], None] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]] | None:
    """Build an exact source-material union using only closed source components."""
    from mesh_metrics import component_face_rows, edge_topology, face_component_labels, mesh_report
    from mesh_vertex_topology import count_non_manifold_vertices
    from source_material_closure import build_source_material
    from source_projected_validation import exact_self_intersection_report

    if progress:
        progress("source_material_fidelity_component_inventory")
    _, edge_faces = edge_topology(faces)
    labels = face_component_labels(len(faces), edge_faces)
    closed_rows = []
    for rows in component_face_rows(labels):
        local_faces = faces[rows]
        topology, local_edges = edge_topology(local_faces)
        if (
            topology["boundary_edges"] == 0
            and topology["non_manifold_edges"] == 0
            and topology["non_manifold_vertices"] == 0
            and count_non_manifold_vertices(local_faces, local_edges) == 0
        ):
            closed_rows.append(rows)
    if not closed_rows:
        return None

    rows = np.concatenate(closed_rows)
    if progress:
        progress("source_material_fidelity_reference_union")
    reference_points, reference_faces, _, construction = build_source_material(
        points, faces[rows], max_error=max_error
    )
    if progress:
        progress("source_material_fidelity_reference_metrics")
    metrics = mesh_report(reference_points, reference_faces)
    topology = metrics["topology"]
    if (
        any(
            topology[name] != 0
            for name in (
                "boundary_edges",
                "non_manifold_edges",
                "non_manifold_vertices",
                "inconsistent_winding_edges",
            )
        )
        or metrics["quality"]["degenerate_faces"] != 0
        or not metrics["volume"]["reliable"]
    ):
        return None
    if progress:
        progress("source_material_fidelity_reference_exact")
    exact = exact_self_intersection_report(
        reference_points, reference_faces, first_only=True
    )
    if exact.get("passed") is not True:
        return None
    raw_mesh = pv.PolyData(
        reference_points,
        np.column_stack((np.full(len(reference_faces), 3, dtype=np.int64), reference_faces)).ravel(),
    )
    if progress:
        progress("source_material_fidelity_reference_normalize")
    normalized = raw_mesh.clean(
        lines_to_points=False, polys_to_lines=False, strips_to_polys=False
    )
    normalized_faces = triangle_faces(normalized)
    normalized_metrics = mesh_report(np.asarray(normalized.points), normalized_faces)
    normalized_topology = normalized_metrics["topology"]
    if (
        any(
            normalized_topology[name] != 0
            for name in (
                "boundary_edges",
                "non_manifold_edges",
                "non_manifold_vertices",
                "inconsistent_winding_edges",
            )
        )
        or normalized_metrics["quality"]["degenerate_faces"] != 0
        or not normalized_metrics["volume"]["reliable"]
    ):
        return None
    if progress:
        progress("source_material_fidelity_reference_normalized_exact")
    normalized_exact = exact_self_intersection_report(
        np.asarray(normalized.points), normalized_faces, first_only=True
    )
    if normalized_exact.get("passed") is not True:
        return None
    return reference_points, reference_faces, {
        "status": "validated_closed_source_material",
        "source_closed_components": int(len(closed_rows)),
        "source_closed_faces": int(len(rows)),
        "reference_faces": int(len(reference_faces)),
        "volume_reliable": True,
        "exact_intersection_pairs": 0,
        "normalized_reference_validated": True,
        "construction": construction,
    }


def _closed_material_winding_soup(
    points: np.ndarray, faces: np.ndarray, max_error: float,
    progress: Callable[[str], None] | None = None,
    *, include_components: bool = False,
) -> tuple | None:
    """Return individually validated closed shells with material-relative winding.

    This avoids requiring touching closed shells to form a valid single CGAL
    union. Every component is independently audited first. Winding remains a
    floating-point classification, so uncertain shells are rejected here and
    query witnesses remain fail-open when callers cannot validate this soup.
    """
    from mesh_metrics import (component_face_rows, edge_topology, face_component_labels,
                              mesh_report, signed_volume_error_bound)
    from mesh_vertex_topology import count_non_manifold_vertices
    from source_material_closure import _build_components
    from source_projected_validation import exact_self_intersection_report

    if progress:
        progress("source_material_fidelity_soup_inventory")
    _, edge_faces = edge_topology(faces)
    labels = face_component_labels(len(faces), edge_faces)
    closed_rows = []
    for rows in component_face_rows(labels):
        local_faces = faces[rows]
        topology, local_edges = edge_topology(local_faces)
        if (topology["boundary_edges"] == 0 and topology["non_manifold_edges"] == 0
                and topology["non_manifold_vertices"] == 0
                and count_non_manifold_vertices(local_faces, local_edges) == 0):
            closed_rows.append(rows)
    if not closed_rows:
        return None

    closed_source_rows = np.concatenate(closed_rows)
    closed_faces = faces[closed_source_rows]
    _, blocks, component_details, _, nested, interactions = _build_components(
        points, closed_faces, max_error=max_error
    )
    if len(blocks) != len(component_details) or any(
        row["mode"] != "preserved_closed_component" or not row["orientable"] or not row["closed"]
        for row in component_details
    ):
        return None
    if progress:
        progress("source_material_fidelity_soup_component_validation")

    soup_points, soup_faces, components = [], [], []
    point_offset = 0
    face_offset = 0
    for block_points, block_faces, block_rows in blocks:
        local = mesh_report(block_points, block_faces)
        topo = local["topology"]
        volume_error = signed_volume_error_bound(block_points, block_faces)
        signed_volume = float(local["volume"]["component_signed"][0])
        component_volume = abs(signed_volume)
        if (topo["boundary_edges"] != 0 or topo["non_manifold_edges"] != 0
                or topo["non_manifold_vertices"] != 0 or topo["inconsistent_winding_edges"] != 0
                or local["quality"]["degenerate_faces"] != 0 or not local["volume"]["reliable"]
                or not local["volume"]["orientation_consistent"] or component_volume <= volume_error):
            return None
        exact = exact_self_intersection_report(block_points, block_faces, first_only=True)
        if exact.get("passed") is not True:
            return None
        soup_points.append(block_points)
        soup_faces.append(block_faces + point_offset)
        components.append((closed_source_rows[block_rows], face_offset, face_offset + len(block_faces), signed_volume > 0))
        point_offset += len(block_points)
        face_offset += len(block_faces)
    if not soup_points:
        return None
    result = np.vstack(soup_points), np.vstack(soup_faces), {
        "status": "validated_closed_component_winding_soup",
        "source_closed_components": int(len(blocks)),
        "source_closed_faces": int(sum(len(row) for row in closed_rows)),
        "nested_component_relations": nested,
        "closed_component_aabb_interactions": interactions,
        "component_self_intersections": "checked_exactly_per_component",
        "cross_component_intersections": "represented by winding sum, not rejected",
        "winding_numeric_method": "libigl floating-point winding number; not an exact point-side proof",
    }
    return (*result, components) if include_components else result


def _closed_soup_interior_witness_mask(
    points: np.ndarray, faces: np.ndarray, query: np.ndarray, max_error: float,
    progress: Callable[[str], None] | None = None,
    *, query_face_rows: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Filter only far-from-boundary witnesses inside validated closed-shell material."""
    soup = _closed_material_winding_soup(points, faces, max_error, progress, include_components=True)
    if soup is None:
        return np.zeros(len(query), dtype=bool), {"status": "unavailable_or_unvalidated"}
    soup_points, soup_faces, details, components = soup
    if progress:
        progress("source_material_fidelity_soup_boundary_distance")
    distance2, _, _ = igl.point_mesh_squared_distance(
        np.ascontiguousarray(query, dtype=np.float64), soup_points, soup_faces
    )
    distance2 = _checked_squared_distances(distance2, len(query), "closed soup distance")
    far = distance2 > max_error ** 2
    buried = np.zeros(len(query), dtype=bool)
    if np.any(far):
        likely, performance = _interior_witness_mask(
            soup_points, soup_faces, query[far], progress, interior_threshold=0.5001, integer_margin=1e-4
        )
        # Keep a wide distance from the winding boundary; this is conservative
        # floating-point evidence, not a formal classification error bound.
        buried[np.flatnonzero(far)] = likely
    else:
        performance = {"winding_queries": 0, "direct_winding_queries": 0,
                       "fast_winding_negative_queries_kept": 0,
                       "winding_numeric_method": "libigl floating-point winding number; not an exact point-side proof"}
    component_filtered = 0
    coordinate_scale = max(float(np.linalg.norm(np.ptp(soup_points, axis=0))),
                           float(np.abs(soup_points).max(initial=0.0)))
    other_clearance = max(coordinate_scale * np.finfo(float).eps * 128, np.nextafter(0., 1.))
    if query_face_rows is not None:
        query_face_rows = np.asarray(query_face_rows, dtype=np.int64)
        if query_face_rows.shape != (len(query),):
            raise ValueError("query face rows must correspond to witnesses")
        for source_rows, start, end, positive in components:
            if not positive or end - start == len(soup_faces):
                continue
            selected = np.flatnonzero(~buried & np.isin(query_face_rows, source_rows))
            if not len(selected):
                continue
            other_faces = np.concatenate((soup_faces[:start], soup_faces[end:]))
            if progress:
                progress("source_material_fidelity_other_shell_distance")
            other_distance, _, _ = igl.point_mesh_squared_distance(query[selected], soup_points, other_faces)
            other_distance = _checked_squared_distances(other_distance, len(selected), "other shell distance")
            far_rows = selected[other_distance > other_clearance ** 2]
            if not len(far_rows):
                continue
            inside, _ = _interior_witness_mask(
                soup_points, other_faces, query[far_rows], progress,
                interior_threshold=0.5001, integer_margin=1e-4,
            )
            buried[far_rows[inside]] = True
            component_filtered += int(np.count_nonzero(inside))
    return buried, {
        **details,
        "status": "validated_closed_component_winding_soup",
        "query_count": int(len(query)),
        "far_from_all_closed_boundaries": int(np.count_nonzero(far)),
        "interior_winding_threshold": 0.5001,
        "interior_winding_integer_margin": 1e-4,
        "witnesses_inside_other_closed_material": component_filtered,
        "other_shell_clearance_input_units": other_clearance,
        "winding_performance": performance,
    }


def _face_errors(
    source_points: np.ndarray,
    source_faces: np.ndarray,
    candidate_points: np.ndarray,
    candidate_faces: np.ndarray,
    max_error: float,
    grid_size: int,
    progress: Callable[[str], None] | None = None,
    prefer_soup: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    if progress:
        progress("source_material_fidelity_raster")
    points_by_view, faces_by_view = _pixel_samples(source_points, source_faces, grid_size)
    if progress:
        progress("source_material_fidelity_witnesses")
    witness_points, witness_faces = _triangle_witnesses(source_points, source_faces)
    pixel_count = sum(len(rows) for rows in faces_by_view)
    points = np.concatenate((*points_by_view, witness_points))
    face_rows = np.concatenate((*faces_by_view, witness_faces))
    if progress:
        progress("source_material_fidelity_candidate_distance")
    distance2, _nearest_face, _closest = igl.point_mesh_squared_distance(
        np.ascontiguousarray(points, dtype=np.float64),
        np.ascontiguousarray(candidate_points, dtype=np.float64),
        np.ascontiguousarray(candidate_faces, dtype=np.int64),
    )
    distance2 = _checked_squared_distances(distance2, len(points), "candidate distance")
    distance = np.sqrt(distance2)
    raw_distance = distance.copy()
    over_budget = distance > max_error
    material_details: dict[str, Any] = {"status": "not_needed_within_error"}
    interior_filtered = 0
    if np.any(over_budget):
        try:
            if progress:
                progress("source_material_fidelity_closed_reference")
            reference = None if prefer_soup else _closed_material_reference(
                source_points, source_faces, max_error, progress
            )
            if reference is None:
                query = points[over_budget]
                fallback_mask, material_details = _closed_soup_interior_witness_mask(
                    source_points, source_faces, query, max_error, progress,
                    query_face_rows=face_rows[over_budget],
                )
                if material_details.get("status") == "validated_closed_component_winding_soup":
                    affected = np.flatnonzero(over_budget)
                    distance[affected[fallback_mask]] = 0.0
                    interior_filtered = int(np.count_nonzero(fallback_mask))
                else:
                    material_details["status"] = "unavailable_or_unvalidated"
            else:
                reference_points, reference_faces, material_details = reference
                query = points[over_budget]
                if progress:
                    progress("source_material_fidelity_reference_distance")
                reference_distance2, _, _ = igl.point_mesh_squared_distance(
                    np.ascontiguousarray(query, dtype=np.float64),
                    np.ascontiguousarray(reference_points, dtype=np.float64),
                    np.ascontiguousarray(reference_faces, dtype=np.int64),
                )
                reference_distance2 = _checked_squared_distances(
                    reference_distance2, len(query), "material reference distance"
                )
                reference_distance = np.sqrt(reference_distance2)
                farther = reference_distance > max_error
                buried = np.zeros(len(query), dtype=bool)
                if np.any(farther):
                    farther_rows = np.flatnonzero(farther)
                    buried[farther_rows], winding_details = _interior_witness_mask(
                        reference_points, reference_faces, query[farther_rows], progress
                    )
                else:
                    winding_details = {"winding_queries": 0, "direct_winding_queries": 0,
                                       "fast_winding_negative_queries_kept": 0,
                                       "winding_numeric_method": "libigl floating-point winding number; not an exact point-side proof"}
                affected = np.flatnonzero(over_budget)
                distance[affected[buried]] = 0.0
                interior_filtered = int(np.count_nonzero(buried))
                material_details["winding_performance"] = winding_details
        except Exception as exc:
            material_details = {
                "status": "failed_open_no_samples_excluded",
                "failure_reason": f"{type(exc).__name__}: {exc}",
            }
    maximum = np.zeros(len(source_faces), dtype=np.float64)
    count = np.bincount(face_rows, minlength=len(source_faces)).astype(np.int64)
    np.maximum.at(maximum, face_rows, distance)
    return maximum, count, distance, {
        "geometry_witness_samples": int(len(witness_points)),
        "witnesses_per_source_face": 7,
        "first_hit_pixel_center_samples": int(pixel_count),
        "source_material_interior_witnesses_filtered": interior_filtered,
        "closed_material_reference": material_details,
        "raw_sample_distance_input_units": {
            "p50": float(np.percentile(raw_distance, 50)),
            "p95": float(np.percentile(raw_distance, 95)),
            "p99": float(np.percentile(raw_distance, 99)),
            "max": float(raw_distance.max(initial=0.0)),
        },
    }


def select_missing_source_facets(
    source_points: np.ndarray,
    source_faces: np.ndarray,
    candidate_points: np.ndarray,
    candidate_faces: np.ndarray,
    *,
    max_error: float,
    grid_size: int,
    progress: Callable[[str], None] | None = None,
    prefer_soup: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    maximum, sample_count, distances, sample_breakdown = _face_errors(
        source_points, source_faces, candidate_points, candidate_faces, max_error, grid_size,
        progress, prefer_soup,
    )
    selected = np.flatnonzero((sample_count > 0) & (maximum > max_error))
    return selected, {
        **sample_breakdown,
        "sample_count": int(distances.size),
        "sample_distance_input_units": {
            "p50": float(np.percentile(distances, 50)),
            "p95": float(np.percentile(distances, 95)),
            "p99": float(np.percentile(distances, 99)),
            "max": float(distances.max(initial=0.0)),
        },
        "source_facets_sampled": int(np.count_nonzero(sample_count)),
        "source_facets_selected": int(len(selected)),
        "selected_source_facets": [int(value) for value in selected],
        "selected_source_face_rows": [int(value) for value in selected],
        "selected_face_max_error_input_units": [float(maximum[row]) for row in selected],
    }
