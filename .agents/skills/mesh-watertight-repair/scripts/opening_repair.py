"""Execute reviewed boundary caps without changing unrelated surface connectivity."""

from __future__ import annotations

import numpy as np
import pyvista as pv
import igl
import trimesh
from mesh_io import triangle_faces
from repair_policy import policy_item_id


def _matching_boundary_map(points, faces, edge_faces, tolerance):
    from scipy.spatial import cKDTree

    directed = []
    for edge, incident in edge_faces.items():
        if len(incident) != 1:
            continue
        face = faces[incident[0]]
        directed.append(next((int(a), int(b)) for a, b in zip(face, np.roll(face, -1))
                             if {int(a), int(b)} == set(edge)))
    mapping = np.arange(len(points))
    if not directed:
        return mapping, 0
    directed = np.asarray(directed, dtype=np.int64)
    vertices = np.unique(directed)
    distances, neighbors = cKDTree(points[vertices]).query(points[vertices], k=2)
    own = neighbors[:, 0] == np.arange(len(vertices))
    nearest = np.where(own, neighbors[:, 1], neighbors[:, 0])
    nearest_distance = np.where(own, distances[:, 1], distances[:, 0])
    for local, other in enumerate(nearest):
        if nearest[other] != local or nearest_distance[local] > tolerance:
            continue
        a, b = sorted((int(vertices[local]), int(vertices[other])))
        if (a, b) not in edge_faces:
            mapping[b] = a
    mapped_edges = mapping[directed]
    _, groups, counts = np.unique(np.sort(mapped_edges, axis=1), axis=0,
                                  return_inverse=True, return_counts=True)
    order = np.argsort(groups, kind="stable")
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    paired = np.flatnonzero(counts == 2)
    left, right = order[starts[paired]], order[starts[paired] + 1]
    valid = np.all(mapped_edges[left] == mapped_edges[right, ::-1], axis=1)
    valid &= mapped_edges[left, 0] != mapped_edges[left, 1]
    involved = np.unique(directed[np.r_[left[valid], right[valid]]])
    approved = np.arange(len(points))
    approved[involved] = mapping[involved]
    return approved, int(valid.sum())


# @entry: explicitly bounded seam sewing; closed source shells have no eligible vertices.
def stitch_boundary_seams(points, faces, *, tolerance):
    from mesh_metrics import edge_topology, mesh_report
    from source_projected_validation import exact_self_intersection_report

    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("stitch tolerance must be finite and positive in input units")
    points, faces = _validated_mesh(points, faces)
    _, edges = edge_topology(faces)
    mapping, paired_edges = _matching_boundary_map(points, faces, edges, tolerance)
    proposal = mapping[faces]
    changed = np.flatnonzero(np.any(proposal != faces, axis=1))
    report = {"status": "unchanged", "paired_boundary_edges": paired_edges,
              "tolerance": float(tolerance), "modified_faces": len(changed), "applied": False}
    if not len(changed):
        return faces, report
    old, new = points[faces[changed]], points[proposal[changed]]
    a = np.cross(old[:, 1] - old[:, 0], old[:, 2] - old[:, 0])
    b = np.cross(new[:, 1] - new[:, 0], new[:, 2] - new[:, 0])
    bounds = np.vstack((old.reshape(-1, 3), new.reshape(-1, 3)))
    triangles = points[proposal]
    nearby = np.flatnonzero(np.all(triangles.min(axis=1) <= bounds.max(axis=0), axis=1)
                           & np.all(triangles.max(axis=1) >= bounds.min(axis=0), axis=1))
    if len(nearby) > 10000 or np.any(np.einsum("ij,ij->i", a, b) <= 0):
        report.update(status="rejected", reason="seam changes face orientation or exceeds local face budget")
        return faces, report
    before, after = mesh_report(points, faces), mesh_report(points, proposal)
    keys = ("non_manifold_edges", "non_manifold_vertices", "inconsistent_winding_edges")
    topology_ok = (after["topology"]["boundary_edges"] < before["topology"]["boundary_edges"]
                   and all(after["topology"][key] <= before["topology"][key] for key in keys)
                   and after["quality"]["degenerate_faces"] <= before["quality"]["degenerate_faces"])
    used, inverse = np.unique(proposal[nearby], return_inverse=True)
    exact = exact_self_intersection_report(points[used], inverse.reshape(-1, 3), first_only=True) if topology_ok else {"passed": False}
    displacement = float(np.linalg.norm(points - points[mapping], axis=1).max(initial=0.))
    passed = topology_ok and exact["passed"] and displacement <= tolerance
    report.update(status="applied" if passed else "rejected", applied=bool(passed),
                  boundary_edges_before=before["topology"]["boundary_edges"],
                  boundary_edges_after=after["topology"]["boundary_edges"],
                  max_surface_displacement=displacement, local_exact=exact,
                  reason=None if passed else "seam topology, intersection or displacement gate failed")
    return (proposal if passed else faces), report


def cap_loop(points: np.ndarray, faces: np.ndarray, region: dict) -> np.ndarray:
    loop = np.asarray(region.get("ordered_vertex_ids", []), dtype=np.int64)
    if not region.get("simple_closed_loop") or len(loop) < 3:
        raise ValueError("cap requires a simple ordered boundary loop")
    if region.get("planarity_ratio", 0.0) > 0.08:
        raise ValueError("nonplanar boundary needs a fitted patch; planar cap refused")
    polygon = pv.PolyData(
        points[loop], np.r_[len(loop), np.arange(len(loop))]
    ).triangulate()
    if polygon.n_cells != len(loop) - 2:
        raise ValueError("boundary triangulation was incomplete")
    patch = loop[triangle_faces(polygon)]
    # The patch must traverse each shared boundary edge opposite to its old face.
    a, b = int(loop[0]), int(loop[1])
    old_direction = any(
        a == x and b == y
        for face in faces[region["face_ids"]]
        for x, y in zip(face, np.roll(face, -1))
    )
    new_direction = any(
        a == x and b == y for face in patch for x, y in zip(face, np.roll(face, -1))
    )
    return patch[:, ::-1] if old_direction == new_direction else patch


def build_preserved_opening_rays(
    points: np.ndarray,
    faces: np.ndarray,
    regions: list[dict],
    resolved_decisions: list[dict],
) -> dict:
    """Build source-clear straight rays for reviewed functional openings."""
    points, faces = _validated_mesh(points, faces)
    by_item = {
        policy_item_id(str(region["id"])): region
        for region in regions
        if region.get("id") is not None
    }
    by_item.update(
        {str(region["id"]): region for region in regions if region.get("id") is not None}
    )
    origins, directions, region_ids = [], [], []
    rows, unsupported = [], []

    for decision in resolved_decisions:
        if not (
            decision.get("status") == "decided"
            and decision.get("decision") == "preserve"
            and decision.get("semantic_label") == "functional_opening"
        ):
            continue
        item_id = str(decision.get("item_id") or decision.get("region_id") or "")
        region = by_item.get(item_id) or by_item.get(str(decision.get("source_region") or ""))
        region_id = str(region.get("id")) if region else item_id
        row = {
            "region_id": region_id,
            "status": "unsupported",
            "probe_count": 0,
            "surface_probe_count": 0,
            "source_clear_ray_count": 0,
        }
        try:
            if region is None:
                raise ValueError("decision does not identify a source boundary region")
            probes, normal, surface_probe_count = _opening_probes(points, faces, region)
            row["probe_count"] = len(probes)
            row["surface_probe_count"] = surface_probe_count
            if not len(probes):
                raise ValueError("all virtual-cap probes lie on the source surface")
            ray_origins = np.repeat(probes, 2, axis=0)
            ray_directions = np.tile(np.vstack((normal, -normal)), (len(probes), 1))
            diagonal = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
            ray_origins -= ray_directions * max(diagonal * 1e-7, 1e-10)
            source_hits = _ray_hits(points, faces, ray_origins, ray_directions)
            clear = ~source_hits
            if not np.any(clear):
                raise ValueError("no virtual-cap probe has a source-clear normal ray")
            clear_origins = ray_origins[clear]
            clear_directions = ray_directions[clear]
            origins.extend(clear_origins.tolist())
            directions.extend(clear_directions.tolist())
            region_ids.extend([region_id] * len(clear_origins))
            row.update(
                status="source_clear",
                source_clear_ray_count=int(np.count_nonzero(clear)),
            )
        except (ValueError, IndexError, KeyError, RuntimeError) as exc:
            row["reason"] = str(exc)
            unsupported.append(region_id)
        rows.append(row)

    return {
        "origins": origins,
        "directions": directions,
        "region_ids": region_ids,
        "regions": rows,
        "unsupported_regions": unsupported,
        "status": "ready" if not unsupported else "unsupported",
        "claim": "checks sampled straight rays only; does not establish volumetric fluid connectivity",
    }


def check_preserved_opening_rays(
    candidate_points: np.ndarray,
    candidate_faces: np.ndarray,
    rays: dict,
) -> dict:
    """Fail if a repair blocks any source-clear ray or leaves a region unverified."""
    unsupported = list(rays.get("unsupported_regions", []))
    origins = np.asarray(rays.get("origins", []), dtype=np.float64).reshape(-1, 3)
    directions = np.asarray(rays.get("directions", []), dtype=np.float64).reshape(-1, 3)
    region_ids = list(rays.get("region_ids", []))
    blocked_by_region: dict[str, int] = {}
    status = "passed"
    reason = None

    if len(origins) != len(directions) or len(origins) != len(region_ids):
        status, reason = "unsupported", "source ray record has inconsistent lengths"
    elif unsupported:
        status, reason = "unsupported", "one or more functional openings have no verifiable source-clear rays"
    elif len(origins):
        try:
            candidate_points, candidate_faces = _validated_mesh(candidate_points, candidate_faces)
            hits = _ray_hits(candidate_points, candidate_faces, origins, directions)
            for region_id, hit in zip(region_ids, hits):
                if hit:
                    blocked_by_region[region_id] = blocked_by_region.get(region_id, 0) + 1
            if blocked_by_region:
                status, reason = "failed", "candidate blocks one or more source-clear opening rays"
        except (ValueError, IndexError, KeyError, RuntimeError) as exc:
            status, reason = "unsupported", str(exc)

    return {
        "status": status,
        "passed": status == "passed",
        "ray_count": len(origins),
        "blocked_ray_count": int(sum(blocked_by_region.values())),
        "blocked_by_region": blocked_by_region,
        "unsupported_regions": unsupported,
        "reason": reason,
        "claim": "checks sampled straight rays only; does not establish volumetric fluid connectivity",
    }


def _validated_mesh(points: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("mesh points must be a finite N by 3 array")
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces):
        raise ValueError("ray validation requires nonempty triangle faces")
    if faces.min() < 0 or faces.max() >= len(points):
        raise ValueError("mesh faces reference an invalid point")
    return np.ascontiguousarray(points), np.ascontiguousarray(faces)


def _opening_probes(
    points: np.ndarray,
    faces: np.ndarray,
    region: dict,
) -> tuple[np.ndarray, np.ndarray, int]:
    loop = np.asarray(region.get("ordered_vertex_ids", []), dtype=np.int64)
    planarity = float(region.get("planarity_ratio", np.inf))
    normal = np.asarray(region.get("normal", []), dtype=np.float64)
    if not region.get("simple_closed_loop") or len(loop) < 3:
        raise ValueError("functional opening needs a simple closed boundary loop")
    if not np.isfinite(planarity) or planarity > 0.08:
        raise ValueError("functional opening loop is nonplanar")
    if loop.min() < 0 or loop.max() >= len(points):
        raise ValueError("opening loop references an invalid point")
    if normal.shape != (3,) or not np.isfinite(normal).all() or np.linalg.norm(normal) <= 1e-12:
        raise ValueError("functional opening needs a finite region normal")
    normal = normal / np.linalg.norm(normal)
    loop_points = points[loop]
    _, singular_values, plane_axes = np.linalg.svd(loop_points - loop_points.mean(axis=0), full_matrices=False)
    if singular_values[-2] <= 1e-12 or singular_values[-1] / singular_values[0] > 0.08:
        raise ValueError("functional opening loop is geometrically nonplanar or degenerate")
    if abs(float(np.dot(normal, plane_axes[-1]))) < 0.9:
        raise ValueError("region normal is not aligned with the opening plane")
    patch = cap_loop(points, faces, region)
    triangles = points[patch]
    samples = np.vstack(
        (
            triangles.mean(axis=1),
            (triangles[:, 0] + triangles[:, 1]) * 0.5,
            (triangles[:, 1] + triangles[:, 2]) * 0.5,
            (triangles[:, 2] + triangles[:, 0]) * 0.5,
        )
    )
    samples = np.unique(np.round(samples, decimals=12), axis=0)
    diagonal = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    tolerance = max(diagonal * 1e-9, 1e-12)
    squared_distances, _, _ = igl.point_mesh_squared_distance(samples, points, faces)
    on_source = np.asarray(squared_distances) <= tolerance * tolerance
    return samples[~on_source], normal, int(np.count_nonzero(on_source))


def _ray_hits(
    points: np.ndarray,
    faces: np.ndarray,
    origins: np.ndarray,
    directions: np.ndarray,
) -> np.ndarray:
    if not len(origins):
        return np.zeros(0, dtype=bool)
    mesh = trimesh.Trimesh(vertices=points, faces=faces, process=False)
    try:
        return np.asarray(mesh.ray.intersects_any(origins, directions), dtype=bool)
    except (ImportError, ModuleNotFoundError):
        return np.asarray(
            [
                bool(igl.ray_mesh_intersect(origin, direction, points, faces, first=False))
                for origin, direction in zip(origins, directions)
            ],
            dtype=bool,
        )


# @entry: resolved semantic choices own only their identified source boundary.
def apply_opening_decisions(
    points: np.ndarray,
    faces: np.ndarray,
    inventory: dict,
    decisions: list[dict],
    *,
    default_action: str = "review",
    preserve_as_wall: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[dict], bool]:
    supplied = {row["item_id"]: row for row in decisions}
    patches, transactions = [], []
    ready = True
    for region in inventory.get("boundary_regions", {}).get("items", []):
        decision = supplied.get(policy_item_id(region["id"]))
        action = (
            decision.get("decision")
            if decision and decision.get("status") == "decided"
            else default_action
        )
        if decision and decision.get("status") != "decided":
            action = "review"
        row = {
            "region_id": region["id"],
            "action": action,
            "applied": False,
            "generated_faces": 0,
        }
        if action == "cap":
            try:
                if decision and decision.get("shape_prior") not in (
                    None,
                    "none",
                    "planar",
                ):
                    raise ValueError(
                        "this cap operator supports planar triangulation; requested prior needs a different operator"
                    )
                patch = cap_loop(points, faces, region)
                patches.append(patch)
                row.update(
                    applied=True,
                    generated_faces=len(patch),
                    operator="boundary_triangulation",
                )
            except ValueError as exc:
                row["reason"] = str(exc)
        elif action == "preserve" and preserve_as_wall:
            row.update(applied=True, operator="preserve_rim_in_thickened_wall")
        else:
            row["reason"] = (
                "an open boundary needs a wall; choose thicken to preserve this rim"
                if action == "preserve"
                else "opening review is pending or the requested operator is unsupported"
            )
        if decision:
            decision["applied_by_pass_id"] = region["id"] if row["applied"] else None
        ready &= row["applied"]
        transactions.append(row)
    result = np.vstack([faces, *patches]) if patches else faces.copy()
    generated = np.r_[
        np.zeros(len(faces), dtype=np.uint8),
        np.ones(len(result) - len(faces), dtype=np.uint8),
    ]
    return result, generated, transactions, bool(ready)
