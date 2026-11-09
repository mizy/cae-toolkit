"""Measured print-scale fidelity, normal-ray wall estimates, and layer contours."""

from __future__ import annotations

import numpy as np
import pyvista as pv
import trimesh


def face_distances(
    mesh: trimesh.Trimesh, target: trimesh.Trimesh
) -> tuple[np.ndarray, dict]:
    if np.array_equal(mesh.vertices, target.vertices) and np.array_equal(
        mesh.faces, target.faces
    ):
        return np.zeros(len(mesh.faces)), {
            "sampled_max_mm": 0.0,
            "p95_face_max_mm": 0.0,
            "samples": 0,
            "identical_arrays": True,
        }
    surface = pv.wrap(target)
    barycentric = np.array(
        [
            [1 / 3, 1 / 3, 1 / 3],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0.5, 0.5, 0],
            [0, 0.5, 0.5],
            [0.5, 0, 0.5],
        ]
    )
    maximum = np.zeros(len(mesh.faces))
    for start in range(0, len(mesh.faces), 20000):
        samples = np.einsum(
            "ki,fij->fkj", barycentric, mesh.triangles[start : start + 20000]
        ).reshape(-1, 3)
        _, closest = surface.find_closest_cell(samples, return_closest_point=True)
        distances = np.linalg.norm(samples - closest, axis=1).reshape(
            -1, len(barycentric)
        )
        maximum[start : start + len(distances)] = distances.max(axis=1)
    return maximum, {
        "sampled_max_mm": float(maximum.max()),
        "p95_face_max_mm": float(np.percentile(maximum, 95)),
        "samples": int(len(mesh.faces) * len(barycentric)),
    }


def fidelity_check(
    source: trimesh.Trimesh,
    candidate: trimesh.Trimesh,
    generated: np.ndarray,
    *,
    error: float,
    mode: str,
    thickness: float,
    merged: bool,
    shells: list[dict],
) -> tuple[np.ndarray, dict]:
    forward, forward_report = face_distances(candidate, source)
    backward, backward_report = face_distances(source, candidate)
    included = ~generated.astype(bool)
    if mode == "hollow":
        components = trimesh.graph.connected_components(
            candidate.face_adjacency, nodes=np.arange(len(candidate.faces))
        )
        for ids, shell in zip(components, shells):
            if shell["depth"] % 2:
                included[ids] = False
    if mode == "thicken":
        allowed = thickness / 2 + error
        a = (
            float(np.max(np.abs(forward[included] - thickness / 2)))
            if np.any(included)
            else 0.0
        )
        b = float(backward.max())
        passed = a <= error and b <= allowed
    else:
        a = float(forward[included].max()) if np.any(included) else 0.0
        b = float(backward.max())
        passed = a <= error and (merged or b <= error)
    return forward, {
        "method": "seven_triangle_samples_including_vertices_and_edge_midpoints; not a Hausdorff certificate",
        "candidate_to_source": forward_report,
        "source_to_candidate": backward_report,
        "max_error_mm": error,
        "forward_checked_max_mm": a,
        "reverse_checked_max_mm": b,
        "generated_faces_excluded_forward": int(np.count_nonzero(~included)),
        "reverse_gate_applies": not merged,
        "authorized_offset_mm": thickness / 2 if mode == "thicken" else 0.0,
        "passed": bool(passed),
    }


def wall_check(
    mesh: trimesh.Trimesh, minimum: float, *, max_samples: int = 4096
) -> tuple[np.ndarray, dict]:
    surface = pv.wrap(mesh)
    ids = np.unique(
        np.linspace(
            0, len(mesh.faces) - 1, min(len(mesh.faces), max_samples), dtype=int
        )
    )
    values = np.full(len(mesh.faces), -1.0, dtype=float)
    diagonal = float(np.linalg.norm(mesh.extents))
    epsilon = max(diagonal * 1e-8, 1e-8)
    triangles = mesh.triangles[ids]
    normals = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    lengths = np.linalg.norm(normals, axis=1)
    normals = np.divide(
        normals,
        lengths[:, None],
        out=np.zeros_like(normals),
        where=lengths[:, None] > 0,
    )
    for index, normal in zip(ids, normals):
        origin = mesh.triangles_center[index]
        direction = -normal
        hit, _ = surface.ray_trace(
            origin + direction * epsilon,
            origin + direction * diagonal * 1.01,
            first_point=True,
        )
        if np.size(hit):
            values[index] = float(
                np.linalg.norm(np.asarray(hit).reshape(-1, 3)[0] - origin)
            )
    measured = values[ids]
    valid = measured >= 0
    thin = valid & (measured < minimum * (1 - 1e-6))
    return values, {
        "method": "deterministic face-centre inward normal first-exit ray; sampled chord estimate, not certified minimum thickness",
        "minimum_required_mm": minimum,
        "samples": len(ids),
        "measured": int(valid.sum()),
        "missing_rays": int((~valid).sum()),
        "thin_samples": int(thin.sum()),
        "sampled_min_mm": float(measured[valid].min()) if np.any(valid) else None,
        "all_face_centres_sampled": len(ids) == len(mesh.faces),
        "passed": bool(np.all(valid) and not np.any(thin)),
    }


def layer_check(
    mesh: trimesh.Trimesh, height: float, *, max_layers: int = 10000
) -> dict:
    import manifold3d as manifold

    solid = manifold.Manifold(
        manifold.Mesh64(
            np.asarray(mesh.vertices, dtype=np.float64),
            np.asarray(mesh.faces, dtype=np.uint64),
        )
    )
    count = int(np.ceil(mesh.extents[2] / height))
    if solid.status() != manifold.Error.NoError or not 0 < count <= max_layers:
        return {
            "passed": False,
            "reason": "invalid manifold or layer count exceeds budget",
            "layers_requested": count,
        }
    layers, invalid, island_starts, unsupported = [], 0, 0, 0.0
    previous = None
    for index in range(count):
        lower = float(mesh.bounds[0, 2] + index * height)
        z = (lower + min(lower + height, float(mesh.bounds[1, 2]))) / 2
        section = solid.slice(z)
        polygons = section.to_polygons()
        areas = [
            float(
                np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1])
                / 2
            )
            for p in polygons
        ]
        bad = sum(
            len(p) < 3 or not np.isfinite(p).all() or abs(a) < 1e-15
            for p, a in zip(polygons, areas)
        )
        invalid += bad
        islands = sum(a > 0 for a in areas)
        if previous is not None:
            unsupported += max(0.0, float((section - previous.offset(height)).area()))
            island_starts += max(0, islands - layers[-1]["islands"])
        layers.append(
            {
                "z_mm": z,
                "islands": islands,
                "area_mm2": float(section.area()),
                "contours_xy_mm": [np.round(p, 6).tolist() for p in polygons],
            }
        )
        previous = section
    return {
        "passed": invalid == 0 and any(row["islands"] for row in layers),
        "method": "Manifold material cross-sections at every layer midpoint; contours are implicitly closed",
        "layer_height_mm": height,
        "invalid_contours": invalid,
        "new_island_count": island_starts,
        "unsupported_area_sum_mm2": unsupported,
        "support_test": "45-degree previous-layer offset heuristic; requires slicer support planning",
        "layers": layers,
        "machine_toolpath_validated": False,
    }
