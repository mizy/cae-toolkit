"""Prepare area-weighted vehicle surface samples from a vehicle-data audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from vehicle_cd_learning.render import canonical_points
from vehicle_cd_similarity_lib.mesh_io import read_mesh


def sample_surface(mesh, geometry, count, seed, *, return_mesh=False, sampling="iid"):
    """@entry Sample the canonical triangulated surface with experimental normals."""
    if count < 1:
        raise ValueError("count must be positive")
    if sampling not in ("iid", "systematic"):
        raise ValueError("sampling must be iid or systematic")
    mesh = mesh.clean(tolerance=0).triangulate()
    points, transform = canonical_points(
        mesh.points,
        unit_scale_to_m=geometry["unit_scale_to_m"],
        front_axis=geometry["front_axis"],
        up_axis=geometry["up_axis"],
        ground_clearance_m=geometry.get("ground_clearance_m", 0.0),
    )
    points = points.astype(np.float64) * transform["length_m"]
    faces = np.asarray(mesh.faces).reshape(-1, 4)[:, 1:]
    triangles = points[faces]
    cross = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    twice_area = np.linalg.norm(cross, axis=1)
    valid = np.isfinite(twice_area) & (twice_area > 0)
    if not np.all(valid):
        raise ValueError(f"Surface contains {int((~valid).sum())} degenerate triangles")
    area = twice_area / 2
    edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    sorted_edges = np.sort(edges, axis=1)
    _, inverse, counts = np.unique(
        sorted_edges, axis=0, return_inverse=True, return_counts=True
    )
    orientation = np.bincount(
        inverse, weights=np.where(edges[:, 0] < edges[:, 1], 1, -1)
    )
    boundary = int(np.count_nonzero(counts == 1))
    nonmanifold = int(np.count_nonzero(counts > 2))
    inconsistent = int(np.count_nonzero((counts == 2) & (orientation != 0)))
    closed = boundary == 0 and nonmanifold == 0
    mesh.points = points
    if closed and inconsistent == 0:
        oriented = mesh.compute_normals(
            cell_normals=True,
            point_normals=False,
            consistent_normals=True,
            auto_orient_normals=True,
            split_vertices=False,
            inplace=False,
        )
        oriented_faces = np.asarray(oriented.faces).reshape(-1, 4)[:, 1:]
        if not np.array_equal(np.sort(faces, axis=1), np.sort(oriented_faces, axis=1)):
            raise ValueError("Normal orientation changed face indices")
        mesh = oriented
        faces = oriented_faces
        triangles = points[faces]
        cross = np.cross(
            triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
        )
        twice_area = np.linalg.norm(cross, axis=1)
        area = twice_area / 2
    volume = float(
        np.einsum(
            "ij,ij->i", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])
        ).sum()
        / 6
    )
    pressure_residual = float(np.linalg.norm(cross.sum(axis=0) / 2) / area.sum())
    rng = np.random.default_rng(seed)
    if sampling == "iid":
        indices = rng.choice(len(faces), count, p=area / area.sum())
    else:
        cdf = np.cumsum(area / area.sum())
        cdf[-1] = 1.0
        positions = (np.arange(count) + rng.random()) / count
        indices = np.searchsorted(cdf, positions, side="right")
    uv = rng.random((count, 2))
    root = np.sqrt(uv[:, 0])
    bary = np.column_stack((1 - root, root * (1 - uv[:, 1]), root * uv[:, 1]))
    samples = np.einsum("ni,nij->nj", bary, triangles[indices])
    arrays = {
        "points_m": samples,
        "normals": cross[indices] / twice_area[indices, None],
        "weights_m2": np.full(count, area.sum() / count),
    }
    diagnostics = {
        "sampling": sampling,
        "normalization": transform,
        "surface_area_m2": float(area.sum()),
        "boundary_edges": boundary,
        "nonmanifold_edges": nonmanifold,
        "inconsistent_winding_edges": inconsistent,
        "closed_by_edge_incidence": closed,
        "signed_volume_m3": volume,
        "constant_pressure_force_residual_fraction": pressure_residual,
        "normals": "VTK auto-oriented"
        if closed and inconsistent == 0
        else "original triangle winding",
        "self_intersections_checked": False,
    }
    return (arrays, diagnostics, mesh) if return_mesh else (arrays, diagnostics)


def prepare(audit_path, case_id, output_dir, count, seed):
    """@entry Save samples and provenance without asserting unknown force references."""
    audit_path, output_dir = Path(audit_path), Path(output_dir)
    audit_bytes = audit_path.read_bytes()
    audit = json.loads(audit_bytes)
    matches = [case for case in audit["cases"] if case["case_id"] == case_id]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one case: {case_id}")
    case = matches[0]
    path = Path(case["geometry"]["path"])
    arrays, diagnostics, mesh = sample_surface(
        read_mesh(path), case["geometry"], count, seed, return_mesh=True
    )
    aref = case.get("assumptions", {}).get("source_force_reference_area_m2")
    if aref is not None and (not np.isfinite(aref) or aref <= 0):
        raise ValueError("Invalid source force reference area")
    experimental = (
        case["condition_profile"] == "drivaerml"
        and aref is not None
        and diagnostics["closed_by_edge_incidence"]
        and diagnostics["inconsistent_winding_edges"] == 0
        and diagnostics["signed_volume_m3"] > 0
        and diagnostics["constant_pressure_force_residual_fraction"] < 1e-6
    )
    metadata = {
        "case_id": case_id,
        "seed": seed,
        "count": count,
        "geometry_path": str(path.resolve()),
        "geometry_sha256": hashlib.file_digest(path.open("rb"), "sha256").hexdigest(),
        "audit_path": str(audit_path.resolve()),
        "audit_sha256": hashlib.sha256(audit_bytes).hexdigest(),
        "source_manifest": audit["source_manifests"][case["condition_profile"]],
        "geometry_convention": case["geometry"],
        "reference_area_m2": aref,
        "experimental_force_quadrature": experimental,
        "coefficient_integration_allowed": experimental,
        "coefficient_integration_note": "Experimental DrivAer source CFD surface assumption; self-intersections and component orientation have not been independently verified.",
        "signed_distance_note": "Canonical mesh available for experimental signed-distance queries only when experimental_force_quadrature is true; sign convention must be verified by the consumer.",
        **diagnostics,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output_dir / "surface.npz", **arrays)
    mesh.save(output_dir / "surface.vtp")
    (output_dir / "surface.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n"
    )
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit", type=Path)
    parser.add_argument("case_id")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--count", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(args.audit, args.case_id, args.output_dir, args.count, args.seed),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
