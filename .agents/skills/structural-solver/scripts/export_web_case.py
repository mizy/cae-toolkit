#!/usr/bin/env python3
"""Export an assembled linear-elastic case for the browser solver prototype."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import spsolve

from assembly import MATERIALS, apply_dirichlet, assemble, elasticity_D
from boundary import build_load_vector, face_nodes
from mesh_tet import build_tet_mesh
from loads import gravity_load, cylinder_load


def export_case(geometry: str, units: str, mesh_size: float, order: int,
                material: str, fix: str | list[str], load: str | list[str], output: Path, *, mesh=None, gravity=None, density=None, cylinder=None) -> dict:
    """Assemble in SI units and export CSR plus an independent CPU reference."""
    fixes = [fix] if isinstance(fix, str) else fix
    load_specs = [load] if isinstance(load, str) else load
    if not fixes or not (load_specs or gravity is not None or cylinder):
        raise ValueError("At least one fixed face and one load are required")
    mesh = mesh if mesh is not None else build_tet_mesh(geometry, mesh_size, order, units)
    mat = MATERIALS[material]
    fixed_nodes = np.unique(np.concatenate([face_nodes(mesh.nodes, f) for f in fixes]))
    fixed_dofs = (3 * fixed_nodes[:, None] + np.arange(3)).ravel()
    rhs, loads = build_load_vector(mesh.nodes, 3 * len(mesh.nodes), load_specs)
    if not np.isfinite(rhs).all():
        raise ValueError("Load must contain finite values")
    for record in loads:
        if np.intersect1d(fixed_nodes, face_nodes(mesh.nodes, record["face"])).size:
            raise ValueError(f"Load face {record['face']} overlaps a fixed face; choose disjoint faces")
    stiffness, _ = assemble(mesh.nodes, mesh.elements, elasticity_D(mat.E, mat.nu), order)
    load_details = {}
    if gravity is not None:
        if density is None:
            raise ValueError("Self weight requires density in kg/m3")
        body_force, load_details["gravity"] = gravity_load(mesh.nodes, mesh.elements, order, density, gravity)
        rhs += body_force
    if cylinder:
        traction, details = cylinder_load(mesh.nodes, mesh.elements, order, cylinder)
        if np.any(traction[fixed_dofs]):
            raise ValueError("Cylinder load overlaps fixed degrees of freedom")
        details.pop("region_nodes")
        load_details["cylinder"] = details
        rhs += traction
    original_stiffness, original_rhs = stiffness, rhs.copy()
    stiffness, rhs = apply_dirichlet(stiffness, rhs, fixed_dofs)
    if not np.any(rhs):
        raise ValueError("No load remains on free degrees of freedom")
    stiffness.eliminate_zeros()
    stiffness.sort_indices()
    reference = spsolve(stiffness, rhs)
    if not np.isfinite(reference).all():
        raise ValueError("Reference solve failed; check the mesh and constraints")
    residual = np.linalg.norm(stiffness @ reference - rhs) / np.linalg.norm(rhs)
    if not np.isfinite(residual) or residual > 1e-7:
        raise ValueError(f"Reference solve residual {residual:.3g} is too large; check constraints and mesh")
    reaction = original_stiffness @ reference - original_rhs
    total_force = original_rhs.reshape(-1,3).sum(axis=0)
    reaction_sum = reaction[fixed_dofs].reshape(-1,3).sum(axis=0)
    force_scale = np.sum(np.linalg.norm(original_rhs.reshape(-1,3),axis=1))
    load_details["total_force_n"] = total_force.tolist()
    load_details["reference_reaction_n"] = reaction_sum.tolist()
    load_details["reference_force_balance"] = float(np.linalg.norm(total_force+reaction_sum)/force_scale)
    case = {
        "rowPtr": stiffness.indptr.tolist(),
        "colIdx": stiffness.indices.tolist(),
        "values": stiffness.data.tolist(),
        "rhs": rhs.tolist(),
        "reference": reference.tolist(),
        "coordinates": mesh.nodes.tolist(),
        "metadata": {
            "geometry": geometry, "units": units, "mesh_size": mesh_size,
            "order": order, "material": material, "fix": fixes[0], "fixes": fixes, "loads": loads,
            "nodes": len(mesh.nodes), "elements": len(mesh.elements),
            "dofs": len(rhs), "nnz": stiffness.nnz, "load_distribution": "equal nodal forces" if loads else "consistent integrated loads",
            "integrated_loads":load_details,
            "coordinate_unit": "m", "displacement_unit": "m", "force_unit": "N",
            "dof_order": "node-major xyz", "reference_relative_residual": float(residual),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(case, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    return case["metadata"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", default="box:0.1,0.01,0.01")
    parser.add_argument("--units", choices=("m", "mm"), default="m")
    parser.add_argument("--mesh-size", type=float, default=0.005)
    parser.add_argument("--order", type=int, choices=(1, 2), default=2)
    parser.add_argument("--material", choices=sorted(MATERIALS), default="steel")
    parser.add_argument("--fix", action="append", help="clamp a face, e.g. x_min; repeatable")
    parser.add_argument("--load", action="append", help="total face force, e.g. 'x_max: fz=-100'; repeatable")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export_case(args.geometry, args.units, args.mesh_size, args.order,
                                 args.material, args.fix or ["x_min"],
                                 args.load or ["x_max: fx=1000"], args.output), indent=2))


if __name__ == "__main__":
    main()
