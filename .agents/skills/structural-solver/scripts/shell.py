"""Linear Kirchhoff–Love thin-shell elements on a planar triangular midsurface."""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pyvista as pv
from scipy.sparse import coo_matrix, diags
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import MatrixRankWarning, spsolve

from assembly import Material, von_mises
from boundary import face_nodes, parse_load
from report import build_report_json, write_json

EDGE_VERTICES = np.array([[0, 1], [1, 2], [2, 0]])
CURVATURE = np.array([[0, 0, 0, 2, 0, 0], [0, 0, 0, 0, 0, 2], [0, 0, 0, 0, 2, 0]])


def quadratic(points):
    x, y = np.moveaxis(points, -1, 0)
    return np.stack((np.ones_like(x), x, y, x*x, x*y, y*y), axis=-1)


def read_midsurface(path):
    surface = pv.read(path)
    if not isinstance(surface, pv.PolyData) or surface.n_cells == 0 or not surface.is_all_triangles:
        raise ValueError("shell input must be one triangular PolyData midsurface (e.g. VTP or STL)")
    points = np.asarray(surface.points, dtype=float)
    triangles = np.asarray(surface.faces.reshape(-1, 4)[:, 1:], dtype=np.intp)
    if not np.isfinite(points).all() or len(np.unique(triangles)) != len(points):
        raise ValueError("shell mesh needs finite points and no unused vertices")
    if len(np.unique(np.sort(triangles, axis=1), axis=0)) != len(triangles):
        raise ValueError("shell mesh has duplicate triangles")
    return surface, points, triangles


def plane_frame(points, triangles):
    span = np.linalg.norm(np.ptp(points, axis=0))
    xyz = points[triangles]
    normals = np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0])
    area = np.linalg.norm(normals, axis=1) / 2
    if span <= 0 or np.any(area <= 1e-14 * span**2):
        raise ValueError("shell mesh has degenerate triangles")
    normal = normals[np.argmax(area)] / (2 * area.max())
    # Make the top face reproducible even if the triangle winding changes.
    normal *= np.sign(normal[np.argmax(abs(normal))])
    axis = np.eye(3)[np.argmin(abs(normal))]
    axis -= (axis @ normal) * normal
    axis /= np.linalg.norm(axis)
    frame = np.column_stack((axis, np.cross(normal, axis), normal))
    local = (points - points[0]) @ frame
    if np.max(abs(local[:, 2])) > 1e-7 * span:
        raise ValueError("shell input must be planar; provide the midsurface, not a closed solid skin")
    return local[:, :2], frame, area, span


def shell_topology(triangles, npoints):
    local_edges = np.sort(triangles[:, EDGE_VERTICES], axis=-1).reshape(-1, 2)
    edges, owner, ids, counts = np.unique(
        local_edges, axis=0, return_index=True, return_inverse=True, return_counts=True
    )
    if np.any(counts > 2):
        raise ValueError("shell mesh has nonmanifold edges")
    # Join elements by shared edges. A vertex-only seam would behave as a hinge.
    ordered = np.argsort(ids).reshape(-1) // 3
    pair = np.cumsum(counts)[counts == 2]
    adjacency = coo_matrix(
        (np.ones(len(pair)), (ordered[pair - 2], ordered[pair - 1])),
        shape=(len(triangles), len(triangles)),
    )
    degree = np.bincount(edges[counts == 1].ravel(), minlength=npoints)
    if connected_components(adjacency, directed=False, return_labels=False) != 1 or np.any(degree > 2):
        raise ValueError("shell mesh must be one edge-connected manifold patch")
    boundary = counts == 1
    return edges, ids.reshape(-1, 3), divmod(owner[boundary], 3), boundary


def element_matrices(xy, triangles, edges, ids):
    triangle = xy[triangles]
    origin = triangle[:, :1]
    size = np.linalg.norm(triangle[:, EDGE_VERTICES[:, 1]] - triangle[:, EDGE_VERTICES[:, 0]], axis=2).max(axis=1)
    p = (triangle - origin) / size[:, None, None]
    l = np.concatenate((np.ones((*p.shape[:2], 1)), p), axis=2)
    d = np.linalg.inv(l)[:, 1:, :] / size[:, None, None]
    membrane = np.zeros((len(p), 3, 6))
    membrane[:, 0, ::2] = membrane[:, 2, 1::2] = d[:, 0]
    membrane[:, 1, 1::2] = membrane[:, 2, ::2] = d[:, 1]
    # The sorted edge gives both neighboring elements the same unit normal.
    tangents = xy[edges[:, 1]] - xy[edges[:, 0]]
    normals = tangents[:, [1, 0]] * np.array([-1, 1])
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    nx, ny = np.moveaxis(normals[ids], -1, 0)
    x, y = np.moveaxis(p[:, EDGE_VERTICES].mean(axis=2), -1, 0)
    slope = np.stack((np.zeros_like(x), nx, ny, 2*x*nx, y*nx+x*ny, 2*y*ny), axis=-1)
    coefficients = np.linalg.inv(np.concatenate((quadratic(p), slope), axis=1))
    coefficients[:, :, 3:] *= size[:, None, None]
    bending = (CURVATURE @ coefficients) / size[:, None, None]**2
    return membrane, bending, coefficients, p


def edge_conditions(surface, points, edges, boundary, owner, fixes, loads, frame, coefficients, p, bend_cells):
    edge_ids = edges[boundary]
    owners, local_edges = owner
    fixed_edges = np.zeros(len(edge_ids), dtype=bool)
    force_m = np.zeros((len(points), 2))
    force_b = np.zeros(len(points) + len(edges))
    records = []

    def selected(region):
        if region in {"x_min", "x_max", "y_min", "y_max", "z_min", "z_max"}:
            ids = face_nodes(points, region)
            mask = np.isin(edge_ids, ids).all(axis=1)
        elif region in surface.point_data:
            tags = np.asarray(surface.point_data[region])
            if tags.shape != (len(points),) or not np.isin(tags, [0, 1]).all():
                raise ValueError(f"edge mask {region!r} must be a scalar point array of 0/1 values")
            mask = tags[edge_ids].all(axis=1)
        else:
            raise ValueError(f"unknown boundary region {region!r}; use a bounding edge or a point mask")
        if not mask.any():
            raise ValueError(f"region {region!r} selects no complete boundary edges")
        return mask

    for region in fixes:
        fixed_edges |= selected(region)
    length = np.linalg.norm(points[edge_ids[:, 1]] - points[edge_ids[:, 0]], axis=1)
    for spec in loads:
        region, force = parse_load(spec)
        if not np.isfinite(force).all():
            raise ValueError("shell loads must be finite")
        mask = selected(region)
        if np.any(mask & fixed_edges):
            raise ValueError(f"loaded edges in {region!r} overlap clamped edges")
        weights = length[mask] / length[mask].sum()
        tangential = (force @ frame)[:2]
        normal = force @ frame[:, 2]
        np.add.at(force_m, edge_ids[mask].ravel(), np.repeat(weights[:, None] * tangential[None, :] / 2, 2, axis=0))
        elem = owners[mask]
        endpoints = p[elem[:, None], EDGE_VERTICES[local_edges[mask]]]
        # Simpson integrates the quadratic edge trace, including the slope DOFs.
        samples = quadratic(endpoints).sum(axis=1) + 4*quadratic(endpoints.mean(axis=1))
        trace = np.einsum("qi,qij->qj", samples / 6, coefficients[elem])
        records.append({"face": region, "force_total": force.tolist(),
                        "nodes": len(np.unique(edge_ids[mask])), "edges": int(mask.sum())})
        np.add.at(force_b, bend_cells[elem].ravel(), (weights[:, None] * normal * trace).ravel())

    return force_m.ravel(), force_b, np.unique(edge_ids[fixed_edges]), np.flatnonzero(boundary)[fixed_edges], records


def component_solve(cells, stiffness, f, fixed):
    rows = np.broadcast_to(cells[:, :, None], stiffness.shape)
    cols = np.broadcast_to(cells[:, None, :], stiffness.shape)
    k = coo_matrix((stiffness.ravel(), (rows.ravel(), cols.ravel())), shape=(len(f), len(f))).tocsr()
    free = np.setdiff1d(np.arange(len(f)), fixed)
    u = np.zeros_like(f)
    if len(free):
        reduced = k[free][:, free]
        diag = reduced.diagonal()
        if np.any(diag <= 0):
            raise ValueError("shell stiffness is singular; check the mesh and supports")
        scale = 1 / np.sqrt(diag)
        with warnings.catch_warnings():
            warnings.simplefilter("error", MatrixRankWarning)
            try:
                u[free] = scale * spsolve((diags(scale) @ reduced @ diags(scale)).tocsc(), scale * f[free])
            except MatrixRankWarning as error:
                raise ValueError("shell stiffness is singular; clamp a complete boundary edge") from error
    residual = k @ u - f
    return u, residual, free


def solve_shell(surface, points, triangles, thickness, material, fixes, loads):
    """@entry Assemble linear CST membrane and Morley bending on a planar patch in SI units."""
    if not np.isfinite(thickness) or thickness <= 0:
        raise ValueError("shell thickness must be finite and positive")
    if not np.isfinite(material.E) or material.E <= 0 or not -1 < material.nu < 0.5:
        raise ValueError("shell material requires E > 0 and -1 < Poisson's ratio < 0.5")
    if not fixes or not loads:
        raise ValueError("shell solve requires at least one clamped edge and one edge force")
    xy, frame, area, span = plane_frame(points, triangles)
    edges, ids, owners, boundary = shell_topology(triangles, len(points))
    membrane, bending, coefficients, p = element_matrices(xy, triangles, edges, ids)
    bend_cells = np.column_stack((triangles, len(points) + ids))
    membrane_cells = (2 * triangles[:, :, None] + np.arange(2)).reshape(-1, 6)
    f_m, f_b, fixed_vertices, fixed_edges, records = edge_conditions(
        surface, points, edges, boundary, owners, fixes, loads, frame, coefficients, p, bend_cells
    )
    nu = material.nu
    d = material.E / (1 - nu**2) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1-nu)/2]])
    k_m = (area * thickness)[:, None, None] * (membrane.transpose(0, 2, 1) @ d @ membrane)
    k_b = (area * thickness**3 / 12)[:, None, None] * (bending.transpose(0, 2, 1) @ d @ bending)
    u_m, r_m, free_m = component_solve(membrane_cells, k_m, f_m, (2*fixed_vertices[:, None] + np.arange(2)).ravel())
    u_b, r_b, free_b = component_solve(bend_cells, k_b, f_b, np.r_[fixed_vertices, len(points)+fixed_edges])
    combined = np.column_stack((u_m.reshape(-1, 2), u_b[:len(points)]))
    reaction = np.r_[r_m.reshape(-1, 2)[fixed_vertices].sum(axis=0), r_b[fixed_vertices].sum()] @ frame.T
    applied = np.asarray([item["force_total"] for item in records])
    force_scale = np.linalg.norm(applied, axis=1).sum()
    residual = np.linalg.norm(np.r_[r_m[free_m], r_b[free_b[free_b < len(points)]],
                                    r_b[free_b[free_b >= len(points)]] / span]) / force_scale

    strain = np.einsum("eij,ej->ei", membrane, u_m[membrane_cells])
    curvature = np.einsum("eij,ej->ei", bending, u_b[bend_cells])
    vm = []
    for side in (1, -1):
        stress = (strain - side * thickness / 2 * curvature) @ d
        voigt = np.zeros((len(triangles), 6))
        voigt[:, [0, 1, 3]] = stress
        vm.append(von_mises(voigt))
    work = float(f_m @ u_m + f_b @ u_b)
    membrane_energy = thickness * np.einsum("e,ei,ij,ej->", area, strain, d, strain)
    bending_energy = thickness**3 / 12 * np.einsum("e,ei,ij,ej->", area, curvature, d, curvature)
    strain_energy = float((membrane_energy + bending_energy) / 2)
    return {
        "displacement": combined @ frame.T, "element_stress": np.array(vm), "area": area,
        "load_edge_mean_displacement_m": work / force_scale, "strain_energy_J": strain_energy,
        "energy_balance": abs(2 * strain_energy - work) / abs(work) if work else float(strain_energy != 0),
        "reaction_N": reaction, "force_balance": np.linalg.norm(reaction + applied.sum(axis=0))/force_scale,
        "free_residual": residual, "load_records": records, "top_normal": frame[:, 2],
        "dofs": len(f_m) + len(f_b),
    }


def run_shell(mesh, thickness, material: Material, units, fixes, loads, output_dir, name, scale=1.0):
    """@entry Run the shell CLI case and write the numeric report and VTP fields."""
    surface, points, triangles = read_midsurface(mesh)
    length = 1e-3 if units == "mm" else 1.0
    stress_scale = 1e-6 if units == "mm" else 1.0
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("deformation scale must be finite and positive")
    result = solve_shell(surface, points * length, triangles, thickness * length, material, fixes, loads)
    displacement = result["displacement"] / length
    stress = result["element_stress"] * stress_scale
    if not np.isfinite(displacement).all() or not np.isfinite(stress).all():
        raise ValueError("shell solution is nonfinite; check the mesh, material and supports")
    original = surface.copy()
    original.point_data["displacement"] = displacement
    original.point_data["displacement_magnitude"] = np.linalg.norm(displacement, axis=1)
    original.cell_data["von_mises_top"] = stress[0]
    original.cell_data["von_mises_bottom"] = stress[1]
    original.cell_data["von_mises_stress"] = stress.max(axis=0)
    deformed = original.copy()
    deformed.points = points + scale * displacement
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    original_path, deformed_path, summary_path = [out / f"{name}_{suffix}" for suffix in ("mesh.vtp", "deformed.vtp", "report.json")]
    original.save(original_path)
    deformed.save(deformed_path)
    gates = {
        "force_balance": {"measured": float(result["force_balance"]), "threshold": "< 1e-6", "passed": bool(result["force_balance"] < 1e-6)},
        "free_residual": {"measured": float(result["free_residual"]), "threshold": "< 1e-6", "passed": bool(result["free_residual"] < 1e-6)},
        "energy_balance": {"measured": result["energy_balance"], "threshold": "< 1e-6", "passed": result["energy_balance"] < 1e-6},
    }
    edge_displacement = result["load_edge_mean_displacement_m"] / length
    report = build_report_json(
        decision={"title": "Planar thin-shell static solve", "kind": "linear_static_shell"},
        input={"geometry": str(mesh), "thickness": thickness, "top_normal": result["top_normal"].tolist(), "nodes": len(points),
               "triangles": len(triangles), "dofs": result["dofs"]},
        material={"name": material.name, "E": material.E, "nu": material.nu},
        units={"length": units, "force": "N", "stress": "MPa" if units == "mm" else "Pa", "energy": "J"},
        load_cases={"fixes": fixes, "loads": result["load_records"]},
        results={"field_summary": {"max_displacement": float(original.point_data["displacement_magnitude"].max()),
                                   "load_edge_mean_displacement": edge_displacement, "max_von_mises": float(stress.max()),
                                   "strain_energy": result["strain_energy_J"], "display_scale": scale},
                 "reaction_sum": result["reaction_N"].tolist()},
        validation={}, gates=gates,
        outputs={"mesh_vtp": str(original_path), "deformed_mesh_vtp": str(deformed_path), "report_json": str(summary_path)},
        runtime={"solver": "scipy sparse direct (CPU)", "element": "CST membrane + Morley Kirchhoff–Love bending"},
    )
    write_json(report, summary_path)
    return report
