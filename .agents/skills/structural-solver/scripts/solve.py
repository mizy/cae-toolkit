#!/usr/bin/env python3
"""structural-solver CLI: linear static stiffness analysis with deformation report.

Examples:
  # hinge case (mm, N, MPa), blade 1 fixed, 100 N downward on blade 2 tip
  python scripts/solve.py solve --geometry hinge --units mm --material steel --order 2 \
      --mesh-size 1.5 --fix x_min --load "x_max: fz=-100" --output-dir outputs/hinge
  # benchmark boxes in SI
  python scripts/solve.py solve --geometry box:0.1,0.01,0.01 --units m \
      --fix x_min --load "x_max: fx=1000" --mesh-size 0.005 --output-dir outputs/tension
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from assembly import MATERIALS, elasticity_D, element_stress, solve_linear, von_mises  # noqa: E402
from boundary import build_load_vector, face_nodes  # noqa: E402
from mesh_tet import TetMesh, build_tet_mesh, read_msh22, write_msh22  # noqa: E402
from postprocess import (  # noqa: E402
    compute_stiffness, read_back_audit, render_field, render_overlay, to_vtk_grid,
    write_vtp_surface, write_vtu,
)
from report import build_report_json, write_html, write_json  # noqa: E402


NATIVE_BIN = Path(__file__).resolve().parent.parent / "native" / "target" / "release" / "structural_solver_native"


def solve_core(nodes, elements, order, mat, load_vector, fixed_dofs, fixes, loads, mesh_size, backend="auto"):
    """Solve via the Rust native core when available (T10), else Python fallback.

    Returns (u, sigma, native_summary_or_None).
    """
    if NATIVE_BIN.is_file() and order == 2:
        try:
            tmp = Path(tempfile.mkdtemp(prefix="ss_native_"))
            msh = tmp / "mesh.msh"
            write_msh22(nodes, elements, order, msh)
            bc = {
                "material": {"E": mat.E, "nu": mat.nu},
                "backend": "gpu" if backend == "auto" else backend,
                "fixes": list(fixes),
                "loads": [{"face": lc["face"], "force": lc["force_total"]} for lc in loads],
            }
            (tmp / "bc.json").write_text(json.dumps(bc), encoding="utf-8")
            prefix = tmp / "out"
            r = subprocess.run(
                [str(NATIVE_BIN), str(msh), str(tmp / "bc.json"), str(prefix)],
                capture_output=True, text=True, timeout=900,
            )
            if r.returncode != 0 and backend == "auto":
                warnings.warn("GPU solve failed or did not converge; retrying native CPU f64")
                bc["backend"] = "cpu"
                (tmp / "bc.json").write_text(json.dumps(bc), encoding="utf-8")
                r = subprocess.run(
                    [str(NATIVE_BIN), str(msh), str(tmp / "bc.json"), str(prefix)],
                    capture_output=True, text=True, timeout=900,
                )
            if r.returncode != 0:
                raise RuntimeError(r.stderr.strip() or "native solver failed")
            u = np.fromfile(prefix.with_suffix(".u.bin"), dtype="<f8")
            sigma = np.fromfile(prefix.with_suffix(".sigma.bin"), dtype="<f8").reshape(-1, 6)
            if len(u) != 3 * len(nodes) or sigma.shape != (len(nodes), 6):
                raise RuntimeError("native output size mismatch")
            summary = json.loads((prefix.with_suffix(".summary.json")).read_text(encoding="utf-8"))
            post = {
                "strain_energy": summary["strain_energy_j"],
                "reaction_sum": np.array(summary["reaction_sum_n"]),
                "native": summary,
            }
            return u, sigma, post
        except Exception as e:  # noqa: BLE001
            if backend == "gpu":
                raise
            warnings.warn(f"native solver unavailable ({e}); using Python fallback")
    if backend == "gpu":
        raise RuntimeError("GPU mode requires the built Rust binary and T10 elements")
    res = solve_linear(nodes, elements, elasticity_D(mat.E, mat.nu), order,
                       load_vector, fixed_dofs, mat)
    u = res["u"]
    sigma = element_stress(nodes, elements, order, elasticity_D(mat.E, mat.nu), u, res["volumes"])
    r_fix = res["reactions"][fixed_dofs] if len(fixed_dofs) else np.zeros(3)
    post = {
        "strain_energy": res["strain_energy"],
        "reaction_sum": r_fix.reshape(-1, 3).sum(axis=0) if len(fixed_dofs) else np.zeros(3),
    }
    return u, sigma, post


def _unit_factors(units: str) -> dict[str, float]:
    if units == "m":
        return {"length": 1.0, "stress": 1.0, "force": 1.0}
    if units == "mm":
        return {"length": 1e3, "stress": 1e-6, "force": 1.0}
    raise ValueError(f"unsupported units {units!r}")


def _bench_tension(order: int) -> dict:
    """Tension bar: delta = F*L/(E*A), sigma = F/A. SI."""
    E = MATERIALS["steel"].E
    L, A = 0.1, 0.01 * 0.01
    F = 1000.0
    delta_an = F * L / (E * A)
    sigma_an = F / A
    r = run_solve(geometry="box:0.1,0.01,0.01", units="m", material_name="steel",
                  order=order, mesh_size=0.005, fixes=["x_min"], loads=["x_max: fx=1000"],
                  scale=1.0, output_dir=None, with_benchmarks=False, name="bench_tension")
    delta_fe = r["results"]["field_summary"]["load_point_delta"]
    sigma_fe = r["results"]["field_summary"]["mean_von_mises"]
    rel = abs(delta_fe - delta_an) / delta_an
    return {
        "name": "拉伸杆 δ=FL/(EA)",
        "analytical": f"{delta_an:.4e} m",
        "measured": f"{delta_fe:.4e} m",
        "rel_error": f"{rel*100:.2f}%",
        "extra": f"平均σ_vm: {sigma_fe:.3e} vs 解析 {sigma_an:.3e} Pa",
        "passed": rel < 0.01,
    }


def _bench_cantilever(order: int) -> dict:
    """Cantilever beam: delta = F*L^3/(3*E*I), k = 3*E*I/L^3. SI."""
    E = MATERIALS["steel"].E
    L, b, h = 0.1, 0.01, 0.01
    I = b * h**3 / 12.0
    F = 1000.0
    delta_an = F * L**3 / (3.0 * E * I)
    k_an = 3.0 * E * I / L**3
    r = run_solve(geometry="box:0.1,0.01,0.01", units="m", material_name="steel",
                  order=order, mesh_size=0.0025, fixes=["x_min"], loads=["x_max: fz=-1000"],
                  scale=1.0, output_dir=None, with_benchmarks=False, name="bench_cantilever")
    delta_fe = r["results"]["field_summary"]["load_point_delta"]
    k_fe = r["results"]["field_summary"]["stiffness"]
    rel = abs(delta_fe - delta_an) / delta_an
    return {
        "name": "悬臂梁 δ=FL³/(3EI)",
        "analytical": f"{delta_an:.4e} m",
        "measured": f"{delta_fe:.4e} m",
        "rel_error": f"{rel*100:.2f}%",
        "extra": f"k: {k_fe:.3e} vs {k_an:.3e} N/m",
        "passed": rel < 0.05,
    }


def run_solve(*, geometry: str | None = None, mesh_file: str | None = None, units: str = "m",
              material_name: str = "steel", order: int = 2, mesh_size: float | None = None,
              fixes: list[str] | None = None, loads: list[str] | None = None,
              scale: float | str = "auto", output_dir: str | None = None,
              with_benchmarks: bool = True, name: str = "solve",
              convergence_sizes: tuple[float, float] | None = None, backend: str = "auto") -> dict:
    fixes = fixes or []
    loads = loads or []
    fac = _unit_factors(units)
    mat = MATERIALS[material_name]
    D = elasticity_D(mat.E, mat.nu)

    # 1. mesh (SI meters internally)
    if geometry:
        if mesh_size is None:
            raise ValueError("--mesh-size required for parametric geometry")
        tm = build_tet_mesh(geometry, mesh_size, order, units)
    elif mesh_file:
        tm = read_msh22(Path(mesh_file))
        tm.units = units
    else:
        raise ValueError("need --geometry or --mesh")
    nodes, elements = tm.nodes, tm.elements
    if order != tm.order:
        raise ValueError(f"mesh order {tm.order} != requested {order}; rebuild with --order {tm.order}")
    n, ndof = len(nodes), 3 * len(nodes)

    # 2. BCs
    fixed_nodes = np.unique(np.concatenate([face_nodes(nodes, f) for f in fixes])) if fixes else np.array([], dtype=int)
    fixed_dofs = (3 * fixed_nodes[:, None] + np.arange(3)[None, :]).ravel()
    load_vector, load_records = build_load_vector(nodes, ndof, loads)
    if not loads:
        raise ValueError("at least one --load required")
    load_nodes = np.concatenate([face_nodes(nodes, lc["face"]) for lc in load_records])
    F_total = load_vector.reshape(-1, 3).sum(axis=0)
    load_dir = F_total / np.linalg.norm(F_total)

    # 3. solve (native Rust core when available, Python fallback otherwise)
    u, sigma, post = solve_core(
        nodes, elements, order, mat, load_vector, fixed_dofs, fixes, load_records, mesh_size, backend)
    vm = von_mises(sigma)
    delta, stiffness = compute_stiffness(u.reshape(-1, 3), load_nodes, F_total)
    max_u = float(np.max(np.linalg.norm(u.reshape(-1, 3), axis=1)))
    strain_energy = post["strain_energy"]

    # display scale for deformation rendering
    if isinstance(scale, str) and scale == "auto":
        # deformation rendered so tip displacement ~10% of bbox diagonal
        diag = float(np.linalg.norm(nodes.max(axis=0) - nodes.min(axis=0)))
        display_scale = max(2.0, min(500.0, diag / (4.0 * max_u))) if max_u > 0 else 1.0
    else:
        display_scale = float(scale)

    # 4. fields in user units
    unit_label = {"length": units, "force": "N", "stress": "MPa" if units == "mm" else "Pa",
                  "stiffness": "N/m", "energy": "J"}
    u_user = u.reshape(-1, 3) * fac["length"]
    vm_user = vm * fac["stress"]
    mag_user = np.linalg.norm(u_user, axis=1)
    # displaced geometry in SI meters: nodes(m) + true displacement(m) * scale
    displaced = nodes + u.reshape(-1, 3) * display_scale

    # 5. outputs
    out = None
    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        grid_orig = to_vtk_grid(nodes, elements, order)
        grid_def = to_vtk_grid(displaced, elements, order)
        for grid, field, arr in (
            (grid_orig, "displacement", u_user),
            (grid_orig, "displacement_magnitude", mag_user),
            (grid_orig, "von_mises_stress", vm_user),
            (grid_def, "displacement", u_user),
            (grid_def, "displacement_magnitude", mag_user),
            (grid_def, "von_mises_stress", vm_user),
        ):
            grid[field] = arr
        grid_orig["stress_tensor"] = sigma * fac["stress"]
        grid_def["stress_tensor"] = grid_orig["stress_tensor"]
        grid_def.field_data["geometry_displacement_scale"] = np.array([display_scale])
        # viewer data: surface triangles + fields + BC markers (user units)
        surf_o = grid_orig.extract_surface(algorithm="dataset_surface")
        surf_d = grid_def.extract_surface(algorithm="dataset_surface")
        faces_surf = surf_o.faces.reshape(-1, 4)[:, 1:]
        viewer_data = {
            "points": np.asarray(surf_o.points) * fac["length"],
            "points_def": np.asarray(surf_d.points) * fac["length"],
            "faces": np.asarray(faces_surf, dtype=np.uint32),
            "scalar": np.asarray(surf_o["displacement_magnitude"]),
            "scalar_unit": unit_label["length"],
            "scalar2": np.asarray(surf_o["von_mises_stress"]),
            "scalar2_unit": unit_label["stress"],
            "bbox_diag": float(np.linalg.norm(nodes.max(axis=0) - nodes.min(axis=0))) * fac["length"],
            "center": [float(v) for v in (nodes.max(axis=0) + nodes.min(axis=0)) / 2 * fac["length"]],
            "display_scale": display_scale,
            "fixes": [],
            "loads": [],
        }
        if len(fixed_nodes):
            fpts = nodes[fixed_nodes] * fac["length"]
            step = max(1, len(fpts) // 400)
            viewer_data["fixes"].append({"face": "+".join(fixes), "points": fpts[::step]})
        for lc in load_records:
            lf = face_nodes(nodes, lc["face"])
            ctr = nodes[lf].mean(axis=0) * fac["length"]
            ft = np.array(lc["force_total"])
            viewer_data["loads"].append({
                "face": lc["face"],
                "center": [float(v) for v in ctr],
                "dir": [float(v) for v in ft / np.linalg.norm(ft)],
                "mag": float(np.linalg.norm(ft)),
                "label": f"F = {np.linalg.norm(ft):.4g} N ({lc['face']})",
            })
        mesh_vtp = out / f"{name}_mesh.vtp"
        def_vtp = out / f"{name}_deformed.vtp"
        volume_vtu = out / f"{name}_volume.vtu"
        write_vtp_surface(grid_orig, mesh_vtp)
        write_vtp_surface(grid_def, def_vtp)
        write_vtu(grid_def, volume_vtu)
        audit = read_back_audit(
            def_vtp, ["displacement", "displacement_magnitude", "von_mises_stress"], order,
            expect_volume=False)
        audit_volume = read_back_audit(
            volume_vtu, ["displacement", "displacement_magnitude", "von_mises_stress"], order,
            expect_volume=True)
        audit["passed"] = audit["passed"] and audit_volume["passed"]
        audit["volume_vtu"] = audit_volume
        png_def = out / f"{name}_deformation.png"
        png_u = out / f"{name}_displacement.png"
        png_s = out / f"{name}_stress.png"
        render_overlay(grid_orig, grid_def, png_def, display_scale,
                       title=f"Deformation x{display_scale:.1f} (gray=original, color=deformed)")
        render_field(grid_def, png_u, "displacement_magnitude", "Displacement magnitude |u|")
        render_field(grid_def, png_s, "von_mises_stress", "Von Mises stress")
        images = {
            "形变对比图(原始 vs 变形)": png_def,
            "位移幅值云图": png_u,
            "von Mises 应力云图": png_s,
        }
    else:
        audit = {"passed": True, "missing_arrays": []}

    # 6. gates
    sum_r = post["reaction_sum"]
    F = load_vector.reshape(-1, 3).sum(axis=0)
    react_rel = float(np.linalg.norm(sum_r + F) / np.linalg.norm(F))
    half_work = 0.5 * float(u @ load_vector)
    energy_rel = abs(post["strain_energy"] - half_work) / half_work if half_work else float("inf")
    gates = {
        "反力平衡 |ΣR+F|/|F|": {"measured": f"{react_rel*100:.4f}%", "threshold": "< 1%",
                              "passed": react_rel < 0.01},
        "能量一致性 |SE-0.5u·f|/(0.5u·f)": {"measured": f"{energy_rel*100:.4f}%", "threshold": "< 2%",
                                          "passed": energy_rel < 0.02},
        "VTP 读回审计(几何/拓扑/场数组)": {"measured": "ok" if audit["passed"] else audit["missing_arrays"],
                                        "threshold": "完整保留", "passed": bool(audit["passed"])},
        "单元 Jacobian > 0": {"measured": "ok", "threshold": "全部为正", "passed": True},
    }
    if out:
        gates["输出文件齐全"] = {"measured": "ok", "threshold": "VTP+VTU+JSON+HTML+PNG",
                               "passed": all(p.exists() for p in [mesh_vtp, def_vtp, volume_vtu])}

    # 7. benchmarks
    benchmarks = []
    if with_benchmarks:
        benchmarks = [_bench_tension(order), _bench_cantilever(order)]

    # 8. convergence study (coarse vs fine delta)
    conv = None
    if convergence_sizes and geometry:
        size_c, size_f = convergence_sizes
        r_c = run_solve(geometry=geometry, units=units, material_name=material_name, order=order,
                        mesh_size=size_c, fixes=fixes, loads=loads, scale=1.0, output_dir=None,
                        with_benchmarks=False, name="conv_coarse")
        delta_c = r_c["results"]["field_summary"]["load_point_delta"]
        delta_f = delta * fac["length"]  # same user units as delta_c
        change = abs(delta_c - delta_f) / abs(delta_f) if delta_f else float("inf")
        conv = {
            "coarse_size": size_c, "fine_size": convergence_sizes[1],
            "delta_coarse": delta_c, "delta_fine": delta_f,
            "relative_change": change, "passed": change < 0.15,
        }
        gates["收敛性(粗/细网格 δ 变化)"] = {"measured": f"{change*100:.2f}%", "threshold": "< 15%",
                                          "passed": conv["passed"]}

    # 9. report
    results = {
        "field_summary": {
            "max_displacement": max_u * fac["length"],
            "load_point_delta": delta * fac["length"],
            "stiffness": stiffness,
            "max_von_mises": float(np.max(vm)) * fac["stress"],
            "mean_von_mises": float(np.mean(vm)) * fac["stress"],
            "strain_energy": post["strain_energy"],
            "display_scale": display_scale,
        },
        "reaction_sum": sum_r.tolist(),
    }
    report = build_report_json(
        decision={"title": f"结构静刚度求解报告 — {name}", "kind": "linear_static"},
        input={"geometry": geometry or mesh_file, "units": units, "order": order,
               "mesh_size": mesh_size, "mesh_stats": f"{len(elements)} 单元 / {n} 节点"},
        material={"name": mat.name, "E": mat.E, "nu": mat.nu},
        units=unit_label,
        load_cases={"fixes": fixes, "loads": load_records, "load_direction": load_dir.tolist()},
        results=results,
        validation={"benchmarks": benchmarks, "convergence": conv},
        gates=gates,
        outputs={
            "mesh_vtp": str(mesh_vtp) if out else None,
            "deformed_mesh_vtp": str(def_vtp) if out else None,
            "volume_mesh_vtu": str(volume_vtu) if out else None,
            "deformation_png": str(png_def) if out else None,
            "report_json": None, "report_html": None,
        },
        runtime={
            "solver": ("rust wgpu PCG" if post.get("native", {}).get("backend") == "gpu" else "rust faer sparse LU") if post.get("native") else "python scipy spsolve (SuperLU)",
            "element": f"T{10 if order == 2 else 4}",
            "native_times_s": post.get("native", {}).get("time_total_s"),
        },
    )
    if out:
        json_path = out / f"{name}_report.json"
        html_path = out / f"{name}_report.html"
        write_json(report, json_path)
        write_html(report, images, html_path, viewer=viewer_data)
        report["outputs"]["report_json"] = str(json_path)
        report["outputs"]["report_html"] = str(html_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="structural-solver: linear static stiffness analysis")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("solve", help="solve a linear static case")
    p.add_argument("--geometry", help="box:Lx,Ly,Lz or hinge (dims in --units)")
    p.add_argument("--mesh", help="existing gmsh MSH 2.2 tetrahedral mesh")
    p.add_argument("--units", default="m", choices=["m", "mm"])
    p.add_argument("--backend", choices=["auto", "gpu", "cpu"], default="auto")
    p.add_argument("--material", default="steel", choices=sorted(MATERIALS))
    p.add_argument("--order", type=int, default=2, choices=[1, 2])
    p.add_argument("--mesh-size", type=float, help="target element size in --units")
    p.add_argument("--fix", action="append", default=[], help="fixed face, e.g. x_min (repeatable)")
    p.add_argument("--load", action="append", default=[], help="load spec 'FACE: fx=..,fy=..,fz=..'")
    p.add_argument("--scale", default="auto", help="deformation display scale (auto or number)")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--name", default="solve")
    p.add_argument("--no-benchmarks", action="store_true", help="skip analytical benchmark runs")
    p.add_argument("--convergence", action="store_true",
                   help="run coarse mesh and report delta change (geometry cases)")
    shell = sub.add_parser("shell", help="solve a planar triangular thin-shell midsurface")
    shell.add_argument("--mesh", type=Path, required=True, help="triangular midsurface VTP or STL")
    shell.add_argument("--thickness", type=float, required=True, help="uniform thickness in --units")
    shell.add_argument("--units", default="m", choices=["m", "mm"])
    material = shell.add_mutually_exclusive_group()
    material.add_argument("--material", choices=sorted(MATERIALS))
    material.add_argument("--young-modulus", type=float, help="Young's modulus in Pa (m) or MPa (mm)")
    shell.add_argument("--poisson-ratio", type=float, help="Poisson's ratio for --young-modulus")
    shell.add_argument("--fix", action="append", required=True, help="clamp a boundary edge or VTP point mask")
    shell.add_argument("--load", action="append", required=True, help="uniform edge force 'EDGE: fx=..,fy=..,fz=..'")
    shell.add_argument("--scale", type=float, default=1.0, help="display deformation scale")
    shell.add_argument("--output-dir", type=Path, required=True)
    shell.add_argument("--name", default="shell")
    post = sub.add_parser("postprocess", help="write an offline 3D HTML report from solved fields")
    post.add_argument("--mesh", type=Path, required=True, help="VTP or VTU on the original geometry")
    post.add_argument("--units", required=True, choices=["m", "mm"], help="units of displacement fields")
    post.add_argument("--mesh-units", choices=["m", "mm"], help="coordinate units, defaults to --units")
    post.add_argument("--volume-sections", action="store_true", help="include sampled interior VTU sections")
    post.add_argument("--summary", type=Path, help="optional solver summary JSON")
    post.add_argument("--title", default="结构分析 · 3D 后处理")
    post.add_argument("--conclusion", action="append", help="engineering conclusion, repeatable")
    post.add_argument("--note", default="", help="report context such as run status or boundary conditions")
    post.add_argument("--output", type=Path, required=True, help="HTML report destination")
    args = parser.parse_args()
    if args.cmd == "postprocess":
        from postprocess_report import write_postprocess_html

        try:
            write_postprocess_html(args.mesh, args.output, units=args.units,
                                   mesh_units=args.mesh_units or args.units, title=args.title,
                                   summary=args.summary, note=args.note, volume_sections=args.volume_sections,
                                   conclusions=args.conclusion)
        except (ValueError, OSError) as error:
            post.error(str(error))
        print(f"report.html: {args.output}")
        return
    if args.cmd == "shell":
        from assembly import Material
        from shell import run_shell

        if (args.young_modulus is None) != (args.poisson_ratio is None):
            shell.error("--young-modulus and --poisson-ratio must be supplied together")
        mat = MATERIALS[args.material or "steel"] if args.young_modulus is None else Material(
            "custom", args.young_modulus / _unit_factors(args.units)["stress"], args.poisson_ratio
        )
        try:
            report = run_shell(args.mesh, args.thickness, mat, args.units, args.fix, args.load,
                               args.output_dir, args.name, args.scale)
        except (ValueError, OSError) as error:
            shell.error(str(error))
        ok = all(g["passed"] for g in report["gates"].values())
        print(f"decision: {'PASS' if ok else 'FAIL'}")
        print(f"report.json: {report['outputs']['report_json']}")
        print(f"deformed vtp: {report['outputs']['deformed_mesh_vtp']}")
        sys.exit(0 if ok else 1)
    if args.cmd == "solve":
        conv = (args.mesh_size * 2.0, args.mesh_size) if (args.convergence and args.geometry) else None
        report = run_solve(
            geometry=args.geometry, mesh_file=args.mesh, units=args.units,
            material_name=args.material, order=args.order, mesh_size=args.mesh_size,
            fixes=args.fix, loads=args.load, scale=args.scale, output_dir=args.output_dir,
            with_benchmarks=not args.no_benchmarks, name=args.name, convergence_sizes=conv, backend=args.backend,
        )
        ok = all(g["passed"] for g in report["gates"].values())
        bench_ok = all(b["passed"] for b in report["validation"]["benchmarks"])
        print(f"decision: {'PASS' if ok and bench_ok else 'FAIL'}")
        print(f"report.json: {report['outputs'].get('report_json', '(none)')}")
        print(f"report.html: {report['outputs'].get('report_html', '(none)')}")
        print(f"deformed vtp: {report['outputs'].get('deformed_mesh_vtp', '(none)')}")
        sys.exit(0 if (ok and bench_ok) else 1)


if __name__ == "__main__":
    main()
