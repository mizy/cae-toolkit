"""Create an offline, interactive HTML report from solved VTP or VTU fields."""
from __future__ import annotations

from html import escape
import json
import re
from pathlib import Path

import numpy as np
import pyvista as pv

from report import _encode_float32, _encode_uint32


def _surface_fields(surface, units):
    point_data = surface.point_data
    movement = "displacement" if "displacement" in point_data else f"u_{units}"
    has_deformation = movement in point_data
    u = np.asarray(point_data[movement], dtype=float) if has_deformation else np.zeros((surface.n_points, 3))
    if u.shape != (surface.n_points, 3) or not np.isfinite(u).all():
        raise ValueError("displacement must contain three finite components per point")

    stress_unit = "MPa" if units == "mm" else "Pa"
    values = [("位移幅值 |u|", units, np.linalg.norm(u, axis=1), "point", False)] if has_deformation else []
    for name, label in (("vertical_mm", "竖向位移"), ("axial_mm", "轴向位移")):
        if units == "mm" and name in point_data:
            values.append((label, units, point_data[name], "point", True))
    if has_deformation:
        for i, axis in enumerate("XYZ"):
            values.append((f"{axis} 向位移", units, u[:, i], "point", True))
    for name, label in (("roller_reaction_N", "节点支撑反力"), ("filament_load_N", "节点施加载荷")):
        if name in point_data:
            force = np.asarray(point_data[name])
            if force.ndim == 2:
                force = np.linalg.norm(force, axis=1)
            values.append((label, "N", force, "point", False))
    for names, label, signed in (
        (("von_mises_stress", f"von_mises_{stress_unit}"), "von Mises 等效应力", False),
        (("von_mises_top",), "壳上表面等效应力", False),
        (("von_mises_bottom",), "壳下表面等效应力", False),
        ((f"sigma_zz_{stress_unit}",), "Z 向正应力", True),
    ):
        for name in names:
            if name in point_data:
                values.append((label, stress_unit, point_data[name], "point", signed))
                break
            if name in surface.cell_data:
                values.append((label, stress_unit, surface.cell_data[name], "cell", signed))
                break
    if "stress_tensor" in point_data:
        tensor = np.asarray(point_data["stress_tensor"], dtype=float)
        if tensor.shape != (surface.n_points, 6) or not np.isfinite(tensor).all():
            raise ValueError("stress_tensor must have six finite Voigt components per point")
        for i, label in enumerate(("XX", "YY", "ZZ", "XY", "YZ", "ZX")):
            values.append((f"{label} 应力", stress_unit, tensor[:, i], "point", True))
        symmetric = tensor[:, [[0, 3, 5], [3, 1, 4], [5, 4, 2]]]
        principal = np.linalg.eigvalsh(symmetric)
        for i, label in ((2, "最大"), (1, "中间"), (0, "最小")):
            values.append((f"{label}主应力", stress_unit, principal[:, i], "point", True))
    # Keep native arrays discoverable without duplicating solver aliases and VTK IDs.
    known = {movement, "displacement_magnitude", "vertical_mm", "axial_mm",
             "stress_tensor", "roller_reaction_N", "filament_load_N",
             "von_mises_stress", "von_mises_top", "von_mises_bottom",
             f"von_mises_{stress_unit}", f"sigma_zz_{stress_unit}"}
    for association, arrays, label in (("point", point_data, "点"), ("cell", surface.cell_data, "单元")):
        for name in arrays.keys():
            if name in known or name.startswith("vtk"):
                continue
            data = np.asarray(arrays[name])
            if data.dtype.kind not in "biuf" or not np.isfinite(data).all():
                continue
            if data.ndim == 2 and data.shape[1] == 3:
                for i, axis in enumerate("XYZ"):
                    values.append((f"{name} · {axis} ({label})", "", data[:, i], association, True))
                data, name = np.linalg.norm(data, axis=1), f"{name} · 幅值"
            if data.ndim == 1:
                values.append((f"{name} ({label})", "", data, association, bool((data < 0).any())))
    if not values:
        values.append(("几何", "", np.zeros(surface.n_points), "point", False))
    fields = []
    for label, unit, data, association, signed in values:
        data = np.asarray(data)
        expected = surface.n_cells if association == "cell" else surface.n_points
        if data.shape != (expected,) or not np.isfinite(data).all():
            raise ValueError(f"{label} must contain one finite value per {association}")
        fields.append({
            "label": label, "unit": unit, "association": association,
            "min": float(data.min()), "max": float(data.max()),
            "p99": float(np.quantile(data, .99)), "signed": signed,
            "values": _encode_float32(data),
        })
    return u, fields, has_deformation


def _surface_payload(surface, units, coordinate_scale):
    surface = surface.triangulate(pass_verts=False, pass_lines=False)
    if not surface.n_faces:
        raise ValueError("the result has no surface faces")
    points = np.asarray(surface.points, dtype=float) * coordinate_scale
    if not np.isfinite(points).all() or not np.ptp(points, axis=0).any():
        raise ValueError("surface geometry must have finite coordinates and a nonzero span")
    u, fields, has_deformation = _surface_fields(surface, units)
    return {
        "points": _encode_float32(points), "polys": _encode_uint32(surface.faces),
        "displacement": _encode_float32(u), "fields": fields, "hasDeformation": has_deformation,
        "pointsCount": surface.n_points, "faceCount": surface.n_faces, "unit": units,
    }


def write_postprocess_html(mesh: Path, output: Path, *, units: str, mesh_units: str,
                           title: str, summary: Path | None = None, note: str = "",
                           volume_sections: bool = False, conclusions: list[str] | None = None) -> None:
    """@entry Package solved fields as an offline HTML report without a new solve."""
    if mesh.suffix.lower() not in {".vtp", ".vtu"}:
        raise ValueError("solved fields must be a VTP or VTU file")
    if output.suffix.lower() != ".html":
        raise ValueError("report output must end in .html")

    grid = pv.read(mesh)
    scale = {"m": 1.0, "mm": .001}
    coordinate_scale = scale[mesh_units] / scale[units]
    # The solver writes an explicit scale on deformed geometry so either result
    # can be postprocessed. Undo the stored display deformation before slicing.
    if "geometry_displacement_scale" in grid.field_data:
        stored_scale = float(grid.field_data["geometry_displacement_scale"][0])
        vectors = np.asarray(grid.point_data["displacement"])
        if not np.isfinite(stored_scale) or vectors.shape != (grid.n_points, 3):
            raise ValueError("invalid deformation metadata")
        grid.points = grid.points - stored_scale * vectors / coordinate_scale
    surface = grid if isinstance(grid, pv.PolyData) else grid.extract_surface(algorithm="dataset_surface")
    report = _surface_payload(surface, units, coordinate_scale)
    report["sections"] = {}
    if volume_sections:
        if not isinstance(grid, pv.UnstructuredGrid) or not np.isin(grid.celltypes, [10, 24]).all():
            raise ValueError("volume sections require a VTU tetrahedral result (T4 or T10)")
        bounds = np.asarray(grid.bounds).reshape(3, 2)
        for i, axis in enumerate("xyz"):
            normal = np.eye(3)[i]
            origin = bounds.mean(axis=1)
            sections = []
            for fraction in (.1, .25, .5, .75, .9):
                origin[i] = (1-fraction)*bounds[i,0] + fraction*bounds[i,1]
                cut = grid.slice(normal=normal, origin=origin)
                if cut.n_cells:
                    sections.append({"position": float(origin[i]*coordinate_scale),
                                     **_surface_payload(cut, units, coordinate_scale)})
            report["sections"][axis] = sections
    metadata = json.loads(summary.read_text(encoding="utf-8")) if summary else {}
    if not isinstance(metadata, dict):
        raise ValueError("summary JSON must contain an object")
    rows = []
    for key, label, unit in (
        ("coil_load_sag_mm", "筒面平均下沉", "mm"),
        ("balance_relative", "相对反力残差", ""),
        ("energy_relative", "相对能量误差", ""),
        ("self_mass_kg", "料盘自重输入", "kg"),
        ("tets", "体单元数", ""),
    ):
        value = metadata.get(key)
        if isinstance(value, (float, int)) and np.isfinite(value):
            rows.append(f"<tr><th>{label}</th><td>{value:.5g} {unit}</td></tr>")
    summary_fields = (metadata.get("results") or {}).get("field_summary") or {}
    units_in_report = metadata.get("units") or {}
    for key, label, kind in (
        ("max_displacement", "模型最大位移", "length"),
        ("max_von_mises", "模型最大等效应力", "stress"),
        ("strain_energy", "应变能", "energy"),
    ):
        value = summary_fields.get(key)
        if isinstance(value, (float, int)) and np.isfinite(value):
            unit = escape(str(units_in_report.get(kind, "")))
            rows.append(f"<tr><th>{label}</th><td>{value:.5g} {unit}</td></tr>")
    details = f'<table class="summary">{"".join(rows)}</table>' if rows else ""
    if conclusions:
        paragraphs = "".join(f"<li>{escape(line)}</li>" for line in conclusions)
    else:
        displacement_peak = report["fields"][0]["max"]
        stress = next((f for f in report["fields"] if f["label"] == "von Mises 等效应力"), None)
        paragraphs = ""
        if report["hasDeformation"]:
            paragraphs = f"<li>所载表面的最大位移幅值为 {displacement_peak:.5g} {units}。</li>"
        if stress is not None:
            paragraphs += f'<li>所载表面的最大等效应力为 {stress["max"]:.5g} {stress["unit"]}。</li>'
        paragraphs += "<li>工程判断需结合工况、材料和验证记录。</li>"

    folder = Path(__file__).resolve().parent
    vtk = folder.parents[1] / "mesh-watertight-repair/assets/vtk-32.12.0"
    asset = (vtk / "vtk.js").read_text(encoding="utf-8").replace("</script", r"<\/script")
    license_text = (vtk / "vtk.js.LICENSE.txt").read_text(encoding="utf-8")
    template = (folder / "postprocess_report.html").read_text(encoding="utf-8")
    replacements = {
        "TITLE": escape(title), "SOURCE": escape(mesh.name),
        "NOTE": f'<p class="note">{escape(note)}</p>' if note else "",
        "SUMMARY": details, "CONCLUSIONS": paragraphs,
        "DATA": json.dumps(report, ensure_ascii=False).replace("<", r"\u003c"),
        "VTK": asset, "LICENSE": escape(license_text),
        "VIEWER": (folder / "postprocess_viewer.js").read_text(encoding="utf-8"),
    }
    # Split once so substituted text can contain placeholders verbatim.
    html = re.sub(r"@@([A-Z]+)@@", lambda match: replacements[match[1]], template)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
