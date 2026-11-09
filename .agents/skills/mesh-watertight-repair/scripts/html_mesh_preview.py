from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv

from mesh_io import triangle_faces


VTK_JS_UNAVAILABLE_STUB = (
    'console.error("vtk.js unavailable: set VTK_JS_PATH to a readable local file");'
)
DEFAULT_VIEWER_TRIANGLE_LIMIT: int | None = None
SOURCE_PROJECTED_OUTPUT_KEY = "source_projected_watertight_candidate_vtp"


def mesh_previews(report: dict[str, Any], report_dir: Path) -> list[dict[str, Any]]:
    specs = preview_specs(report)
    previews = []
    for label, raw_path in specs:
        if not raw_path:
            continue
        path = resolve_path(raw_path, report_dir)
        if path and path.exists():
            try:
                previews.append(mesh_preview(label, path))
            except Exception as exc:
                previews.append({"label": label, "source": str(path), "error": str(exc), "points": [], "polys": []})
        else:
            previews.append({"label": label, "source": str(path), "error": "mesh file is missing", "points": [], "polys": []})
    if previews and (report.get("target", {}).get("purpose") == "print"
                     or report.get("target", {}).get("name") == "watertight-material"):
        regions = report.get("unresolved_policy_packet", {}).get("items", [])
        if previews[0].get("full_resolution"):
            previews[0]["regions"] = regions
        elif regions:
            previews[0]["opening_review_unavailable"] = "Opening review requires a full-resolution source view; decimated face IDs cannot identify source regions."
        for preview in previews:
            preview["minimum_wall_mm"] = report.get("parameters", {}).get("print_min_wall_mm")
    return previews


def preview_specs(report: dict[str, Any]) -> list[tuple[str, str | None]]:
    outputs = report.get("outputs", {})
    if outputs.get("mesh_previews"):
        return [(item["label"], item["path"]) for item in outputs["mesh_previews"]]
    if outputs.get("cfd_surface_candidate_vtp"):
        return [
            ("Accepted watertight input", outputs.get("watertight_mesh_vtp")),
            ("Accepted CFD surface refinement" if outputs.get("accepted_mesh_vtp") else "Rejected CFD surface candidate", outputs["cfd_surface_candidate_vtp"]),
        ]
    original_comparison = outputs.get("original_mesh_vtp")
    processed_comparison = outputs.get("processed_mesh_vtp")
    if original_comparison and processed_comparison:
        specs = [
            ("Original mesh with watertight issue arrays", original_comparison),
            ("Processed candidate with watertight issue arrays", processed_comparison),
        ]
        if outputs.get("original_issue_faces_vtp"):
            specs.append(("Original issue-adjacent faces", outputs.get("original_issue_faces_vtp")))
        if outputs.get("processed_issue_faces_vtp"):
            specs.append(("Processed issue-adjacent faces", outputs.get("processed_issue_faces_vtp")))
        return specs
    stage1 = outputs.get("stage1_exterior_candidate_vtp") or input_path(report)
    proxy = outputs.get("closure_proxy_vtp")
    if accepted := outputs.get("accepted_mesh_vtp"):
        projected = outputs.get(SOURCE_PROJECTED_OUTPUT_KEY)
        if projected == accepted:
            label = "Accepted source-projected watertight shell"
        else:
            label = "Accepted repair"
        return [
            ("Source exterior", stage1),
            (label, accepted),
            ("Diagnostic closure proxy", proxy),
        ]
    if outputs.get("rejected_candidate_vtp") and outputs.get("closure_proxy_vtp"):
        return [
            ("Rejected candidate", outputs.get("rejected_candidate_vtp")),
            ("Diagnostic closure proxy", outputs.get("closure_proxy_vtp")),
        ]
    if proxy:
        return [("Diagnostic closure proxy", proxy)]
    if refined := outputs.get("adaptive_refined_source_vtp"):
        return [("Original input", input_path(report)), ("Adaptive refined source", refined)]
    if candidate := outputs.get(SOURCE_PROJECTED_OUTPUT_KEY) or outputs.get("rejected_candidate_vtp"):
        return [("Source exterior", stage1), ("Unaccepted candidate — diagnostic only", candidate)]
    return []


def input_path(report: dict[str, Any]) -> str | None:
    raw_input = report.get("input")
    if isinstance(raw_input, dict):
        return raw_input.get("path")
    if isinstance(raw_input, str):
        return raw_input
    return None


def resolve_path(raw_path: str | None, report_dir: Path) -> Path | None:
    if not raw_path:
        return None
    path = Path(raw_path)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    candidate = report_dir / path
    return candidate if candidate.exists() else path


def mesh_preview(label: str, path: Path) -> dict[str, Any]:
    # Preview source defects too: cleaning can collapse degenerate faces to lines.
    if path.suffix.lower() in (".glb", ".gltf"):
        from gltf_surface import read_gltf_surface
        mesh = read_gltf_surface(path)
    else:
        mesh = pv.read(path)
    if isinstance(mesh, pv.MultiBlock):
        mesh = mesh.combine()
    if not isinstance(mesh, pv.PolyData):
        mesh = mesh.extract_surface()
    mesh = mesh.triangulate()
    faces = triangle_faces(mesh)
    original_triangles = int(faces.shape[0])
    viewer_triangle_limit = viewer_triangle_limit_from_env()
    if viewer_triangle_limit and original_triangles > viewer_triangle_limit:
        reduction = 1.0 - (viewer_triangle_limit / float(original_triangles))
        mesh = mesh.decimate_pro(reduction, preserve_topology=False).triangulate().clean()
        faces = triangle_faces(mesh)
    polys = np.column_stack((np.full((faces.shape[0], 1), 3, dtype=np.uint32), faces)).ravel()
    points = np.asarray(mesh.points, dtype=np.float32)
    displayed_triangles = int(faces.shape[0])
    fields = {
        name: np.nan_to_num(np.asarray(mesh.cell_data[name], dtype=float), nan=-1, posinf=-1, neginf=-1).tolist()
        for name in ("print_generated_face", "print_source_distance_mm", "print_thickness_mm")
        if name in mesh.cell_data and len(mesh.cell_data[name]) == displayed_triangles
    }
    if "source_projected_vertex_fraction" in mesh.cell_data:
        fraction = np.asarray(mesh.cell_data["source_projected_vertex_fraction"], dtype=float).reshape(-1)
        if fraction.size == displayed_triangles:
            fields["projection_fallback_fraction"] = np.where(np.isfinite(fraction), 1. - np.clip(fraction, 0., 1.), -1.).tolist()
    normals = np.asarray(mesh.point_data["Normals"], dtype=float) if "Normals" in mesh.point_data else None
    geometry_normals = None
    if "corner_normals" in mesh.cell_data:
        corner_normals = np.asarray(mesh.cell_data["corner_normals"], dtype=float)
        if corner_normals.shape != (displayed_triangles, 9) or not np.all(np.isfinite(corner_normals)):
            raise ValueError("corner_normals must contain three finite normals per triangle")
        # Split display vertices only at normal seams; keep triangle IDs and physical VTP intact.
        geometric = mesh.compute_normals(cell_normals=False, split_vertices=False)
        vertex_ids = faces.ravel()
        corner_normals = np.round(corner_normals.reshape(-1, 3), 6)
        keys = np.column_stack((vertex_ids, corner_normals))
        _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
        display_ids = vertex_ids[first]
        geometry_normals = np.asarray(geometric.point_data["Normals"])[display_ids]
        points = points[display_ids]
        normals = corner_normals[first]
        polys = np.column_stack((np.full(displayed_triangles, 3), inverse.reshape(-1, 3))).ravel()
    valid_normals = normals is not None and normals.shape == points.shape and np.all(np.isfinite(normals))
    return {
        "label": label,
        "source": str(path),
        "normals": np.round(normals, 6).tolist() if valid_normals else None,
        "geometry_normals": np.round(geometry_normals, 6).tolist() if geometry_normals is not None else None,
        "points": np.round(points, 6).tolist(),
        "polys": polys.astype(np.uint32).tolist(),
        "triangles": displayed_triangles,
        "original_triangles": original_triangles,
        "full_resolution": displayed_triangles == original_triangles,
        "downsampled": displayed_triangles != original_triangles,
        "viewer_triangle_limit": viewer_triangle_limit,
        "points_count": int(points.shape[0]),
        "proxy_weights": proxy_weight_values(mesh, displayed_triangles),
        "issue_values": issue_values(mesh, displayed_triangles),
        "fields": fields,
    }


def viewer_triangle_limit_from_env() -> int | None:
    raw = os.environ.get("CAD_SURFACE_MESHER_VIEWER_TRIANGLES")
    if raw:
        try:
            value = int(raw)
        except ValueError:
            return DEFAULT_VIEWER_TRIANGLE_LIMIT
        return value if value > 0 else None
    return DEFAULT_VIEWER_TRIANGLE_LIMIT


def proxy_weight_values(mesh: Any, cell_count: int) -> list[float]:
    for name in ("proxy_weight", "sdf_blend_weight"):
        if name not in mesh.cell_data:
            continue
        values = np.asarray(mesh.cell_data[name], dtype=np.float64).reshape(-1)
        if values.size != cell_count:
            continue
        return np.round(np.clip(values, 0.0, 1.0), 6).tolist()
    return []


def issue_values(mesh: Any, cell_count: int) -> list[float]:
    name = "watertight_audit_issue_mask"
    if name not in mesh.cell_data:
        return []
    values = np.asarray(mesh.cell_data[name]).reshape(-1)
    if values.size != cell_count:
        return []
    return (values != 0).astype(np.float32).tolist()


def vtk_js_source() -> str:
    env_path = os.environ.get("VTK_JS_PATH")
    path = Path(env_path) if env_path else Path(__file__).resolve().parents[1] / "assets/vtk-32.12.0/vtk.js"
    if not path.is_file():
        return VTK_JS_UNAVAILABLE_STUB
    try:
        source = path.read_text(encoding="utf-8")
        if not env_path:
            notices = "\n".join(
                (path.parent / name).read_text(encoding="utf-8")
                for name in ("LICENSE", "Copyright.txt", "vtk.js.LICENSE.txt")
            )
            source = "/*\n" + notices.replace("*/", "* /") + "\n*/\n" + source
        return source
    except (OSError, UnicodeError):
        return VTK_JS_UNAVAILABLE_STUB


def vtk_script_section(vtk_js: str, preview_meshes: list[dict[str, Any]]) -> str:
    if not preview_meshes:
        return ""
    return "\n".join([
        "<script>", script_source(vtk_js), "</script>",
        f'<script type="application/json" id="mesh-preview-data">{script_json(preview_meshes)}</script>',
        "<script>", viewer_js(), "</script>",
    ])


def script_source(source: str) -> str:
    return source.replace("</script", "<\\/script")


def script_json(value: Any) -> str:
    data = json.dumps(value, separators=(",", ":"), ensure_ascii=False)
    return data.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")


def viewer_js() -> str:
    return Path(__file__).with_name("html_mesh_viewer.js").read_text(encoding="utf-8")
