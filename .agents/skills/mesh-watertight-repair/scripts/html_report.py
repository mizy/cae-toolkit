from __future__ import annotations

import base64
import json
import mimetypes
from html import escape
from pathlib import Path
from typing import Any

from html_mesh_preview import DEFAULT_VIEWER_TRIANGLE_LIMIT, mesh_preview, preview_specs  # noqa: F401
from html_mesh_preview import mesh_previews, resolve_path, vtk_js_source, vtk_script_section


def write_html_report(report: dict[str, Any], path: Path, title: str) -> None:
    path.write_text(render_html(report, title, path.parent), encoding="utf-8")


def render_html(report: dict[str, Any], title: str, report_dir: Path) -> str:
    vtk_js = vtk_js_source()
    preview_meshes = mesh_previews(report, report_dir)
    body = [
        hero_section(title, report),
        viewer_section(preview_meshes),
        print_artifact_section(report, report_dir),
        summary_section(report),
        diagnostics_section(report),
        contract_section(report),
        stages_section(report),
        watertight_artifact_section(report, report_dir),
        trace_section(report),
        change_section(report),
        defect_section(report),
        requested_section(report),
        outputs_section(report),
        visual_section(report, report_dir),
        full_json_section(report),
        vtk_script_section(vtk_js, preview_meshes),
    ]
    return "\n".join([
        "<!doctype html>",
        "<html>",
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{text(title)}</title>",
        "<style>",
        css(),
        "</style>",
        "</head>",
        "<body>",
        *body,
        "</body>",
        "</html>",
    ])


def hero_section(title: str, report: dict[str, Any]) -> str:
    status = report.get("decision", {}).get("status")
    label = f'<span class="status status-{text(str(status or "visual"))}">{text(str(status or "visual only"))}</span>'
    return "\n".join([
        '<header class="report-hero">',
        f"<h1>{text(title)}</h1>",
        f"{label}",
        "</header>",
    ])


def summary_section(report: dict[str, Any]) -> str:
    rows = []
    if "decision" in report:
        rows.append(("Decision", report["decision"]))
    if "input" in report:
        rows.append(("Input", report["input"]))
    if "input_truth" in report:
        rows.append(("Input Truth", report["input_truth"]))
    if "target" in report:
        rows.append(("Target", report["target"]))
    if "output_contract" in report:
        rows.append(("Output Contract", report["output_contract"]))
    if "method" in report:
        rows.append(("Method", report["method"]))
    if "gates" in report:
        rows.extend((f"Gate: {key}", value) for key, value in report["gates"].items())
    return section("Summary", kv_table(rows))


def diagnostics_section(report: dict[str, Any]) -> str:
    return section_from_keys(report, "Diagnostics", [
        "parameters",
        "acceptance_checks",
        "limitations",
        "group_filter",
        "workflow_decision",
        "view_reports",
        "size_field",
        "refinement_iterations",
        "comparisons",
        "cfd_surface_refinement",
        "assembly_context_validation",
        "printing",
        "opening_transactions",
        "diagnostics",
    ])


def contract_section(report: dict[str, Any]) -> str:
    return section_from_keys(report, "Source-Preserving Contract", [
        "inventory_before",
        "deterministic_passes",
        "unresolved_policy_packet",
        "policy_decisions",
        "inventory_after",
        "ignored_outputs",
        "unhandled_items",
    ])


def section_from_keys(report: dict[str, Any], title: str, keys: list[str]) -> str:
    rows = [(key, report[key]) for key in keys if key in report]
    if not rows:
        return ""
    return section(title, kv_table(rows))


def stages_section(report: dict[str, Any]) -> str:
    stages = report.get("stages")
    if not isinstance(stages, dict) or not stages:
        return ""
    rows = []
    for name, raw_details in stages.items():
        details = raw_details if isinstance(raw_details, dict) else {}
        metrics = details.get("metrics") if isinstance(details.get("metrics"), dict) else {}
        topology = metrics.get("topology") if isinstance(metrics.get("topology"), dict) else {}
        components = topology.get("components") if isinstance(topology.get("components"), dict) else {}
        rows.append({
            "stage": name,
            "status": details.get("status"),
            "role": details.get("role"),
            "source": details.get("source"),
            "triangles": metrics.get("triangles"),
            "boundary_edges": topology.get("boundary_edges"),
            "non_manifold_edges": topology.get("non_manifold_edges"),
            "non_manifold_vertices": topology.get("non_manifold_vertices"),
            "components": components.get("count"),
            "accepted_final_geometry": details.get("accepted_final_geometry"),
            "path": details.get("path"),
        })
    columns = [
        "stage",
        "status",
        "role",
        "source",
        "triangles",
        "boundary_edges",
        "non_manifold_edges",
        "non_manifold_vertices",
        "components",
        "accepted_final_geometry",
        "path",
    ]
    return section("Stages", row_table(rows, columns))


def trace_section(report: dict[str, Any]) -> str:
    repair = report.get("repair_report") or report.get("change_report") or {}
    rows = repair.get("geometry_to_mesh_trace")
    if not rows:
        return ""
    columns = [
        "stage",
        "operation",
        "status",
        "triangles",
        "removed_triangles",
        "points",
        "boundary_edges",
        "non_manifold_edges",
        "non_manifold_vertices",
        "components",
        "output",
    ]
    return section("Geometry To Mesh Trace", row_table(rows, columns))


def change_section(report: dict[str, Any]) -> str:
    repair = report.get("repair_report") or report.get("change_report") or {}
    change = repair.get("change_summary")
    if not change:
        return ""
    return section("Change Summary", object_table(change))


def defect_section(report: dict[str, Any]) -> str:
    repair = report.get("repair_report") or report.get("change_report") or {}
    defects = repair.get("defect_matrix")
    if not defects:
        return ""
    return section("Defect Matrix", object_table(defects))


def requested_section(report: dict[str, Any]) -> str:
    repair = report.get("repair_report", {})
    requested = repair.get("requested_capabilities")
    if not requested:
        return ""
    return section("Requested Capabilities", object_table(requested))


def outputs_section(report: dict[str, Any]) -> str:
    outputs = report.get("outputs")
    if not outputs:
        return ""
    return section("Outputs", object_table(outputs))


def print_artifact_section(report: dict[str, Any], report_dir: Path) -> str:
    outputs = report.get("outputs", {})
    rows = outputs.get("print_files", [])
    if not rows:
        return ""
    accepted = bool(outputs.get("accepted_mesh_vtp"))
    links = []
    for row in rows:
        path = resolve_path(row["path"], report_dir)
        if path and path.is_file():
            try:
                href = path.resolve().relative_to(report_dir.resolve()).as_posix()
            except ValueError:
                href = path.resolve().as_uri()
            links.append(f'<a download href="{text(href)}">{text(path.name)}</a>')
    role = "几何验收通过；设备切片状态见下方报告。" if accepted else "诊断候选，未通过打印几何验收。"
    return section("Print files / 打印文件", f'<p>{role}</p><p>' + ' · '.join(links) + '</p>')


def visual_section(report: dict[str, Any], report_dir: Path) -> str:
    items = visual_items(report)
    if not items:
        return ""
    cards = []
    for image, caption in items:
        cards.append(f'<figure><img src="{image_src(image, report_dir)}"><figcaption>{text(caption)}</figcaption></figure>')
    return section("Visual Evidence", '<div class="image-grid">' + "\n".join(cards) + "</div>")


def visual_items(report: dict[str, Any]) -> list[tuple[str, str]]:
    items = [(image, image) for image in report.get("outputs", {}).get("previews", [])]
    for view in report.get("view_reports", []):
        image = view.get("image")
        if image:
            caption = f"{view.get('view', 'view')} critical depth-gradient pixels: {view.get('critical_pixels')}"
            items.append((image, caption))
    return items


def viewer_section(preview_meshes: list[dict[str, Any]]) -> str:
    if not preview_meshes:
        return ""
    cards = []
    for index, mesh in enumerate(preview_meshes):
        if mesh.get("error"):
            cards.append(f'<div><h3>{text(mesh["label"])}</h3><p>3D model unavailable: {text(mesh["error"])}</p><p>{text(mesh["source"])}</p></div>')
            continue
        cards.append("\n".join([
            "<div>",
            f"<h3>{text(mesh['label'])}</h3>",
            f'<div class="vtk-viewer" id="vtk-viewer-{index}"></div>',
            '<div class="viewer-controls">',
            f'<button type="button" id="vtk-reset-{index}">Reset view / 重置视角</button>',
            f'<label><input type="checkbox" id="vtk-edges-{index}"> Mesh edges / 网格边</label>',
            f'<label><input type="checkbox" id="vtk-smooth-{index}" checked> 平滑光照（不移动顶点）</label>',
            f'<label><input type="checkbox" id="vtk-source-normals-{index}" checked> 使用文件法线（不移动顶点）</label>' if mesh.get("normals") else '',
            f'<label>Opacity / 不透明度 <input type="range" id="vtk-opacity-{index}" min="0.1" max="1" step="0.05" value="1"></label>',
            f'<label>剖切 <select id="vtk-cut-axis-{index}"><option value="">关闭</option><option value="0">X</option><option value="1">Y</option><option value="2">Z</option></select><input aria-label="剖切位置" type="range" id="vtk-cut-position-{index}" min="0" max="1" step="0.01" value="0.5"></label>',
            f'<label><input type="checkbox" id="vtk-overlay-{index}"> 叠加首个模型</label>' if index else '',
            f'<label>着色 <select id="vtk-field-{index}"><option value="">默认</option>' + ''.join(f'<option value="{text(key)}">{text(label)}</option>' for key, label in (("print_generated_face", "新增补面"), ("print_source_distance_mm", "源表面距离 mm"), ("print_thickness_mm", "采样壁厚 mm，灰色为未采样"), ("projection_fallback_fraction", "体素回退比例（红色=未投影）")) if key in mesh.get("fields", {})) + '</select></label>',
            f'<span id="vtk-field-range-{index}"></span>',
            opening_controls(mesh, index),
            '</div>',
            proxy_weight_legend(mesh),
            viewer_mesh_caption(mesh),
            "</div>",
        ]))
    instructions = '<p>拖动旋转 · 滚轮缩放 · Shift + 拖动平移。各阶段可独立检查；候选模型的展示不代表验收通过。</p>'
    sync = '<label><input type="checkbox" id="vtk-sync-cameras"> 同步各阶段视角</label>'
    return section("3D Mesh Evidence", instructions + sync + '<div class="viewer-grid">' + "\n".join(cards) + "</div>")


def opening_controls(mesh: dict[str, Any], index: int) -> str:
    regions = mesh.get("regions", [])
    if not regions:
        return ""
    options = ''.join(f'<option value="{i}">{text(row["id"])} · {text(str(row.get("geometry_classification")))}</option>' for i, row in enumerate(regions))
    return (
        f'<label>孔洞区域 <select id="vtk-region-{index}"><option value="">选择区域</option>{options}</select></label>'
        f'<label>下一轮决策 <select id="vtk-region-action-{index}"><option value="defer">待定</option><option value="cap">封闭</option><option value="preserve">保留开口（需壁厚）</option><option value="reject">拒绝</option></select></label>'
        f'<button type="button" id="vtk-policy-export-{index}">下载绑定当前模型的决策 JSON</button>'
        '<small>选区高亮相邻面；导出的决策需由几何 worker 执行并重新验收。</small>'
    )


def viewer_mesh_caption(mesh: dict[str, Any]) -> str:
    parts = [
        f"{mesh['triangles']} triangles",
        "full resolution" if mesh["full_resolution"] else f"downsampled from {mesh['original_triangles']}",
    ]
    if mesh.get("viewer_triangle_limit"):
        parts.append(f"viewer triangle limit: {mesh['viewer_triangle_limit']}")
    if mesh.get("opening_review_unavailable"):
        parts.append(text(mesh["opening_review_unavailable"]))
    if mesh.get("proxy_weights"):
        parts.append("source/proxy blend heatmap from proxy_weight cell data")
    if mesh.get("issue_values"):
        parts.append("red cells are adjacent to a recorded watertightness defect")
    parts.append(f"source: {text(mesh['source'])}")
    return f"<p class=\"muted\">{'; '.join(parts)}</p>"


def proxy_weight_legend(mesh: dict[str, Any]) -> str:
    if mesh.get("issue_values"):
        return "\n".join([
            '<div class="proxy-weight-legend" aria-label="watertight issue legend">',
            '<span>Watertight issue adjacency</span>',
            '<div class="proxy-weight-labels"><span style="color:#8c9eb3">gray clean face</span><span style="color:#f5421f">red issue-adjacent face</span></div>',
            "</div>",
        ])
    if not mesh.get("proxy_weights"):
        return ""
    return "\n".join([
        '<div class="proxy-weight-legend" aria-label="source proxy blend weight legend">',
        '<span>Source/proxy blend weight</span>',
        '<div class="proxy-weight-ramp"></div>',
        '<div class="proxy-weight-labels"><span>0 source</span><span>0.5 seam</span><span>1 proxy</span></div>',
        "</div>",
    ])


def watertight_artifact_section(report: dict[str, Any], report_dir: Path) -> str:
    artifacts = report.get("outputs", {}).get("watertight_issue_artifacts", [])
    if not artifacts:
        return ""
    rows = []
    for artifact in artifacts:
        raw_path = artifact.get("path")
        resolved = resolve_path(raw_path, report_dir)
        if resolved and resolved.exists():
            try:
                href = resolved.resolve().relative_to(report_dir.resolve()).as_posix()
            except ValueError:
                href = resolved.resolve().as_uri()
            path_html = f'<a href="{text(href)}">{text(resolved.name)}</a>'
        else:
            path_html = text(raw_path or "missing")
        rows.append(
            "<tr>"
            f"<td>{text(str(artifact.get('stage', '')))}</td>"
            f"<td>{text(str(artifact.get('kind', '')))}</td>"
            f"<td>{value_html(artifact.get('count'))}</td>"
            f"<td>{value_html(artifact.get('regions'))}</td>"
            f"<td>{text(str(artifact.get('representation', '')))}</td>"
            f"<td>{path_html}</td>"
            "</tr>"
        )
    table = (
        "<table><thead><tr><th>Stage</th><th>Issue/artifact</th><th>Count</th>"
        "<th>Regions</th><th>Representation</th><th>Full-resolution VTP</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )
    return section("Watertight Issue VTP Artifacts", table)


def full_json_section(report: dict[str, Any]) -> str:
    content = text(json.dumps(report, indent=2, ensure_ascii=False))
    return section("Full JSON", f"<details><summary>Open full machine-readable report</summary><pre>{content}</pre></details>")


def section(title: str, content: str) -> str:
    if not content:
        return ""
    return f"<section><h2>{text(title)}</h2>{content}</section>"


def kv_table(rows: list[tuple[str, Any]]) -> str:
    body = "\n".join(f"<tr><th>{text(key)}</th><td>{value_html(value)}</td></tr>" for key, value in rows)
    return f"<table>{body}</table>"


def row_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    head = "".join(f"<th>{text(column)}</th>" for column in columns)
    body_rows = []
    for row in rows:
        cells = "".join(f"<td>{value_html(row.get(column))}</td>" for column in columns)
        body_rows.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


def object_table(value: Any) -> str:
    if isinstance(value, dict):
        return kv_table(list(value.items()))
    if isinstance(value, list):
        rows = [(str(index), item) for index, item in enumerate(value)]
        return kv_table(rows)
    return f"<p>{value_html(value)}</p>"


def value_html(value: Any) -> str:
    if isinstance(value, bool):
        return f'<span class="bool-{str(value).lower()}">{str(value).lower()}</span>'
    if isinstance(value, (dict, list)):
        return f"<pre>{text(json.dumps(value, indent=2, ensure_ascii=False))}</pre>"
    if value is None:
        return '<span class="muted">null</span>'
    return text(str(value))


def relative_image(path: str) -> str:
    parts = Path(path).parts
    if "visual" in parts:
        index = parts.index("visual")
        return str(Path(*parts[index:]))
    return path


def image_src(raw_path: str, report_dir: Path) -> str:
    path = resolve_path(raw_path, report_dir)
    if not path or not path.exists():
        return text(relative_image(raw_path))
    mime_type = mimetypes.guess_type(path)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def text(value: str) -> str:
    return escape(value, quote=True)


def css() -> str:
    return """
body {
  color: #e8edf2;
  background:
    radial-gradient(circle at 8% 0%, rgba(43, 129, 140, 0.32), transparent 30%),
    linear-gradient(135deg, rgba(255,255,255,0.035) 25%, transparent 25%) 0 0 / 18px 18px,
    #101417;
  font-family: "Aptos", "Avenir Next", "Helvetica Neue", sans-serif;
  line-height: 1.45;
  margin: 0;
  padding: 30px;
}
h1, h2, h3 {
  color: #f8fafc;
  font-family: "DIN Alternate", "Bahnschrift", "Aptos Display", sans-serif;
  letter-spacing: 0;
}
h1 {
  font-size: clamp(34px, 5vw, 72px);
  line-height: 0.95;
  margin: 0;
  max-width: 920px;
}
h2 {
  color: #a7d8cf;
  font-size: 15px;
  font-weight: 700;
  margin: 0 0 14px;
  text-transform: uppercase;
}
h3 {
  font-size: 18px;
  margin: 0 0 8px;
}
.report-hero {
  align-items: end;
  border-bottom: 1px solid rgba(167, 216, 207, 0.28);
  display: flex;
  gap: 18px;
  justify-content: space-between;
  margin-bottom: 28px;
  padding-bottom: 20px;
}
.status {
  border: 1px solid rgba(167, 216, 207, 0.42);
  color: #a7d8cf;
  font-family: "SF Mono", "IBM Plex Mono", monospace;
  font-size: 12px;
  letter-spacing: 0.08em;
  padding: 7px 10px;
  text-transform: uppercase;
  white-space: nowrap;
}
.status-accepted { color: #95e6b8; }
.status-rejected { color: #ffb38a; }
section {
  margin: 26px 0;
}
table {
  border-collapse: collapse;
  width: 100%;
}
th, td {
  border: 1px solid #d8dee4;
  padding: 8px;
  text-align: left;
  vertical-align: top;
}
th { background: #f1f5f9; color: #1f2937; width: 220px; }
a { color: #7dd3fc; }
pre {
  background: #0f172a;
  border-radius: 6px;
  color: #e5e7eb;
  max-height: 460px;
  overflow: auto;
  padding: 12px;
  white-space: pre-wrap;
}
.bool-true { color: #067647; font-weight: 700; }
.bool-false { color: #b42318; font-weight: 700; }
.muted { color: #667085; }
.image-grid {
  display: grid;
  gap: 16px;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
}
.viewer-grid {
  display: grid;
  gap: 18px;
  grid-template-columns: repeat(2, minmax(0, 1fr));
}
.viewer-controls { display: flex; flex-wrap: wrap; gap: 12px; margin: 10px 0; align-items: center; }
.viewer-controls label { display: flex; align-items: center; gap: 6px; font-size: 13px; }
.viewer-controls button { padding: 6px 10px; cursor: pointer; }
.viewer-controls label { max-width: 100%; min-width: 0; }
.viewer-controls select { max-width: 350px; }
.viewer-grid p { overflow-wrap: anywhere; }
.vtk-viewer {
  background: #0b0f12;
  border: 1px solid rgba(167, 216, 207, 0.32);
  height: 420px;
  min-width: 0;
}
figure { margin: 0; }
img {
  background: #111827;
  border: 1px solid rgba(167, 216, 207, 0.22);
  max-width: 100%;
}
figcaption {
  color: #aab7bd;
  font-size: 12px;
  margin-top: 4px;
  word-break: break-all;
}
.viewer-grid > div {
  border-top: 2px solid rgba(255, 179, 138, 0.64);
  padding-top: 10px;
}
.viewer-grid p {
  color: #aab7bd;
  font-family: "SF Mono", "IBM Plex Mono", monospace;
  font-size: 12px;
}
.proxy-weight-legend {
  color: #dce7ec;
  font-family: "SF Mono", "IBM Plex Mono", monospace;
  font-size: 12px;
  margin: 8px 0;
}
.proxy-weight-ramp {
  background: linear-gradient(90deg, #3894d1 0%, #f2b32e 50%, #e13d29 100%);
  height: 10px;
  margin: 6px 0 4px;
}
.proxy-weight-labels {
  display: flex;
  justify-content: space-between;
}
@media (max-width: 720px) {
  body { padding: 18px; }
  .report-hero {
    align-items: flex-start;
    flex-direction: column;
  }
  .viewer-grid { grid-template-columns: minmax(0, 1fr); }
}
"""
