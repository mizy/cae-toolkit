"""Create one auditable 3D report for the width trial and mirror sensitivity probe."""

import argparse
import json
import os
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("probe", type=Path)
    p.add_argument("width_experiment", type=Path)
    args = p.parse_args()
    root = args.probe.resolve()
    geometry = json.loads((root / "geometry-report.json").read_text())
    sensitivity = json.loads((root / "sensitivity.json").read_text())
    comparison = json.loads((args.width_experiment / "comparison.json").read_text())
    summary = {}
    for size, model in sensitivity["models"].items():
        summary[size] = {}
        for mode in ("actual_dimensions", "image_only", "numeric_only"):
            subset = [
                r
                for r in model["predictions"]
                if r["variant"] == "mirrors_minus30" and r["mode"] == mode
            ]
            nominal = next(r for r in subset if r["offset"] == 0)
            summary[size][mode] = {
                "nominal": nominal,
                "delta_cd_render_range": [
                    min(r["delta_cd"] for r in subset),
                    max(r["delta_cd"] for r in subset),
                ],
                "delta_cl_render_range": [
                    min(r["delta_cl"] for r in subset),
                    max(r["delta_cl"] for r in subset),
                ],
            }
    plot_changes(root, sensitivity)
    primary = geometry["variants"][-1]
    os.environ["CAD_SURFACE_MESHER_VIEWER_TRIANGLES"] = "200000"
    sys.path.insert(
        0, str(Path(__file__).resolve().parents[3] / "mesh-watertight-repair/scripts")
    )
    from html_report import write_html_report

    report = {
        "decision": {
            "status": "diagnostic_only",
            "reason": "Neural response and single-seed width test; no paired CFD validation of modified car",
        },
        "input": {
            "path": geometry["source_mesh"],
            "kind": "mesh",
            "units": "m",
            "case": geometry["case"]["case_id"],
            "source_fingerprint": geometry["selection"]["source_sha256"],
        },
        "target": {
            "purpose": "coefficient_sensitivity",
            "name": "full-car mirror-protrusion edit",
        },
        "output_contract": {
            "kind": "mesh_to_mesh_design_variants",
            "full_mesh_triangles": 753250,
            "preview_only_triangle_limit": 200000,
            "training_input": "unchanged full five-view 256 depth/mask contract; no component crops",
        },
        "method": {
            "edit": "Both mirror outboard extensions shortened15% and30% via monotone Y mapping; x,z unchanged; root fixed",
            "conditions": geometry["conditions"],
            "reference_area": geometry["reference_area_policy"],
            "geometry_vs_numeric": "actual_dimensions = changed images+updated global width; image_only = changed images+original numeric geometry; numeric_only = original images+changed numeric width. Last two are input-path diagnostics, not physical contributions; effects need not add linearly.",
        },
        "geometry_to_mesh_trace": [
            "Read original held-out source STL without repair",
            "Identify both source mirror seed regions and save source face/point IDs",
            "Apply fixed-root lateral-only deformation to copies",
            "VTP serialized readback verifies identical connectivity and unchanged x/z",
            "Exact all-face self-intersection comparison and full-frame image audit",
            "Run frozen32/128 single-seed models with zero-change and ±0.1px camera controls",
        ],
        "change_summary": {
            "modified_feature": "Exterior mirror outboard span only",
            "maximum_inward_motion_each_side_m": primary["geometry_metrics"][
                "max_displacement_m"
            ],
            "moved_vertices": primary["geometry_metrics"]["moved_points"],
            "unmodified": "outside source region; wheels, roof, tail; original topology/connectivity",
            "selection": geometry["selection"],
        },
        "defect_matrix": [
            {
                "variant": v["label"],
                **v["geometry_metrics"],
                "exact_intersection_pairs": v["self_intersection"][
                    "intersection_pairs"
                ],
                "new_exact_intersection_pairs": v["introduced_intersection_pairs"],
                "reliable_enclosed_volume": v["enclosed_volume_reliable"],
            }
            for v in geometry["variants"]
        ],
        "diagnostics": {
            "input_changes_minus30": primary["input_changes"],
            "source_cache_replay": geometry["source_cache_replay"],
            "coefficients_and_robustness": summary,
            "baseline_CFD_labels_original_only": geometry["case"]["targets"],
            "modified_geometry_CFD_labels": None,
        },
        "limitations": [
            geometry["geometry_acceptance"],
            "A numerical response is not proof of aerodynamic accuracy or causal contribution.",
            "The five camera offsets measure deterministic rendering sensitivity, not a statistical confidence interval.",
            "Width comparison has one seed and reuses historical within-family geometry holdouts.",
            "3D display is decimated to200000 triangles per stage; full-resolution VTP, numerical inputs and geometry audits remain753250 triangles.",
        ],
        "outputs": {
            "mesh_previews": [
                {
                    "label": "原车：源模型，含3对既有自交",
                    "path": str(root / "original.vtp"),
                },
                {
                    "label": "外后视镜外伸缩短30%：设计副本，未做CFD验证",
                    "path": str(root / "mirrors_minus30.vtp"),
                },
            ],
            "previews": [
                str(root / "full-view-change.png"),
                str(root / "sensitivity.png"),
            ],
            "geometry_report": str(root / "geometry-report.json"),
            "sensitivity": str(root / "sensitivity.json"),
            "width_comparison": str(
                (args.width_experiment / "comparison.json").resolve()
            ),
        },
    }
    (root / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    path = root / "report.html"
    write_html_report(report, path, "车辆后视镜细节敏感性：32维与128维")
    rows = []
    for size in ("32", "128"):
        for mode, label in (
            ("actual_dimensions", "修改图像＋更新整体宽度"),
            ("image_only", "只改图像，数值输入固定"),
            ("numeric_only", "只改宽度数值，图像固定"),
        ):
            r = summary[size][mode]["nominal"]
            rows.append(
                f"<tr><td>{size}</td><td>{label}</td><td>{r['cd']:.7f}</td><td>{r['delta_cd']:+.7f}</td><td>{r['cl']:.7f}</td><td>{r['delta_cl']:+.7f}</td></tr>"
            )
    score_rows = []
    for size in ("32", "128"):
        score = comparison["runs"][size]["scores"]
        for domain in ("vehicle", "aircraft"):
            m = score[f"test_source_macro/{domain}"]
            score_rows.append(
                f"<tr><td>{size}</td><td>{domain}</td><td>{m['cd_mae']:.6f}</td><td>{m['cl_mae']:.6f}</td><td>{m['signed_cl_over_cd_mae']:.6f}</td></tr>"
            )
    intro = (
        "<section class='section'><h2>单种子宽度对照：来源等权测试MAE</h2><table><tr><th>每视图维数</th><th>领域</th><th>Cd MAE</th><th>Cl MAE</th><th>Cl/Cd MAE</th></tr>"
        + "".join(score_rows)
        + "</table><h2>后视镜外伸缩短30%的预测响应</h2><p>以下是模型预测变化，修改版没有配对CFD真值。系数变化不可直接解释为真实减阻收益。原车基准、相机、参考面积和实际工况保持固定。</p><table><tr><th>维数</th><th>输入路径</th><th>Cd预测</th><th>ΔCd</th><th>Cl预测</th><th>ΔCl</th></tr>"
        + "".join(rows)
        + "</table></section>"
    )
    html = path.read_text()
    # The generic mesh viewer uses Y-up; these source vehicles explicitly use Z-up.
    if "camera.setViewUp(0, 1, 0);" not in html:
        raise ValueError("Generic viewer camera contract changed")
    html = html.replace(
        "camera.setPosition(1, 0.7, 1);", "camera.setPosition(-1, 1, 0.65);"
    )
    html = html.replace("camera.setViewUp(0, 1, 0);", "camera.setViewUp(0, 0, 1);")
    html = html.replace(
        '<meta charset="utf-8">',
        '<meta charset="utf-8"><link rel="icon" href="data:,">',
        1,
    )
    path.write_text(html.replace("</header>", "</header>" + intro, 1))
    print(
        json.dumps(
            {"report": str(path), "primary_mirror_sensitivity": summary}, indent=2
        )
    )


def plot_changes(root, sensitivity):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    labels = ["original", "mirrors_minus15", "mirrors_minus30"]
    for row, metric in enumerate(("delta_cd", "delta_cl")):
        for col, mode in enumerate(("actual_dimensions", "image_only")):
            ax = axes[row, col]
            for size, color in (("32", "#64748b"), ("128", "#2563eb")):
                data = sensitivity["models"][size]["predictions"]
                values = [
                    [
                        x[metric]
                        for x in data
                        if x["variant"] == label and x["mode"] == mode
                    ]
                    for label in labels
                ]
                nominal = [
                    next(
                        x[metric]
                        for x in data
                        if x["variant"] == label
                        and x["mode"] == mode
                        and x["offset"] == 0
                    )
                    for label in labels
                ]
                ax.plot(
                    [0, 15, 30],
                    nominal,
                    "o-",
                    color=color,
                    label=f"{size} features/view",
                )
                ax.fill_between(
                    [0, 15, 30],
                    [min(v) for v in values],
                    [max(v) for v in values],
                    color=color,
                    alpha=0.15,
                )
            ax.axhline(0, color="black", lw=0.7)
            ax.set(
                title=f"{metric}: {mode}",
                xlabel="Mirror outboard shortening (%)",
                ylabel=metric,
            )
            ax.grid(alpha=0.2)
            ax.legend()
    fig.suptitle(
        "Predicted mirror sensitivity, not validated aero gain\nBands: deterministic ±0.1px camera-offset range"
    )
    fig.savefig(root / "sensitivity.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
