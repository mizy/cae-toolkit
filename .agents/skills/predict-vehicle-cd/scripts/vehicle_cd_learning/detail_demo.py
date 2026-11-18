"""Build a portable full-geometry and paired-CFD detail contribution report."""

import argparse
import base64
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from .geometry_diagnostics import pairwise_ranking
from .pair_evaluation import delta_metrics
from .pair_experiment import load


def build(experiment):
    """@entry Render audited real geometry and complete historical-query results."""
    protocol, dataset = load(experiment)
    out = experiment / "demo"
    comparison = json.loads((experiment / "comparison.json").read_text())
    audit = json.loads((out / "geometry-audit.json").read_text())
    radiant_path = experiment / "radiant-status.json"
    radiant = (
        json.loads(radiant_path.read_text())
        if radiant_path.exists()
        else {
            "status": "本轮尚未获得当前 Radiant 同一基准、相同预处理的可核验结果，未列入精度排名。"
        }
    )
    images = {
        identity: "data:image/png;base64,"
        + base64.b64encode(
            (out / audit["cases"][identity]["input_image"]).read_bytes()
        ).decode()
        for identity in protocol["display_query_ids"]
    }
    data = {
        "comparison": comparison,
        "geometryAudit": audit,
        "images": images,
        "radiant": radiant,
        "baselines": baseline_scores(dataset, comparison),
        "pairwise": pairwise_scores(comparison),
    }
    template = Path(__file__).resolve().parents[2] / "assets/detail-demo"
    sys.path.insert(
        0, str(Path(__file__).resolve().parents[3] / "mesh-watertight-repair/scripts")
    )
    from html_mesh_preview import vtk_js_source

    def script_json(value):
        return json.dumps(
            value, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).replace("<", "\\u003c")

    html = (template / "report.html").read_text()
    for key, value in {
        "__CSS__": (template / "report.css").read_text(),
        "__DATA__": script_json(data),
        "__GEOMETRY__": (out / "geometry.json").read_text().replace("<", "\\u003c"),
        "__VTK__": vtk_js_source().replace("</script", "<\\/script"),
        "__JS__": (template / "report.js").read_text(),
    }.items():
        if html.count(key) != 1:
            raise ValueError(f"Report template marker mismatch: {key}")
        html = html.replace(key, value)
    if "__DATA__" in html or "__GEOMETRY__" in html:
        raise ValueError("Unexpanded report data")
    (out / "report.html").write_text(html)
    (out / "evidence.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    write_csv(out / "queries.csv", comparison)
    files = [
        out / "report.html",
        out / "evidence.json",
        out / "queries.csv",
        out / "geometry-audit.json",
    ]
    manifest = {
        "files_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files
        },
        "portable": "All renderer, geometry, images and data embedded in report.html; no CDN or network inference.",
        "case_selection": protocol["anchor_selection"],
        "reproduction": "pair_experiment freeze/train A1/train A2; pair_evaluation; detail_geometry; detail_demo",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def baseline_scores(dataset, comparison):
    """Supplementary constant-output diagnostics; never used for model selection."""
    train = [
        c
        for c in dataset["cases"]
        if c["source"] == "drivaerrim" and c["split"] == "train"
    ]
    mean = np.asarray([c["targets"] for c in train]).mean(0)
    truth = np.asarray(
        [
            r["truth"]
            for r in comparison["models"]["A0"]["rows"]
            if r["source"] == "drivaerrim"
        ]
    )
    anchor = np.asarray(comparison["models"]["A0"]["anchors"][0]["truth"])
    return {
        "zero_change": delta_metrics(
            truth, np.broadcast_to(anchor, truth.shape), anchor, anchor
        ),
        "train_mean": delta_metrics(
            truth, np.broadcast_to(mean, truth.shape), anchor, anchor
        ),
        "train_mean_coefficients": mean.tolist(),
        "scope": "Post-training diagnostic baselines: unchanged known anchor, or always output train-only mean. Both ignore geometry; not model-selection candidates.",
    }


def pairwise_scores(comparison):
    """Anchor-independent diagnostic pairs share92 queries, not4186 independent trials."""
    result = {}
    for name in ("A0", "A1", "A2"):
        rows = [
            dict(r, split="test")
            for r in comparison["models"][name]["rows"]
            if r["source"] == "drivaerrim"
        ]
        truth, pred = (
            np.asarray([r[k] for r in rows]) for k in ("truth", "prediction")
        )
        a, b = np.triu_indices(len(rows), 1)
        result[name] = {
            "query_cases": len(rows),
            "dependent_pairs": len(a),
            "delta_mae": abs((pred[b] - pred[a]) - (truth[b] - truth[a]))
            .mean(0)
            .tolist(),
            "cd_ranking": pairwise_ranking(rows)["drivaerrim"],
        }
    return result


def write_csv(path, comparison):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "case_id",
                "model",
                "anchor_id",
                "true_cd",
                "predicted_cd",
                "true_delta_cd",
                "predicted_delta_cd",
                "anchored_cd",
                "true_cl",
                "predicted_cl",
                "true_delta_cl",
                "predicted_delta_cl",
                "anchored_cl",
            ]
        )
        for name in ("A0", "A1", "A2"):
            model = comparison["models"][name]
            base = model["anchors"][0]
            for row in model["rows"]:
                if row["source"] != "drivaerrim":
                    continue
                fields = [row["case_id"], name, base["case_id"]]
                for i in range(2):
                    delta = row["prediction"][i] - base["prediction"][i]
                    fields.extend(
                        [
                            row["truth"][i],
                            row["prediction"][i],
                            row["truth"][i] - base["truth"][i],
                            delta,
                            base["truth"][i] + delta,
                        ]
                    )
                writer.writerow(fields)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.experiment), indent=2))
