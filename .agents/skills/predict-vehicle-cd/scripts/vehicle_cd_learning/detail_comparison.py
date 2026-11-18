"""Summarize information probes and independently exported external Cd predictions."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def model_scores(model, anchor_id):
    rows = [r for r in model["rows"] if r["source"] == "drivaerrim"]
    truth, pred = (np.asarray([r[k][0] for r in rows]) for k in ("truth", "prediction"))
    a, b = np.triu_indices(len(rows), 1)
    actual, estimated = truth[b] - truth[a], pred[b] - pred[a]
    valid = abs(actual) > 1e-12
    return {
        "anchor": model["anchor_scores"][anchor_id],
        "pair_mae": float(abs(estimated - actual).mean()),
        "ranking": float(np.mean(np.sign(actual[valid]) == np.sign(estimated[valid]))),
    }


def collect(diagnosis, paired, external):
    own = json.loads((diagnosis / "comparison.json").read_text())
    comparison = json.loads((paired / "comparison.json").read_text())
    imported = json.loads(external.read_text())
    anchor = comparison["protocol"]["anchor_id"]
    truth = {
        r["case_id"]: r["truth"][0]
        for r in comparison["models"]["A0"]["rows"]
        if r["source"] == "drivaerrim"
    }
    for rows in imported["rows_by_anchor"].values():
        if {r["case_id"] for r in rows} != set(truth):
            raise ValueError(
                "External query geometries do not match our92-case evaluation"
            )
        for row in rows:
            if row["truth"] != truth[row["case_id"]]:
                raise ValueError("External coefficient labels do not match")
    for base, rows in imported["rows_by_anchor"].items():
        expected = float(
            np.mean([abs(r["delta_prediction"] - r["delta_truth"]) for r in rows])
        )
        if not np.isclose(
            expected, imported["metrics_by_anchor"][base]["delta_cd_mae"], atol=1e-12
        ):
            raise ValueError("External aggregate error differs from its query rows")
    if imported["primary_anchor"] != anchor or imported["primary_anchor_cd"] != 0.24:
        raise ValueError("External reference differs")
    models = {
        name: model_scores(comparison["models"][name], anchor)
        for name in ("A0", "A1", "A2")
    }
    models["Radiant_SDF"] = {
        "anchor": imported["metrics_by_anchor"][anchor],
        "pair_mae": imported["direct_test_pair_metrics"][
            "both_directions_delta_cd_mae"
        ],
        "ranking": imported["direct_test_pair_metrics"]["ranking_accuracy"],
    }
    values = np.asarray(list(truth.values()))
    i, j = np.triu_indices(len(values), 1)
    baseline = {
        "zero_anchor_delta_mae": float(abs(values - 0.24).mean()),
        "zero_pair_delta_mae": float(abs(values[j] - values[i]).mean()),
    }
    return {
        "diagnosis": own,
        "paired": comparison,
        "external": imported,
        "models": models,
        "baselines": baseline,
        "source_sha256": {
            str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                diagnosis / "comparison.json",
                paired / "comparison.json",
                external,
            )
        },
        "scope": "Same public92-query, known training-reference Cd task. External preprocessing is model-level full-source SDF; training exposure and live deployment unverified. Not a same-training-data architecture benchmark.",
    }


def plot(result, output):
    probes, native = result["diagnosis"]["probes"], result["diagnosis"]["native"]
    stages = ["raw", "A2-pooled", "A2-projected", "A2-fused"]
    labels = [
        "Raw views",
        "Pooled 5120",
        "Projected 160",
        "Fused 168",
        "Original A2 head",
    ]
    train = [probes[s]["scores"]["train"]["absolute_mae"][0] for s in stages] + [
        native["A2"]["train"]["absolute_mae"][0]
    ]
    test = [probes[s]["scores"]["test"]["absolute_mae"][0] for s in stages] + [
        native["A2"]["test"]["absolute_mae"][0]
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    y = np.arange(len(labels))
    axes[0].barh(y - 0.18, train, height=0.34, label="Train", color="#9cbce6")
    axes[0].barh(y + 0.18, test, height=0.34, label="Historical test", color="#2867ac")
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Absolute Cd MAE")
    axes[0].set_title("Frozen representation / local information probes")
    axes[0].legend(frameon=False)
    names = list(result["models"])
    anchor = [
        result["models"][n]["anchor"].get(
            "delta_mae", [result["models"][n]["anchor"].get("delta_cd_mae")]
        )[0]
        for n in names
    ]
    pair = [result["models"][n]["pair_mae"] for n in names]
    x = np.arange(len(names))
    axes[1].bar(
        x - 0.18, anchor, width=0.34, label="Fixed anchor Rim0001", color="#376bac"
    )
    axes[1].bar(x + 0.18, pair, width=0.34, label="All query pairs", color="#e59b41")
    axes[1].set_xticks(x, ["A0", "A1", "A2", "Radiant SDF"])
    axes[1].set_ylabel("Delta Cd MAE")
    axes[1].set_title("Same query geometry / coefficient / reference")
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x" if ax is axes[0] else "y", alpha=0.18)
        ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(output / "comparison.png", dpi=170)
    plt.close(fig)


def write(result, output):
    output.mkdir(exist_ok=True)
    (output / "comparison-summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    plot(result, output)
    rows = []
    for name, score in result["models"].items():
        anchor = score["anchor"]
        mae = (
            anchor["delta_mae"][0] if "delta_mae" in anchor else anchor["delta_cd_mae"]
        )
        rows.append(
            f"| {name} | {mae:.6f} | {score['pair_mae']:.6f} | {100 * score['ranking']:.1f}% |"
        )
    stages = []
    for name, values in result["diagnosis"]["probes"].items():
        train, test = values["scores"]["train"], values["scores"]["test"]
        stages.append(
            f"| {name} | {train['absolute_mae'][0]:.6f} | {test['absolute_mae'][0]:.6f} | {test['pair_delta_mae'][0]:.6f} |"
        )
    content = "# 细节预测误差诊断与外部模型对照\n\n" + result["scope"] + "\n\n"
    content += (
        "固定主基准Rim0001，92个历史查询；4186个两两查询对相互依赖。\n\n| 方法 | 主基准ΔCd MAE | 两两ΔCd MAE | 非零差值排序正确率 |\n|---|---:|---:|---:|\n"
        + "\n".join(rows)
    )
    content += (
        "\n\n冻结特征探针，使用相同训练侧归一化与固定验证选择网格；源特定探针与产品模型训练预算不同。\n\n| 表示阶段 | train Cd MAE | historical test Cd MAE | test两两ΔCd MAE |\n|---|---:|---:|---:|\n"
        + "\n".join(stages)
    )
    content += f"\n\n零变化基线：主基准ΔCd MAE {result['baselines']['zero_anchor_delta_mae']:.6f}，两两ΔCd MAE {result['baselines']['zero_pair_delta_mae']:.6f}。模型差异仅为这批历史查询的描述统计，不代表同训练数据架构优劣。\n"
    content += "\n\n![comparison](comparison.png)\n\n完整数据及适用范围见comparison-summary.json。没有复制外部模型代码、权重或隐藏参考池；仅消费导出的公开几何系数预测。原权重和默认模型保留。\n"
    (output / "README.md").write_text(content)
    with (output / "external-queries.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "case_id",
                "truth",
                "prediction",
                "delta_prediction",
                "delta_truth",
            ],
        )
        writer.writeheader()
        writer.writerows(
            result["external"]["rows_by_anchor"][result["external"]["primary_anchor"]]
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("diagnosis", type=Path)
    parser.add_argument("paired", type=Path)
    parser.add_argument("external", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    write(collect(args.diagnosis, args.paired, args.external), args.output)
