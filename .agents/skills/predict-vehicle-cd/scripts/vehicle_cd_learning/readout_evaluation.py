"""Verify and score readout arms using absolute Cd and Cl errors only."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .readout import initialize, load_model
from .readout_experiment import PooledCases, absolute_scores, load, predict


def verify(experiment, p, data):
    draws = np.load(experiment / "draws.npy", allow_pickle=False)
    trace = hashlib.sha256()
    for row in draws:
        for start in range(0, len(row), p["batch_size"]):
            trace.update(row[start : start + p["batch_size"]].tobytes())
    counts = np.bincount(draws.ravel(), minlength=draws.shape[1])
    steps = p["epochs"] * ((len(draws[0]) + p["batch_size"] - 1) // p["batch_size"])
    original = torch.load(p["checkpoint"], map_location="cpu", weights_only=True)
    for arm in p["runs"]:
        report = json.loads((experiment / arm / "training_report.json").read_text())
        if len(report["history"]) != p["epochs"] or report["optimizer_steps"] != steps:
            raise ValueError("Complete all matched arms before test evaluation")
        if (
            report["draw_trace_sha256"] != trace.hexdigest()
            or list(report["exposures"].values()) != counts.tolist()
        ):
            raise ValueError("Training draws/exposures differ")
        model, ck = load_model(experiment / arm / "model.pt")
        for name, value in original["model_state"].items():
            if (
                name.startswith(("image_encoder.", "bottom_encoder."))
                and ".project." not in name
                and not torch.equal(model.state_dict()[name], value)
            ):
                raise ValueError("A frozen visual parameter changed")
        eligible = [h for h in report["history"] if h["eligible"]]
        best = (
            min(eligible, key=lambda h: sum(h["validation"]["macro"].values()))
            if eligible
            else None
        )
        initial = sum(p["initial_validation"]["macro"].values())
        selected = (
            best["epoch"]
            if best and sum(best["validation"]["macro"].values()) < initial
            else 0
        )
        if ck["epoch"] != selected:
            raise ValueError(
                "Checkpoint does not follow absolute-error validation selector"
            )
    return {
        "epochs_per_arm": p["epochs"],
        "optimizer_steps_per_arm": steps,
        "same_draws_and_exposures": True,
        "minimum_geometry_exposures": int(counts.min()),
        "frozen_visual_parameters_unchanged": True,
        "selection_uses_absolute_cd_cl_mae_only": True,
        "missing_cl_mask_preserved": True,
    }


def evaluate(experiment):
    """@entry Test all arms after training, retaining validation-only selection."""
    p, data, pooled = load(experiment)
    proof = verify(experiment, p, data)
    torch.set_num_threads(2)
    values = PooledCases(
        pooled, data["cases"], p["normalization"], range(len(data["cases"]))
    )
    original = torch.load(p["checkpoint"], map_location="cpu", weights_only=True)
    models = {"initial": initialize(original)}
    checkpoints = {}
    for arm in p["runs"]:
        models[arm], checkpoints[arm] = load_model(experiment / arm / "model.pt")
    result = {}
    for name, model in models.items():
        y = predict(model, values, p["normalization"])
        scores = {}
        for split in ("train", "val", "test"):
            mask = np.array([c["split"] == split for c in data["cases"]])
            scores[split] = absolute_scores(
                [c for c in data["cases"] if c["split"] == split], y[mask]
            )
        result[name] = {
            "epoch": checkpoints[name]["epoch"]
            if name in checkpoints
            else original["epoch"],
            "scores": scores,
            "rows": [
                {
                    "case_id": c["case_id"],
                    "split": c["split"],
                    "source": c["source"],
                    "truth": c["targets"],
                    "prediction": v.tolist(),
                }
                for c, v in zip(data["cases"], y)
            ],
        }
    winner = min(
        p["runs"], key=lambda n: sum(result[n]["scores"]["val"]["macro"].values())
    )
    comparison = {
        "protocol": p,
        "proof": proof,
        "models": result,
        "validation_selected": winner,
        "selection": "Absolute Cd MAE+Cl MAE only; test used once after all4 runs complete",
        "scope": "Single-seed cached-feature continuation, historical geometry-family splits, not a new independent vehicle test",
    }
    (experiment / "comparison.json").write_text(
        json.dumps(comparison, indent=2, allow_nan=False) + "\n"
    )
    lines = [
        "# 绝对Cd/Cl读出优化\n",
        "固定完整五视图CNN，只训练读出侧；评估仅报告原始系数单位的MAE。\n",
        "| 组 | 验证Cd MAE | 验证Cl MAE | 历史test Cd MAE | 历史test Cl MAE |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, item in result.items():
        val, test = item["scores"]["val"]["macro"], item["scores"]["test"]["macro"]
        lines.append(
            f"| {name} | {val['cd_mae']:.6f} | {val['cl_mae']:.6f} | {test['cd_mae']:.6f} | {test['cl_mae']:.6f} |"
        )
    lines.extend(
        [
            f"\n仅按验证集选出：{winner}。",
            "\n主指标按CFD来源等权；完整逐来源/逐案例数值保存在comparison.json。未知参考车Cl不参与损失或指标。原权重保留，本次尚未自动替换默认模型。",
        ]
    )
    (experiment / "README.md").write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            {
                "selected": winner,
                "scores": {
                    n: {s: d["scores"][s]["macro"] for s in ("val", "test")}
                    for n, d in result.items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    evaluate(parser.parse_args().experiment)
