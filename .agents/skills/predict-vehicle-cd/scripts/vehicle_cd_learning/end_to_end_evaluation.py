"""Absolute Cd/Cl and same-query Radiant Cd evaluation after all paired runs finish."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from aircraft_learning.scene_experiment import SceneCases

from . import dimensionless as dim
from .end_to_end_experiment import load
from .head_experiment import physical
from .kernel_experiment import bootstrap
from .pair_experiment import predictions
from .pooled_residual import load_model
from .readout_experiment import absolute_scores


def radiant_scores(cases, prediction, external):
    """Use the same training-reference CFD Cd for both methods; no query-label input."""
    by_id = {c["case_id"]: i for i, c in enumerate(cases)}
    expected = {
        c["case_id"]
        for c in cases
        if c["source"] == "drivaerrim" and c["split"] == "test"
    }
    ids = sorted(expected)
    ours, theirs, by_anchor = [], [], {}
    for anchor, rows in external["rows_by_anchor"].items():
        if {r["case_id"] for r in rows} != expected:
            raise ValueError("Radiant geometry membership differs")
        reference = cases[by_id[anchor]]
        if (
            reference["split"] != "train"
            or reference["targets"][0] != external["anchors"][anchor]
        ):
            raise ValueError("Reference coefficient or split differs")
        mapping = {r["case_id"]: r for r in rows}
        indices = [by_id[k] for k in ids]
        truth = np.array([cases[i]["targets"][0] for i in indices])
        for k, actual in zip(ids, truth):
            if mapping[k]["truth"] != actual:
                raise ValueError("Radiant labels differ")
            if (
                abs(
                    mapping[k]["prediction"]
                    - reference["targets"][0]
                    - mapping[k]["delta_prediction"]
                )
                > 1e-12
            ):
                raise ValueError(
                    "Radiant absolute prediction disagrees with its reference"
                )
        predicted = (
            reference["targets"][0]
            + prediction[indices, 0]
            - prediction[by_id[anchor], 0]
        )
        external_pred = np.array([mapping[k]["prediction"] for k in ids])
        a, b = abs(predicted - truth), abs(external_pred - truth)
        ours.append(a)
        theirs.append(b)
        by_anchor[anchor] = {
            "our_cd_mae": float(a.mean()),
            "radiant_cd_mae": float(b.mean()),
            "rows": [
                {"case_id": k, "truth": float(y), "ours": float(u), "radiant": float(v)}
                for k, y, u, v in zip(ids, truth, predicted, external_pred)
            ],
        }
    diff = np.mean(ours, axis=0) - np.mean(theirs, axis=0)
    rng = np.random.default_rng(20261116)
    samples = diff[rng.integers(0, len(ids), (5000, len(ids)))].mean(1)
    return {
        "query_count": len(ids),
        "reference_count": len(ours),
        "our_cd_mae_mean_references": float(np.mean(ours)),
        "radiant_cd_mae_mean_references": float(np.mean(theirs)),
        "our_minus_radiant_mae_95_interval": np.quantile(
            samples, [0.025, 0.975]
        ).tolist(),
        "by_anchor": by_anchor,
        "scope": "Absolute anchored Cd, same92 queries and three known training references. Bootstrap groups references by query. Radiant exposure/live deployment and Cl unverified; no architecture or unseen-family ranking.",
    }


def evaluate(experiment, device_name="mps"):
    """@entry Choose by validation first, then report every seed and external comparison."""
    p, data = load(experiment)
    torch.set_num_threads(2)
    paths, proof = {"baseline": Path(p["checkpoint"])}, {}
    for seed in p["seeds"]:
        draws = np.load(experiment / f"draws-{seed}.npy", allow_pickle=False)
        counts = np.bincount(draws.ravel(), minlength=draws.shape[1]).tolist()
        trace = hashlib.sha256()
        for row in draws:
            for start in range(0, len(row), p["batch_size"]):
                trace.update(row[start : start + p["batch_size"]].tobytes())
        for arm in p["arms"]:
            name = f"{arm}-{seed}"
            report = json.loads(
                (experiment / name / "training_report.json").read_text()
            )
            expected_steps = p["epochs"] * (
                (draws.shape[1] + p["batch_size"] - 1) // p["batch_size"]
            )
            if (
                len(report["history"]) != p["epochs"]
                or report["optimizer_steps"] != expected_steps
                or report["exposures"] != counts
                or report["draw_trace_sha256"] != trace.hexdigest()
            ):
                raise ValueError("Incomplete or unmatched paired training")
            paths[name] = experiment / name / "model.pt"
            proof[name] = {
                "epochs": p["epochs"],
                "optimizer_steps": expected_steps,
                "exposures": sum(counts),
                "minimum_exposures": min(counts),
            }
    device = torch.device(device_name)
    values = SceneCases(Path(p["prepared"]), data["cases"], p["normalization"])
    outputs, selections = {}, {}
    original = torch.load(p["checkpoint"], map_location="cpu", weights_only=True)
    for name, path in paths.items():
        model, ck = dim.load_model(path) if name == "baseline" else load_model(path)
        if name != "baseline":
            changed = [
                k
                for k, v in original["model_state"].items()
                if not torch.equal(v, ck["model_state"][k])
            ]
            proof[name]["selected_epoch"] = ck["epoch"]
            proof[name]["changed_cnn_tensors"] = sum(
                k.startswith(("image_encoder.blocks", "bottom_encoder.blocks"))
                for k in changed
            )
        model.to(device).eval()
        outputs[name] = physical(predictions(model, values, device), p["normalization"])
        selections[name] = {
            "epoch": ck.get("epoch"),
            "checkpoint": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        del model
        if device.type == "mps":
            torch.mps.empty_cache()
        print(json.dumps({"predicted": name, "cases": len(data["cases"])}), flush=True)
    for arm in p["arms"]:
        outputs[arm + "-ensemble"] = np.mean(
            [outputs[f"{arm}-{seed}"] for seed in p["seeds"]], axis=0
        )
    val = np.array([c["split"] == "val" for c in data["cases"]])
    val_cases = [c for c in data["cases"] if c["split"] == "val"]
    choices = {
        n: absolute_scores(val_cases, outputs[n][val])
        for n in ["baseline", *[a + "-ensemble" for a in p["arms"]]]
    }
    limits = p["baseline_validation"]["macro"]
    eligible = [
        n
        for n, scores in choices.items()
        if n == "baseline" or all(scores["macro"][k] <= limits[k] for k in limits)
    ]
    winner = min(eligible, key=lambda n: sum(choices[n]["macro"].values()))
    selection = {
        "selected": winner,
        "validation": choices,
        "checkpoints": selections,
        "test_used": False,
        "external_used": False,
    }
    (experiment / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    if winner != "baseline":
        arm = winner.removesuffix("-ensemble")
        (experiment / "ensemble.json").write_text(
            json.dumps(
                {
                    "checkpoints": [
                        str(paths[f"{arm}-{seed}"].resolve()) for seed in p["seeds"]
                    ],
                    "aggregation": "equal mean",
                    "selected_by": "validation absolute Cd/Cl MAE",
                },
                indent=2,
            )
            + "\n"
        )
    external = json.loads(Path(p["external"]).read_text())
    models = {}
    for name, pred in outputs.items():
        scores = {}
        for split in ("train", "val", "test"):
            mask = np.array([c["split"] == split for c in data["cases"]])
            scores[split] = absolute_scores(
                [c for c in data["cases"] if c["split"] == split], pred[mask]
            )
        models[name] = {
            "scores": scores,
            "radiant": radiant_scores(data["cases"], pred, external),
            "rows": [
                {
                    "case_id": c["case_id"],
                    "source": c["source"],
                    "split": c["split"],
                    "truth": c["targets"],
                    "prediction": v.tolist(),
                }
                for c, v in zip(data["cases"], pred)
            ],
        }
    test = np.array([c["split"] == "test" for c in data["cases"]])
    uncertainty = bootstrap(
        [c for c in data["cases"] if c["split"] == "test"],
        outputs["baseline"][test],
        outputs[winner][test],
    )
    summary = {}
    for arm in p["arms"]:
        scores = np.array(
            [
                list(models[f"{arm}-{s}"]["scores"]["test"]["macro"].values())
                for s in p["seeds"]
            ]
        )
        summary[arm] = {
            "mean_cd_cl_mae": scores.mean(0).tolist(),
            "std_cd_cl_mae": scores.std(0, ddof=1).tolist(),
        }
    result = {
        "selection": selection,
        "proof": proof,
        "models": models,
        "three_seed_test_summary": summary,
        "selected_vs_baseline": uncertainty,
    }
    (experiment / "comparison.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "selected": winner,
                "three_seed_test": summary,
                "selected_test": models[winner]["scores"]["test"]["macro"],
                "selected_radiant_cd": {
                    k: v
                    for k, v in models[winner]["radiant"].items()
                    if k != "by_anchor"
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("experiment", type=Path)
    p.add_argument("--device", default="mps")
    a = p.parse_args()
    evaluate(a.experiment, a.device)
