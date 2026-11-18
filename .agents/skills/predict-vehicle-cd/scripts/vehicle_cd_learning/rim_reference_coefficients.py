"""Audit absolute Cd/Cl and their differences using the frozen paired predictor and labeled Rim references."""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from . import sampling_experiment
from .cl_sampling_repair import encode_requests
from .delta_experiment import sha256, write_json
from .delta_metrics import score_delta
from .design_change_audit import pair_outputs
from .joint_sampling import load, load_checkpoint


def coefficient_metrics(truth, predictions):
    a, b = np.asarray(list(itertools.combinations(range(len(truth)), 2))).T
    return {
        "nominal_mae": np.abs(predictions - truth).mean(0).tolist(),
        "signed_bias": (predictions - truth).mean(0).tolist(),
        "implied_delta": score_delta(
            truth[b] - truth[a], predictions[b] - predictions[a]
        ),
    }


def estimate(changes, references, offset):
    """Return each query's physical coefficient using the same fixed train references."""
    return (np.asarray(references)[:, None, :] + np.asarray(changes)).mean(0) + offset


def run(experiment, output):
    p, _ = load(experiment)
    sp, cases = sampling_experiment.load(p["sampling_parent"])
    r = json.loads((experiment / "retention.json").read_text())
    if sha256(r["checkpoint"]) != r["sha256"]:
        raise ValueError("Retained checkpoint changed")
    refs = sp["anchors"]
    groups = {
        s: sorted(
            c["case_id"] for c in cases if c["split"] == s and c["case_id"] not in refs
        )
        for s in ("train", "val", "test")
    }
    if [len(groups[s]) for s in groups] != [717, 92, 92]:
        raise ValueError("Unexpected Rim split")
    if len({tuple(c["parameters"]) for c in cases}) != 1:
        raise ValueError("Rim physical conditions differ")
    output.mkdir(parents=True, exist_ok=False)
    write_json(
        output / "plan.json",
        {
            "checkpoint_sha256": r["sha256"],
            "code_sha256": sha256(Path(__file__)),
            "references": refs,
            "fit_ids": groups["train"],
            "input": "full 512 nominal Rim views",
            "estimator": "mean of three measured anchor coefficients plus frozen paired differences, then a per-coefficient median residual fitted on 717 other train cases",
            "controls": [
                "zero delta with the same references",
                "training median of 717",
                "anchored without learned offset",
            ],
            "selection": "Fixed comparison. Validation and historical test used only for reporting; no scale fitting or model selection.",
        },
    )
    torch.set_num_threads(2)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model, _ = load_checkpoint(r["checkpoint"], device)
    model.eval()
    data = sampling_experiment.SamplingCases(sp, cases, with_phases=False)

    def targets(ids):
        return np.asarray([cases[data.index[i]]["targets"] for i in ids], float)

    anchor_targets = targets(refs)
    median = np.median(targets(groups["train"]), axis=0)
    with torch.inference_mode():
        anchors = encode_requests(
            model, data, [(data.index[i], None) for i in refs], device
        )
        numeric = data.parameters[data.index[refs[0]]].to(device)
        results = {}
        for split, ids in groups.items():
            actual = targets(ids)
            bank = encode_requests(
                model, data, [(data.index[i], None) for i in ids], device
            )
            bank.update(anchors)
            pairs = [
                ((data.index[a], None), (data.index[b], None))
                for a, b in itertools.product(refs, ids)
            ]
            changes = pair_outputs(
                model, bank, pairs, numeric, sp["delta_std"], device
            ).reshape(len(refs), -1, 2)
            uncalibrated = estimate(changes, anchor_targets, 0.0)
            if split == "train":
                offset = np.median(actual - uncalibrated, axis=0)
                write_json(
                    output / "train-calibration.json",
                    {
                        "offset_cd_cl": offset.tolist(),
                        "fit_records": len(ids),
                        "anchors": refs,
                        "train_median": median.tolist(),
                        "source": "drivaerrim",
                        "reference_targets": anchor_targets.tolist(),
                    },
                )
            pred = estimate(changes, anchor_targets, offset)
            np.savez_compressed(
                output / f"{split}.npz",
                case_ids=np.array(ids),
                truth=actual,
                prediction=pred,
                uncalibrated=uncalibrated,
                direct_changes=changes,
            )
            if split != "train":
                results[split] = {
                    "anchor_constant_mae": np.abs(actual - anchor_targets.mean(0))
                    .mean(0)
                    .tolist(),
                    "train_median_mae": np.abs(actual - median).mean(0).tolist(),
                    "uncalibrated": coefficient_metrics(actual, uncalibrated),
                    "calibrated": coefficient_metrics(actual, pred),
                }
            print(split, len(ids), np.abs(actual - pred).mean(0), flush=True)
    write_json(output / "results.json", results)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("experiment", type=Path)
    cli.add_argument("output", type=Path)
    a = cli.parse_args()
    run(a.experiment, a.output)
