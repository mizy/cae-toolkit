"""Four-arm ΔCd/ΔCl evaluation on common geometries, references and pair metrics."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .delta_experiment import batch, dataset_for, load, sha256, write_json
from .delta_metrics import (
    compare_radiant,
    score_anchors,
    score_delta,
    zero_change_controls,
)
from .paired_delta import PairedDelta


def encode_all(model, dataset, device):
    collected = {}
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(dataset), 8):
            views, _, _ = batch(
                dataset, list(range(start, min(start + 8, len(dataset)))), device
            )
            for name, value in model.encode(views).items():
                collected.setdefault(name, []).append(value.cpu())
    return {name: torch.cat(values) for name, values in collected.items()}


def predict_pairs(model, encoded, parameters, left, right, scale, device):
    output = []
    with torch.inference_mode():
        for start in range(0, len(left), 64):
            a = torch.as_tensor(left[start : start + 64], dtype=torch.long)
            b = torch.as_tensor(right[start : start + 64], dtype=torch.long)
            first = {n: v[a].to(device) for n, v in encoded.items()}
            second = {n: v[b].to(device) for n, v in encoded.items()}
            output.append(
                model.predict_pair(first, second, parameters[a].to(device))
                .cpu()
                .numpy()
            )
    return np.concatenate(output) * np.asarray(scale)


def anchor_predictions(model, encoded, dataset, cases, anchors, split, scale, device):
    queries = [i for i, c in enumerate(cases) if c["split"] == split]
    result = {}
    for anchor in anchors:
        reference = dataset.index[anchor]
        pred = predict_pairs(
            model,
            encoded,
            dataset.parameters,
            np.full(len(queries), reference),
            np.array(queries),
            scale,
            device,
        )
        result[anchor] = {
            cases[i]["case_id"]: p.tolist() for i, p in zip(queries, pred)
        }
    return result


def control_scores(model, encoded, dataset, cases, anchors, predictions, scale, device):
    queries = np.array([i for i, c in enumerate(cases) if c["split"] == "test"])
    self_pred = predict_pairs(
        model, encoded, dataset.parameters, queries, queries, scale, device
    )
    reverse = []
    for anchor in anchors:
        a = dataset.index[anchor]
        backward = predict_pairs(
            model,
            encoded,
            dataset.parameters,
            queries,
            np.full(len(queries), a),
            scale,
            device,
        )
        forward = np.array([predictions[anchor][cases[i]["case_id"]] for i in queries])
        reverse.extend(backward + forward)
    a, b = [dataset.index[x] for x in anchors[:2]]
    ab = predict_pairs(
        model, encoded, dataset.parameters, np.array([a]), np.array([b]), scale, device
    )[0]
    cycle = np.array(
        [
            np.array(predictions[anchors[0]][cases[i]["case_id"]])
            - np.array(predictions[anchors[1]][cases[i]["case_id"]])
            - ab
            for i in queries
        ]
    )
    return {
        "self": zero_change_controls(self_pred),
        "reverse_max_abs": abs(np.array(reverse)).max(0).tolist(),
        "cycle_mae": abs(cycle).mean(0).tolist(),
        "cycle_max_abs": abs(cycle).max(0).tolist(),
        "scope": "Self/reverse are architecture checks; three-geometry cycle is measured, not guaranteed or an accuracy certificate.",
    }


def prior_predictions(cases, anchors, rows):
    lookup = {r["case_id"]: r for r in rows}
    queries = [c for c in cases if c["split"] == "test"]
    for c in cases:
        if c["case_id"] not in lookup or lookup[c["case_id"]]["truth"] != c["targets"]:
            raise ValueError("Previous baseline case/label mismatch")
    return {
        a: {
            c["case_id"]: (
                np.array(lookup[c["case_id"]]["prediction"]) - lookup[a]["prediction"]
            ).tolist()
            for c in queries
        }
        for a in anchors
    }


def evaluate(experiment):
    """@entry Freeze the validation winner before computing held-out errors."""
    root = Path(experiment)
    p, cases = load(root)
    if (root / "comparison.json").exists():
        raise FileExistsError("Comparison already recorded")
    plan = json.loads((root / "evaluation-plan.json").read_text())
    for file, expected_sha in plan["sha256"].items():
        if sha256(file) != expected_sha:
            raise ValueError(f"Frozen evaluation input changed: {file}")
    with np.load(root / "pairs.npz", allow_pickle=False) as pairs:
        left, right = pairs["left"], pairs["right"]
    expected = hashlib.sha256(
        np.stack((left, right), axis=2).astype(np.int32).tobytes()
    ).hexdigest()
    selected, proof = {}, {}
    for arm in p["arms"]:
        report = json.loads((root / arm / "training.json").read_text())
        if (
            len(report["history"]) != p["epochs"]
            or report["optimizer_steps"] != p["epochs"] * 90
            or report["executed_pair_sha256"] != expected
        ):
            raise ValueError("Incomplete/mismatched executed pair budget")
        if set(report["endpoint_exposures"]) != set(p["train_ids"]) or set(
            report["endpoint_exposures"].values()
        ) != {2 * p["epochs"]}:
            raise ValueError("Supervised geometry exposure differs")
        ck = torch.load(root / arm / "model.pt", map_location="cpu", weights_only=True)
        if (
            ck["protocol_sha256"] != sha256(root / "protocol.json")
            or ck["epoch"] != report["selected_epoch"]
        ):
            raise ValueError("Selected checkpoint/protocol differs")
        if ck["data512_lock_sha256"] is not None and ck[
            "data512_lock_sha256"
        ] != sha256(root / "data512-lock.json"):
            raise ValueError("512 input lock differs")
        selected[arm] = {
            "epoch": ck["epoch"],
            "validation": ck["validation"],
            "model_sha256": sha256(root / arm / "model.pt"),
        }
        proof[arm] = {
            "steps": report["optimizer_steps"],
            "actual_pair_trace": expected,
            "exposures_per_geometry": 2 * p["epochs"],
            "initial_tensor_sha256": report["initial_tensor_sha256"],
        }
    winner = min(selected, key=lambda a: selected[a]["validation"]["normalized_mae"])
    write_json(
        root / "selection.json",
        {
            "selected": winner,
            "arms": selected,
            "test_used": False,
            "radiant_used": False,
        },
    )
    external = json.loads(Path(p["external"]).read_text())
    external_pairs = Path(p["external"]).with_name("Radiant-SDF-pairs.csv")
    pair_rows = list(csv.DictReader(external_pairs.open()))
    queries = np.array([i for i, c in enumerate(cases) if c["split"] == "test"])
    i, j = np.triu_indices(len(queries), 1)
    qa, qb = queries[i], queries[j]
    truth = np.asarray([c["targets"] for c in cases], float)
    all_truth = truth[qb] - truth[qa]
    mapping = {(row["reference_id"], row["query_id"]): row for row in pair_rows}
    if len(mapping) != len(qa):
        raise ValueError("External unordered pair count differs")
    external_delta = []
    for n, (a, b) in enumerate(zip(qa, qb)):
        row = mapping[(cases[a]["case_id"], cases[b]["case_id"])]
        if not np.isclose(
            float(row["true_delta"]), all_truth[n, 0], atol=1e-12, rtol=0
        ):
            raise ValueError("External pair labels differ")
        external_delta.append(float(row["predicted_delta"]))
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    torch.set_num_threads(2)
    result = {}
    for arm, config in p["arms"].items():
        ck = torch.load(root / arm / "model.pt", map_location="cpu", weights_only=True)
        model = PairedDelta(**ck["model_config"]).to(device)
        model.load_state_dict(ck["model_state"], strict=True)
        model.eval()
        dataset = dataset_for(p, cases, config["resolution"])
        encoded = encode_all(model, dataset, device)
        val = anchor_predictions(
            model, encoded, dataset, cases, p["anchors"], "val", p["delta_std"], device
        )
        val_scores = score_anchors(cases, p["anchors"], val, "val", p["delta_std"])
        if (
            abs(val_scores["normalized_mae"] - ck["validation"]["normalized_mae"])
            > 2e-5
        ):
            raise ValueError("Validation replay differs")
        test = anchor_predictions(
            model, encoded, dataset, cases, p["anchors"], "test", p["delta_std"], device
        )
        train_pred = predict_pairs(
            model,
            encoded,
            dataset.parameters,
            left[0],
            right[0],
            p["delta_std"],
            device,
        )
        all_pred = predict_pairs(
            model, encoded, dataset.parameters, qa, qb, p["delta_std"], device
        )
        result[arm] = {
            "config": config,
            "epoch": ck["epoch"],
            "validation": val_scores,
            "train_pairs": score_delta(truth[right[0]] - truth[left[0]], train_pred),
            "test": score_anchors(cases, p["anchors"], test, "test", p["delta_std"]),
            "radiant": compare_radiant(cases, p["anchors"], test, external),
            "controls": control_scores(
                model,
                encoded,
                dataset,
                cases,
                p["anchors"],
                test,
                p["delta_std"],
                device,
            ),
            "test_pairwise": score_delta(all_truth, all_pred),
            "predictions": test,
            "pairwise_predictions": all_pred.tolist(),
        }
        write_json(root / arm / "evaluation.json", result[arm])
        print(
            json.dumps(
                {
                    "evaluated": arm,
                    "delta_mae": result[arm]["test"]["mean_anchors"],
                    "Cd_amplitude": result[arm]["test"]["pooled_correlated_pairs"][
                        "Cd"
                    ]["slope_through_zero"],
                }
            ),
            flush=True,
        )
        del model, encoded, dataset
        if device.type == "mps":
            torch.mps.empty_cache()
    base = json.loads(
        Path("outputs/vehicle-end-to-end-20261116/comparison.json").read_text()
    )["models"]["detail-ensemble"]
    kernel = json.loads(
        Path("outputs/vehicle-multiscale-readout-20261116/comparison.json").read_text()
    )
    baselines = {}
    for name, old in [("previous-neural", base), ("previous-multiscale", kernel)]:
        pred = prior_predictions(cases, p["anchors"], old["rows"])
        baselines[name] = {
            "test": score_anchors(cases, p["anchors"], pred, "test", p["delta_std"]),
            "radiant": compare_radiant(cases, p["anchors"], pred, external),
        }
    zeros = {
        a: {c["case_id"]: [0.0, 0.0] for c in cases if c["split"] == "test"}
        for a in p["anchors"]
    }
    baselines["zero-change"] = score_anchors(
        cases, p["anchors"], zeros, "test", p["delta_std"]
    )
    radiant_pairs = score_delta(
        np.column_stack((all_truth[:, 0], np.zeros(len(all_truth)))),
        np.column_stack((external_delta, np.zeros(len(all_truth)))),
    )["Cd"]
    write_json(
        root / "comparison.json",
        {
            "selected": winner,
            "proof": proof,
            "models": result,
            "baselines": baselines,
            "radiant_test_pairwise_Cd": radiant_pairs,
            "test_query_ids": [cases[x]["case_id"] for x in queries],
            "external_pairs_sha256": sha256(external_pairs),
            "scope": "Single-seed12epoch quick factorial; Rim720/92/92 fixedbody/conditions. All-test pairs share92geometries, not4186independent samples. Cl external comparison unavailable.",
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    evaluate(parser.parse_args().experiment)
