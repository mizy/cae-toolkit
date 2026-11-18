"""Validate sampled-buffer registration on the absolute and direct design-change paths."""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from .absolute_view_audit import predict_grid, view_metrics
from .delta_experiment import sha256, write_json
from .joint_sampling import FullCases, load, load_checkpoint
from .model import VIEWS
from .phase_decoder_scores import (
    groups_and_pairs,
    pair_grid,
    patterns,
    stability,
    summarize,
)
from .sampling_experiment import SamplingCases
from .sampling_experiment import load as load_sampling
from .subpixel_geometry import METHODS, STATES


def feature_bank(root, size, geometry, hashes):
    rows = []
    for gid in geometry:
        key = f"{size}-{gid}"
        path = root / "features" / f"{key}.npz"
        if sha256(path) != hashes[key]["feature_sha256"]:
            raise ValueError(f"Feature hash mismatch: {gid}")
        with np.load(path, allow_pickle=False) as f:
            rows.append({k: f[k] for k in f.files})
    return {k: torch.from_numpy(np.stack([item[k] for item in rows])) for k in rows[0]}


def absolute_output(readout, features, rows, params, grid, stats, device):
    groups = features["front"].shape[0]
    tokens = []
    for v in VIEWS:
        projection = readout.bottom_project if v == "bottom" else readout.image_project
        values = projection(features[v].to(device).flatten(0, 1)).reshape(
            groups, len(STATES), -1
        )
        tokens.append(values)
    bank = torch.stack(tokens, 2)[rows]
    if all(torch.equal(bank[:, 0], bank[:, i]) for i in range(1, len(STATES))):
        values = predict_grid(readout, bank, params, grid[:1], stats)
        return np.broadcast_to(values, (len(params), len(grid), 2))
    return predict_grid(readout, bank, params, grid, stats)


def absolute_scores(readout, data, groups, root, hashes, p, split, device):
    geometry = [gid for gid, record in groups.items() if record["split"] == split]
    ids = [cid for gid in geometry for cid in groups[gid]["case_ids"]]
    rows = [j for j, gid in enumerate(geometry) for _ in groups[gid]["case_ids"]]
    cases = [data.cases[data.index[cid]] for cid in ids]
    truth = np.asarray([c["targets"] for c in cases])
    sources = np.asarray([c["source"] for c in cases])
    params = data.parameters[[data.index[i] for i in ids]].to(device)
    all_features = feature_bank(root, 256, geometry, hashes)
    grid = np.array(
        list(itertools.product(range(len(STATES)), repeat=5)), dtype=np.int16
    )
    quarters = (grid < 5).all(1)
    result = {}
    baseline = {}
    examples = {}
    for mode, method in enumerate(METHODS):
        values = absolute_output(
            readout,
            {k: v[:, mode] for k, v in all_features.items()},
            rows,
            params,
            grid,
            p["normalization"],
            device,
        )
        metrics = {}
        for s in sorted(set(sources)):
            mask = sources == s
            record = view_metrics(values[mask], truth[mask], grid)
            q = view_metrics(values[mask][:, quarters], truth[mask], grid[quarters])
            if method == "point":
                baseline[s] = np.array(record["nominal_mae"])
            record["quarter_grid_worst_drift_mean"] = q[
                "mean_case_worst_independent_drift"
            ]
            record["passes_stability"] = bool(
                (
                    np.array(record["mean_case_worst_independent_drift"])
                    <= np.array(p["delta_std"]) * 0.05
                ).all()
                and (
                    np.array(record["p95_case_worst_independent_drift"])
                    <= np.array(p["delta_std"]) * 0.1
                ).all()
            )
            record["nominal_mae_preserved"] = bool(
                (np.array(record["nominal_mae"]) <= baseline[s] * 1.05 + 1e-9).all()
            )
            metrics[s] = record
        result[method] = metrics
        examples[method] = np.stack(
            (values[:, 0], values.min(1), values.max(1)), axis=1
        )
    return {
        "metrics": result,
        "case_ids": ids,
        "geometry_count": len(geometry),
        "grid_patterns": len(grid),
    }, examples


def paired_scores(model, rim, groups, root, hashes, p, split, device):
    queries = sorted(
        cid
        for g in groups.values()
        if g["source"] == "drivaerrim" and g["split"] == split
        for cid in g["case_ids"]
    )
    ids = p["anchors"] + queries
    geom = [rim.cases[rim.index[c]]["geometry_id"] for c in ids]
    bank = feature_bank(root, 512, geom, hashes)
    groups = groups_and_pairs(queries, p["anchors"])
    pairs = list(itertools.chain.from_iterable(groups.values()))
    labels = {cid: rim.cases[rim.index[cid]]["targets"] for cid in ids}
    params = rim.parameters[rim.index[queries[0]]].to(device)
    grid = patterns(len(STATES), exhaustive_singles=True)
    result = {}
    predictions = {}
    baseline = None
    worst_parity = 0.0
    for mode, method in enumerate(METHODS):
        features = {k: v[:, mode] for k, v in bank.items()}
        repeated = all(
            torch.equal(v[:, 0:1].expand_as(v), v) for v in features.values()
        )
        used = grid[:1] if repeated else grid
        values = pair_grid(model, features, ids, pairs, used, params, p["delta_std"])
        if repeated:
            values = np.broadcast_to(values, (len(grid), len(pairs), 2))
        # Recheck actual model.predict_pair on a mixed pattern, outside the factored scorer.
        for ix in (0, len(grid) - 1):
            a, b = pairs[len(queries)]
            pa, pb = grid[ix]
            ends = []
            for cid, phases in ((a, pa), (b, pb)):
                ends.append(
                    {
                        k: features[k][ids.index(cid) : ids.index(cid) + 1, state].to(
                            device
                        )
                        for v, state in zip(VIEWS, phases)
                        for k in (v, v + "_mask")
                    }
                )
            ref = (
                model.predict_pair(*ends, params[None]).cpu().numpy()[0]
                * p["delta_std"]
            )
            worst_parity = max(
                worst_parity, float(np.max(np.abs(ref - values[ix, len(queries)])))
            )
        if baseline is None:
            baseline = values
        metrics = summarize(values, grid, groups, labels, baseline)
        metrics["stability"] = stability(metrics, p["delta_std"])
        metrics["max_abs_difference_from_original_nominal"] = (
            np.abs(values - baseline[0:1]).max((0, 1)).tolist()
        )
        result[method] = metrics
        predictions[method] = values
    if worst_parity > 2e-6:
        raise ValueError(
            f"Cached pair readout disagrees with full readout: {worst_parity}"
        )
    return {
        "metrics": result,
        "queries": queries,
        "pair_ids": pairs,
        "patterns": grid.tolist(),
        "direct_forward_max_diff": worst_parity,
    }, predictions


def run(experiment, root, output):
    """@entry Score prespecified acquisition methods; no weights or case labels enter the sampler."""
    p, cases = load(experiment)
    plan = json.loads((root / "plan.json").read_text())
    m = json.loads((root / "manifest.json").read_text())
    if (
        sha256(root / "plan.json") != m["plan_sha256"]
        or plan["methods"] != list(METHODS)
        or sha256(plan["checkpoint"]) != plan["checkpoint_sha256"]
    ):
        raise ValueError("Acquisition identity mismatch")
    for f, h in plan["sources"].items():
        if sha256(f) != h:
            raise ValueError(f"Changed acquisition source: {f}")
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model, readout = load_checkpoint(plan["checkpoint"], device)
    model.eval()
    readout.eval()
    data = FullCases(p, cases)
    sp, rc = load_sampling(p["sampling_parent"])
    rim = SamplingCases(sp, rc, with_phases=False)
    result = {}
    with torch.inference_mode():
        for split in ("val", "test"):
            a, absolute = absolute_scores(
                readout, data, plan["groups"], root, m["records"], p, split, device
            )
            b, pairs = paired_scores(
                model, rim, plan["groups"], root, m["records"], p, split, device
            )
            np.savez_compressed(
                output / f"{split}-predictions.npz",
                **{"absolute_" + n: v for n, v in absolute.items()},
                **{"pair_" + n: v for n, v in pairs.items()},
            )
            result[split] = {"absolute": a, "paired": b}
            print(split, {n: b["metrics"][n]["stability"] for n in METHODS}, flush=True)
    write_json(output / "comparison.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("experiment", "buffers", "output"):
        parser.add_argument(field, type=Path)
    a = parser.parse_args()
    run(a.experiment, a.buffers, a.output)
