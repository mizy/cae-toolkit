"""Evaluate absolute coefficients over a finite product of independent view phases."""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from .absolute_view_renders import PHASES
from .delta_experiment import sha256, write_json
from .delta_metrics import score_delta
from .image_perturbation import translate_views
from .joint_sampling import FullCases, batch_cases, load, load_checkpoint
from .model import VIEWS


def project(model, readout, views):
    pooled = model.encode(views)
    body = readout.image_project(torch.cat([pooled[n] for n in VIEWS[:4]]))
    return torch.stack(
        [*body.chunk(4), readout.bottom_project(pooled["bottom"])], dim=1
    )


def render_batch(data, indices, phase, manifest, device):
    if phase == 0 or manifest is None:
        views, _, _ = batch_cases(data, indices, device)
        if phase:
            shift = torch.tensor(PHASES[phase], dtype=torch.float32, device=device)
            views = translate_views(views, shift.expand(len(indices), 2))
        return views
    arrays = []
    for idx in indices:
        gid = data.cases[idx]["geometry_id"]
        record = manifest[gid]["views"][phase]
        if sha256(record["path"]) != record["sha256"]:
            raise ValueError(f"Raw phase cache changed for {gid}")
        with np.load(record["path"], allow_pickle=False) as f:
            arrays.append({n: torch.from_numpy(f[n]) for n in VIEWS})
    return {n: torch.stack([r[n] for r in arrays]).to(device) for n in VIEWS}


def representations(model, readout, data, indices, manifest, device):
    representatives = {data.cases[i]["geometry_id"]: i for i in indices}
    ids = list(representatives.values())
    bank = []
    for start in range(0, len(ids), 8):
        batch = ids[start : start + 8]
        bank.append(
            torch.stack(
                [
                    project(
                        model,
                        readout,
                        render_batch(data, batch, phase, manifest, device),
                    )
                    for phase in range(len(PHASES))
                ],
                dim=1,
            ).cpu()
        )
    lookup = {gid: pos for pos, gid in enumerate(representatives)}
    return torch.cat(bank)[[lookup[data.cases[i]["geometry_id"]] for i in indices]].to(
        device
    )


def predict_grid(readout, embeddings, parameters, grid, stats):
    """Keep the existing nonlinear five-view attention and head for every combination."""
    device = parameters.device
    condition = readout.parameter_encoder(parameters)
    predictions = []
    for start in range(0, len(grid), 16):
        block = torch.as_tensor(
            grid[start : start + 16], dtype=torch.long, device=device
        )
        inputs = [
            embeddings[:, block[:, j], j].flatten(0, 1) for j in range(len(VIEWS))
        ]
        repeated = condition[:, None].expand(-1, len(block), -1).flatten(0, 1)
        scaled, _ = readout.view_attention(inputs, repeated)
        result = readout.head(torch.cat([*scaled, repeated], dim=1))
        predictions.append(result.reshape(len(parameters), len(block), 2).cpu().numpy())
    return (
        np.concatenate(predictions, axis=1).astype(float) * stats["target_std"]
        + stats["target_mean"]
    )


def direct_check(
    model, readout, data, indices, manifest, grid, predictions, stats, device
):
    batch = indices[:4]
    # Include nominal, a single shifted view, and five independently chosen phases.
    checks = [(0, 0, 0, 0, 0), (0, 0, 0, 0, 1), (1, 2, 3, 4, 0), (4, 4, 4, 4, 4)]
    phase_inputs = [
        render_batch(data, batch, j, manifest, device) for j in range(len(PHASES))
    ]
    params = data.parameters[batch].to(device)
    worst = 0.0
    for pattern in checks:
        views = {n: phase_inputs[state][n] for n, state in zip(VIEWS, pattern)}
        expected = readout(model.encode(views), params).cpu().numpy().astype(float)
        expected = expected * stats["target_std"] + stats["target_mean"]
        col = int(np.flatnonzero(np.all(grid == pattern, axis=1))[0])
        worst = max(
            worst, float(np.max(np.abs(expected - predictions[: len(batch), col])))
        )
    if worst > 5e-6:
        raise ValueError(f"Cached grid predictions disagree with full forward: {worst}")
    return worst


def view_metrics(values, truth, grid):
    nominal = values[:, 0]
    distance = np.abs(values - nominal[:, None])
    same = np.all(grid == grid[:, :1], axis=1)
    by_view = {}
    for j, name in enumerate(VIEWS):
        single = (grid[:, j] != 0) & (np.count_nonzero(grid, axis=1) == 1)
        moved = distance[:, single].max(1)
        by_view[name] = {
            "mean_case_worst_drift": moved.mean(0).tolist(),
            "p95_case_worst_drift": np.quantile(moved, 0.95, axis=0).tolist(),
        }
    return {
        "records": len(values),
        "single_view": by_view,
        "nominal_mae": np.abs(nominal - truth).mean(0).tolist(),
        "mean_case_worst_independent_mae": np.abs(values - truth[:, None])
        .max(1)
        .mean(0)
        .tolist(),
        "mean_case_worst_common_drift": distance[:, same].max(1).mean(0).tolist(),
        "mean_case_worst_independent_drift": distance.max(1).mean(0).tolist(),
        "p95_case_worst_independent_drift": np.quantile(
            distance.max(1), 0.95, axis=0
        ).tolist(),
    }


def detail_intervals(truth, nominal, lower, upper, pairs, margin=5e-6):
    """Tight coefficient-wise difference bounds over independent endpoint choices."""
    a, b = np.asarray(pairs, dtype=int).T
    actual, clean = truth[b] - truth[a], nominal[b] - nominal[a]
    lo, hi = lower[b] - upper[a], upper[b] - lower[a]
    scores = score_delta(actual, clean)
    for j, target in enumerate(("Cd", "Cl")):
        signal = actual[:, j]
        valid = np.abs(signal) > 1e-12
        correct = ((signal > 0) & (lo[:, j] > margin)) | (
            (signal < 0) & (hi[:, j] < -margin)
        )
        flip = (lo[:, j] < -margin) & (hi[:, j] > margin)
        scores[target].update(
            nonzero_truth=int(valid.sum()),
            always_correct=int((valid & correct).sum()),
            nominal_correct=int((valid & (signal * clean[:, j] > 0)).sum()),
            can_reverse=int((valid & flip).sum()),
            worst_delta_mae=float(
                np.maximum(np.abs(lo[:, j] - signal), np.abs(hi[:, j] - signal)).mean()
            ),
        )
    return scores


def evaluate(model, readout, data, indices, manifest, grid, p, output, device):
    stats = p["normalization"]
    cases = [data.cases[i] for i in indices]
    sources = np.array([c["source"] for c in cases])
    truth = np.array([c["targets"] for c in cases], float)
    if not np.isfinite(truth).all():
        raise ValueError("Expected finite validation labels for both coefficients")
    with torch.inference_mode():
        bank = representations(model, readout, data, indices, manifest, device)
        predictions = predict_grid(
            readout, bank, data.parameters[indices].to(device), grid, stats
        )
        parity = direct_check(
            model, readout, data, indices, manifest, grid, predictions, stats, device
        )
    if not np.isfinite(predictions).all():
        raise ValueError("Nonfinite independent-view predictions")
    np.savez_compressed(
        output,
        case_ids=np.array([c["case_id"] for c in cases]),
        sources=sources,
        truth=truth,
        grid=grid,
        prediction=predictions,
    )
    groups = {n: sources == n for n in sorted(set(sources))}
    groups["vehicle"] = np.array([c["scene"] == 1 for c in cases])
    groups["aircraft"] = np.array([c["scene"] == 2 for c in cases])
    metrics = {
        n: view_metrics(predictions[mask], truth[mask], grid)
        for n, mask in groups.items()
        if mask.any()
    }
    detail = {}
    for source in ("drivaerrim", "hilift"):
        source_ids = np.flatnonzero(sources == source)
        matched = {}
        for i in source_ids:
            # HiLift sweeps share source Re_MAC and Mach; compare designs at the same AoA.
            alpha = cases[i]["conditions"]["alpha_deg"]
            matched.setdefault(alpha, []).append(i)
        pairs = [
            pair for ids in matched.values() for pair in itertools.combinations(ids, 2)
        ]
        if pairs:
            detail[source] = detail_intervals(
                truth, predictions[:, 0], predictions.min(1), predictions.max(1), pairs
            )
    return {
        "rows": len(cases),
        "geometries": len({c["geometry_id"] for c in cases}),
        "metrics": metrics,
        "detail_from_absolute_outputs": detail,
        "max_full_forward_error": parity,
        "arrays_sha256": sha256(output),
    }


def run(experiment, renders, output):
    p, cases = load(experiment)
    retained = json.loads((experiment / "retention.json").read_text())
    if sha256(retained["checkpoint"]) != retained["sha256"]:
        raise ValueError("Retained coefficients checkpoint changed")
    raw = json.loads((renders / "manifest.json").read_text())
    if raw["plan_sha256"] != sha256(renders / "plan.json") or raw["phases_pixels"] != [
        list(s) for s in PHASES
    ]:
        raise ValueError("Physical phase manifest disagrees with audit design")
    val = [i for i, c in enumerate(cases) if c["split"] == "val"]
    raw_ids = {cid for g in raw["geometries"].values() for cid in g["case_ids"]}
    raw_val = [i for i in val if cases[i]["case_id"] in raw_ids]
    if len(raw_val) != len(raw_ids) or len(val) != 614:
        raise ValueError("Validation membership changed")
    output.mkdir(parents=True, exist_ok=False)
    grid = np.array(
        list(itertools.product(range(len(PHASES)), repeat=len(VIEWS))), dtype=np.int16
    )
    write_json(
        output / "plan.json",
        {
            "checkpoint_sha256": retained["sha256"],
            "views": list(VIEWS),
            "states_pixels": PHASES,
            "val_ids": [cases[i]["case_id"] for i in val],
            "raw_ids": sorted(raw_ids),
            "grid_configurations": len(grid),
            "ordering_margin": 5e-6,
            "joint_protocol_sha256": sha256(experiment / "protocol.json"),
            "raw_manifest_sha256": sha256(renders / "manifest.json"),
            "code_sha256": sha256(Path(__file__)),
            "scope": "Frozen absolute head, validation, independent camera-plane sampling only. HiLift/Rim same-source boundary conditions. No fitting or selection.",
        },
    )
    torch.set_num_threads(2)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model, readout = load_checkpoint(retained["checkpoint"], device)
    model.eval(), readout.eval()
    data = FullCases(p, cases)
    result = {}
    for kind, indices, manifest in [
        ("image", val, None),
        ("geometry", raw_val, raw["geometries"]),
    ]:
        result[kind] = evaluate(
            model,
            readout,
            data,
            indices,
            manifest,
            grid,
            p,
            output / f"{kind}.npz",
            device,
        )
        print(
            kind,
            result[kind]["rows"],
            "cases, forward parity",
            result[kind]["max_full_forward_error"],
            flush=True,
        )
    reference = json.loads((experiment / "validation-metrics.json").read_text())[
        "candidates"
    ]
    ref = next(
        c["full_absolute"]["metrics"]
        for c in reference
        if c["arm"] == "online" and c["epoch"] == 5
    )
    for source, measures in result["image"]["metrics"].items():
        expected = [ref[source]["targets"][k]["mae"] for k in ("Cd", "Cl")]
        np.testing.assert_allclose(measures["nominal_mae"], expected, rtol=0, atol=5e-6)
    write_json(output / "metrics.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("experiment", "renders", "output"):
        parser.add_argument(field, type=Path)
    a = parser.parse_args()
    run(a.experiment, a.renders, a.output)
