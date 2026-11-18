"""Score sampling drift and labeled changes of the existing antisymmetric pair readout."""

import itertools

import numpy as np
import torch

from .delta_metrics import score_delta
from .model import VIEWS
from .paired_delta import _masked_pool
from .phase_metrics import preservation_gate


def patterns(states, mixed=128, exhaustive_singles=False):
    """Perturb both endpoints independently, alongside single-view and common-shift controls."""
    zero = (0,) * len(VIEWS)
    result = {(zero, zero)}
    for a, b in itertools.product(range(states), repeat=2):
        if exhaustive_singles or a == 0 or b == 0 or a == b:
            result.add(((a,) * 5, (b,) * 5))
            for v in range(5):
                u = list(zero)
                w = list(zero)
                u[v] = a
                w[v] = b
                result.add((tuple(u), tuple(w)))
    rng = np.random.default_rng(42)
    for a, b in rng.integers(states, size=(mixed, 2, 5)):
        result.add((tuple(a), tuple(b)))
    return np.asarray(sorted(result), dtype=np.int16)


def pair_grid(model, bank, ids, pairs, grid, condition, scale):
    """Cache the exact per-view adapter outputs. Fusion and the Cd/Cl head see every phase mix."""
    device = condition.device
    indices = {cid: i for i, cid in enumerate(ids)}
    a, b = torch.tensor([(indices[i], indices[j]) for i, j in pairs], device=device).T
    n = len(a)
    local = []
    selectors = []
    for v, name in enumerate(VIEWS):
        distinct, lookup = np.unique(grid[:, :, v], axis=0, return_inverse=True)
        features = bank[name].to(device)
        masks = bank[name + "_mask"].to(device)
        vectors = []
        for start in range(0, len(distinct), 4):
            phases = torch.as_tensor(
                distinct[start : start + 4], dtype=torch.long, device=device
            )
            left, right = phases.T[:, :, None]
            fa, fb = features[a, left].flatten(0, 1), features[b, right].flatten(0, 1)
            mask = (
                (torch.maximum(masks[a, left], masks[b, right]) > 0)
                .to(fa.dtype)
                .flatten(0, 1)
            )
            context, diff = (fa + fb) * 0.5, fb - fa
            hidden = model.adapter(
                torch.cat(
                    [torch.cat((context, diff), 1), torch.cat((context, -diff), 1)]
                )
            )
            pos, neg = _masked_pool(hidden, mask.repeat(2, 1, 1, 1), 2).chunk(2)
            vectors.append(
                torch.stack([pos.reshape(-1, n, 128), neg.reshape(-1, n, 128)], 2).cpu()
            )
        local.append(torch.cat(vectors))
        selectors.append(lookup)
    param = model.parameter_encoder(condition[None])
    outputs = []
    for start in range(0, len(grid), 8):
        block = [
            value[torch.from_numpy(sel[start : start + 8])].to(device)
            for value, sel in zip(local, selectors)
        ]
        k = block[0].shape[0]
        full = torch.cat([*block, param.expand(k, n, 2, -1)], -1)
        values = model.head(full)
        outputs.append(((values[:, :, 0] - values[:, :, 1]) * 0.5).cpu().numpy())
    return np.concatenate(outputs).astype(float) * scale


def groups_and_pairs(queries, anchors):
    groups = {
        "self": [(q, q) for q in queries],
        "pairs": list(itertools.combinations(queries, 2)),
    }
    groups.update({a: [(a, q) for q in queries] for a in anchors})
    return groups


def drift_statistics(worst_per_case):
    return {
        "mae": worst_per_case.mean(0).tolist(),
        "p95": np.quantile(worst_per_case, 0.95, axis=0).tolist(),
        "max": worst_per_case.max(0).tolist(),
    }


def summarize(values, grid, groups, labels, compare=None, strict_patterns=False):
    """Keep the worst pattern per pair before averaging; test each real-response group."""
    if np.any(grid[0]) or not np.isfinite(values).all():
        raise ValueError("Invalid phase predictions")
    result = {}
    start = 0
    all_drift = []
    failures = []
    for group, pairs in groups.items():
        end = start + len(pairs)
        rows = values[:, start:end]
        ref = None if compare is None else compare[:, start:end]
        start = end
        nominal = rows[0]
        drift = np.abs(rows - nominal[None]).max(0)
        if group == "self":
            result[group] = drift_statistics(np.abs(rows).max(0))
            continue
        truth = np.array([np.asarray(labels[b]) - labels[a] for a, b in pairs])
        predicted = score_delta(truth, nominal)
        result[group] = {
            "nominal": predicted,
            "worst_drift": drift_statistics(drift),
            "can_flip": (
                (rows.min(0) < -5e-6) & (rows.max(0) > 5e-6) & (np.abs(truth) > 1e-12)
            )
            .sum(0)
            .tolist(),
            "nonzero": (np.abs(truth) > 1e-12).sum(0).tolist(),
        }
        all_drift.append(drift)
        if ref is not None:
            # The nominal group is a geometry response. For each shifted group compare like phases.
            checks = range(len(grid)) if strict_patterns else [0]
            for k in checks:
                gate = preservation_gate(
                    score_delta(truth, ref[k]),
                    score_delta(truth, rows[k]),
                    min_amplitude_fraction=0.95,
                )
                if not gate["pass"]:
                    failures.append(
                        {"group": group, "pattern": k, "reasons": gate["reasons"]}
                    )
    result["combined_drift"] = drift_statistics(np.concatenate(all_drift))
    result["response_failures"] = len(failures)
    result["first_failures"] = failures[:8]
    return result


def stability(metrics, scales):
    scores = [
        np.asarray(metrics[group][stat]) / (np.asarray(scales) * limit)
        for group in ("self", "combined_drift")
        for stat, limit in [("mae", 0.05), ("p95", 0.10)]
    ]
    value = float(np.max(scores))
    return {"passes": value <= 1.0, "worst_threshold_multiple": value}
