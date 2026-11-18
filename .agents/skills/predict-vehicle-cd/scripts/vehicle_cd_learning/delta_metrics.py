"""Frozen direct-pair delta metrics; no per-example relative-error division."""

import numpy as np

THRESHOLDS = (0.0001, 0.0005, 0.001)
# Absorb coefficient-subtraction roundoff only; this is not CFD uncertainty.
THRESHOLD_TOLERANCE = 1e-12
SCOPE = (
    "Historical known-family Rim queries, shared body and fixed source conditions. "
    "Pairs sharing queries or anchors are correlated, not independent samples. "
    "Direction thresholds are diagnostics, not verified CFD noise bounds. "
    "Radiant training exposure and live identity are unknown; no architecture superiority claim."
)


def _arrays(truth, pred):
    truth, pred = np.asarray(truth, float), np.asarray(pred, float)
    if truth.shape != pred.shape or truth.ndim != 2 or truth.shape[1] != 2:
        raise ValueError("Expected matching nonempty N x 2 Cd/Cl arrays")
    if not len(truth) or not np.isfinite(truth).all() or not np.isfinite(pred).all():
        raise ValueError("Expected nonempty finite Cd/Cl arrays")
    return truth, pred


def _target(truth, pred):
    error = pred - truth
    direction = {}
    for threshold in (0.0, *THRESHOLDS):
        active = abs(truth) > threshold + THRESHOLD_TOLERANCE
        direction[str(threshold)] = {
            "count": int(active.sum()),
            "accuracy": float(np.mean(np.sign(pred[active]) == np.sign(truth[active])))
            if active.any()
            else None,
        }
    true_std, pred_std = float(truth.std()), float(pred.std())
    norm = float(truth @ truth)
    return {
        "n": len(truth),
        "mae": float(abs(error).mean()),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "signed_bias": float(error.mean()),
        "direction": direction,
        "slope_through_zero": float(truth @ pred / norm) if norm > 0 else None,
        "true_std": true_std,
        "predicted_std": pred_std,
        "std_ratio": pred_std / true_std if true_std > 0 else None,
        "correlation": float(np.corrcoef(truth, pred)[0, 1])
        if true_std > 0 and pred_std > 0
        else None,
        "zero_change_baseline_mae": float(abs(truth).mean()),
    }


def score_delta(truth, pred):
    """@entry Score physical coefficient deltas; exact-zero truth has no direction."""
    truth, pred = _arrays(truth, pred)
    return {
        "n": len(truth),
        "Cd": _target(truth[:, 0], pred[:, 0]),
        "Cl": _target(truth[:, 1], pred[:, 1]),
        "direction_primary_threshold": 0.0,
        "threshold_scope": "Diagnostic thresholds, not verified CFD noise bounds.",
    }


def normalized_delta_mae(truth, pred, delta_std):
    """Validation selector: equally weight targets using frozen train-only scales."""
    truth, pred = _arrays(truth, pred)
    scales = np.asarray(delta_std, float)
    if scales.shape != (2,) or not np.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError("delta_std must contain two positive frozen train-only scales")
    return float(np.mean(abs(pred - truth).mean(0) / scales))


def _pair_arrays(cases, anchor_ids, predictions, split):
    by_id = {c["case_id"]: c for c in cases}
    if (
        len(by_id) != len(cases)
        or not anchor_ids
        or len(set(anchor_ids)) != len(anchor_ids)
    ):
        raise ValueError("Case and anchor IDs must be unique")
    queries = sorted(
        c["case_id"]
        for c in cases
        if c["source"] == "drivaerrim" and c["split"] == split
    )
    if not queries:
        raise ValueError("No Rim queries in requested split")
    truth, pred = [], []
    for anchor in anchor_ids:
        reference = by_id[anchor]
        if reference["source"] != "drivaerrim" or reference["split"] != "train":
            raise ValueError("Anchors must be training Rim cases")
        if set(predictions[anchor]) != set(queries):
            raise ValueError("Prediction query membership differs")
        t = np.asarray([by_id[q]["targets"] for q in queries]) - reference["targets"]
        p = np.asarray([predictions[anchor][q] for q in queries])
        t, p = _arrays(t, p)
        truth.append(t)
        pred.append(p)
    return by_id, queries, np.asarray(truth), np.asarray(pred)


def score_anchors(cases, anchor_ids, predictions, split="test", delta_std=None):
    """Score direct pair predictions mapping anchor -> query -> [deltaCd, deltaCl]."""
    _, ids, truth, pred = _pair_arrays(cases, anchor_ids, predictions, split)
    scores = {a: score_delta(t, p) for a, t, p in zip(anchor_ids, truth, pred)}
    pooled = score_delta(truth.reshape(-1, 2), pred.reshape(-1, 2))
    return {
        "query_count": len(ids),
        "anchor_count": len(anchor_ids),
        "pair_count": len(ids) * len(anchor_ids),
        "by_anchor": scores,
        "mean_anchors": {
            name: {
                metric: float(np.mean([s[name][metric] for s in scores.values()]))
                for metric in ("mae", "rmse", "signed_bias", "zero_change_baseline_mae")
            }
            for name in ("Cd", "Cl")
        },
        "pooled_correlated_pairs": pooled,
        "normalized_mae": normalized_delta_mae(
            truth.reshape(-1, 2), pred.reshape(-1, 2), delta_std
        )
        if delta_std is not None
        else None,
        "scope": SCOPE,
    }


def zero_change_controls(pred):
    """Actual same-geometry direct-pair outputs supplied by the caller."""
    pred = np.asarray(pred, float)
    truth, pred = _arrays(np.zeros_like(pred), pred)
    return {
        "scores": score_delta(truth, pred),
        "max_abs": abs(pred).max(0).tolist(),
        "scope": "Algebraic consistency control; not aerodynamic accuracy.",
    }


def compare_radiant(
    cases, anchor_ids, predictions, external, bootstrap_samples=2000, seed=20261117
):
    """@entry Compare identical held-out Cd pairs; bootstrap whole query clusters."""
    by_id, ids, truth, ours = _pair_arrays(cases, anchor_ids, predictions, "test")
    if set(external["rows_by_anchor"]) != set(anchor_ids):
        raise ValueError("Radiant anchor membership differs")
    theirs, scores = [], {}
    for i, anchor in enumerate(anchor_ids):
        rows = external["rows_by_anchor"][anchor]
        mapping = {r["case_id"]: r for r in rows}
        if len(mapping) != len(rows) or set(mapping) != set(ids):
            raise ValueError("Radiant query membership differs")
        base = by_id[anchor]["targets"][0]
        if external["anchors"][anchor] != base:
            raise ValueError("Radiant anchor truth differs")
        predicted = []
        for j, query in enumerate(ids):
            row = mapping[query]
            if (
                row["truth"] != by_id[query]["targets"][0]
                or row["delta_truth"] != truth[i, j, 0]
            ):
                raise ValueError("Radiant query/delta truth differs")
            if (
                not np.isfinite(row["prediction"])
                or abs(row["prediction"] - base - row["delta_prediction"]) > 1e-12
            ):
                raise ValueError("Radiant anchored prediction identity differs")
            predicted.append(row["delta_prediction"])
        p = np.asarray(predicted, float)
        if not np.isfinite(p).all():
            raise ValueError("Radiant predictions must be finite")
        theirs.append(p)
        scores[anchor] = {
            "ours": _target(truth[i, :, 0], ours[i, :, 0]),
            "radiant": _target(truth[i, :, 0], p),
        }
    theirs = np.asarray(theirs)
    # First average all anchors within a query, then resample queries together.
    difference = (
        abs(ours[:, :, 0] - truth[:, :, 0]) - abs(theirs - truth[:, :, 0])
    ).mean(0)
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    draws = np.random.default_rng(seed).integers(
        len(ids), size=(bootstrap_samples, len(ids))
    )
    return {
        "query_count": len(ids),
        "anchor_count": len(anchor_ids),
        "correlated_pair_count": len(ids) * len(anchor_ids),
        "by_anchor": scores,
        "ours": _target(truth[:, :, 0].ravel(), ours[:, :, 0].ravel()),
        "radiant": _target(truth[:, :, 0].ravel(), theirs.ravel()),
        "our_minus_radiant_cd_mae": float(difference.mean()),
        "our_minus_radiant_cd_mae_95_interval": np.quantile(
            difference[draws].mean(1), [0.025, 0.975]
        ).tolist(),
        "Cl_supported": False,
        "bootstrap": {
            "unit": "query with all anchors kept together",
            "samples": bootstrap_samples,
            "seed": seed,
        },
        "zero_change_baseline_cd_mae": float(abs(truth[:, :, 0]).mean()),
        "scope": SCOPE,
    }
