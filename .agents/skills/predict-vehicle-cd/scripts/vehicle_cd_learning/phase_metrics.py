"""Score camera-phase stability without treating paired rows as independent cases."""

import numpy as np

from .delta_metrics import score_delta

SCOPE = (
    "Camera phases and pair rows are repeated measurements on fixed geometries; "
    "they are not independent geometry samples. Stability scores describe only "
    "the supplied phase set and pair set."
)


def _finite_array(value, shape, name):
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(
            f"{name} must have shape {shape} and contain only finite values"
        )
    return array


def sampling_scores(truth_pairs, nominal_pred, phase_pred, self_phase_pred, delta_std):
    """Score fixed-pair deltas and same-geometry outputs across camera phases."""
    truth = np.asarray(truth_pairs, dtype=float)
    if truth.ndim != 2 or truth.shape[0] < 1 or truth.shape[1] != 2:
        raise ValueError("truth_pairs must be a nonempty P x 2 array")
    pairs = truth.shape[0]
    nominal = _finite_array(nominal_pred, (pairs, 2), "nominal_pred")
    phase = np.asarray(phase_pred, dtype=float)
    self_phase = np.asarray(self_phase_pred, dtype=float)
    if phase.ndim != 3 or phase.shape[1:] != (pairs, 2) or phase.shape[0] < 1:
        raise ValueError("phase_pred must have shape K x P x 2")
    if (
        self_phase.ndim != 3
        or self_phase.shape[0] != phase.shape[0]
        or self_phase.shape[2] != 2
        or self_phase.shape[1] < 1
    ):
        raise ValueError("self_phase_pred must have shape K x N x 2 with matching K")
    if (
        not np.isfinite(truth).all()
        or not np.isfinite(phase).all()
        or not np.isfinite(self_phase).all()
    ):
        raise ValueError(
            "all prediction and truth arrays must contain only finite values"
        )
    scales = np.asarray(delta_std, dtype=float)
    if scales.shape != (2,) or not np.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError("delta_std must contain two positive finite scales")

    phase_truth = np.broadcast_to(truth, phase.shape).reshape(-1, 2)
    phase_flat = phase.reshape(-1, 2)
    drift = phase - nominal[None, :, :]
    self_abs = np.abs(self_phase)
    drift_abs = np.abs(drift)
    nominal_score = score_delta(truth, nominal)
    phase_score = score_delta(phase_truth, phase_flat)
    targets = {}
    stability = []
    for index, name in enumerate(("Cd", "Cl")):
        self_mae = float(np.abs(self_phase[:, :, index]).mean())
        drift_mae = float(drift_abs[:, :, index].mean())
        stability.append((self_mae + drift_mae) / scales[index])
        targets[name] = {
            "self_phase_mae": self_mae,
            "self_phase_p95_abs": float(np.quantile(self_abs[:, :, index], 0.95)),
            "common_phase_drift_mae": drift_mae,
            "common_phase_drift_p95_abs": float(
                np.quantile(drift_abs[:, :, index], 0.95)
            ),
            "phase_delta_mae": phase_score[name]["mae"],
            "nominal_delta_mae": nominal_score[name]["mae"],
            "nominal_slope_through_zero": nominal_score[name]["slope_through_zero"],
            "nominal_direction": nominal_score[name]["direction"]["0.0"],
            "phase_slope_through_zero": phase_score[name]["slope_through_zero"],
            "phase_direction": phase_score[name]["direction"]["0.0"],
        }
    return {
        "targets": targets,
        "normalized_stability_score": float(np.mean(stability)),
        "nominal_score": nominal_score,
        "phase_score": phase_score,
        "scope": SCOPE,
    }


def preservation_gate(
    reference_nominal_score,
    candidate_nominal_score,
    tolerance=0.05,
    min_amplitude_fraction=0.9,
    direction_slack=0.03,
):
    """Require nominal score metrics to preserve error, response, and direction."""
    reasons = []
    for name in ("Cd", "Cl"):
        try:
            reference = reference_nominal_score[name]
            candidate = candidate_nominal_score[name]
            ref_mae, cand_mae = float(reference["mae"]), float(candidate["mae"])
            ref_slope = reference["slope_through_zero"]
            cand_slope = candidate["slope_through_zero"]
            ref_direction = reference["direction"]["0.0"]["accuracy"]
            cand_direction = candidate["direction"]["0.0"]["accuracy"]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid score_delta-style metrics for {name}") from exc
        values = [ref_mae, cand_mae]
        if ref_slope is not None:
            values.append(float(ref_slope))
        if cand_slope is not None:
            values.append(float(cand_slope))
        if ref_direction is not None:
            values.append(float(ref_direction))
        if cand_direction is not None:
            values.append(float(cand_direction))
        if not np.isfinite(values).all():
            raise ValueError(f"non-finite score_delta-style metrics for {name}")
        if cand_mae > ref_mae * (1 + tolerance):
            reasons.append(f"{name} nominal MAE exceeds reference tolerance")
        if (
            ref_slope is not None
            and float(ref_slope) > 0
            and (
                cand_slope is None
                or float(cand_slope) < min_amplitude_fraction * float(ref_slope)
            )
        ):
            reasons.append(f"{name} positive nominal slope loses too much amplitude")
        if ref_direction is not None and (
            cand_direction is None
            or float(cand_direction) < float(ref_direction) - direction_slack
        ):
            reasons.append(f"{name} nominal direction falls below reference allowance")
    return {"pass": not reasons, "reasons": reasons}
