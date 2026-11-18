import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from vehicle_cd_learning.delta_metrics import (
    compare_radiant,
    normalized_delta_mae,
    score_anchors,
    score_delta,
    zero_change_controls,
)


def test_zero_prediction_exposes_collapsed_response():
    truth = np.array([[-0.002, -0.1], [0.001, 0.2], [0, 0]])
    result = score_delta(truth, np.zeros_like(truth))
    cd = result["Cd"]
    assert cd["mae"] == cd["zero_change_baseline_mae"]
    assert cd["slope_through_zero"] == cd["std_ratio"] == 0
    assert cd["direction"]["0.0"] == {"count": 2, "accuracy": 0.0}
    assert cd["correlation"] is None
    json.dumps(result, allow_nan=False)


def test_reverse_sign_and_amplitude():
    truth = np.array([[-0.002, -0.1], [0.001, 0.2]])
    for target in score_delta(truth, -truth).values():
        if isinstance(target, dict) and "slope_through_zero" in target:
            assert target["slope_through_zero"] == -1
            assert target["std_ratio"] == 1
            assert target["correlation"] == pytest.approx(-1)
            assert target["direction"]["0.0"]["accuracy"] == 0
    assert normalized_delta_mae(truth, truth, [0.01, 0.1]) == 0
    assert normalized_delta_mae(
        truth, np.zeros_like(truth), [0.01, 0.1]
    ) == pytest.approx(0.825)


def test_direction_thresholds_strictly_exclude_boundary_and_zero():
    values = np.array([0, 0.0001, -0.0005, 0.001, -0.002])
    truth = np.column_stack([values, values])
    result = score_delta(truth, truth)["Cd"]["direction"]
    assert [result[str(t)]["count"] for t in (0.0, 0.0001, 0.0005, 0.001)] == [
        4,
        3,
        2,
        1,
    ]
    control = zero_change_controls([[0, 0], [0.01, -0.02]])
    assert control["max_abs"] == [0.01, 0.02]
    assert control["scores"]["Cd"]["direction"]["0.0"]["accuracy"] is None
    json.dumps(control, allow_nan=False)


def test_direction_threshold_tolerates_rounded_coefficient_subtraction():
    at_threshold = 0.2401 - 0.24
    just_above = 0.24010000001 - 0.24
    numerical_zero = 0.24000000000000002 - 0.24
    assert at_threshold > 0.0001  # Binary subtraction lands just above the decimal.
    assert 0 < numerical_zero < 1e-12
    truth = np.array([[at_threshold, numerical_zero], [just_above, at_threshold]])

    result = score_delta(truth, truth)

    assert result["Cd"]["direction"]["0.0"]["count"] == 2
    assert result["Cd"]["direction"]["0.0001"]["count"] == 1
    assert result["Cl"]["direction"]["0.0"]["count"] == 1
    assert result["Cl"]["direction"]["0.0001"]["count"] == 0


def comparison_fixture():
    cases = [
        {"case_id": a, "source": "drivaerrim", "split": "train", "targets": [v, 0.1]}
        for a, v in (("a", 0.24), ("b", 0.25), ("c", 0.26))
    ]
    cases += [
        {"case_id": q, "source": "drivaerrim", "split": "test", "targets": [v, 0.2]}
        for q, v in (("q1", 0.23), ("q2", 0.27))
    ]
    predictions, external = {}, {"anchors": {}, "rows_by_anchor": {}}
    for case in cases[:3]:
        anchor, base = case["case_id"], case["targets"][0]
        predictions[anchor], external["rows_by_anchor"][anchor] = {}, []
        external["anchors"][anchor] = base
        for q, error in zip(cases[3:], (0.001, 0.003)):
            delta = q["targets"][0] - base
            predictions[anchor][q["case_id"]] = [delta + error, 0.1]
            external["rows_by_anchor"][anchor].append(
                {
                    "case_id": q["case_id"],
                    "truth": q["targets"][0],
                    "prediction": q["targets"][0],
                    "delta_prediction": delta,
                    "delta_truth": delta,
                }
            )
    return cases, ["a", "b", "c"], predictions, external


def test_joint_anchor_bootstrap_and_exact_query_matching():
    cases, anchors, predictions, external = comparison_fixture()
    report = compare_radiant(cases, anchors, predictions, external)
    assert report["query_count"] == 2
    assert report["correlated_pair_count"] == 6
    assert report["our_minus_radiant_cd_mae"] == pytest.approx(0.002)
    assert report["our_minus_radiant_cd_mae_95_interval"] == pytest.approx(
        [0.001, 0.003]
    )
    assert report["Cl_supported"] is False
    for rows in external["rows_by_anchor"].values():
        rows.reverse()
    assert compare_radiant(cases, anchors, predictions, external) == report
    ours = score_anchors(cases, anchors, predictions, delta_std=[0.01, 0.1])
    assert ours["mean_anchors"]["Cd"]["mae"] == pytest.approx(0.002)
    assert ours["normalized_mae"] == pytest.approx(0.1)
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("field", ["case_id", "truth", "delta_truth", "prediction"])
def test_external_identity_guard(field):
    cases, anchors, predictions, external = comparison_fixture()
    bad = copy.deepcopy(external)
    bad["rows_by_anchor"]["a"][0][field] = "other" if field == "case_id" else 1.0
    with pytest.raises(ValueError):
        compare_radiant(cases, anchors, predictions, bad)


def test_duplicate_query_and_invalid_scale_rejected():
    cases, anchors, predictions, external = comparison_fixture()
    external["rows_by_anchor"]["a"].append(external["rows_by_anchor"]["a"][0])
    with pytest.raises(ValueError):
        compare_radiant(cases, anchors, predictions, external)
    with pytest.raises(ValueError):
        normalized_delta_mae([[0, 0]], [[0, 0]], [0, 1])
