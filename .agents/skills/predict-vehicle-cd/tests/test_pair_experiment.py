import sys
from pathlib import Path

import numpy as np
import pytest
import torch

SKILL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL / "scripts"))
sys.path.insert(0, str(SKILL.parent / "predict-aircraft-coefficients/scripts"))

from vehicle_cd_learning.pair_evaluation import delta_metrics
from vehicle_cd_learning.pair_experiment import pair_loss, pair_schedule


def test_paired_endpoints_use_only_training_same_condition_rim_and_are_replayable():
    cases = [
        {"source": name, "scene": 1, "split": "train"}
        for name in [
            "drivaerrim",
            "builtin_assumed",
            "drivaerml",
            "drivaerrim",
            "drivaerrim",
        ]
    ]
    left, right = pair_schedule(cases, 20, 42)
    again = pair_schedule(cases, 20, 42)
    np.testing.assert_array_equal(left, again[0])
    np.testing.assert_array_equal(right, again[1])
    for a, b in zip(left.ravel(), right.ravel()):
        if cases[a]["source"] == "drivaerrim":
            assert a != b and cases[b]["source"] == "drivaerrim"
        else:
            assert a == b


def test_delta_gradient_reaches_both_endpoints_and_ignores_missing_cl():
    raw = torch.tensor([[0.1, 0.2], [0.2, 0.3], [0.2, 0.4], [0.2, 0.3]])
    targets = torch.tensor(
        [[0.0, 0.1], [0.2, float("nan")], [0.4, 0.2], [0.2, float("nan")]]
    )
    gradients = []
    for weight in (0, 1):
        pred = raw.clone().requires_grad_()
        loss, _ = pair_loss(
            pred,
            targets,
            torch.tensor([True, False]),
            torch.ones(2),
            torch.tensor([0.1, 0.2]),
            weight,
        )
        loss.backward()
        assert torch.isfinite(pred.grad).all()
        assert pred.grad[1, 1] == pred.grad[3, 1] == 0
        gradients.append(pred.grad)
    delta = gradients[1] - gradients[0]
    torch.testing.assert_close(delta[0], -delta[2])
    assert (delta[0].abs() > 0).all()
    assert (delta[1] == 0).all() and (delta[3] == 0).all()


def test_fixed_anchor_metrics_separate_absolute_bias_and_design_change_error():
    result = delta_metrics(
        np.array([[0.2416, -0.1305]]),
        np.array([[0.251, -0.138]]),
        np.array([0.24, -0.1326]),
        np.array([0.25, -0.14]),
    )
    assert result["delta_mae"] == pytest.approx([0.0006, 0.0001])
    assert result["anchored_mae"] == pytest.approx(result["delta_mae"])
    assert result["absolute_mae"] == pytest.approx([0.0094, 0.0075])
    assert result["direction_accuracy_nonzero"] == [1.0, 1.0]
    assert result["cd_delta_bins"][1]["count"] == 1
