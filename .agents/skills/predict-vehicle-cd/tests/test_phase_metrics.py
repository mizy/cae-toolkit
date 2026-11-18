import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from vehicle_cd_learning.delta_metrics import score_delta
from vehicle_cd_learning.phase_metrics import preservation_gate, sampling_scores


def test_zero_model_metrics_expose_response_collapse():
    truth = np.array([[-1.0, -2.0], [1.0, 2.0]])
    result = sampling_scores(
        truth, np.zeros_like(truth), np.zeros((2, 2, 2)), np.zeros((2, 3, 2)), [1, 2]
    )
    assert result["targets"]["Cd"]["nominal_delta_mae"] == 1
    assert result["targets"]["Cd"]["nominal_slope_through_zero"] == 0
    assert result["targets"]["Cd"]["nominal_direction"]["accuracy"] == 0
    assert result["normalized_stability_score"] == 0
    assert "not independent geometry samples" in result["scope"]


def test_phase_perturbations_report_self_and_common_drift():
    truth = np.array([[1.0, 2.0], [-1.0, -2.0]])
    nominal = truth.copy()
    phase = np.stack((nominal + [0.2, -0.4], nominal + [0.0, 0.2]))
    self_phase = np.array([[[0.1, -0.2]], [[-0.3, 0.0]]])
    result = sampling_scores(truth, nominal, phase, self_phase, [1, 2])
    cd, cl = result["targets"]["Cd"], result["targets"]["Cl"]
    assert cd["self_phase_mae"] == pytest.approx(0.2)
    assert cd["common_phase_drift_mae"] == pytest.approx(0.1)
    assert cd["phase_delta_mae"] == pytest.approx(0.1)
    assert cl["common_phase_drift_mae"] == pytest.approx(0.3)
    assert cl["phase_delta_mae"] == pytest.approx(0.3)


def test_shape_and_finiteness_checks():
    truth = np.ones((2, 2))
    with pytest.raises(ValueError):
        sampling_scores(
            truth, np.ones((1, 2)), np.ones((1, 2, 2)), np.ones((1, 1, 2)), [1, 1]
        )
    with pytest.raises(ValueError):
        sampling_scores(truth, truth, np.ones((1, 3, 2)), np.ones((1, 1, 2)), [1, 1])
    with pytest.raises(ValueError):
        sampling_scores(truth, truth, np.ones((1, 2, 2)), np.ones((2, 1, 2)), [1, 1])
    with pytest.raises(ValueError):
        sampling_scores(
            truth, truth, np.full((1, 2, 2), np.nan), np.ones((1, 1, 2)), [1, 1]
        )


def test_preservation_gate_checks_mae_amplitude_and_direction():
    truth = np.array([[1.0, 2.0], [-1.0, -2.0]])
    reference = score_delta(truth, truth)
    assert preservation_gate(reference, reference)["pass"]
    candidate = score_delta(truth, truth * 0.8)
    gate = preservation_gate(reference, candidate)
    assert not gate["pass"]
    assert any("slope" in reason for reason in gate["reasons"])
