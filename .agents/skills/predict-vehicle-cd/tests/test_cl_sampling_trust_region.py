import copy
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[2] / "predict-aircraft-coefficients/scripts"),
)

from vehicle_cd_learning import cl_sampling_trust_region as trust
from vehicle_cd_learning.phase_metrics import sampling_scores


@pytest.mark.parametrize("radius", [0.05, 0.1, 0.25])
def test_ill_conditioned_fit_cannot_escape_deployed_float32_radius(radius):
    design = np.array([[1, 1e-5], [1, -1e-5], [-1, 1e-5], [-1, -1e-5]])
    truth = np.array([2, 0, 0, -2.0])
    fidelity = np.array([[1, 0], [-1, 0.0]])
    nuisance = np.array([[0, 1e-5], [0, -1e-5]])
    original = np.array([1, 0.1])
    row, report = trust.fit_trust_region(
        design, truth, fidelity, nuisance, original, 100, 0.01, radius
    )
    assert report["unconstrained_displacement_over_original_norm"] > 10
    assert report["lagrange_multiplier"] > 0
    assert np.linalg.norm(row - original) <= radius * np.linalg.norm(original)
    assert np.linalg.norm(row) <= (1 + radius) * np.linalg.norm(original)
    np.testing.assert_allclose(fidelity @ row, fidelity @ original, atol=1e-5)
    np.testing.assert_array_equal(row, row.astype(np.float32).astype(np.float64))


def test_one_dimensional_solution_reaches_known_boundary():
    row, report = trust.fit_trust_region(
        np.ones((2, 1)),
        np.array([10, 10]),
        np.zeros((1, 1)),
        np.zeros((1, 1)),
        np.array([1.0]),
        1,
        1,
        0.1,
    )
    assert row[0] == pytest.approx(1.1, abs=2e-6)
    assert report["lagrange_multiplier"] > 0


def test_one_worsened_phase_is_rejected_despite_better_cross_phase_average():
    truth = np.array([[-0.3, 0.2], [-0.1, -0.3], [0.1, 0.4], [0.3, -0.2]])
    prediction = truth * 0.9

    def phase(self_noise):
        self_prediction = np.zeros((1, 3, 2))
        self_prediction[:, :, 1] = self_noise
        score = sampling_scores(
            truth, prediction, prediction[None], self_prediction, [1, 1]
        )
        return {name: copy.deepcopy(score) for name in trust.repair.PATTERNS}

    reference = {"phase-a": phase(1), "phase-b": phase(1)}
    candidate = {"phase-a": phase(0), "phase-b": phase(1.06)}
    gate = trust.phase_guards(
        reference,
        candidate,
        {"tolerance": 0.05, "min_amplitude_fraction": 0.9, "direction_slack": 0.03},
    )
    assert not gate["pass"]
    assert any(
        "phase-b/common/Cl/self_phase_mae" in reason for reason in gate["reasons"]
    )
    assert not any("phase-a" in reason for reason in gate["reasons"])
