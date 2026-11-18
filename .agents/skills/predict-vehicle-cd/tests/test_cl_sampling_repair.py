import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[2] / "predict-aircraft-coefficients/scripts"),
)

from vehicle_cd_learning import cl_sampling_repair as repair
from vehicle_cd_learning.delta_metrics import score_delta
from vehicle_cd_learning.model import VIEWS
from vehicle_cd_learning.paired_delta import PairedDelta
from vehicle_cd_learning.phase_metrics import sampling_scores


def test_capture_reproduces_original_pair_prediction_without_altering_weights():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    model = PairedDelta(8, 36).eval().requires_grad_(False)
    before = {n: t.clone() for n, t in model.state_dict().items()}

    def images():
        return {
            n: torch.cat((torch.rand(3, 1, 32, 32), torch.ones(3, 1, 32, 32)), dim=1)
            for n in VIEWS
        }

    with torch.inference_mode():
        left, right = model.encode(images()), model.encode(images())
        parameters = torch.randn(3, 36)
        z, error = repair.capture_features(model, left, right, parameters)
        expected = model.predict_pair(left, right, parameters)
        torch.testing.assert_close(
            z @ model.head[-1].weight.T, expected, atol=2e-7, rtol=2e-5
        )
    assert z.shape == (3, 64)
    assert error < 2e-7
    assert not model.head[-1]._forward_pre_hooks
    assert all(torch.equal(t, before[n]) for n, t in model.state_dict().items())


def test_separable_nuisance_removed_while_geometry_and_nominal_fidelity_preserved():
    design = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]], float)
    truth = design[:, 0]
    fidelity = np.array([[1, 0], [-1, 0]], float)
    nuisance = np.array([[0, 1], [0, -1]], float)
    original = np.array([1, 1.0])
    row, ridge = repair.fit_row(design, truth, fidelity, nuisance, original, 100, 100)
    assert ridge > 0
    assert np.mean(np.abs(nuisance @ row)) < 0.01
    np.testing.assert_allclose(fidelity @ row, fidelity @ original, atol=1e-6)
    np.testing.assert_allclose(row[0], 1, atol=1e-6)
    # Nominal-only fidelity must not preserve the original unit phase error.
    assert abs(row[1]) < 1e-5


def test_proof_rejects_any_non_cl_change_and_confirms_exact_cd_and_auxiliary():
    original = {
        "model_state": {"head.2.weight": torch.ones(2, 64), "cnn": torch.ones(3)},
        "absolute_state": {"weight": torch.ones(4)},
    }
    candidate = copy.deepcopy(original)
    candidate["model_state"]["head.2.weight"][1] *= 0.9
    proof = repair.changed_tensor_proof(original, candidate)
    assert proof["changed_parameter_count"] == 64
    assert proof["Cd_row_exact"] and proof["absolute_tensors_exact"]
    candidate["model_state"]["head.2.weight"][0, 0] += 1
    with pytest.raises(ValueError, match="Forbidden tensor change"):
        repair.changed_tensor_proof(original, candidate)
    candidate = copy.deepcopy(original)
    candidate["absolute_state"]["weight"][0] += 1
    with pytest.raises(ValueError, match="Forbidden tensor change"):
        repair.changed_tensor_proof(original, candidate)


def test_multireference_guards_reject_zero_cl_amplitude_despite_perfect_stability():
    truth = np.array([[-0.3, 0.2], [-0.1, -0.3], [0.1, 0.4], [0.3, -0.2]])

    def metrics(prediction):
        score = score_delta(truth, prediction)
        phase = sampling_scores(
            truth, prediction, np.stack([prediction] * 4), np.zeros((4, 3, 2)), [1, 1]
        )
        return {
            "nominal_anchors": {
                "pooled_correlated_pairs": score,
                "by_anchor": {"a": score, "b": score},
            },
            "patterns": {name: copy.deepcopy(phase) for name in repair.PATTERNS},
        }

    original = metrics(truth * 0.9)
    collapsed = truth * 0.9
    collapsed[:, 1] = 0
    plan = {
        "anchors": ["a", "b"],
        "preservation": {
            "tolerance": 0.05,
            "min_amplitude_fraction": 0.9,
            "direction_slack": 0.03,
        },
        "stability": {"mae_over_delta_std": 0.05, "p95_over_delta_std": 0.1},
    }
    gates = repair.repair_gates(original, metrics(collapsed), plan, [1, 1], True)
    assert gates["Cl_sampling_score"] == 0
    assert not gates["response_preserved"]
    assert all(
        not gate["pass"]
        for gate in gates["strict_inherited_gates"]["nominal_anchor_gates"].values()
    )
    damaged_anchor = copy.deepcopy(original)
    damaged_anchor["nominal_anchors"]["by_anchor"]["b"] = metrics(collapsed)[
        "nominal_anchors"
    ]["by_anchor"]["b"]
    assert not repair.repair_gates(original, damaged_anchor, plan, [1, 1], True)[
        "response_preserved"
    ]
