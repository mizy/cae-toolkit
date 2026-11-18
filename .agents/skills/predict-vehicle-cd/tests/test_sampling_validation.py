import copy
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from vehicle_cd_learning import sampling_validation as validation
from vehicle_cd_learning.delta_metrics import score_delta
from vehicle_cd_learning.model import VIEWS
from vehicle_cd_learning.phase_metrics import sampling_scores


def plan():
    return {
        "anchors": ["anchor"],
        "preservation": {"tolerance": 0.05, "min_amplitude_fraction": 0.9, "direction_slack": 0.03},
        "stability": {"mae_over_delta_std": 0.05, "p95_over_delta_std": 0.1},
    }


def metrics(fraction=0.9):
    truth = np.array([[-0.3, 0.2], [-0.1, -0.3], [0.1, 0.4], [0.3, -0.2]])
    prediction = truth * fraction
    score = score_delta(truth, prediction)
    sampling = sampling_scores(truth, prediction, np.stack([prediction] * 4), np.zeros((4, 3, 2)), [1.0, 1.0])
    return {
        "nominal_anchors": {"pooled_correlated_pairs": score, "by_anchor": {"anchor": score}},
        "patterns": {name: copy.deepcopy(sampling) for name in validation.PATTERNS},
    }


def test_zero_output_collapse_fails_nominal_and_phase_amplitude_gates():
    reference, collapsed = metrics(), metrics(0)
    gates = validation.candidate_gates(reference, collapsed, plan(), [1, 1])
    assert gates["stability_passed"] and gates["score"] == 0
    assert not gates["nominal_preserved"] and not gates["validation_passed"]
    assert any("amplitude" in reason for reason in gates["reasons"])
    # A model preserving nominal output also fails if its shifted output collapses.
    shifted = metrics()
    for pattern in shifted["patterns"].values():
        pattern["phase_score"] = collapsed["patterns"]["common"]["phase_score"]
    gates = validation.candidate_gates(reference, shifted, plan(), [1, 1])
    assert gates["nominal_preserved"] and not gates["phase_preserved"]
    assert all(not gate["pass"] for gate in gates["phase_gates"].values())


@pytest.mark.parametrize(
    "metric,value",
    [
        ("self_phase_mae", 0.051),
        ("self_phase_p95_abs", 0.101),
        ("common_phase_drift_mae", 0.051),
        ("common_phase_drift_p95_abs", 0.101),
    ],
)
def test_hard_stability_gate_checks_each_pattern_and_both_targets(metric, value):
    candidate = metrics()
    candidate["patterns"]["reference_shifted"]["targets"]["Cl"][metric] = value
    gates = validation.candidate_gates(metrics(), candidate, plan(), [1, 1])
    assert gates["nominal_preserved"] and gates["phase_preserved"]
    assert not gates["stability_passed"] and not gates["validation_passed"]
    assert any(f"reference_shifted/Cl/{metric}" in reason for reason in gates["reasons"])


def test_full_anchor_and_subset_nominal_metrics_both_must_preserve():
    candidate = metrics()
    candidate["nominal_anchors"]["by_anchor"]["anchor"] = score_delta([[0.1, 0.2]], [[0, 0]])
    gates = validation.candidate_gates(metrics(), candidate, plan(), [1, 1])
    assert not gates["nominal_preserved"]
    candidate = metrics()
    candidate["patterns"]["common"]["nominal_score"] = score_delta([[0.1, 0.2]], [[0, 0]])
    gates = validation.candidate_gates(metrics(), candidate, plan(), [1, 1])
    assert not gates["nominal_preserved"]


def candidate(name, score, nominal=True, phase=True, stable=True):
    return {
        "checkpoint": "/" + name + ".pt",
        "checkpoint_sha256": name + "-sha",
        "arm": name,
        "epoch": 3,
        "gates": {
            "nominal_preserved": nominal,
            "phase_preserved": phase,
            "stability_passed": stable,
            "validation_passed": nominal and phase and stable,
            "score": score,
            "reasons": [] if nominal and phase and stable else ["failed gate"],
        },
    }


def test_selection_prefers_pass_then_preservation_and_marks_provisional_failure():
    baseline = candidate("original_B", 0.4, phase=False, stable=False)
    collapsed = candidate("collapse", 0, nominal=False)
    passed = candidate("pass", 0.1)
    preserved = candidate("preserved", 0.01, stable=False)
    selected = validation.choose_candidate([collapsed, passed, preserved], baseline)
    assert selected["chosen_arm"] == "pass" and selected["validation_passed"]
    assert selected["selected_model_sha256"] == "pass-sha"
    selected = validation.choose_candidate([collapsed, preserved], baseline)
    assert selected["chosen_arm"] == "preserved" and not selected["validation_passed"]
    assert selected["selection_tier"] == "preservation_only"
    selected = validation.choose_candidate([collapsed], baseline)
    assert selected["baseline_fallback"] and not selected["validation_passed"]
    assert selected["selection_tier"] == "nominal_only_provisional"
    assert selected["reasons"] == ["failed gate"]


class Difference(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def encode(self, views):
        return {"latent": views["front"][:, :, 0, 0]}

    def predict_pair(self, a, b, parameters):
        assert parameters.shape[1] == 36
        return (b["latent"] - a["latent"]) * self.weight


def test_scores_encode_each_nominal_phase_once_and_never_read_test_images():
    cases = [
        {
            "case_id": str(i),
            "source": "drivaerrim",
            "split": "train" if i < 3 else "val" if i < 6 else "test",
            "targets": [0.1 * i, 0.2 * i],
        }
        for i in range(7)
    ]
    reads = []

    class Data:
        def __init__(self):
            self.index = {str(i): i for i in range(7)}
            self.parameters = torch.zeros(7, 36)

        def get(self, index, phase=None):
            assert index < 6
            reads.append((index, phase))
            offset = 0 if phase is None else 0.01 if phase == "positive" else -0.01
            value = torch.tensor(cases[index]["targets"]) + offset
            return (
                {name: value[:, None, None] for name in VIEWS},
                self.parameters[index],
                torch.tensor(cases[index]["targets"]),
            )

    result = validation.score_checkpoint(
        Difference(),
        Data(),
        {"anchors": ["0", "1", "2"], "delta_std": [1.0, 1.0]},
        cases,
        ["3", "4"],
        ["positive", "negative"],
    )
    assert len(reads) == len(set(reads)) == result["encoded_image_count"] == 10
    assert result["nominal_anchors"]["query_count"] == 3
    assert result["phase_scoring"]["pairs"] == [["3", "4"]]
    arrays = result["phase_scoring"]
    np.testing.assert_allclose(arrays["truth_pairs"], [[0.1, 0.2]], atol=1e-7)
    np.testing.assert_allclose(arrays["common_pred"], [arrays["nominal_pred"]] * 2, atol=1e-7)
    nominal = np.array(arrays["nominal_pred"])
    query_drift = np.array(arrays["query_shifted_pred"]) - nominal
    reference_drift = np.array(arrays["reference_shifted_pred"]) - nominal
    np.testing.assert_allclose(query_drift, -reference_drift, atol=1e-7)
    np.testing.assert_allclose(query_drift[:, 0], [[0.01, 0.01], [-0.01, -0.01]], atol=1e-7)
    assert all(value == 1 for value in Counter(reads).values())


def test_selection_rejects_incomplete_training_before_any_scoring(tmp_path):
    arm = tmp_path / "control"
    arm.mkdir()
    (arm / "training.json").write_text(json.dumps({"optimizer_steps": 90, "history": [{"epoch": 1}]}))
    with pytest.raises(ValueError, match="All six epochs"):
        validation.completed_checkpoints(tmp_path, {"arms": {"control": {"phases": False}}})


def test_prepare_freezes_thresholds_and_code_without_scoring(monkeypatch, tmp_path):
    phase = tmp_path / "phases"
    phase.mkdir()
    ids = [str(i) for i in range(12)]
    (phase / "plan.json").write_text(json.dumps({"selection": {"val": ids}}))
    p = {"phase_root": str(phase), "phase_ids": ["p1", "p2", "p3", "p4"], "anchors": ["a"], "arms": {"control": {}}}
    (tmp_path / "protocol.json").write_text(json.dumps(p))
    monkeypatch.setattr(validation.experiment, "load", lambda root: (p, [{"case_id": i, "split": "val"} for i in ids]))
    frozen = validation.prepare(tmp_path)
    assert frozen["stability"] == {"mae_over_delta_std": 0.05, "p95_over_delta_std": 0.1}
    assert frozen["preservation"]["min_amplitude_fraction"] == 0.9
    assert any(path.endswith("phase_metrics.py") for path in frozen["code_sha256"])
    assert validation.prepare(tmp_path) == frozen
    changed = json.loads((tmp_path / "validation-plan.json").read_text())
    changed["stability"]["mae_over_delta_std"] = 0.2
    (tmp_path / "validation-plan.json").write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="Frozen validation plan differs"):
        validation.prepare(tmp_path)
