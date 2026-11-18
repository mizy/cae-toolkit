import json
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

from vehicle_cd_learning import joint_sampling_evaluation as evaluation
from vehicle_cd_learning.model import VIEWS


def test_absolute_metrics_mask_missing_labels_and_keep_source_denominators():
    truth = np.array([[1.0, np.nan], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0]])
    prediction = np.array([[1.1, 100.0], [2.2, 4.4], [3.3, 6.6], [4.4, 8.5]])
    phases = np.stack((prediction + 0.1, prediction - 0.1))
    score = evaluation.absolute_scores(
        truth, prediction, ["builtin_assumed", "cocoanet", "cocoanet", "hilift"], phases
    )
    assert score["builtin_assumed"]["targets"]["Cl"]["label_count"] == 0
    assert score["builtin_assumed"]["targets"]["Cl"]["mae"] is None
    assert score["aircraft"]["case_count"] == 3
    assert score["vehicle"]["case_count"] == 1
    assert score["all"]["targets"]["Cl"]["label_count"] == 3
    assert score["all"]["targets"]["Cl"]["mae"] == pytest.approx(0.5)
    assert score["all"]["targets"]["Cl"]["sampling_drift_mae"] == pytest.approx(0.1)


def test_absolute_guard_rejects_source_regression_hidden_by_pooled_improvement():
    truth = np.ones((2, 2))
    reference = {
        "metrics": evaluation.absolute_scores(truth, [[2, 2], [11, 11]], ["a", "b"])
    }
    improved_pool = {
        "metrics": evaluation.absolute_scores(truth, [[3, 3], [1, 1]], ["a", "b"])
    }
    gate = evaluation.absolute_guard(reference, improved_pool, 0.05)
    assert not gate["pass"]
    assert len(gate["reasons"]) == 2
    assert all(reason.startswith("a/") for reason in gate["reasons"])


def test_full_evaluation_consumes_only_requested_split_and_exports_physical_arrays(
    monkeypatch, tmp_path
):
    reads = []

    class Data:
        def __init__(self):
            self.cases = [
                {
                    "case_id": str(i),
                    "split": "val" if i < 2 else "test",
                    "source": "cocoanet",
                }
                for i in range(3)
            ]
            self.targets = torch.tensor(
                [[10.0, 20.0], [12.0, float("nan")], [100, 200]]
            )

    def batch(data, indices, device):
        reads.extend(indices)
        image = torch.ones((len(indices), 2, 6, 6), dtype=torch.float32)
        return (
            {name: image for name in VIEWS},
            torch.zeros(len(indices), 36),
            torch.zeros(len(indices), 2),
        )

    class Model(torch.nn.Module):
        def encode(self, views):
            return views

    class Readout(torch.nn.Module):
        def forward(self, pooled, params):
            return pooled["front"].mean((2, 3))

    monkeypatch.setattr(evaluation.experiment, "batch_cases", batch)
    path = tmp_path / "arrays.npz"
    result = evaluation.evaluate_full(
        Model(),
        Readout(),
        Data(),
        {"normalization": {"target_mean": [10, 20], "target_std": [2, 3]}},
        "val",
        torch.device("cpu"),
        evaluation.FIXED_OFFSETS,
        path,
    )
    assert reads == [0, 1]
    assert result["metrics"]["all"]["targets"]["Cl"]["label_count"] == 1
    with np.load(path) as arrays:
        assert arrays["case_ids"].tolist() == ["0", "1"]
        np.testing.assert_allclose(arrays["prediction"], [[12, 23], [12, 23]])
        assert arrays["shifted_prediction"].shape == (3, 2, 2)
        assert np.isnan(arrays["truth"][1, 1])
    assert result["arrays_sha256"] == evaluation.sha256(path)


def test_freeze_plan_does_not_score_or_access_test_images(monkeypatch, tmp_path):
    phase = tmp_path / "phase"
    phase.mkdir()
    val_ids = [str(i) for i in range(12)]
    (phase / "plan.json").write_text(json.dumps({"selection": {"val": val_ids}}))
    p = {"sampling_parent": str(tmp_path), "epochs": 6, "arms": ["control", "online"]}
    (tmp_path / "protocol.json").write_text(json.dumps(p))
    cases = [{"case_id": i, "split": "val"} for i in val_ids] + [
        {"case_id": "test", "split": "test"}
    ]
    monkeypatch.setattr(evaluation.experiment, "load", lambda root: (p, cases))
    monkeypatch.setattr(
        evaluation.rim_experiment,
        "load",
        lambda root: (
            {
                "phase_root": str(phase),
                "phase_ids": ["a", "b", "c", "d"],
                "anchors": ["anchor"],
            },
            cases,
        ),
    )
    monkeypatch.setattr(
        evaluation,
        "score_checkpoint",
        lambda *args: pytest.fail("Preparation scored a candidate"),
    )
    plan = evaluation.freeze_val_plan(tmp_path)
    assert plan["full_validation_case_ids"] == val_ids
    assert plan == evaluation.freeze_val_plan(tmp_path)
    assert "Adopt only an all-gates" in plan["selection"]
    changed = json.loads((tmp_path / "validation-plan.json").read_text())
    changed["absolute_mae_tolerance"] = 1
    (tmp_path / "validation-plan.json").write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="Frozen validation plan differs"):
        evaluation.freeze_val_plan(tmp_path)


def _candidate(arm, epoch, score, preserved):
    return {
        "arm": arm,
        "epoch": epoch,
        "checkpoint": f"/{arm}-{epoch}.pt",
        "checkpoint_sha256": f"{arm}-{epoch}",
        "gates": {
            "validation_passed": False,
            "nominal_preserved": preserved,
            "phase_preserved": False,
            "stability_passed": False,
            "score": score,
            "reasons": ["failed"],
        },
    }


def test_failed_low_stability_attempt_remains_diagnostic_and_does_not_replace_b():
    baseline = _candidate("original_B", None, 1, True)
    candidates = [
        _candidate(arm, epoch, 0.01 if epoch else 1, False)
        for arm in ("control", "online")
        for epoch in (0, 1)
    ]
    validation = {"baseline": baseline, "candidates": candidates}
    rows = evaluation.test_representatives(
        validation, {"selected_checkpoint": baseline["checkpoint"]}
    )
    assert rows["selected"] == baseline
    assert rows["online"] == baseline
    assert rows["control"] == baseline
    assert rows["online-diagnostic"]["epoch"] == 1
    assert rows["control-diagnostic"]["epoch"] == 1
    assert rows["control-matched"]["epoch"] == rows["online-diagnostic"]["epoch"]
    assert rows["control-matched"]["arm"] == "control"
    assert not rows["online-diagnostic"]["gates"]["nominal_preserved"]


def test_select_retains_b_when_provisional_candidate_fails_stability(
    monkeypatch, tmp_path
):
    p = {"sampling_parent": str(tmp_path), "delta_std": [1, 1]}
    (tmp_path / "protocol.json").write_text(json.dumps(p))
    (tmp_path / "validation-plan.json").write_text(
        json.dumps(
            {
                "source_sha256": {},
                "code_sha256": {},
                "full_validation_case_ids": [],
                "validation_case_ids": ["val"],
                "phase_ids": ["phase"],
                "absolute_mae_tolerance": 0.05,
            }
        )
    )
    paths = []
    for arm, value in (("original_B", 1.0), ("control", 0.2), ("online", 0.1)):
        path = tmp_path / ("initial.pt" if arm == "original_B" else arm + ".pt")
        torch.save(
            {
                "arm": arm,
                "epoch": 1,
                "protocol_sha256": evaluation.sha256(tmp_path / "protocol.json"),
                "value": value,
            },
            path,
        )
        if arm != "original_B":
            paths.append((arm, 1, path))

    def models(path, device):
        layer = torch.nn.Linear(1, 1, bias=False).to(device)
        with torch.no_grad():
            layer.weight.fill_(torch.load(path, weights_only=True)["value"])
        return layer, torch.nn.Linear(1, 1).to(device)

    def gates(reference, candidate, plan, scale):
        return {
            "validation_passed": False,
            "nominal_preserved": True,
            "phase_preserved": True,
            "stability_passed": False,
            "score": candidate["value"],
            "reasons": ["stability failed"],
        }

    absolute = {"metrics": evaluation.absolute_scores([[1, 1]], [[2, 2]], ["source"])}
    monkeypatch.setattr(evaluation.experiment, "load", lambda root: (p, []))
    monkeypatch.setattr(evaluation.experiment, "load_checkpoint", models)
    monkeypatch.setattr(evaluation.experiment, "FullCases", lambda *args: None)
    monkeypatch.setattr(evaluation.rim_experiment, "load", lambda root: ({}, []))
    monkeypatch.setattr(evaluation.rim_experiment, "SamplingCases", lambda *args: None)
    monkeypatch.setattr(evaluation, "completed_checkpoints", lambda *args: paths)
    monkeypatch.setattr(
        evaluation,
        "score_checkpoint",
        lambda model, *args: {"value": float(model.weight.detach().cpu()[0, 0])},
    )
    monkeypatch.setattr(evaluation, "evaluate_full", lambda *args: absolute)
    monkeypatch.setattr(evaluation, "candidate_gates", gates)
    result = evaluation.select(tmp_path)
    assert result["chosen_arm"] == "original_B"
    assert result["baseline_fallback"]
    assert not result["validation_passed"]
    assert result["provisional_recommendation"]["chosen_arm"] == "online"
    assert (tmp_path / "selection.json").is_file()
    assert not (tmp_path / "test-plan.json").exists()
