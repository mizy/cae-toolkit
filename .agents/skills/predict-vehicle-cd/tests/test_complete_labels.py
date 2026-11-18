import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

SKILL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL / "scripts"))
sys.path.insert(0, str(SKILL.parent / "predict-aircraft-coefficients/scripts"))

from aircraft_learning.scene_experiment import SceneCases, normalization
from vehicle_cd_learning import complete_labels as labels
from vehicle_cd_learning import dimensionless as dim
from vehicle_cd_learning.model import VIEWS


@pytest.mark.parametrize(
    "source,mu", [("drivaerrim", 1.813e-5), ("builtin_assumed", 1.81e-5)]
)
def test_nominal_re_has_explicit_assumptions_and_body_length(source, mu):
    case = {
        "source": source,
        "parameters": [None] * 18,
        "conditions": {"velocity_m_s": 30},
        "source_evidence": {},
    }
    labels.fill_re(case, 4.5)
    expected = 1.204 * 30 * 4.5 / mu
    assert case["conditions"]["re_body"] == expected
    assert case["parameters"][2] == np.log(expected)
    evidence = case["source_evidence"]["re_completion"]
    assert (
        evidence["status"] == "estimated" and not evidence["source_conditions_verified"]
    )
    assert evidence["audit_inputs"]["length_m"] == 4.5
    before = json.dumps(case)
    labels.fill_re(case, 100)
    assert json.dumps(case) == before


def test_no_silent_assumption_for_other_sources_or_invalid_geometry():
    case = {
        "source": "new_source",
        "parameters": [None] * 18,
        "conditions": {"velocity_m_s": 30},
        "source_evidence": {},
    }
    with pytest.raises(ValueError, match="No approved"):
        labels.fill_re(case, 4.5)
    case["source"] = "drivaerrim"
    for length in (0, -1, np.nan, np.inf):
        with pytest.raises(ValueError, match="finite and positive"):
            labels.fill_re(case, length)


def test_completion_keeps_truth_mask_and_training_ignores_pseudo_labels(
    tmp_path, monkeypatch
):
    source = tmp_path / "source"
    source.mkdir()
    cases = []
    for i, name in enumerate(("drivaerml", "drivaerrim", "builtin_assumed")):
        path = source / f"{i}.npz"
        np.savez_compressed(path, **{k: np.zeros((2, 2, 2), np.float32) for k in VIEWS})
        values = [0.4, 0.3, np.log(3e6) if i == 0 else None] + [None] * 15
        cases.append(
            {
                "case_id": f"{name}__car",
                "geometry_id": f"{name}__car",
                "geometry_hash": str(i),
                "source": name,
                "scene": 1,
                "split": "train",
                "parameters": values,
                "conditions": {
                    "velocity_m_s": 30,
                    **({"re_body": 3e6} if i == 0 else {}),
                },
                "source_evidence": {},
                "targets": [0.25 + i * 0.01, None if i == 2 else 0.1 + i * 0.01],
                "views": path.name,
                "views_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    data = {
        "input_contract": dim.CONTRACT,
        "parameter_names": dim.PARAMETERS,
        "source_counts": {c["source"]: {"train": 1} for c in cases},
        "cases": cases,
    }
    manifest = source / "dataset.json"
    manifest.write_text(json.dumps(data))
    original = manifest.read_bytes()
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps([{"id": "car", "dimensions": {"length_m": 4.5}}]))
    rim = tmp_path / "rim"
    rim.mkdir()
    (rim / "dataset.json").write_text(
        json.dumps({"geometry": {"car": {"dimensions": [4.6, 2, 1.4]}}})
    )
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"frozen teacher fixture")
    monkeypatch.setattr(labels, "load_model", lambda _: (None, {}))
    monkeypatch.setattr(
        labels,
        "predict_views",
        lambda m, ck, v, p: np.array([0.25, 0.02 if p[2] is None else 0.03]),
    )
    output = tmp_path / "completed"
    result = labels.complete(source, audit, rim, checkpoint, output)
    completed = dim.read_prepared(output / "prepared")
    assert manifest.read_bytes() == original
    assert result["remaining_missing_re"] == 0 and result["estimated_cl"] == 1
    assert result["observed_cl"] == 2
    for before, after in zip(cases, completed["cases"]):
        assert before["targets"] == after["targets"]
        assert before["geometry_hash"] == after["geometry_hash"]
    last = completed["cases"][-1]
    estimate = last["target_estimates"]["cl"]
    assert estimate["value"] == 0.03 and estimate["before_re_completion"] == 0.02
    assert (
        estimate["eligible_for_supervised_loss"] is False
        and estimate["uncertainty"] is None
    )
    dataset = SceneCases(
        output / "prepared", completed["cases"], normalization(completed["cases"])
    )
    assert np.isnan(dataset[2][2].numpy()[1])
    assert (
        ",model_pseudo_label,False," in (output / "builtin-estimates.csv").read_text()
    )
    with pytest.raises(FileExistsError):
        labels.complete(source, audit, rim, checkpoint, output)
