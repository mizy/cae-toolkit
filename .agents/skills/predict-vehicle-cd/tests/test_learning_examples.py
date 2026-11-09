from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from vehicle_cd_learning.validate_examples import image_arrays, stable_prediction
from vehicle_cd_similarity_lib.types import SideArtifacts


class FixedModel(torch.nn.Module):
    variant = "five"

    def forward(self, views, parameters):
        return parameters[:, :1]


class DriftingModel(torch.nn.Module):
    variant = "five"

    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, views, parameters):
        self.calls += 1
        return parameters[:, :1] + self.calls * 1e-3


def _case_inputs():
    views = {
        name: np.stack(
            (np.zeros((16, 16), dtype=np.float32), np.ones((16, 16), dtype=np.float32))
        )
        for name in ("front", "rear", "right", "top", "bottom")
    }
    checkpoint = {
        "parameter_mean": [0.0] * 5,
        "parameter_std": [1.0] * 5,
        "target_mean": [0.0],
        "target_std": [1.0],
        "target_names": ["cd"],
    }
    return views, checkpoint, np.asarray([0.2, 0.3, 0.4, 0.0, 1.0])


def test_example_validation_rejects_empty_masks_and_nonfinite_depth():
    views = {
        name: SideArtifacts(
            mask=np.zeros((16, 16), dtype=bool),
            depth=np.zeros((16, 16), dtype=np.float32),
            normal=np.zeros((16, 16, 3), dtype=np.float32),
        )
        for name in ("front", "rear", "right", "top", "bottom")
    }
    with pytest.raises(ValueError, match="mask"):
        image_arrays(views)
    views["front"].mask[0, 0] = True
    views["front"].depth[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        image_arrays(views)


def test_repeated_inference_reports_latency_and_identical_predictions():
    views, checkpoint, parameters = _case_inputs()
    prediction, durations, spread = stable_prediction(
        FixedModel(), checkpoint, views, parameters, repeats=5
    )
    assert prediction.tolist() == pytest.approx([parameters[0]])
    assert len(durations) == 5 and all(value >= 0 for value in durations)
    assert spread == [0.0]


def test_repeated_inference_detects_a_drifting_model():
    views, checkpoint, parameters = _case_inputs()
    with pytest.raises(AssertionError, match="unstable"):
        stable_prediction(DriftingModel(), checkpoint, views, parameters, repeats=4)


class NegativeCdModel(FixedModel):
    def forward(self, views, parameters):
        return -torch.ones((parameters.shape[0], 1), dtype=parameters.dtype)


def test_nonvehicle_negative_cd_is_a_stable_supported_domain_rejection():
    from vehicle_cd_learning.validate_examples import stable_nonvehicle_prediction

    views, checkpoint, parameters = _case_inputs()
    checkpoint["target_mean"] = [0.0]
    result, durations, spread, rejection = stable_nonvehicle_prediction(
        NegativeCdModel(), checkpoint, views, parameters, repeats=4
    )
    assert result is None
    assert (
        rejection
        == "Model predicted negative drag; input is outside the supported passive-vehicle domain"
    )
    assert len(durations) == 4 and spread == []


def test_unseen_geometry_requires_explicit_diagnostic_opt_in():
    from vehicle_cd_learning.predict import predict
    from vehicle_cd_learning.validate_examples import predict_case

    views, checkpoint, parameters = _case_inputs()
    checkpoint["parameter_names"] = [
        "width_ratio",
        "height_ratio",
        "log_re",
        "yaw_sin",
        "yaw_cos",
    ]
    checkpoint["target_names"] = ["cd"]
    checkpoint["parameter_ranges"] = {
        "width_ratio": [0.3, 0.5],
        "height_ratio": [0.2, 0.5],
        "log_re": [0.0, 1.0],
        "yaw_sin": [0.0, 1.0],
        "yaw_cos": [0.0, 1.0],
    }
    model = FixedModel()
    with pytest.raises(ValueError, match="Unsupported geometry: width_ratio"):
        predict(model, checkpoint, views, parameters)
    assert predict(
        model,
        checkpoint,
        views,
        parameters,
        allow_geometry_extrapolation=True,
    ).shape == (1,)
    case = {
        "case_id": "unseen-geometry",
        "family_id": "unseen-geometry",
        "split": "test",
        "parameters": parameters.tolist(),
        "targets": [0.3],
    }
    report = predict_case(model, checkpoint, case, views, repeats=3)
    assert report["status"].startswith("diagnostic-only")
    assert (
        "Unsupported geometry: width_ratio" in report["default_mode_rejection_reason"]
    )
    assert report["prediction"]["cd"] == pytest.approx(parameters[0])
