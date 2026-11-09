"""Adversarial boundary checks for learned coefficient inference."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from vehicle_cd_learning.data import PARAMETER_NAMES, VIEWS, CaseDataset
from vehicle_cd_learning.model import CoefficientModel
from vehicle_cd_learning.predict import load_model, predict, prediction_warnings


def test_within_family_checkpoint_exposes_its_prediction_scope():
    checkpoint = {
        "split_policy": "geometry_disjoint",
        "train_family_ids": ["drivaer-notchback"],
        "data_kind": "cfd",
    }
    warnings = prediction_warnings(checkpoint, [0.4, 0.3, 15, 0, 1])
    assert len(warnings) == 1
    assert "unseen vehicle families" in warnings[0]
    assert "drivaer-notchback" in warnings[0]
    assert prediction_warnings({"data_kind": "cfd"}, [0.4, 0.3, 15, 0, 1]) == []


def test_training_fixed_inputs_match_inference_zero_without_hiding_extrapolation(
    tmp_path,
):
    mean = [0.4, 0.3, 15.0, 0.0, 1.0]
    cases = [
        {"parameters": [0.4 + 1e-10, 0.35, 15 + 2e-9, 0.0, 1.0], "targets": [0.3]},
        {"parameters": [0.4, 0.35, 16.0, 0.0, 1.0], "targets": [0.3]},
    ]
    dataset = CaseDataset(
        tmp_path,
        cases,
        mean,
        [1e-8, 0.05, 1e-8, 1e-8, 1e-8],
        [0.3],
        [0.1],
        "parameters",
    )
    _, normalized, _ = dataset[0]
    assert normalized.tolist() == pytest.approx([0, 1, 0, 0, 0])
    _, extrapolated, _ = dataset[1]
    assert extrapolated[2] > 1e6


@pytest.fixture
def inference_checkpoint():
    torch.manual_seed(3)
    model = CoefficientModel(parameter_count=5, variant="five", target_count=1)
    checkpoint = {
        "parameter_names": PARAMETER_NAMES,
        "model_config": {"parameter_count": 5, "variant": "five", "target_count": 1},
        "model_state": model.state_dict(),
        "parameter_mean": [0.4, 0.32, 14.91, 0.0, 1.0],
        "parameter_std": [0.04, 0.03, 1e-8, 1e-8, 1e-8],
        "target_names": ["cd"],
        "target_mean": [0.3],
        "target_std": [0.05],
        "parameter_ranges": {
            "width_ratio": [0.3, 0.5],
            "height_ratio": [0.25, 0.4],
            "log_re": [14.91, 14.91],
            "yaw_sin": [0.0, 0.0],
            "yaw_cos": [1.0, 1.0],
        },
        "render_config": {"ordinary_size": 32, "bottom_size": 64},
        "conditions": {"ground": "assumed fixed"},
        "data_kind": "reference_assumed",
        "dataset_sha256": "fixture",
    }
    views = {
        name: np.stack(
            (
                np.full((64, 64) if name == "bottom" else (32, 32), 0.5, np.float32),
                np.ones((64, 64) if name == "bottom" else (32, 32), np.float32),
            )
        )
        for name in VIEWS
    }
    return model, checkpoint, views


def test_repeated_inference_is_deterministic_and_finite(inference_checkpoint):
    model, checkpoint, views = inference_checkpoint
    values = [0.4, 0.32, 14.91, 0.0, 1.0]
    first = predict(model, checkpoint, views, values)
    second = predict(model, checkpoint, views, values)
    assert np.isfinite(first).all()
    np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize(
    "values",
    [
        [0.4, 0.32, np.nan, 0.0, 1.0],
        [0.4, 0.32, np.inf, 0.0, 1.0],
        [0.4, 0.32, 14.91],
    ],
)
def test_invalid_query_parameters_fail_before_network(inference_checkpoint, values):
    model, checkpoint, views = inference_checkpoint
    with pytest.raises(ValueError, match="parameter|finite"):
        predict(model, checkpoint, views, values)


def test_changed_fixed_reynolds_condition_is_rejected(inference_checkpoint):
    model, checkpoint, views = inference_checkpoint
    values = [0.4, 0.32, float(np.log(1e6)), 0.0, 1.0]
    with pytest.raises(ValueError, match="range|parameter"):
        predict(model, checkpoint, views, values)


def test_invalid_depth_masks_and_empty_projections_are_rejected(inference_checkpoint):
    model, checkpoint, views = inference_checkpoint
    malformed = {key: value.copy() for key, value in views.items()}
    malformed["front"][1, 0, 0] = 0.25
    with pytest.raises(ValueError, match="mask|view"):
        predict(model, checkpoint, malformed, [0.4, 0.32, 14.91, 0.0, 1.0])

    empty = {key: value.copy() for key, value in views.items()}
    for value in empty.values():
        value[1] = 0
    with pytest.raises(ValueError, match="empty|mask|view"):
        predict(model, checkpoint, empty, [0.4, 0.32, 14.91, 0.0, 1.0])


def test_checkpoint_rejects_zero_scale_nonfinite_weights_and_wrong_shape(
    inference_checkpoint, tmp_path
):
    _, checkpoint, _ = inference_checkpoint
    path = tmp_path / "checkpoint.pt"

    invalid_scale = {
        **checkpoint,
        "parameter_std": [0.0, *checkpoint["parameter_std"][1:]],
    }
    torch.save(invalid_scale, path)
    with pytest.raises(ValueError, match="scale|finite|checkpoint"):
        load_model(path)

    bad_weights = {
        key: value.clone() for key, value in checkpoint["model_state"].items()
    }
    first = next(iter(bad_weights))
    bad_weights[first].view(-1)[0] = float("nan")
    torch.save({**checkpoint, "model_state": bad_weights}, path)
    with pytest.raises(ValueError, match="finite|weight|checkpoint"):
        load_model(path)

    wrong_shape = {**checkpoint, "parameter_mean": [0.4, 0.32]}
    torch.save(wrong_shape, path)
    with pytest.raises(ValueError, match="shape|parameter|checkpoint"):
        load_model(path)


def test_exact_trained_condition_endpoints_are_supported(inference_checkpoint):
    model, checkpoint, views = inference_checkpoint
    parameters = [0.3, 0.25, 14.91, 0.0, 1.0]
    assert np.isfinite(predict(model, checkpoint, views, parameters)).all()


def test_changed_yaw_is_rejected_when_training_yaw_is_fixed(inference_checkpoint):
    model, checkpoint, views = inference_checkpoint
    parameters = [
        0.4,
        0.32,
        14.91,
        float(np.sin(np.deg2rad(5))),
        float(np.cos(np.deg2rad(5))),
    ]
    with pytest.raises(ValueError, match="range|condition|parameter"):
        predict(model, checkpoint, views, parameters)


@pytest.mark.parametrize(
    "key,value",
    [
        ("parameter_mean", [np.nan, 0.32, 14.91, 0.0, 1.0]),
        ("parameter_std", [np.inf, 0.03, 1e-8, 1e-8, 1e-8]),
        ("parameter_std", [-1.0, 0.03, 1e-8, 1e-8, 1e-8]),
        ("target_std", [np.nan]),
        ("target_mean", [np.inf]),
        ("target_mean", [0.3, 0.4]),
    ],
)
def test_checkpoint_rejects_nonfinite_nonpositive_or_misshaped_stats(
    inference_checkpoint, tmp_path, key, value
):
    _, checkpoint, _ = inference_checkpoint
    path = tmp_path / "checkpoint.pt"
    torch.save({**checkpoint, key: value}, path)
    with pytest.raises(ValueError, match="finite|scale|length"):
        load_model(path)


@pytest.mark.parametrize(
    "corrupt,pattern",
    [
        ("mask", "mask|view"),
        ("depth", "depth|view"),
        ("background", "background|depth|view"),
        ("shape", "size|view"),
        ("missing", "Missing view|view"),
        ("empty", "mask|view"),
    ],
)
def test_depth_view_contract_rejects_corrupted_inputs(
    inference_checkpoint, corrupt, pattern
):
    model, checkpoint, views = inference_checkpoint
    changed = {key: value.copy() for key, value in views.items()}
    if corrupt == "mask":
        changed["front"][1, 0, 0] = 0.25
    elif corrupt == "depth":
        changed["front"][0, 0, 0] = 1.1
    elif corrupt == "background":
        changed["front"][1, 0, 0] = 0
        changed["front"][0, 0, 0] = 0.1
    elif corrupt == "shape":
        changed["front"] = changed["front"][:, :-1]
    elif corrupt == "missing":
        changed.pop("front")
    else:
        for view in changed.values():
            view[1] = 0
    with pytest.raises(ValueError, match=pattern):
        predict(model, checkpoint, changed, [0.4, 0.32, 14.91, 0.0, 1.0])


@pytest.mark.parametrize(
    "model_config",
    [
        {"parameter_count": 4, "variant": "five", "target_count": 1},
        {"parameter_count": 5, "variant": "four", "target_count": 1},
        {"parameter_count": 5, "variant": "five", "target_count": 2},
    ],
)
def test_checkpoint_rejects_incompatible_model_shape(
    inference_checkpoint, tmp_path, model_config
):
    _, checkpoint, _ = inference_checkpoint
    path = tmp_path / "checkpoint.pt"
    torch.save({**checkpoint, "model_config": model_config}, path)
    with pytest.raises(
        (ValueError, RuntimeError),
        match="weight|shape|size|checkpoint|missing|unexpected|dimensions",
    ):
        load_model(path)


def test_tiny_numeric_drift_on_fixed_conditions_does_not_amplify(inference_checkpoint):
    model, checkpoint, views = inference_checkpoint
    baseline = np.asarray([0.4, 0.32, 14.91, 0.0, 1.0])
    jittered = baseline.copy()
    jittered[2] += 1e-7
    jittered[3] += 1e-7
    first = predict(model, checkpoint, views, baseline)
    second = predict(model, checkpoint, views, jittered)
    np.testing.assert_allclose(second, first, rtol=0, atol=1e-8)


def test_negative_predicted_drag_is_rejected_for_passive_vehicle(inference_checkpoint):
    model, checkpoint, views = inference_checkpoint
    with torch.no_grad():
        model.head[-1].weight.zero_()
        model.head[-1].bias.fill_(-10.0)
    with pytest.raises(ValueError, match="negative drag|passive-vehicle"):
        predict(model, checkpoint, views, [0.4, 0.32, 14.91, 0.0, 1.0])


def test_geometry_extrapolation_is_rejected_by_default(inference_checkpoint):
    model, checkpoint, views = inference_checkpoint
    out_of_range = [0.6, 0.32, 14.91, 0.0, 1.0]
    with pytest.raises(
        ValueError, match="Unsupported geometry|width_ratio|trained range"
    ):
        predict(model, checkpoint, views, out_of_range)


def test_geometry_extrapolation_override_is_explicit_and_diagnostic_only(
    inference_checkpoint,
):
    model, checkpoint, views = inference_checkpoint
    with torch.no_grad():
        model.head[-1].weight.zero_()
        model.head[-1].bias.zero_()
    out_of_range = [0.6, 0.32, 14.91, 0.0, 1.0]
    predicted = predict(
        model, checkpoint, views, out_of_range, allow_geometry_extrapolation=True
    )
    assert predicted.shape == (1,)
    assert np.isfinite(predicted).all()
    assert predicted[0] == pytest.approx(checkpoint["target_mean"][0])


def test_geometry_extrapolation_override_does_not_bypass_fixed_conditions(
    inference_checkpoint,
):
    model, checkpoint, views = inference_checkpoint
    out_of_range = [0.6, 0.32, float(np.log(1e6)), 0.0, 1.0]
    with pytest.raises(ValueError, match="condition|range|parameter"):
        predict(
            model, checkpoint, views, out_of_range, allow_geometry_extrapolation=True
        )
