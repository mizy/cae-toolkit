from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from vehicle_cd_learning import predict as predict_module
from vehicle_cd_learning.data import PARAMETER_NAMES, VIEWS
from vehicle_cd_learning.model import CoefficientModel
from vehicle_cd_learning.predict import load_model, predict, prediction_warnings

PROFILES = {
    "drivaerml": {
        "conditions": {"ground": "moving", "inlet": "uniform"},
        "provenance": {"source": "CFD"},
        "data_kind": "cfd",
        "labeled_targets": ["cd", "cl"],
    },
    "ahmedml": {
        "conditions": {"ground": "fixed", "inlet": "uniform", "wall": "no-slip"},
        "provenance": {"source": "Ahmed ML"},
        "data_kind": "cfd",
        "labeled_targets": ["cd", "cl"],
    },
    "windsorml": {
        "conditions": {"ground": "moving", "inlet": "uniform", "wall": "no-slip"},
        "provenance": {"source": "Windsor ML"},
        "data_kind": "cfd",
        "labeled_targets": ["cd", "cl"],
    },
    "builtin_assumed": {
        "conditions": {"ground": "fixed", "inlet": "uniform"},
        "provenance": {"source": "public reference"},
        "data_kind": "reference_assumed",
        "labeled_targets": ["cd"],
    },
}


@pytest.fixture
def profile_checkpoint(tmp_path):
    torch.manual_seed(11)
    parameter_names = PARAMETER_NAMES + [f"profile_{name}" for name in PROFILES]
    config = {
        "parameter_count": len(parameter_names),
        "variant": "five",
        "target_count": 2,
        "size": "small",
        "encoder": "pyramid_detail_attention",
        "fusion": "attention",
    }
    profile_mean = [1.0 / len(PROFILES)] * len(PROFILES)
    profile_std = [
        float(np.sqrt((1.0 / len(PROFILES)) * (1.0 - 1.0 / len(PROFILES))))
    ] * len(PROFILES)
    profile_reynolds = {
        "drivaerml": 12_000_000,
        "ahmedml": 2_900_000,
        "windsorml": 3_500_000,
        "builtin_assumed": 2_900_000,
    }
    profile_ranges = {
        name: {
            "width_ratio": [0.35, 0.45],
            "height_ratio": [0.3, 0.35],
            "log_re": [float(np.log(re)), float(np.log(re))],
            "yaw_sin": [0.0, 0.0],
            "yaw_cos": [1.0, 1.0],
        }
        for name, re in profile_reynolds.items()
    }
    checkpoint_profiles = {
        name: {**profile, "parameter_ranges": profile_ranges[name]}
        for name, profile in PROFILES.items()
    }
    log_re_lo = float(np.log(min(profile_reynolds.values())))
    log_re_hi = float(np.log(max(profile_reynolds.values())))
    model = CoefficientModel(**config)
    checkpoint = {
        "model_state": model.state_dict(),
        "model_config": config,
        "parameter_names": parameter_names,
        "condition_profiles": checkpoint_profiles,
        "target_names": ["cd", "cl"],
        "parameter_mean": [
            0.4,
            0.32,
            (log_re_lo + log_re_hi) / 2,
            0.0,
            1.0,
            *profile_mean,
        ],
        "parameter_std": [0.04, 0.03, 0.7, 1e-8, 1e-8, *profile_std],
        "target_mean": [0.3, 0.0],
        "target_std": [0.02, 0.02],
        "parameter_ranges": {
            "width_ratio": [0.35, 0.45],
            "height_ratio": [0.3, 0.35],
            "log_re": [log_re_lo, log_re_hi],
            "yaw_sin": [0.0, 0.0],
            "yaw_cos": [1.0, 1.0],
            **{f"profile_{name}": [0.0, 1.0] for name in PROFILES},
        },
        "render_config": {"ordinary_size": 32, "bottom_size": 64},
        "conditions": {"profiled": True},
        "data_kind": "mixed_sources",
        "dataset_sha256": "test-dataset",
    }
    checkpoint_path = tmp_path / "model.pt"
    torch.save(checkpoint, checkpoint_path)
    views = {
        name: np.stack(
            (
                np.full((64, 64) if name == "bottom" else (32, 32), 0.5, np.float32),
                np.ones((64, 64) if name == "bottom" else (32, 32), np.float32),
            )
        )
        for name in VIEWS
    }
    return checkpoint_path, checkpoint, views


def test_profile_checkpoint_loads_nine_parameter_attention_model(profile_checkpoint):
    path, checkpoint, views = profile_checkpoint
    model, loaded = load_model(path)
    assert model.parameter_count == 9
    assert model.image_encoder.detail_pooling
    assert model.image_encoder.channel is not None
    assert model.fusion == "attention"
    values = [0.4, 0.32, float(np.log(2_900_000)), 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    result = predict(model, loaded, views, values)
    assert result.shape == (2,)
    assert np.isfinite(result).all()
    assert prediction_warnings(checkpoint, values) == [
        "Mixed-source checkpoint: prediction support depends on the selected condition profile",
        "Condition profile builtin_assumed uses assumed conditions and public Cd labels",
        "Cl is unsupported for condition profile builtin_assumed; its source has no Cl label",
    ]


@pytest.mark.parametrize(
    "profile_values",
    [[0.0, 0.0, 0.0, 0.0], [0.25, 0.25, 0.25, 0.25], [1.0, 0.0, 0.0, 1.0]],
)
def test_profile_features_must_be_an_exact_declared_one_hot(
    profile_checkpoint, profile_values
):
    path, _, views = profile_checkpoint
    model, checkpoint = load_model(path)
    values = [0.4, 0.32, float(np.log(2_900_000)), 0.0, 1.0, *profile_values]
    with pytest.raises(ValueError, match="exact declared one-hot"):
        predict(model, checkpoint, views, values, allow_geometry_extrapolation=True)


def test_profile_specific_reynolds_range_overrides_global_union(profile_checkpoint):
    path, checkpoint, views = profile_checkpoint
    model, checkpoint = load_model(path)
    values = [0.4, 0.32, float(np.log(2_900_000)), 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="Unsupported condition: log_re"):
        predict(model, checkpoint, views, values)
    assert any(
        warning.startswith("log_re=") and "outside training range" in warning
        for warning in prediction_warnings(checkpoint, values)
    )


def test_checkpoint_rejects_profile_range_outside_global_union(
    profile_checkpoint, tmp_path
):
    _, checkpoint, _ = profile_checkpoint
    invalid_profiles = {
        **checkpoint["condition_profiles"],
        "drivaerml": {
            **checkpoint["condition_profiles"]["drivaerml"],
            "parameter_ranges": {
                **checkpoint["condition_profiles"]["drivaerml"]["parameter_ranges"],
                "log_re": [0.0, 1.0],
            },
        },
    }
    path = tmp_path / "bad-model.pt"
    torch.save({**checkpoint, "condition_profiles": invalid_profiles}, path)
    with pytest.raises(ValueError, match="falls outside global range"):
        load_model(path)


def test_geometry_override_can_extrapolate_fixed_geometry_only(profile_checkpoint):
    path, _, views = profile_checkpoint
    model, checkpoint = load_model(path)
    checkpoint["parameter_std"][:2] = [1e-8, 1e-8]
    values = [0.5, 0.4, float(np.log(2_900_000)), 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    with pytest.raises(ValueError, match="Unsupported geometry"):
        predict(model, checkpoint, views, values)
    assert np.isfinite(
        predict(model, checkpoint, views, values, allow_geometry_extrapolation=True)
    ).all()


def test_cli_selects_explicit_profile_and_filters_unlabeled_cl(
    profile_checkpoint, tmp_path, monkeypatch, capsys
):
    path, _, _ = profile_checkpoint
    geometry_views = {
        name: SimpleNamespace(
            depth=np.full((64, 64) if name == "bottom" else (32, 32), 0.5),
            mask=np.ones((64, 64) if name == "bottom" else (32, 32)),
        )
        for name in VIEWS
    }
    monkeypatch.setattr(
        predict_module,
        "render_geometry",
        lambda *_args: (
            geometry_views,
            {"length_m": 4.5, "width_m": 1.8, "height_m": 1.4},
        ),
    )
    case_path = tmp_path / "case.json"
    case_path.write_text(
        json.dumps(
            {
                "condition_profile": "builtin_assumed",
                "geometry": {"path": "car.vtp"},
                "reynolds": 2_900_000,
                "yaw_deg": 0,
                "conditions": PROFILES["builtin_assumed"]["conditions"],
            }
        )
    )
    monkeypatch.setattr(sys, "argv", ["predict", str(path), str(case_path)])

    predict_module.main()

    output = json.loads(capsys.readouterr().out)
    assert output["condition_profile"] == "builtin_assumed"
    assert output["conditions"] == PROFILES["builtin_assumed"]["conditions"]
    assert list(output["coefficients"]) == ["cd"]
    assert output["unsupported_targets"] == ["cl"]
    assert any("Cl is unsupported" in warning for warning in output["warnings"])


def test_cli_rejects_explicit_profile_when_conditions_differ(
    profile_checkpoint, tmp_path, monkeypatch
):
    path, _, _ = profile_checkpoint
    case_path = tmp_path / "case.json"
    case_path.write_text(
        json.dumps(
            {
                "condition_profile": "builtin_assumed",
                "geometry": {"path": "car.vtp"},
                "reynolds": 2_900_000,
                "yaw_deg": 0,
                "conditions": PROFILES["drivaerml"]["conditions"],
            }
        )
    )
    monkeypatch.setattr(sys, "argv", ["predict", str(path), str(case_path)])
    with pytest.raises(ValueError, match="differ from condition profile"):
        predict_module.main()


def test_cli_requires_explicit_profile_for_mixed_checkpoint(
    profile_checkpoint, tmp_path, monkeypatch
):
    path, _, _ = profile_checkpoint
    case_path = tmp_path / "case.json"
    case_path.write_text(
        json.dumps(
            {
                "geometry": {"path": "car.vtp"},
                "reynolds": 2_900_000,
                "yaw_deg": 0,
                "conditions": PROFILES["builtin_assumed"]["conditions"],
            }
        )
    )
    monkeypatch.setattr(sys, "argv", ["predict", str(path), str(case_path)])
    with pytest.raises(ValueError, match="explicit condition_profile"):
        predict_module.main()


def test_cli_rejects_unknown_profile_without_condition_fallback(
    profile_checkpoint, tmp_path, monkeypatch
):
    path, _, _ = profile_checkpoint
    case_path = tmp_path / "case.json"
    case_path.write_text(
        json.dumps(
            {
                "condition_profile": "unknown",
                "geometry": {"path": "car.vtp"},
                "reynolds": 2_900_000,
                "yaw_deg": 0,
                "conditions": PROFILES["builtin_assumed"]["conditions"],
            }
        )
    )
    monkeypatch.setattr(sys, "argv", ["predict", str(path), str(case_path)])
    with pytest.raises(ValueError, match="Unknown condition profile"):
        predict_module.main()
