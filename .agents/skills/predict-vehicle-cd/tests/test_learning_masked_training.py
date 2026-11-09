from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

SKILL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL / "scripts"))

from vehicle_cd_learning.model import VIEWS, CoefficientModel
from vehicle_cd_learning.train import (
    condition_profiles_with_train_ranges,
    masked_mse_loss,
    train_model,
    training_normalization,
)


class TinyCases(Dataset):
    def __init__(self, parameters, targets):
        self.parameters = torch.tensor(parameters, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return {}, self.parameters[index], self.targets[index]


class TinyViewCases(TinyCases):
    def __getitem__(self, index):
        image = torch.zeros(2, 32, 32)
        image[0] = torch.linspace(0, 1, 32).view(1, 32).expand(32, -1)
        image[1, 4:28, 5:27] = 1
        return (
            {name: image for name in VIEWS},
            self.parameters[index],
            self.targets[index],
        )


def test_masked_loss_gives_missing_cl_zero_gradient():
    prediction = torch.tensor([[1.0, 100.0], [2.0, 200.0]], requires_grad=True)
    targets = torch.tensor([[0.0, float("nan")], [0.0, 0.0]])

    loss = masked_mse_loss(prediction, targets)
    loss.backward()

    assert torch.isfinite(loss)
    assert prediction.grad[0, 1].item() == 0.0
    assert prediction.grad[0, 0].item() != 0.0
    assert prediction.grad[1, 1].item() != 0.0


def test_training_normalization_uses_finite_training_labels_only():
    parameters = np.arange(27, dtype=np.float64).reshape(3, 9)
    targets = [[0.2, None], [0.4, -0.5], [0.6, 0.5]]

    stats = training_normalization(parameters, targets, ["cd", "cl"])

    assert stats["target_counts"] == [3, 2]
    assert stats["target_mean"] == pytest.approx([0.4, 0.0])
    assert stats["target_std"] == pytest.approx([np.std([0.2, 0.4, 0.6]), 0.5])
    assert len(stats["parameter_mean"]) == len(stats["parameter_std"]) == 9
    with pytest.raises(ValueError, match="no observed values: cl"):
        training_normalization(
            parameters, [[0.2, None], [0.4, None], [0.6, None]], ["cd", "cl"]
        )


def test_profile_ranges_use_only_training_cases_and_copy_source_metadata():
    source_profiles = {
        "builtin": {
            "conditions": {"ground": "assumed"},
            "provenance": {"source": "built-in"},
            "data_kind": "reference_assumed",
            "labeled_targets": ["cd"],
        },
        "cfd": {
            "conditions": {"ground": "wind tunnel"},
            "provenance": {"source": "cfd"},
            "data_kind": "cfd",
            "labeled_targets": ["cd", "cl"],
        },
    }
    train_cases = [
        {"condition_profile": "builtin", "parameters": [0.4, 0.3, 13, 0, 1, 1, 0]},
        {"condition_profile": "builtin", "parameters": [0.5, 0.4, 14, 0.1, 0.9, 1, 0]},
        {"condition_profile": "cfd", "parameters": [0.6, 0.2, 15, -0.1, 0.9, 0, 1]},
    ]
    result = condition_profiles_with_train_ranges(source_profiles, train_cases)

    assert "parameter_ranges" not in source_profiles["builtin"]
    assert result["builtin"]["conditions"] == source_profiles["builtin"]["conditions"]
    assert result["builtin"]["parameter_ranges"] == {
        "width_ratio": [0.4, 0.5],
        "height_ratio": [0.3, 0.4],
        "log_re": [13.0, 14.0],
        "yaw_sin": [0.0, 0.1],
        "yaw_cos": [0.9, 1.0],
    }


def test_training_mixed_cd_only_and_joint_cases_records_source_coverage(tmp_path):
    raw_parameters = np.arange(63, dtype=np.float64).reshape(7, 9)
    raw_targets = np.asarray(
        [[-1.0, None], [1.0, -1.0], [0.5, None], [-0.5, 1.0]],
        dtype=np.float64,
    )
    stats = training_normalization(raw_parameters[:4], raw_targets, ["cd", "cl"])
    x = (raw_parameters - stats["parameter_mean"]) / stats["parameter_std"]
    y = (
        np.asarray(
            [
                [-1.0, None],
                [1.0, -1.0],
                [0.5, None],
                [-0.5, 1.0],
                [0.3, None],
                [0.6, None],
                [-0.4, None],
            ],
            dtype=np.float64,
        )
        - stats["target_mean"]
    ) / stats["target_std"]
    train = TinyCases(x[:4], y[:4])
    validation = TinyCases(x[4:], y[4:])
    condition_profiles = {"legacy": {"reynolds": 1.0e6}, "cfd-a": {"reynolds": 2.0e6}}
    metadata = {
        "render_config": {},
        "conditions": {},
        "condition_profiles": condition_profiles,
        "parameter_ranges": {},
        "data_kind": "mixed_sources",
        "provenance": {},
        "train_ids": ["legacy-0", "legacy-1", "cfd-a-0", "cfd-a-1"],
        "train_source_counts": {
            "legacy": {"cases": 2, "cd": 2, "cl": 1},
            "cfd-a": {"cases": 2, "cd": 2, "cl": 1},
        },
        "train_geometry_hashes": ["g0", "g1", "g2", "g3"],
        "heldout_ids": ["held-0", "held-1", "held-2"],
        "dataset_sha256": "fixture",
    }

    checkpoint_path = train_model(
        train,
        validation,
        parameter_names=[f"p{i}" for i in range(9)],
        target_names=["cd", "cl"],
        normalization=stats,
        metadata=metadata,
        output_dir=tmp_path / "mixed-model",
        variant="parameters",
        epochs=1,
        batch_size=4,
        seed=8,
        model_size="small",
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    report = json.loads((checkpoint_path.parent / "training_report.json").read_text())
    assert checkpoint["model_config"]["parameter_count"] == 9
    assert checkpoint["condition_profiles"] == condition_profiles
    assert checkpoint["data_kind"] == "mixed_sources"
    assert checkpoint["train_ids"] == metadata["train_ids"]
    assert report["train_source_counts"] == metadata["train_source_counts"]
    assert report["train_target_counts"] == {"cd": 4, "cl": 2}
    assert report["validation_target_counts"] == {"cd": 3, "cl": 0}
    assert np.isfinite(report["best_validation_mse_standardized"])
    assert report["validation_mae"]["cl"] is None
    assert report["validation_rmse"]["cl"] is None
    assert "NaN" not in (checkpoint_path.parent / "training_report.json").read_text()


def test_encoder_checkpoint_initializes_only_shared_pyramid_encoders(tmp_path):
    torch.manual_seed(17)
    source_model = CoefficientModel(
        5,
        "five",
        2,
        size="small",
        encoder="pyramid_detail_attention",
        fusion="attention",
    )
    source_state = source_model.state_dict()
    for key in source_state:
        if key.startswith("image_encoder."):
            source_state[key].fill_(0.125)
        elif key.startswith("bottom_encoder."):
            source_state[key].fill_(0.25)
        elif key.startswith("head."):
            source_state[key].fill_(7.0)
    source_path = tmp_path / "source.pt"
    torch.save(
        {
            "model_config": {
                "parameter_count": 5,
                "variant": "five",
                "target_count": 2,
                "size": "small",
                "encoder": "pyramid_detail_attention",
                "fusion": "attention",
            },
            "model_state": source_state,
        },
        source_path,
    )

    raw_parameters = np.arange(36, dtype=np.float64).reshape(4, 9)
    raw_targets = np.asarray([[0.2, -1.0], [0.4, 1.0], [0.6, -0.5], [0.8, 0.5]])
    stats = training_normalization(raw_parameters, raw_targets, ["cd", "cl"])
    x = (raw_parameters - stats["parameter_mean"]) / stats["parameter_std"]
    y = (raw_targets - stats["target_mean"]) / stats["target_std"]
    train = TinyViewCases(x[:2], y[:2])
    validation = TinyViewCases(x[2:], y[2:])
    metadata = {
        "render_config": {},
        "conditions": {},
        "parameter_ranges": {},
        "data_kind": "synthetic",
        "provenance": {},
        "train_ids": ["train-0", "train-1"],
        "train_geometry_hashes": ["g0", "g1"],
        "heldout_ids": ["val-0", "val-1"],
        "dataset_sha256": "fixture",
    }

    checkpoint_path = train_model(
        train,
        validation,
        parameter_names=[f"p{i}" for i in range(9)],
        target_names=["cd", "cl"],
        normalization=stats,
        metadata=metadata,
        output_dir=tmp_path / "warm-model",
        variant="five",
        epochs=1,
        batch_size=2,
        seed=18,
        model_size="small",
        encoder="pyramid_detail_attention",
        fusion="attention",
        encoder_checkpoint=source_path,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    initialization = checkpoint["encoder_initialization"]
    assert initialization["mode"] == "shared_encoders_only"
    assert initialization["checkpoint"] == str(source_path)
    assert initialization["loaded_modules"] == ["image_encoder", "bottom_encoder"]
    assert initialization["loaded_keys"]
    assert all(
        key.startswith(("image_encoder.", "bottom_encoder."))
        for key in initialization["loaded_keys"]
    )
    assert not torch.all(checkpoint["model_state"]["head.0.weight"] == 7.0)
