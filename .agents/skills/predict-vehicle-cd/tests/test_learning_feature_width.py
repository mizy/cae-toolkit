import json

import numpy as np
import pytest
import torch
from test_learning_masked_training import TinyViewCases
from vehicle_cd_learning.compare_feature_width import paired_bootstrap
from vehicle_cd_learning.model import VIEWS, CoefficientModel
from vehicle_cd_learning.train import train_model, training_normalization


def test_width_comparison_pairs_cases_and_error_direction():
    first = [
        {
            "case_id": str(i),
            "split": "val",
            "profile": str(i),
            "truth": [1.0, 1.0],
            "prediction": [1.2, 0.8],
        }
        for i in range(2)
    ]
    second = [{**r, "prediction": [1.1, 0.9]} for r in first]
    result = paired_bootstrap(first, second)
    np.testing.assert_allclose(result["difference"][:2], [-0.1, -0.1])
    np.testing.assert_allclose(
        result["ci95"], [result["difference"], result["difference"]]
    )
    with pytest.raises(ValueError, match="alignment"):
        paired_bootstrap(first, second[::-1])


def test_default_and_explicit_32_preserve_legacy_state():
    torch.manual_seed(42)
    legacy = CoefficientModel(
        9, size="small", encoder="pyramid_detail_attention", fusion="attention"
    )
    torch.manual_seed(42)
    explicit = CoefficientModel(
        9,
        size="small",
        encoder="pyramid_detail_attention",
        fusion="attention",
        view_feature_size=32,
    )
    for key, value in legacy.state_dict().items():
        torch.testing.assert_close(value, explicit.state_dict()[key], rtol=0, atol=0)
    with pytest.raises(ValueError):
        CoefficientModel(9, view_feature_size=0)


def test_width_training_pairs_trunk_preserves_mask_and_checkpoint(tmp_path):
    torch.set_num_threads(2)
    source = CoefficientModel(
        5, size="small", encoder="pyramid_detail_attention", fusion="attention"
    )
    source_path = tmp_path / "encoder.pt"
    torch.save(
        {
            "model_config": {
                "variant": "five",
                "size": "small",
                "encoder": "pyramid_detail_attention",
            },
            "model_state": source.state_dict(),
        },
        source_path,
    )
    x = np.arange(36).reshape(4, 9)
    y = np.array([[0.2, np.nan], [0.3, -0.1], [0.4, 0.2], [0.5, 0.3]])
    stats = training_normalization(x, y, ["cd", "cl"])
    cases = TinyViewCases(
        (x - stats["parameter_mean"]) / stats["parameter_std"],
        (y - stats["target_mean"]) / stats["target_std"],
    )
    metadata = {
        "render_config": {},
        "conditions": {},
        "parameter_ranges": {},
        "data_kind": "synthetic",
        "provenance": {},
        "train_ids": ["0", "1", "2", "3"],
        "train_geometry_hashes": ["a", "b", "c", "d"],
        "heldout_ids": [],
        "dataset_sha256": "fixture",
    }
    saved = []
    for width in (32, 128):
        path = train_model(
            cases,
            cases,
            parameter_names=[f"p{i}" for i in range(9)],
            target_names=["cd", "cl"],
            normalization=stats,
            metadata=metadata,
            output_dir=tmp_path / str(width),
            epochs=1,
            batch_size=4,
            learning_rate=1e-12,
            seed=42,
            model_size="small",
            encoder="pyramid_detail_attention",
            fusion="attention",
            encoder_checkpoint=source_path,
            view_feature_size=width,
        )
        ck = torch.load(path, weights_only=True)
        report = json.loads((path.parent / "training_report.json").read_text())
        assert report["train_target_counts"] == {"cd": 4, "cl": 3}
        assert (
            ck["encoder_initialization"]["mode"] == "shared_pyramid_before_projection"
        )
        assert all(
            ".project." not in k for k in ck["encoder_initialization"]["loaded_keys"]
        )
        model = CoefficientModel(**ck["model_config"])
        model.load_state_dict(ck["model_state"], strict=True)
        assert model.image_encoder.project[1].out_features == width
        assert model.parameter_encoder[-2].out_features == 8
        assert model.head[0].out_features == 64 and model.head[2].out_features == 16
        assert model.view_attention.score[0].out_features == 16
        views, parameters, _ = cases[0]
        views = {k: v[None] for k, v in views.items()}
        pooled = {
            k: (
                model.bottom_encoder if k == "bottom" else model.image_encoder
            ).pooled_features(views[k])
            for k in VIEWS
        }
        torch.testing.assert_close(
            model(views, parameters[None]),
            model.head(model.encode_pooled(pooled, parameters[None])),
        )
        saved.append(ck)
    assert saved[0]["common_initial_sha256"] == saved[1]["common_initial_sha256"]
