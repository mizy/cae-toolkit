import sys
from pathlib import Path

import numpy as np
import pytest
import torch

SKILL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL / "scripts"))
sys.path.insert(0, str(SKILL.parent / "predict-aircraft-coefficients/scripts"))

from vehicle_cd_learning import dimensionless as dim
from vehicle_cd_learning.model import VIEWS, CoefficientModel
from vehicle_cd_learning.readout import CONTRACT, initialize, load_model
from vehicle_cd_learning.readout_experiment import coefficient_loss, feature_statistics
from vehicle_cd_learning.readout_export import export


def test_shared_projection_normalization_preserves_initial_function_and_serializes(
    tmp_path,
):
    torch.set_num_threads(2)
    torch.manual_seed(42)
    base = CoefficientModel(**dim.MODEL).eval()
    source = {"model_config": dim.MODEL, "model_state": base.state_dict()}
    rng = np.random.default_rng(42)
    pooled = rng.normal(2, 0.3, size=(6, 5, 16, 8, 8)).astype(np.float32)
    stats = feature_statistics(pooled, [0, 1, 2, 3])
    model = initialize(source, stats).eval()
    params = torch.randn(6, 36)
    features = {n: torch.from_numpy(pooled[:, i]) for i, n in enumerate(VIEWS)}
    with torch.inference_mode():
        expected = base.head(base.encode_pooled(features, params))
        actual = model.head(model.encode_pooled(features, params))
    torch.testing.assert_close(expected, actual, atol=1e-5, rtol=1e-5)
    for name, value in model.named_parameters():
        if (
            name.startswith(("image_encoder.", "bottom_encoder."))
            and ".project." not in name
        ):
            assert not value.requires_grad
    path = tmp_path / "model.pt"
    torch.save(
        {
            "input_contract": CONTRACT,
            "parameter_names": dim.NAMES,
            "raw_parameter_names": dim.PARAMETERS,
            "model_config": dim.MODEL,
            "model_state": model.state_dict(),
            "parameter_mean": [0.0] * 36,
            "parameter_std": [1.0] * 36,
        },
        path,
    )
    loaded, _ = load_model(path)
    with torch.inference_mode():
        torch.testing.assert_close(
            loaded.head(loaded.encode_pooled(features, params)), actual
        )
    with pytest.raises(ValueError, match="dimensionless vehicle"):
        dim.load_model(path)


def test_feature_statistics_use_only_train_and_preserve_ordinary_view_sharing():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(6, 5, 16, 8, 8)).astype(np.float32)
    expected = feature_statistics(x, [0, 1, 2, 3])
    x[4:] = 1000
    assert expected == feature_statistics(x, [0, 1, 2, 3])
    assert np.all(np.array(expected["body_std"]) >= 0.05)
    np.testing.assert_allclose(
        expected["body_mean"], x[:4, :4].reshape(-1, 1024).mean(0, dtype=np.float64)
    )


def test_absolute_loss_masks_missing_cl_and_balances_only_observed_targets():
    weight = torch.tensor([1.0, 2.0], requires_grad=True)
    pred = weight[None].expand(2, 2)
    target = torch.tensor([[0.0, 0.0], [0.0, float("nan")]])
    loss, values, weights = coefficient_loss(
        pred, target, torch.tensor([1.0, 4.0]), [weight], True
    )
    assert values[0].item() == 1 and values[1].item() == 8
    assert sum(weights) == pytest.approx(2)
    assert 0.25 <= weights[0] <= 1.75
    loss.backward()
    assert torch.isfinite(weight.grad).all()
    assert weights[0] > weights[1]


def test_full_view_and_cached_pooled_execution_match_and_are_batch_independent():
    torch.set_num_threads(2)
    torch.manual_seed(43)
    base = CoefficientModel(**dim.MODEL).eval()
    model = initialize(
        {"model_config": dim.MODEL, "model_state": base.state_dict()}
    ).eval()
    views = {n: torch.rand(2, 2, 64, 64) for n in VIEWS}
    params = torch.randn(2, 36)
    with torch.inference_mode():
        body = model.image_encoder.pooled_features(
            torch.cat([views[n] for n in VIEWS[:4]])
        )
        parts = dict(zip(VIEWS[:4], body.chunk(4)))
        parts["bottom"] = model.bottom_encoder.pooled_features(views["bottom"])
        direct = model(views, params)
        cached = model.head(model.encode_pooled(parts, params))
        torch.testing.assert_close(direct, cached, atol=1e-5, rtol=1e-5)
        separate = torch.cat(
            [
                model({n: v[i : i + 1] for n, v in views.items()}, params[i : i + 1])
                for i in range(2)
            ]
        )
        torch.testing.assert_close(direct, separate, atol=1e-5, rtol=1e-5)


def test_identity_export_preserves_weights_and_rejects_scaled_features(tmp_path):
    torch.manual_seed(44)
    base = CoefficientModel(**dim.MODEL)
    original = {"model_config": dim.MODEL, "model_state": base.state_dict()}
    model = initialize(original)
    metadata = {
        "input_contract": CONTRACT,
        "parameter_names": dim.NAMES,
        "raw_parameter_names": dim.PARAMETERS,
        "model_config": dim.MODEL,
        "model_state": model.state_dict(),
        "parameter_mean": [0.0] * 36,
        "parameter_std": [1.0] * 36,
    }
    source = tmp_path / "source.pt"
    target = tmp_path / "compatible.pt"
    torch.save(metadata, source)
    export(source, target)
    compatible, _ = dim.load_model(target)
    for key, value in base.state_dict().items():
        torch.testing.assert_close(compatible.state_dict()[key], value, rtol=0, atol=0)
    with pytest.raises(FileExistsError):
        export(source, target)
    metadata["model_state"]["body_std"] = torch.full((1, 16, 8, 8), 0.5)
    torch.save(metadata, source)
    with pytest.raises(ValueError, match="cannot remove normalization"):
        export(source, tmp_path / "invalid.pt")
