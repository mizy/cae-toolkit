import sys
from pathlib import Path

import numpy as np
import pytest
import torch

SKILL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL / "scripts"))
sys.path.insert(0, str(SKILL.parent / "predict-aircraft-coefficients/scripts"))

from vehicle_cd_learning import dimensionless as dim
from vehicle_cd_learning.detail_kernel import kernel
from vehicle_cd_learning.kernel_readout import CONTRACT, initialize, load_model
from vehicle_cd_learning.model import VIEWS, CoefficientModel


@pytest.mark.parametrize("stage", ["pooled", "fused"])
def test_full_views_replay_cached_features_and_keep_conditions(stage):
    torch.set_num_threads(2)
    torch.manual_seed(52)
    source = CoefficientModel(**dim.MODEL).eval()
    ck = {"model_state": source.state_dict(), "model_config": dim.MODEL}
    model = initialize(ck, stage, (5120 if stage == "pooled" else 168) + 36)
    views = {n: torch.randn(2, 2, 32, 32) for n in VIEWS}
    params = torch.randn(2, 36)
    with torch.inference_mode():
        torch.testing.assert_close(model(views, params), source(views, params))
        body = model.image_encoder.pooled_features(
            torch.cat([views[n] for n in VIEWS[:4]])
        )
        pooled = dict(zip(VIEWS[:4], body.chunk(4)))
        pooled["bottom"] = model.bottom_encoder.pooled_features(views["bottom"])
        x, base = model.features(pooled, params)
        torch.testing.assert_close(x[:, -36:], params)
        model.kernel_reference = x.double().clone()
        model.kernel_dual = torch.tensor([[0.2, 0.6], [0.7, -0.3]], dtype=torch.float64)
        model.kernel_bias.fill_(0.1)
        torch.testing.assert_close(
            model(views, params), model.predict_features(x, base)
        )
    assert all(
        torch.equal(value, model.state_dict()[key])
        for key, value in source.state_dict().items()
    )
    assert not any(p.requires_grad for p in model.parameters())


@pytest.mark.parametrize("residual", [False, True])
def test_saved_kernel_matches_numpy_in_both_absolute_prediction_modes(
    tmp_path, residual
):
    torch.manual_seed(53)
    source = CoefficientModel(**dim.MODEL)
    ck = {"model_state": source.state_dict(), "model_config": dim.MODEL}
    model = initialize(ck, "fused", 204)
    model.feature_mean.fill_(0.3)
    model.feature_scale.fill_(1.3)
    model.kernel_reference = torch.randn(6, 204, dtype=torch.float64)
    model.kernel_dual = torch.randn(6, 2, dtype=torch.float64)
    model.kernel_bias.fill_(-0.2)
    model.kernel_bandwidth.fill_(0.9)
    model.base_weight.fill_(int(residual))
    features, base = torch.randn(3, 204), torch.randn(3, 2)
    x = (features.numpy().astype(float) - 0.3) / 1.3
    expected = (
        kernel(x, model.kernel_reference.numpy(), "rbf", 0.9)
        @ model.kernel_dual.numpy()
        - 0.2
        + int(residual) * base.numpy()
    )
    path = tmp_path / "model.pt"
    torch.save(
        {
            **ck,
            "model_state": model.state_dict(),
            "input_contract": CONTRACT,
            "readout_stage": "fused",
            "parameter_names": dim.NAMES,
            "raw_parameter_names": dim.PARAMETERS,
        },
        path,
    )
    loaded, _ = load_model(path)
    actual = loaded.predict_features(features, base).numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
