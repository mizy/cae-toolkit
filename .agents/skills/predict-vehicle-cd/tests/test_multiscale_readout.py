import sys
from pathlib import Path

import numpy as np
import pytest
import torch

SKILL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL / "scripts"))
sys.path.insert(0, str(SKILL.parent / "predict-aircraft-coefficients/scripts"))

from vehicle_cd_learning import dimensionless as dim
from vehicle_cd_learning.detail_kernel import distances
from vehicle_cd_learning.multiscale_readout import (
    CONTRACT,
    SCALES,
    MultiScaleReadout,
    load_model,
    mixture,
)


@pytest.mark.parametrize("residual", [False, True])
def test_multiscale_kernel_psd_and_saved_inference(tmp_path, residual):
    torch.manual_seed(55)
    model = MultiScaleReadout("pooled", 5156, 5, **dim.MODEL).eval()
    model.feature_mean.fill_(0.2)
    model.feature_scale.fill_(0.7)
    model.kernel_reference.normal_()
    model.kernel_dual.normal_()
    model.kernel_bias.fill_(0.05)
    model.base_weight.fill_(int(residual))
    features, base = torch.randn(3, 5156), torch.randn(3, 2)
    x = (features.numpy().astype(float) - 0.2) / 0.7
    ref = model.kernel_reference.numpy()
    gram = mixture(distances(ref, ref))
    assert np.linalg.eigvalsh(gram).min() >= -1e-10
    np.testing.assert_allclose(np.diag(gram), 1.0, atol=1e-10)
    expected = (
        mixture(distances(x, ref)) @ model.kernel_dual.numpy()
        + 0.05
        + int(residual) * base.numpy()
    )
    path = tmp_path / "model.pt"
    torch.save(
        {
            "input_contract": CONTRACT,
            "kernel_scales": SCALES,
            "parameter_names": dim.NAMES,
            "model_config": dim.MODEL,
            "model_state": model.state_dict(),
        },
        path,
    )
    loaded, _ = load_model(path)
    np.testing.assert_allclose(
        loaded.predict_features(features, base).numpy(), expected, atol=1e-6, rtol=1e-6
    )
