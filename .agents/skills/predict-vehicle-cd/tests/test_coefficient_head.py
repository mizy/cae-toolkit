import sys
from pathlib import Path

import numpy as np
import torch

SKILL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL / "scripts"))
sys.path.insert(0, str(SKILL.parent / "predict-aircraft-coefficients/scripts"))

from vehicle_cd_learning import dimensionless as dim
from vehicle_cd_learning.coefficient_head import CONTRACT, KINDS, initialize, load_model
from vehicle_cd_learning.head_experiment import rbf_kernel
from vehicle_cd_learning.model import CoefficientModel


def test_head_variants_start_at_identical_predictions_and_freeze_the_encoder():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    base = CoefficientModel(**dim.MODEL).eval()
    source = {"model_config": dim.MODEL, "model_state": base.state_dict()}
    features = torch.randn(5, 168)
    for kind in KINDS:
        model = initialize(source, kind, np.ones(168), np.full(168, 0.3)).eval()
        with torch.no_grad():
            torch.testing.assert_close(
                model.predict_features(features), base.head(features), atol=0, rtol=0
            )
        assert all(
            not p.requires_grad
            for n, p in model.named_parameters()
            if not n.startswith(("head.", "correction."))
        )


def test_normalized_head_is_independent_of_other_query_rows():
    torch.manual_seed(43)
    base = CoefficientModel(**dim.MODEL)
    model = initialize(
        {"model_config": dim.MODEL, "model_state": base.state_dict()},
        "normalized",
        np.zeros(168),
        np.ones(168),
    ).eval()
    with torch.no_grad():
        model.correction[-1].weight.normal_(0, 0.1)
    features = torch.randn(4, 168)
    with torch.no_grad():
        together = model.predict_features(features)
        separate = torch.cat([model.predict_features(f[None]) for f in features])
    torch.testing.assert_close(together, separate, atol=1e-6, rtol=1e-6)


def test_rbf_head_matches_double_precision_kernel_and_checkpoint(tmp_path):
    torch.manual_seed(44)
    base = CoefficientModel(**dim.MODEL)
    mean = np.ones(168, dtype=np.float32)
    std = np.full(168, 0.3, dtype=np.float32)
    model = initialize(
        {"model_config": dim.MODEL, "model_state": base.state_dict()}, "rbf", mean, std
    ).eval()
    ref = torch.randn(8, 168, dtype=torch.float64)
    dual = torch.randn(8, 2, dtype=torch.float64)
    model.kernel_reference = ref
    model.kernel_dual = dual
    model.kernel_scale.fill_(1.2)
    features = torch.tensor(mean) + torch.tensor(std) * torch.randn(3, 168)
    x = (features.numpy().astype(float) - mean) / std
    expected = rbf_kernel(x, ref.numpy(), 1.2) @ dual.numpy()
    with torch.no_grad():
        torch.testing.assert_close(
            model.predict_features(features),
            base.head(features) + torch.tensor(expected, dtype=torch.float32),
            atol=1e-6,
            rtol=1e-6,
        )
    path = tmp_path / "model.pt"
    torch.save(
        {
            "input_contract": CONTRACT,
            "head_kind": "rbf",
            "parameter_names": dim.NAMES,
            "raw_parameter_names": dim.PARAMETERS,
            "model_config": dim.MODEL,
            "model_state": model.state_dict(),
        },
        path,
    )
    loaded, _ = load_model(path)
    with torch.no_grad():
        torch.testing.assert_close(
            model.predict_features(features),
            loaded.predict_features(features),
            atol=0,
            rtol=0,
        )
