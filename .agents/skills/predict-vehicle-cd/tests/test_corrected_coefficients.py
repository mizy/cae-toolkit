import sys
from pathlib import Path

import torch

SKILL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL / "scripts"))
sys.path.insert(0, str(SKILL.parent / "predict-aircraft-coefficients/scripts"))

from vehicle_cd_learning import dimensionless as dim
from vehicle_cd_learning.corrected_coefficients import CorrectedCoefficients
from vehicle_cd_learning.model import VIEWS, CoefficientModel
from vehicle_cd_learning.multiscale_readout import MultiScaleReadout


def test_correction_uses_the_selected_neural_base_not_the_feature_encoders_head():
    torch.set_num_threads(2)
    base = CoefficientModel(**dim.MODEL).eval()
    kernel = MultiScaleReadout("pooled", 5156, 1, **dim.MODEL).eval()
    with torch.no_grad():
        for p in base.parameters():
            p.zero_()
        base.head[-1].bias.copy_(torch.tensor([2.0, 3.0]))
        kernel.head[-1].bias.fill_(9.0)
        kernel.kernel_bias.copy_(torch.tensor([0.1, 0.2], dtype=torch.float64))
    model = CorrectedCoefficients([base], kernel).eval()
    views = {n: torch.randn(2, 2, 32, 32) for n in VIEWS}
    with torch.inference_mode():
        actual = model(views, torch.zeros(2, 36))
    torch.testing.assert_close(actual, torch.tensor([[2.1, 3.2], [2.1, 3.2]]))


def test_zero_correction_skips_feature_encoding_and_preserves_the_ensemble():
    class UnusedEncoder(torch.nn.Module):
        kernel_reference = ()

        def forward(self, *args):
            raise AssertionError("Zero correction must not encode geometry again")

    base = CoefficientModel(**dim.MODEL).eval()
    model = CorrectedCoefficients([base, base], UnusedEncoder()).eval()
    views = {n: torch.randn(2, 2, 32, 32) for n in VIEWS}
    params = torch.randn(2, 36)
    with torch.inference_mode():
        torch.testing.assert_close(model(views, params), base(views, params))
