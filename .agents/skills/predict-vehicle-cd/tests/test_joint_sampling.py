import sys
from pathlib import Path

import numpy as np
import torch

SKILL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL / "scripts"))
sys.path.insert(0, str(SKILL.parent / "predict-aircraft-coefficients/scripts"))

from aircraft_learning.scene_experiment import normalization
from vehicle_cd_learning.delta_experiment import sha256
from vehicle_cd_learning.joint_sampling import AbsoluteReadout, FullCases
from vehicle_cd_learning.model import VIEWS, CoefficientModel


def test_auxiliary_readout_reuses_existing_pooled_coefficient_path():
    torch.manual_seed(42)
    readout = AbsoluteReadout()
    torch.manual_seed(42)
    reference = CoefficientModel(
        parameter_count=36,
        variant="five",
        size="small",
        encoder="pyramid_detail_attention",
        fusion="attention",
    )
    pooled = {n: torch.randn(2, 16, 8, 8, requires_grad=True) for n in VIEWS}
    params = torch.randn(2, 36)
    output = readout(pooled, params)
    expected = reference.head(reference.encode_pooled(pooled, params))
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    output.square().mean().backward()
    assert all(
        torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0 for x in pooled.values()
    )
    assert not any(
        "blocks" in n or "lateral" in n for n, _ in readout.named_parameters()
    )


def test_full_cases_preserve_missing_cl_and_use_only_train_normalization(tmp_path):
    path = tmp_path / "geometry.npz"
    image = np.zeros((2, 256, 256), dtype=np.float32)
    image[:, 30:50, 60:80] = 1
    np.savez_compressed(path, **{n: image for n in VIEWS})
    cases = [
        {
            "case_id": f"case{i}",
            "views": path.name,
            "views_sha256": sha256(path),
            "source": "builtin_assumed",
            "split": "train" if i < 2 else "val",
            "parameters": [float(i + 1), None, *([0.0] * 16)],
            "targets": [[0.2, None], [0.3, -0.2], [5.0, 10.0]][i],
        }
        for i in range(3)
    ]
    stats = normalization(cases[:2])
    p = {
        "joint_prepared": str(tmp_path),
        "normalization": stats,
        "source_case_weights": {"builtin_assumed": 1.0},
    }
    data = FullCases(p, cases)
    assert stats["parameter_mean"][0] == 1.5
    assert data.parameters.shape == (3, 36)
    assert torch.isfinite(data.parameters).all()
    assert data.parameters[0, 1] == data.parameters[0, 19] == 0
    assert data.parameters[0, 18] == 1
    first, params, target = data.get(0)
    second, _, _ = data.get(1)
    assert first is second
    assert torch.isnan(target[1]) and torch.isnan(data.targets[0, 1])
    assert params[0] == -1 and data.parameters[2, 0] == 3
