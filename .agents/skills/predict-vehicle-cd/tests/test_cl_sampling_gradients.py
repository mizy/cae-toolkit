import sys
from pathlib import Path

import torch

SKILL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL / "scripts"))
sys.path.insert(0, str(SKILL.parent / "predict-aircraft-coefficients/scripts"))

from vehicle_cd_learning.cl_sampling_gradients import target_objectives
from vehicle_cd_learning.joint_sampling_train import absolute_objective


def test_target_split_preserves_actual_loss_and_missing_cl_gradient():
    torch.manual_seed(42)
    prediction = torch.randn(6, 2, requires_grad=True)
    target = torch.tensor([[0.2, float("nan")], [0.3, -0.1], [0.1, 0.2]])
    source_weights = torch.tensor([3.0, 0.8, 1.2])
    weights = {
        "absolute_nominal": 0.25,
        "absolute_augmented": 0.25,
        "absolute_consistency": 0.1,
    }
    split = target_objectives(prediction, target, source_weights, weights)
    full = sum(
        weights[k] * v
        for k, v in absolute_objective(prediction, target, source_weights).items()
    )
    torch.testing.assert_close(split.sum(), full)
    split_gradient = torch.autograd.grad(split.sum(), prediction, retain_graph=True)[0]
    full_gradient = torch.autograd.grad(full, prediction)[0]
    torch.testing.assert_close(split_gradient, full_gradient)
    assert split_gradient[0, 1] == split_gradient[3, 1] == 0
    assert split_gradient[0, 0] != 0 and split_gradient[3, 0] != 0
