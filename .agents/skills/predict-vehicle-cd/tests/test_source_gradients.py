import sys
from pathlib import Path

import torch

SKILL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL / "scripts"))
sys.path.insert(0, str(SKILL.parent / "predict-aircraft-coefficients/scripts"))

from vehicle_cd_learning.readout_experiment import coefficient_loss
from vehicle_cd_learning.source_gradients import (
    backward,
    project_pair,
    source_losses,
    update_average,
)


def test_source_losses_preserve_the_original_masked_absolute_objective():
    pred = torch.tensor([[0.2, 0.3], [0.4, 0.6], [0.1, 0.8]], requires_grad=True)
    target = torch.tensor([[0.3, float("nan")], [0.2, 0.4], [0.5, 0.1]])
    std = torch.tensor([0.04, 0.2])
    losses = source_losses(pred, target, std, torch.tensor([0, 1, 1]))
    expected, _, _ = coefficient_loss(pred, target, std, [pred], False)
    torch.testing.assert_close(sum(losses.values()), expected)
    a = torch.autograd.grad(sum(losses.values()), pred, retain_graph=True)[0]
    b = torch.autograd.grad(expected, pred)[0]
    torch.testing.assert_close(a, b)
    assert a[0, 1] == 0


def test_pair_projection_removes_opposition_and_preserves_nonconflicting_vectors():
    a, b = torch.tensor([1.0, -1.0]), torch.tensor([-2.0, -1.0])
    x, y = project_pair(a, b)
    assert x @ y >= -1e-7 and (x + y) @ a >= -1e-7 and (x + y) @ b >= -1e-7
    c = torch.tensor([3.0, 2.0])
    x, y = project_pair(a, c)
    torch.testing.assert_close(x, a, atol=0, rtol=0)
    torch.testing.assert_close(y, c, atol=0, rtol=0)


def test_backward_changes_only_shared_parameters_and_preserves_other_sources():
    shared = torch.tensor([0.5, 0.2], requires_grad=True)
    private = torch.tensor(0.3, requires_grad=True)
    losses = {
        0: shared @ torch.tensor([1.0, -1.0]) + 2 * private,
        1: shared @ torch.tensor([-2.0, -1.0]) + 3 * private,
        2: shared @ torch.tensor([0.4, 0.6]) + 4 * private,
    }
    info = backward(losses, [shared], (0, 1))
    a, b = project_pair(torch.tensor([1.0, -1.0]), torch.tensor([-2.0, -1.0]))
    torch.testing.assert_close(shared.grad, a + b + torch.tensor([0.4, 0.6]))
    torch.testing.assert_close(private.grad, torch.tensor(9.0))
    assert info["projected"] and info["cosine_after"] >= -1e-7


def test_parameter_average_does_not_mutate_the_trainable_model():
    model = torch.nn.Linear(2, 1)
    average = torch.nn.Linear(2, 1)
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(2)
        for p in average.parameters():
            p.zero_()
    update_average(average, model, 0.75)
    for p in average.parameters():
        torch.testing.assert_close(p, torch.full_like(p, 0.5))
    for p in model.parameters():
        torch.testing.assert_close(p, torch.full_like(p, 2))
