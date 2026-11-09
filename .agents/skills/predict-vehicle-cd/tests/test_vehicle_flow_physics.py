import sys
from argparse import Namespace
from pathlib import Path

import pytest
import torch

SKILL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL / "scripts"))

from vehicle_flow_learning import channel, manufactured_2d


def test_manufactured_solution_satisfies_full_navier_stokes():
    class Exact(torch.nn.Module):
        def forward(self, xy, amplitude, reynolds):
            return manufactured_2d.exact_solution(xy, amplitude)

    torch.manual_seed(7)
    xy = torch.rand(30, 2, dtype=torch.float64)
    amplitude = torch.rand(30, 1, dtype=torch.float64) * 0.2 + 0.1
    reynolds = torch.rand(30, 1, dtype=torch.float64) * 8 + 2
    error = manufactured_2d.physics_residual(Exact(), xy, amplitude, reynolds)
    assert error.abs().max().item() < 1e-12


def test_channel_momentum_and_wall_conditions():
    class Exact(torch.nn.Module):
        def forward(self, y, reynolds, gradient, height, wall):
            return wall * y / height + reynolds * gradient * y * (height - y) / 2

    y = torch.linspace(0, 1.1, 101, dtype=torch.float64).reshape(-1, 1)
    re, gradient = torch.full_like(y, 35), torch.full_like(y, 0.013)
    height, wall = torch.full_like(y, 1.1), torch.full_like(y, 0.3)
    assert channel.residual(Exact(), y, re, gradient, height, wall).abs().max() < 1e-12
    model = channel.ChannelOperator().double()
    values = model(y, re, gradient, height, wall).detach()
    assert values[0].item() == pytest.approx(0.0, abs=1e-12)
    assert values[-1].item() == pytest.approx(0.3, abs=1e-12)


def test_manufactured_training_does_not_read_interior_field_labels(
    tmp_path, monkeypatch
):
    def forbidden(*args, **kwargs):
        raise AssertionError("training must not read the analytic interior solution")

    monkeypatch.setattr(manufactured_2d, "exact_solution", forbidden)
    monkeypatch.setattr(manufactured_2d, "evaluate", lambda *args: {})
    args = Namespace(
        output_dir=tmp_path / "model",
        seed=42,
        threads=2,
        device="cpu",
        width=8,
        lr=0.001,
        steps=2,
        batch_size=8,
        log_every=2,
    )
    manufactured_2d.train(args)
    checkpoint = torch.load(args.output_dir / "model.pt", weights_only=True)
    assert "no interior field labels" in checkpoint["training_labels"]


def test_channel_inference_rejects_untrained_condition(tmp_path):
    checkpoint = tmp_path / "model.pt"
    torch.save(
        {
            "width": 8,
            "model_state": channel.ChannelOperator(8).state_dict(),
            "train_re": channel.TRAIN_RE,
            "train_gradient": channel.TRAIN_G,
            "train_height": channel.TRAIN_HEIGHT,
            "train_wall": channel.TRAIN_WALL,
        },
        checkpoint,
    )
    result = channel.predict_grid(checkpoint, 30, 0.015, 1.0, 0.2, 8)
    assert all(value.shape == (8, 8) for value in result.values())
    with pytest.raises(ValueError, match="outside training range"):
        channel.predict_grid(checkpoint, 100, 0.015, 1.0, 0.2, 8)


def test_manufactured_inference_returns_grid_and_guards_conditions(tmp_path):
    checkpoint = tmp_path / "model.pt"
    torch.save(
        {
            "width": 8,
            "model_state": manufactured_2d.FlowOperator(8).state_dict(),
            "train_amplitudes": manufactured_2d.TRAIN_AMPLITUDES,
            "train_reynolds": manufactured_2d.TRAIN_REYNOLDS,
        },
        checkpoint,
    )
    fields = manufactured_2d.predict_grid(checkpoint, 0.15, 3.5, 8)
    assert set(fields) == {"x", "y", "u", "v", "p"}
    assert all(field.shape == (8, 8) for field in fields.values())
    with pytest.raises(ValueError, match="outside training range"):
        manufactured_2d.predict_grid(checkpoint, 0.15, 20, 8)
