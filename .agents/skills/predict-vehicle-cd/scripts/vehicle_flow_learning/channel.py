"""Train one physics-only operator for nondimensional laminar channel flow.

The domain is x in [0, 1], y in [0, h]. Fully developed steady flow has
u=(u(y), 0), p=-g*x, and viscosity 1/Re. Thus u_yy + Re*g = 0.
Wall velocities are enforced exactly by u=wall*(y/h)+(y/h)*(1-y/h)*network(...).
Training uses this equation only; the analytical solution appears exclusively
in validation. This deliberately restricted model is not a vehicle solver.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

TRAIN_RE = (20.0, 40.0, 60.0, 80.0)
TRAIN_G = (0.005, 0.01, 0.02, 0.025)
VAL_RE = (30.0, 50.0, 70.0)
VAL_G = (0.0075, 0.015, 0.0225)
TRAIN_HEIGHT = (0.8, 1.0, 1.2)
VAL_HEIGHT = (0.9, 1.1)
TRAIN_WALL = (0.0, 0.2, 0.4)
VAL_WALL = (0.1, 0.3)
TEST_RE = (25.0, 35.0, 45.0, 55.0, 65.0, 75.0)
TEST_G = (0.00625, 0.0125, 0.0175, 0.02375)
TEST_HEIGHT = (0.85, 1.05, 1.15)
TEST_WALL = (0.05, 0.15, 0.25, 0.35)


class ChannelOperator(nn.Module):
    """Shared operator for a family of fully developed flows."""

    def __init__(self, width: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, 1),
        )

    def forward(self, y, reynolds, gradient, height, wall):
        eta = y / height
        inputs = torch.cat(
            (
                2 * eta - 1,
                (reynolds - 50) / 30,
                (gradient - 0.015) / 0.01,
                (height - 1) / 0.2,
                (wall - 0.2) / 0.2,
            ),
            dim=-1,
        )
        return wall * eta + eta * (1 - eta) * self.net(inputs)


def residual(model, y, reynolds, gradient, height, wall):
    """Compute Re-scaled axial NS residual with automatic differentiation."""
    y = y.detach().requires_grad_(True)
    u = model(y, reynolds, gradient, height, wall)
    dy = torch.autograd.grad(u.sum(), y, create_graph=True)[0]
    dyy = torch.autograd.grad(dy.sum(), y, create_graph=True)[0]
    return dyy + reynolds * gradient


def synchronize(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def predict_grid(
    checkpoint_path, reynolds, gradient, height, wall, size, device_name="cpu"
):
    """Load weights and produce one complete 2D field without optimization."""
    if (
        not np.isfinite([reynolds, gradient, height, wall]).all()
        or min(reynolds, gradient, height) <= 0
        or size < 2
    ):
        raise ValueError(
            "Re, gradient, height and grid size must be positive, all inputs finite"
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    for value, key in (
        (reynolds, "train_re"),
        (gradient, "train_gradient"),
        (height, "train_height"),
        (wall, "train_wall"),
    ):
        if not min(checkpoint[key]) <= value <= max(checkpoint[key]):
            raise ValueError(f"{key} outside training range")
    device = torch.device(device_name)
    model = ChannelOperator(checkpoint["width"]).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    x_axis = torch.linspace(0, 1, size, device=device)
    y_axis = torch.linspace(0, height, size, device=device)
    y, x = torch.meshgrid(y_axis, x_axis, indexing="ij")
    y_flat = y.reshape(-1, 1)
    with torch.inference_mode():
        u = model(
            y_flat,
            torch.full_like(y_flat, reynolds),
            torch.full_like(y_flat, gradient),
            torch.full_like(y_flat, height),
            torch.full_like(y_flat, wall),
        ).reshape(size, size)
        v = torch.zeros_like(u)
        p = -gradient * x
    synchronize(device)
    return {
        name: value.cpu().numpy()
        for name, value in {"x": x, "y": y, "u": u, "v": v, "p": p}.items()
    }


def train(args):
    if (
        min(args.steps, args.batch_size, args.width, args.threads, args.log_every) < 1
        or args.lr <= 0
    ):
        raise ValueError("training sizes and learning rate must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    model = ChannelOperator(args.width).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_re = torch.tensor(TRAIN_RE, device=device)
    train_g = torch.tensor(TRAIN_G, device=device)
    train_height = torch.tensor(TRAIN_HEIGHT, device=device)
    train_wall = torch.tensor(TRAIN_WALL, device=device)
    history = []
    start = time.perf_counter()
    for step in range(1, args.steps + 1):
        height = train_height[
            torch.randint(len(train_height), (args.batch_size, 1), device=device)
        ]
        y = torch.rand(args.batch_size, 1, device=device) * height
        re = train_re[torch.randint(len(train_re), (args.batch_size, 1), device=device)]
        g = train_g[torch.randint(len(train_g), (args.batch_size, 1), device=device)]
        wall = train_wall[
            torch.randint(len(train_wall), (args.batch_size, 1), device=device)
        ]
        loss = residual(model, y, re, g, height, wall).square().mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % args.log_every == 0 or step == args.steps:
            history.append({"step": step, "physics_loss": float(loss.detach().cpu())})
            print(json.dumps(history[-1]), flush=True)
    synchronize(device)
    elapsed = time.perf_counter() - start
    checkpoint = {
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "width": args.width,
        "seed": args.seed,
        "steps": args.steps,
        "train_re": TRAIN_RE,
        "train_gradient": TRAIN_G,
        "train_height": TRAIN_HEIGHT,
        "train_wall": TRAIN_WALL,
        "equations": "steady fully developed incompressible 2D channel: u_yy + Re*g = 0; p=-g*x; v=0",
        "training_labels": "none; residual and hard boundary/continuity constraints",
    }
    ckpt = args.output_dir / "model.pt"
    torch.save(checkpoint, ckpt)
    (args.output_dir / "training.json").write_text(
        json.dumps(
            {
                "history": history,
                "elapsed_s": elapsed,
                "config": vars(args) | {"output_dir": str(args.output_dir)},
            },
            indent=2,
        )
        + "\n"
    )
    report = evaluate(ckpt, args.device, args.threads)
    (args.output_dir / "validation.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(report, indent=2), flush=True)


def evaluate(checkpoint_path, device_name="cpu", threads=2, split="val"):
    """Evaluate unseen Re and gradient combinations against an analytical solution."""
    torch.set_num_threads(threads)
    start = time.perf_counter()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    device = torch.device(device_name)
    model = ChannelOperator(checkpoint["width"]).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    rows = []
    if split not in ("val", "test"):
        raise ValueError("split must be val or test")
    re_values, g_values, heights, walls = (
        (VAL_RE, VAL_G, VAL_HEIGHT, VAL_WALL)
        if split == "val"
        else (TEST_RE, TEST_G, TEST_HEIGHT, TEST_WALL)
    )
    for re in re_values:
        for g in g_values:
            for height in heights:
                for wall in walls:
                    y = torch.linspace(0, height, 257, device=device).unsqueeze(-1)
                    r = torch.full_like(y, re)
                    grad = torch.full_like(y, g)
                    h = torch.full_like(y, height)
                    wall_input = torch.full_like(y, wall)
                    with torch.no_grad():
                        prediction = model(y, r, grad, h, wall_input)
                        reference = wall * y / height + re * g * y * (height - y) / 2
                        error = prediction - reference
                        relative_l2 = torch.linalg.vector_norm(
                            error
                        ) / torch.linalg.vector_norm(reference)
                        max_bc = torch.max(
                            torch.abs(
                                prediction[[0, -1], 0]
                                - torch.tensor([0.0, wall], device=device)
                            )
                        )
                        mae = error.abs().mean()
                        flow = torch.trapezoid(prediction[:, 0], y[:, 0])
                        exact_flow = wall * height / 2 + re * g * height**3 / 12
                    eq = residual(model, y, r, grad, h, wall_input)
                    x = torch.linspace(0, 1, 257, device=device)
                    pressure = -g * x
                    pressure_reference = (
                        -g * x
                    )  # Gauge p(0)=0; pressure is imposed structurally.
                    rows.append(
                        {
                            "re": re,
                            "pressure_gradient_magnitude": g,
                            "height": height,
                            "top_wall_velocity": wall,
                            "u_relative_l2": float(relative_l2.cpu()),
                            "u_mae": float(mae.cpu()),
                            "u_max_abs_error": float(error.abs().max().cpu()),
                            "flow_rate": float(flow.cpu()),
                            "flow_rate_relative_error": float(
                                abs(flow.item() - exact_flow) / exact_flow
                            ),
                            "wall_velocity_max_abs": float(max_bc.cpu()),
                            "pressure_max_abs_error_structural": float(
                                (pressure - pressure_reference).abs().max().cpu()
                            ),
                            "momentum_residual_rms": float(
                                (eq / re).square().mean().sqrt().detach().cpu()
                            ),
                        }
                    )
    synchronize(device)
    end_to_end = time.perf_counter() - start
    n = 10000
    grid = torch.linspace(0, 1, n, device=device).unsqueeze(-1)
    re = torch.full_like(grid, 50)
    grad = torch.full_like(grid, 0.015)
    height = torch.full_like(grid, 1.1)
    wall = torch.full_like(grid, 0.1)
    with torch.inference_mode():
        for _ in range(5):
            model(grid, re, grad, height, wall)
        synchronize(device)
        t0 = time.perf_counter()
        for _ in range(20):
            model(grid, re, grad, height, wall)
        synchronize(device)
    forward = (time.perf_counter() - t0) / 20
    t0 = time.perf_counter()
    fields = predict_grid(checkpoint_path, 50.0, 0.015, 1.1, 0.1, 100, device_name)
    cold_prediction = time.perf_counter() - t0
    return {
        "scope": f"{split}: unseen Re, pressure-gradient, height and wall-speed combinations, 2D steady fully developed laminar channel reduced to an ODE; analytical fields never used in training",
        "cases": rows,
        "mean_relative_l2": float(np.mean([row["u_relative_l2"] for row in rows])),
        "max_relative_l2": float(max(row["u_relative_l2"] for row in rows)),
        "max_wall_velocity": float(max(row["wall_velocity_max_abs"] for row in rows)),
        "flow_rate_mean_relative_error": float(
            np.mean([row["flow_rate_relative_error"] for row in rows])
        ),
        "flow_rate_max_relative_error": float(
            max(row["flow_rate_relative_error"] for row in rows)
        ),
        "continuity": "identically zero by fully developed u(y), v=0 architectural assumption, not a learned general divergence-free field",
        "pressure": "p=-g*x by structural assumption, pressure error zero is not a learned result",
        "inductive_bias": "y/h*(1-y/h) encodes the no-slip wall shape and coincides with the exact parabolic profile up to a constant coefficient; the trainable network is not a general 2D NS solver",
        "inference": {
            "device": str(device),
            "points": n,
            "mean_warm_forward_s": forward,
            "cold_checkpoint_to_2d_numpy_field_s": cold_prediction,
            "cold_load_and_case_evaluations_s": end_to_end,
            "finite_output": bool(all(np.isfinite(v).all() for v in fields.values())),
            "scope": "cold includes loading checkpoint, grid construction and u/v/p transfer; excludes disk export, geometry preprocessing, full 3D volume and vehicle physics",
        },
        "checkpoint": str(Path(checkpoint_path).resolve()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("train")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--log-every", type=int, default=500)
    q = sub.add_parser("evaluate")
    q.add_argument("checkpoint", type=Path)
    q.add_argument("--device", default="cpu")
    q.add_argument("--threads", type=int, default=2)
    q.add_argument("--split", choices=("val", "test"), default="val")
    q.add_argument("--output", type=Path)
    s = sub.add_parser("predict")
    s.add_argument("checkpoint", type=Path)
    s.add_argument("--re", type=float, required=True)
    s.add_argument("--gradient", type=float, required=True)
    s.add_argument("--height", type=float, required=True)
    s.add_argument("--wall", type=float, required=True)
    s.add_argument("--size", type=int, default=100)
    s.add_argument("--device", default="cpu")
    s.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "train":
        train(args)
    elif args.command == "evaluate":
        report = (
            json.dumps(
                evaluate(args.checkpoint, args.device, args.threads, args.split),
                indent=2,
                allow_nan=False,
            )
            + "\n"
        )
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(report)
        print(report)
    else:
        fields = predict_grid(
            args.checkpoint,
            args.re,
            args.gradient,
            args.height,
            args.wall,
            args.size,
            args.device,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.output,
            **fields,
            re=args.re,
            gradient=args.gradient,
            height=args.height,
            wall=args.wall,
        )
        print(
            json.dumps(
                {
                    "path": str(args.output.resolve()),
                    "shape": [args.size, args.size],
                    "parameters": {
                        "re": args.re,
                        "gradient": args.gradient,
                        "height": args.height,
                        "wall": args.wall,
                    },
                }
            )
        )


if __name__ == "__main__":
    main()
