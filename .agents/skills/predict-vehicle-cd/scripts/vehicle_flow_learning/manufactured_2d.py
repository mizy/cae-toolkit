"""Physics-only conditional 2D Navier–Stokes manufactured-solution experiment.

Unit square, steady incompressible flow, prescribed body force and velocity
on all walls. The force and wall values are derived from a manufactured
solution, so they encode knowledge of that solution. Interior field values
are never used as training labels. This is not a vehicle or turbulence model.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

TRAIN_AMPLITUDES = (0.1, 0.2, 0.3)
TRAIN_REYNOLDS = (2.0, 5.0, 10.0)
VALID_AMPLITUDES = (0.15, 0.25)
VALID_REYNOLDS = (3.5, 7.5)


class FlowOperator(nn.Module):
    """Predict u, v and p with shared parameters for all cases."""

    def __init__(self, width=48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, 3),
        )

    def forward(self, xy, amplitude, reynolds):
        features = torch.cat(
            (2 * xy - 1, (amplitude - 0.2) / 0.1, (reynolds - 6.0) / 4.0),
            dim=1,
        )
        return self.net(features)


def body_force(xy, amplitude, reynolds):
    """Manufactured source for the stated nondimensional NS convention."""
    x, y = xy[:, :1], xy[:, 1:2]
    sx, cx = torch.sin(math.pi * x), torch.cos(math.pi * x)
    sy, cy = torch.sin(math.pi * y), torch.cos(math.pi * y)
    ux = amplitude * math.pi * sx * cy
    vy = -amplitude * math.pi * cx * sy
    fx = amplitude.square() * math.pi**3 * sx * cx
    fx = fx + amplitude * math.pi * cx * sy + 2 * math.pi**2 * ux / reynolds
    fy = amplitude.square() * math.pi**3 * sy * cy
    fy = fy + amplitude * math.pi * sx * cy + 2 * math.pi**2 * vy / reynolds
    return torch.cat((fx, fy), dim=1)


def boundary_velocity(xy, amplitude):
    """Prescribed velocity at the square's boundary, not interior labels."""
    x, y = xy[:, :1], xy[:, 1:2]
    return torch.cat(
        (
            amplitude * math.pi * torch.sin(math.pi * x) * torch.cos(math.pi * y),
            -amplitude * math.pi * torch.cos(math.pi * x) * torch.sin(math.pi * y),
        ),
        dim=1,
    )


def exact_solution(xy, amplitude):
    """Analytic field used only for evaluation."""
    x, y = xy[:, :1], xy[:, 1:2]
    return torch.cat(
        (
            amplitude * math.pi * torch.sin(math.pi * x) * torch.cos(math.pi * y),
            -amplitude * math.pi * torch.cos(math.pi * x) * torch.sin(math.pi * y),
            amplitude * torch.sin(math.pi * x) * torch.sin(math.pi * y),
        ),
        dim=1,
    )


def physics_residual(model, xy, amplitude, reynolds):
    """Evaluate steady 2D continuity and momentum by automatic differentiation."""
    xy = xy.detach().requires_grad_(True)
    prediction = model(xy, amplitude, reynolds)
    u, v, p = (prediction[:, i : i + 1] for i in range(3))
    grads = [torch.autograd.grad(q.sum(), xy, create_graph=True)[0] for q in (u, v, p)]
    gu, gv, gp = grads
    laplacians = []
    for grad in (gu, gv):
        xx = torch.autograd.grad(grad[:, 0].sum(), xy, create_graph=True)[0][:, :1]
        yy = torch.autograd.grad(grad[:, 1].sum(), xy, create_graph=True)[0][:, 1:2]
        laplacians.append(xx + yy)
    force = body_force(xy, amplitude, reynolds)
    ru = (
        u * gu[:, :1]
        + v * gu[:, 1:2]
        + gp[:, :1]
        - laplacians[0] / reynolds
        - force[:, :1]
    )
    rv = (
        u * gv[:, :1]
        + v * gv[:, 1:2]
        + gp[:, 1:2]
        - laplacians[1] / reynolds
        - force[:, 1:2]
    )
    return torch.cat((gu[:, :1] + gv[:, 1:2], ru, rv), dim=1)


def _parameters(count, device):
    amplitude = torch.tensor(TRAIN_AMPLITUDES, device=device)
    reynolds = torch.tensor(TRAIN_REYNOLDS, device=device)
    return (
        amplitude[torch.randint(len(amplitude), (count, 1), device=device)],
        reynolds[torch.randint(len(reynolds), (count, 1), device=device)],
    )


def train(args):
    if (
        min(args.steps, args.batch_size, args.width, args.threads, args.log_every) < 1
        or args.lr <= 0
    ):
        raise ValueError(
            "steps, batch size, width, threads and learning rate must be positive"
        )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    model = FlowOperator(args.width).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history = []
    start = time.perf_counter()
    for step in range(1, args.steps + 1):
        n = args.batch_size
        amplitude, reynolds = _parameters(n, device)
        xy = torch.rand(n, 2, device=device)
        r = physics_residual(model, xy, amplitude, reynolds)
        boundary = torch.rand(n, 2, device=device)
        side = torch.randint(4, (n,), device=device)
        boundary[side == 0, 0] = 0
        boundary[side == 1, 0] = 1
        boundary[side == 2, 1] = 0
        boundary[side == 3, 1] = 1
        pred_boundary = model(boundary, amplitude, reynolds)[:, :2]
        wall_loss = (
            (pred_boundary - boundary_velocity(boundary, amplitude)).square().mean()
        )
        origin = torch.zeros(n, 2, device=device)
        gauge_loss = model(origin, amplitude, reynolds)[:, 2].square().mean()
        loss = r.square().mean() + 10 * wall_loss + gauge_loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % args.log_every == 0 or step == args.steps:
            row = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "residual_mse": float(r.square().mean().detach().cpu()),
                "boundary_mse": float(wall_loss.detach().cpu()),
                "gauge_mse": float(gauge_loss.detach().cpu()),
            }
            history.append(row)
            print(json.dumps(row), flush=True)
    checkpoint = {
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "width": args.width,
        "seed": args.seed,
        "steps": args.steps,
        "train_amplitudes": TRAIN_AMPLITUDES,
        "train_reynolds": TRAIN_REYNOLDS,
        "scope": "2D steady incompressible manufactured body force, prescribed wall velocities",
        "training_labels": "no interior field labels; manufactured source and boundary are analytic",
    }
    path = args.output_dir / "model.pt"
    torch.save(checkpoint, path)
    config = vars(args) | {"output_dir": str(args.output_dir)}
    (args.output_dir / "training.json").write_text(
        json.dumps(
            {
                "config": config,
                "history": history,
                "elapsed_s": time.perf_counter() - start,
            },
            indent=2,
        )
        + "\n"
    )
    report = evaluate(path, args.device, args.threads)
    (args.output_dir / "validation.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(report, indent=2), flush=True)


def evaluate(path, device_name="cpu", threads=2):
    """Evaluate four unseen interpolating condition pairs on a fixed grid."""
    torch.set_num_threads(threads)
    start = time.perf_counter()
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    device = torch.device(device_name)
    model = FlowOperator(checkpoint["width"]).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    axis = torch.linspace(0, 1, 33, device=device)
    x, y = torch.meshgrid(axis, axis, indexing="ij")
    xy = torch.stack((x.flatten(), y.flatten()), dim=1)
    cases = []
    for amplitude in VALID_AMPLITUDES:
        for reynolds in VALID_REYNOLDS:
            a = torch.full((len(xy), 1), amplitude, device=device)
            re = torch.full_like(a, reynolds)
            with torch.no_grad():
                pred = model(xy, a, re)
                target = exact_solution(xy, a)
                error = pred - target
                rel = torch.linalg.vector_norm(error, dim=0) / torch.linalg.vector_norm(
                    target, dim=0
                )
            res = physics_residual(model, xy, a, re)
            cases.append(
                {
                    "amplitude": amplitude,
                    "reynolds": reynolds,
                    "relative_l2_u_v_p": rel.cpu().tolist(),
                    "mae_u_v_p": error.abs().mean(0).cpu().tolist(),
                    "residual_rms_div_momentum_x_y": res.square()
                    .mean(0)
                    .sqrt()
                    .detach()
                    .cpu()
                    .tolist(),
                }
            )
    elapsed = time.perf_counter() - start
    xyb = torch.rand(10000, 2, device=device)
    a = torch.full((10000, 1), 0.15, device=device)
    re = torch.full_like(a, 3.5)
    with torch.inference_mode():
        for _ in range(3):
            model(xyb, a, re)
        if device.type == "mps":
            torch.mps.synchronize()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(10):
            model(xyb, a, re)
        if device.type == "mps":
            torch.mps.synchronize()
        if device.type == "cuda":
            torch.cuda.synchronize()
        forward = (time.perf_counter() - t) / 10
    return {
        "scope": "2D manufactured steady laminar square, interpolated amplitude/Re; not vehicle physics",
        "checkpoint": str(Path(path).resolve()),
        "validation_cases": cases,
        "mean_relative_l2_u_v_p": np.mean(
            [c["relative_l2_u_v_p"] for c in cases], axis=0
        ).tolist(),
        "inference": {
            "points": 10000,
            "device": str(device),
            "warm_forward_s": forward,
            "cold_load_plus_validation_s": elapsed,
            "limitations": "forward only excludes geometry, postprocessing and 3D",
        },
        "warning": "Manufactured forcing and wall values are derived from the analytic solution; no interior field labels used for training.",
    }


def predict_grid(path, amplitude, reynolds, size, device_name="cpu"):
    """Decode a square field for a supported condition with fixed weights."""
    if not np.isfinite([amplitude, reynolds]).all() or size < 2:
        raise ValueError("condition values must be finite and grid size at least 2")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    for value, key in (
        (amplitude, "train_amplitudes"),
        (reynolds, "train_reynolds"),
    ):
        if not min(checkpoint[key]) <= value <= max(checkpoint[key]):
            raise ValueError(f"{key} outside training range")
    device = torch.device(device_name)
    model = FlowOperator(checkpoint["width"]).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    axis = torch.linspace(0, 1, size, device=device)
    x, y = torch.meshgrid(axis, axis, indexing="ij")
    xy = torch.stack((x.flatten(), y.flatten()), dim=1)
    a = torch.full((len(xy), 1), amplitude, device=device)
    re = torch.full_like(a, reynolds)
    with torch.inference_mode():
        output = model(xy, a, re).reshape(size, size, 3).cpu().numpy()
    if not np.isfinite(output).all():
        raise ValueError("nonfinite predicted field")
    return {
        "x": x.cpu().numpy(),
        "y": y.cpu().numpy(),
        "u": output[:, :, 0],
        "v": output[:, :, 1],
        "p": output[:, :, 2],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("train")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--width", type=int, default=48)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--log-every", type=int, default=500)
    q = sub.add_parser("evaluate")
    q.add_argument("checkpoint", type=Path)
    q.add_argument("--device", default="cpu")
    q.add_argument("--threads", type=int, default=2)
    predict = sub.add_parser("predict")
    predict.add_argument("checkpoint", type=Path)
    predict.add_argument("--amplitude", type=float, required=True)
    predict.add_argument("--re", type=float, required=True)
    predict.add_argument("--size", type=int, default=100)
    predict.add_argument("--device", default="cpu")
    predict.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "train":
        train(args)
    elif args.command == "evaluate":
        print(
            json.dumps(evaluate(args.checkpoint, args.device, args.threads), indent=2)
        )
    else:
        fields = predict_grid(
            args.checkpoint, args.amplitude, args.re, args.size, args.device
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output, **fields)
        print(
            json.dumps(
                {"path": str(args.output.resolve()), "shape": [args.size, args.size]}
            )
        )


if __name__ == "__main__":
    main()
