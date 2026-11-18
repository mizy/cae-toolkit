"""Train-only gradient diagnostics for the frozen paired continuation objectives."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from aircraft_learning.scene_experiment import SceneCases
from torch.utils.data import default_collate

from .dimensionless import load_model
from .pair_experiment import load, pair_loss
from .train import move_batch


def category(name):
    if name.startswith(("image_encoder", "bottom_encoder")):
        return "projection" if ".project." in name else "image_features"
    return "head" if name.startswith("head.") else "conditions_and_attention"


def run(experiment, output, device_name="mps"):
    if output.exists():
        raise FileExistsError("Use a fresh gradient report")
    protocol, data = load(experiment)
    cases = [c for c in data["cases"] if c["split"] == "train"]
    values = SceneCases(Path(protocol["prepared"]), cases, protocol["normalization"])
    with np.load(experiment / "pairs.npz", allow_pickle=False) as f:
        left, right = f["left"][0], f["right"][0]
    device = torch.device(device_name)
    torch.set_num_threads(2)
    stats = protocol["normalization"]
    std = torch.tensor(stats["target_std"], dtype=torch.float32, device=device)
    delta_std = torch.tensor(protocol["delta_std"], dtype=torch.float32, device=device)
    results = {}
    for arm, checkpoint in (
        ("A0", Path(protocol["checkpoint"])),
        ("A2", experiment / "A2/model.pt"),
    ):
        model, _ = load_model(checkpoint)
        model.to(device).eval()
        named = list(model.named_parameters())
        groups = {
            g: [i for i, (name, _) in enumerate(named) if category(name) == g]
            for g in (
                "image_features",
                "projection",
                "conditions_and_attention",
                "head",
            )
        }
        records = []
        for start in range(0, 6 * protocol["batch_size"], protocol["batch_size"]):
            a, b = (
                left[start : start + protocol["batch_size"]],
                right[start : start + protocol["batch_size"]],
            )
            batch = default_collate([values[int(i)] for i in np.concatenate((a, b))])
            views, x, targets = move_batch(batch[:3], device)
            rim = torch.tensor(
                [cases[i]["source"] == "drivaerrim" for i in a], device=device
            )
            pred = model(views, x)
            _, delta = pair_loss(pred, targets, rim, std, delta_std, 0.1)
            observed = torch.isfinite(targets)
            squares = (pred - torch.nan_to_num(targets)).square()
            rim_rows = torch.cat((rim, rim))[:, None]
            absolute_rim = squares[observed & rim_rows].sum() / observed.sum()
            absolute_other = squares[observed & ~rim_rows].sum() / observed.sum()
            losses = {
                "absolute_rim": absolute_rim,
                "absolute_other": absolute_other,
                "weighted_delta": 0.1 * delta,
            }
            gradients = {}
            for name, loss in losses.items():
                gradient = torch.autograd.grad(
                    loss, [p for _, p in named], retain_graph=True, allow_unused=True
                )
                gradients[name] = {
                    g: torch.cat(
                        [
                            (
                                gradient[i]
                                if gradient[i] is not None
                                else torch.zeros_like(named[i][1])
                            ).flatten()
                            for i in ids
                        ]
                    ).detach()
                    for g, ids in groups.items()
                }
            record = {
                "batch": start // protocol["batch_size"],
                "rim_pairs": int(rim.sum()),
                "losses": {k: float(v.detach()) for k, v in losses.items()},
                "groups": {},
            }
            for group in groups:
                d, other, same = (
                    gradients[k][group]
                    for k in ("weighted_delta", "absolute_other", "absolute_rim")
                )
                norm = {k: float(gradients[k][group].norm()) for k in losses}

                def cosine(a, b):
                    denominator = a.norm() * b.norm()
                    return float((a @ b) / denominator) if denominator > 0 else None

                record["groups"][group] = {
                    "gradient_norm": norm,
                    "delta_vs_other_cosine": cosine(d, other),
                    "delta_vs_rim_absolute_cosine": cosine(d, same),
                }
            records.append(record)
        results[arm] = {
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "records": records,
        }
        del model
        if device.type == "mps":
            torch.mps.empty_cache()
    summary = {
        arm: {
            g: {
                "mean_weighted_delta_grad_norm": float(
                    np.mean(
                        [
                            r["groups"][g]["gradient_norm"]["weighted_delta"]
                            for r in values["records"]
                        ]
                    )
                ),
                "mean_other_absolute_grad_norm": float(
                    np.mean(
                        [
                            r["groups"][g]["gradient_norm"]["absolute_other"]
                            for r in values["records"]
                        ]
                    )
                ),
                "mean_delta_vs_other_cosine": float(
                    np.mean(
                        [
                            r["groups"][g]["delta_vs_other_cosine"]
                            for r in values["records"]
                            if r["groups"][g]["delta_vs_other_cosine"] is not None
                        ]
                    )
                ),
            }
            for g in groups
        }
        for arm, values in results.items()
    }
    output.write_text(
        json.dumps(
            {
                "scope": "First6 actual frozen training batches; gradients only, no optimizer step. Descriptive local interference/scale evidence, not proof of a unique cause.",
                "results": results,
                "summary": summary,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("experiment", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--device", default="mps")
    a = p.parse_args()
    run(a.experiment, a.output, a.device)
