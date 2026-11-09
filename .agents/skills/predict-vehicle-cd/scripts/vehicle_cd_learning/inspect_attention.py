"""Inspect learned gates on five complete views of a frozen dataset case."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.nn import functional as F

from .data import VIEWS, CaseDataset, load_dataset
from .predict import load_model, predict


def inspect(checkpoint_path, prepared, case_id, output):
    """@entry Export complete-view spatial gates and per-case view weights."""
    dataset = load_dataset(prepared)
    model, checkpoint = load_model(checkpoint_path)
    if (
        checkpoint["dataset_sha256"]
        != hashlib.sha256((prepared / "dataset.json").read_bytes()).hexdigest()
    ):
        raise ValueError(
            "Attention case must come from the checkpoint's frozen dataset"
        )
    if (
        model.variant != "five"
        or model.view_attention is None
        or getattr(model.image_encoder, "spatial", None) is None
    ):
        raise ValueError("Inspection requires five-view spatial and view attention")
    cases = [case for case in dataset["cases"] if case["case_id"] == case_id]
    if not cases:
        raise ValueError(f"Unknown dataset case: {case_id}")
    case = cases[0]
    data = CaseDataset(
        prepared,
        cases,
        checkpoint["parameter_mean"],
        checkpoint["parameter_std"],
        checkpoint["target_mean"],
        checkpoint["target_std"],
    )
    views, parameters, _ = data[0]
    body, bottom, weights = [], [], []
    handles = [
        model.image_encoder.spatial.register_forward_hook(
            lambda m, i, o: body.append(o.detach().numpy())
        ),
        model.bottom_encoder.spatial.register_forward_hook(
            lambda m, i, o: bottom.append(o.detach().numpy())
        ),
        model.view_attention.register_forward_hook(
            lambda m, i, o: weights.append(o[1].detach().numpy()[0])
        ),
    ]
    with torch.inference_mode():
        raw = model(
            {name: value[None] for name, value in views.items()}, parameters[None]
        ).numpy()[0]
    for handle in handles:
        handle.remove()
    predicted = raw * np.asarray(checkpoint["target_std"]) + checkpoint["target_mean"]
    if not np.isfinite(predicted).all():
        raise ValueError("Nonfinite diagnostic prediction")
    maps = dict(zip(VIEWS, [*body[0][:, 0], bottom[0][0, 0]]))
    try:
        predict(
            model,
            checkpoint,
            {name: value.numpy() for name, value in views.items()},
            case["parameters"],
        )
        rejection = None
    except ValueError as error:
        rejection = str(error)
    output.mkdir(parents=True, exist_ok=False)
    tile = 256
    canvas = Image.new("RGB", (tile * 5, tile * 2 + 64), "#20242a")
    draw = ImageDraw.Draw(canvas)
    summary = {}
    for index, name in enumerate(VIEWS):
        image, gate = views[name].numpy(), maps[name]
        mask = image[1].astype(bool)
        gray = (np.clip(image[0], 0, 1) * 255).astype(np.uint8)
        depth = np.repeat(gray[..., None], 3, axis=2)
        depth[~mask] = (32, 36, 42)
        mask_small = (
            F.adaptive_max_pool2d(views[name][None, 1:2], gate.shape)[0, 0]
            .numpy()
            .astype(bool)
        )
        summary[name] = {
            "view_weight": float(weights[0][index]),
            "spatial_foreground_mean": float(gate[mask_small].mean()),
            "spatial_background_mean": float(gate[~mask_small].mean())
            if (~mask_small).any()
            else None,
            "spatial_min": float(gate.min()),
            "spatial_max": float(gate.max()),
        }
        color = np.stack(
            (gate * 255, np.full_like(gate, 64), (1 - gate) * 255), axis=-1
        ).astype(np.uint8)
        heat = np.asarray(
            Image.fromarray(color).resize(
                (image.shape[2], image.shape[1]), Image.Resampling.BILINEAR
            )
        ).copy()
        heat[~mask] = (32, 36, 42)
        for row, pixels in enumerate((depth, heat)):
            canvas.paste(
                Image.fromarray(pixels).resize((tile, tile)),
                (index * tile, row * (tile + 24) + 24),
            )
        draw.text((index * tile + 8, 6), name, fill="white")
        draw.text(
            (index * tile + 8, tile + 30),
            f"view gate {weights[0][index]:.3f}",
            fill="white",
        )
    draw.text(
        (8, tile * 2 + 52),
        "Spatial sigmoid: fixed blue=0 / red=1. Gates are model weights, not aerodynamic contribution percentages.",
        fill="white",
    )
    canvas.save(output / "complete-view-gates.png")
    np.savez_compressed(output / "spatial-gates.npz", **maps, view_weights=weights[0])
    profile = checkpoint.get("condition_profiles", {}).get(
        case.get("condition_profile")
    )
    labeled = profile["labeled_targets"] if profile else checkpoint["target_names"]
    report = {
        "case_id": case_id,
        "split": case["split"],
        "data_kind": dataset["data_kind"],
        "family_id": case["family_id"],
        "dataset_sha256": checkpoint["dataset_sha256"],
        "truth": dict(zip(checkpoint["target_names"], case["targets"])),
        "prediction": {
            name: float(value)
            for name, value in zip(checkpoint["target_names"], predicted)
            if name in labeled
        },
        "unsupported_targets": [
            name for name in checkpoint["target_names"] if name not in labeled
        ],
        **({"condition_profile": case["condition_profile"]} if profile else {}),
        "default_rejection": rejection,
        "views": summary,
        "spatial_multiplier": 2.0 if model.image_encoder.detail_pooling else 1.0,
        "interpretation": "Learned complete-view gates; not physical drag attribution or validated delta-Cd sensitivity",
    }
    (output / "attention.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("prepared", type=Path)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    print(
        json.dumps(
            inspect(args.checkpoint, args.prepared, args.case_id, args.output_dir),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
