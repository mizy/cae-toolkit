"""Fixed-epoch grouped out-of-fold evaluation, independent of final all-sample fitting."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from .data import CaseDataset, load_dataset
from .evaluate import metrics
from .predict import load_model, predict


def run(
    prepared,
    output,
    folds=3,
    epochs=60,
    seed=42,
    augment=False,
    model_size="standard",
    encoder="plain",
    device_name="cpu",
    fusion="concat",
):
    source = load_dataset(prepared)
    families = sorted({c["family_id"] for c in source["cases"]})
    if not 2 <= folds <= len(families):
        raise ValueError("folds must be between 2 and the number of families")
    np.random.default_rng(seed).shuffle(families)
    output.mkdir(parents=True, exist_ok=False)
    results = []
    torch.set_num_threads(2)
    for index, heldout in enumerate(np.array_split(families, folds)):
        train = [
            dict(c, split="train")
            for c in source["cases"]
            if c["family_id"] not in heldout
        ]
        test = [c for c in source["cases"] if c["family_id"] in heldout]
        if {c["geometry_hash"] for c in train} & {c["geometry_hash"] for c in test}:
            raise ValueError("Cross-validation geometry appears in both train and test")
        folder = output / f"fold-{index}"
        (folder / "data/views").mkdir(parents=True)
        for case in train:
            destination = folder / "data" / case["views"]
            if not destination.exists():
                os.link(prepared / case["views"], destination)
        (folder / "data/dataset.json").write_text(
            json.dumps({**source, "cases": train}, indent=2) + "\n"
        )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "vehicle_cd_learning.train",
                str(folder / "data"),
                "--output-dir",
                str(folder / "model"),
                "--fit-all",
                "--epochs",
                str(epochs),
                "--batch-size",
                "4",
                "--seed",
                str(seed),
                "--model-size",
                model_size,
                "--encoder",
                encoder,
                "--fusion",
                fusion,
                "--device",
                device_name,
            ]
            + (["--augment"] if augment else []),
            check=True,
        )
        model, checkpoint = load_model(folder / "model/model.pt")
        assert not set(checkpoint["train_ids"]) & {c["case_id"] for c in test}
        ds = CaseDataset(
            prepared,
            test,
            checkpoint["parameter_mean"],
            checkpoint["parameter_std"],
            checkpoint["target_mean"],
            checkpoint["target_std"],
        )
        for i, case in enumerate(test):
            views, parameters, _ = ds[i]
            with torch.inference_mode():
                standardized = model(
                    {k: v[None] for k, v in views.items()}, parameters[None]
                ).numpy()[0]
            predicted = (
                standardized * np.asarray(checkpoint["target_std"])
                + checkpoint["target_mean"]
            )
            try:
                predict(
                    model,
                    checkpoint,
                    {k: v.numpy() for k, v in views.items()},
                    case["parameters"],
                )
                rejection = None
            except ValueError as error:
                rejection = str(error)
            results.append(
                {
                    "case_id": case["case_id"],
                    "fold": index,
                    "truth": case["targets"],
                    "prediction": predicted.tolist(),
                    "train_mean": checkpoint["target_mean"],
                    "default_rejection": rejection,
                }
            )
        print("completed fold", index, flush=True)
    report = {
        "scope": "Fixed-epoch grouped out-of-fold predictions against mixed-source reference labels, not verified CFD accuracy",
        "folds": folds,
        "augmentation": "sampling_jitter_v1" if augment else "none",
        "model_size": model_size,
        "encoder": encoder,
        "fusion": fusion,
        "device": device_name,
        "dataset_sha256": hashlib.sha256(
            (prepared / "dataset.json").read_bytes()
        ).hexdigest(),
        "epochs": epochs,
        "seed": seed,
        "count": len(results),
        "metrics": metrics(
            [r["truth"] for r in results],
            [r["prediction"] for r in results],
            source["target_names"],
        ),
        "mean_baseline": metrics(
            [r["truth"] for r in results],
            [r["train_mean"] for r in results],
            source["target_names"],
        ),
        "cases": results,
    }
    (output / "out-of-fold.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument(
        "--model-size", choices=("standard", "small"), default="standard"
    )
    parser.add_argument(
        "--encoder",
        choices=(
            "plain",
            "pyramid",
            "pyramid_attention",
            "pyramid_detail",
            "pyramid_detail_attention",
        ),
        default="plain",
    )
    parser.add_argument("--fusion", choices=("concat", "attention"), default="concat")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.prepared,
                args.output_dir,
                args.folds,
                args.epochs,
                augment=args.augment,
                model_size=args.model_size,
                encoder=args.encoder,
                device_name=args.device,
                fusion=args.fusion,
            )["metrics"]
        )
    )


if __name__ == "__main__":
    main()
