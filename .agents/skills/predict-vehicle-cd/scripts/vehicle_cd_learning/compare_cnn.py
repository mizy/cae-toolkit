"""Compare whole-view CNN pooling and view attention on one frozen labeled split."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .data import load_dataset
from .evaluate import evaluate

ARMS = (
    ("parameters", "parameters", "pyramid", "concat"),
    ("pyramid", "five", "pyramid", "concat"),
    ("detail", "five", "pyramid_detail", "concat"),
    ("detail_attention", "five", "pyramid_detail", "attention"),
)


def compare(prepared, output, *, seeds, epochs=60, batch_size=4, device="cpu"):
    """@entry Train all fixed ablations before opening their common test split."""
    data = load_dataset(prepared)
    torch.set_num_threads(2)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Supply distinct seeds")
    counts = {
        split: sum(c["split"] == split for c in data["cases"])
        for split in ("train", "val", "test")
    }
    if not all(counts.values()):
        raise ValueError("Comparison requires nonempty train, val and test splits")
    output.mkdir(parents=True, exist_ok=False)
    fingerprint = hashlib.sha256((prepared / "dataset.json").read_bytes()).hexdigest()
    protocol = {
        "dataset_sha256": fingerprint,
        "data_kind": data["data_kind"],
        "split_policy": data.get("split_policy", "family_disjoint"),
        "counts": counts,
        "render_config": data["render_config"],
        "seeds": seeds,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": 0.001,
        "model_size": "small",
        "device": device,
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "code_sha256": {
            name: hashlib.sha256(
                (Path(__file__).parent / name).read_bytes()
            ).hexdigest()
            for name in (
                "model.py",
                "train.py",
                "data.py",
                "predict.py",
                "evaluate.py",
                "compare_cnn.py",
            )
        },
        "augmentation": "none",
        "checkpoint_selection": "minimum validation MSE within fixed epoch budget",
        "test_policy": "evaluate every arm once after all training completes",
        "prediction_mode": "raw diagnostics for all cases; retain strict inference rejections",
        "arms": [
            {"name": name, "variant": variant, "encoder": encoder, "fusion": fusion}
            for name, variant, encoder, fusion in ARMS
        ],
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    runs = []
    for seed in seeds:
        for name, variant, encoder, fusion in ARMS:
            folder = output / f"seed-{seed}" / name
            folder.mkdir(parents=True)
            command = [
                sys.executable,
                "-m",
                "vehicle_cd_learning.train",
                str(prepared),
                "--output-dir",
                str(folder / "model"),
                "--variant",
                variant,
                "--encoder",
                encoder,
                "--fusion",
                fusion,
                "--model-size",
                "small",
                "--epochs",
                str(epochs),
                "--batch-size",
                str(batch_size),
                "--seed",
                str(seed),
                "--device",
                device,
            ]
            start = time.perf_counter()
            with (folder / "train.log").open("w") as log:
                subprocess.run(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=1800,
                )
            runs.append(
                {
                    "seed": seed,
                    "arm": name,
                    "folder": folder,
                    "training_seconds": time.perf_counter() - start,
                }
            )
            print(f"trained seed={seed} arm={name}", flush=True)
    if (
        hashlib.sha256((prepared / "dataset.json").read_bytes()).hexdigest()
        != fingerprint
    ):
        raise ValueError("Prepared dataset changed during comparison")
    for row in runs:
        folder = row.pop("folder")
        result = evaluate(
            folder / "model/model.pt", prepared, folder / "test", diagnostic=True
        )
        row.update(
            metrics=result["metrics"],
            mean_baseline=result["mean_baseline"],
            count=result["count"],
            checkpoint=str(folder / "model/model.pt"),
            median_inference_seconds=result["median_inference_seconds"],
            default_rejection_count=result["default_rejection_count"],
            inference_timing_scope=result["inference_timing_scope"],
        )
        print(
            f"tested seed={row['seed']} arm={row['arm']} {row['metrics']}", flush=True
        )
    summary = {}
    for name, *_ in ARMS:
        selected = [r for r in runs if r["arm"] == name]
        summary[name] = {}
        for target in data["target_names"]:
            summary[name][target] = {
                metric: {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "by_seed": dict(zip(map(str, seeds), values)),
                }
                for metric in ("mae", "rmse", "p95_absolute_error")
                for values in [[r["metrics"][target][metric] for r in selected]]
            }
    report = {
        "protocol": protocol,
        "summary": summary,
        "runs": runs,
        "interpretation": "Repeated training seeds on one fixed test set; not independent datasets or proof of unseen-family accuracy",
    }
    (output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 42, 73])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    result = compare(
        args.prepared,
        args.output_dir,
        seeds=args.seeds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
    )
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
