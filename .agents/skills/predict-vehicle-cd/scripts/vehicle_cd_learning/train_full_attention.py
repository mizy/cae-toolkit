"""Train the full-data attention model as each verified CFD source becomes ready."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

from .data import load_dataset
from .evaluate import evaluate
from .predict import load_model
from .prepare import prepare

SOURCES = (
    ("ahmedml", "full-source.json"),
    ("windsorml", "full-source-deduplicated.json"),
    ("drivaerml", "full-training_manifest.json"),
)


def run(data_root, output, *, epochs=80, batch_size=8, seed=42, device="mps"):
    """@entry Keep source splits fixed and retain spatial/channel/view attention."""
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    output.mkdir(parents=True, exist_ok=True)
    config = {
        "epochs": epochs,
        "batch_size": batch_size,
        "seed": seed,
        "device": device,
        "encoder": "pyramid_detail_attention",
        "fusion": "attention",
        "model_size": "small",
        "preload": True,
        "learning_rate": 0.001,
        "render_config": {"ordinary_size": 512, "bottom_size": 512, "window_l": 1.3},
        "selection": "minimum validation standardized Cd/Cl MSE; test is never fitted",
        "scope": "separate CFD pools, within-family geometry holdout; raw diagnostic coverage retained",
    }
    protocol_path = output / "protocol.json"
    code_hashes = {
        name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
        for name in (
            "model.py",
            "train.py",
            "prepare.py",
            "render.py",
            "data.py",
            "evaluate.py",
            "predict.py",
        )
    }
    if protocol_path.exists():
        protocol = json.loads(protocol_path.read_text())
        if protocol["config"] != config:
            raise ValueError("Existing run uses another training protocol")
        if protocol["code_sha256"] != code_hashes:
            raise ValueError(
                "Training/preprocessing code changed; use a fresh experiment folder"
            )
    else:
        protocol = {
            "config": config,
            "sources": {},
            "runtime": {"python": sys.version, "torch": str(torch.__version__)},
            "code_sha256": code_hashes,
        }
        protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
    (output / "runner.pid").write_text(str(os.getpid()) + "\n")
    torch.set_num_threads(2)
    results = {}
    for name, filename in SOURCES:
        source = data_root / name / filename
        deadline = time.monotonic() + 4 * 3600
        print(f"waiting for verified full source: {name} {source}", flush=True)
        while True:
            try:
                manifest = json.loads(source.read_text())
                if manifest["cases"]:
                    break
            except (FileNotFoundError, json.JSONDecodeError):
                pass
            if time.monotonic() > deadline:
                raise TimeoutError(f"Full source not ready after four hours: {source}")
            time.sleep(5)
        fingerprint = hashlib.sha256(source.read_bytes()).hexdigest()
        previous = protocol["sources"].get(name)
        if previous and previous["sha256"] != fingerprint:
            raise ValueError(f"Source changed within an experiment: {source}")
        counts = {
            split: sum(c["split"] == split for c in manifest["cases"])
            for split in ("train", "val", "test")
        }
        protocol["sources"][name] = {
            "path": str(source),
            "sha256": fingerprint,
            "counts": counts,
        }
        protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
        folder = output / name
        prepared = folder / "prepared"
        if not (prepared / "dataset.json").exists():
            if prepared.exists():
                prepared.rename(folder / f"prepared-interrupted-{time.time_ns()}")
            print(f"preparing {name}: {counts}", flush=True)
            prepare(source, prepared, config["render_config"])
        dataset = load_dataset(prepared)
        if len(dataset["cases"]) != len(manifest["cases"]):
            raise ValueError("Prepared case count differs from full source")
        checkpoint_path = folder / "model/model.pt"
        if not (folder / "model/training_report.json").exists():
            if (folder / "model").exists():
                (folder / "model").rename(
                    folder / f"model-interrupted-{time.time_ns()}"
                )
            command = [
                sys.executable,
                "-u",
                "-m",
                "vehicle_cd_learning.train",
                str(prepared),
                "--output-dir",
                str(folder / "model"),
                "--encoder",
                config["encoder"],
                "--fusion",
                "attention",
                "--model-size",
                "small",
                "--preload",
                "--epochs",
                str(epochs),
                "--batch-size",
                str(batch_size),
                "--seed",
                str(seed),
                "--device",
                device,
            ]
            print(
                f"training {name}: spatial/channel/view attention, {counts}", flush=True
            )
            started = time.perf_counter()
            with (folder / "train.log").open("w") as log:
                subprocess.run(
                    command,
                    check=True,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=7200,
                )
            print(f"trained {name} in {time.perf_counter() - started:.1f}s", flush=True)
        _, checkpoint = load_model(checkpoint_path)
        trained_ids = {c["case_id"] for c in dataset["cases"] if c["split"] == "train"}
        if set(checkpoint["train_ids"]) != trained_ids:
            raise ValueError(
                "Checkpoint training IDs differ from official training split"
            )
        if (
            checkpoint["dataset_sha256"]
            != hashlib.sha256((prepared / "dataset.json").read_bytes()).hexdigest()
        ):
            raise ValueError("Checkpoint was trained on another dataset")
        evaluation_path = folder / "test/evaluation.json"
        if evaluation_path.exists():
            report = json.loads(evaluation_path.read_text())
        else:
            report = evaluate(
                checkpoint_path,
                prepared,
                folder / "test",
                pair_tolerance=0.001,
                diagnostic=True,
            )
        training = json.loads((folder / "model/training_report.json").read_text())
        results[name] = {
            "counts": counts,
            "checkpoint": str(checkpoint_path),
            "best_epoch": training["best_epoch"],
            "parameter_count": training["parameter_count"],
            "metrics": report["metrics"],
            "mean_baseline": report["mean_baseline"],
            "default_rejection_count": report["default_rejection_count"],
            "paired_changes": report["paired_changes"],
            "family_evaluation_scope": report["family_evaluation_scope"],
        }
        (output / "results.json").write_text(
            json.dumps({"config": config, "pools": results}, indent=2) + "\n"
        )
        print(
            f"evaluated {name}: {report['metrics']}; default_rejections={report['default_rejection_count']}",
            flush=True,
        )
    print("FULL_ATTENTION_TRAINING_COMPLETE", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", type=Path, default=Path("outputs/predict-vehicle-cd/public-data")
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    run(
        args.data_root,
        args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()
