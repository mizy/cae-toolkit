"""Paired three-seed full-network continuation with a pre-compression residual path."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from aircraft_learning.scene_experiment import SceneCases, sampling_weights
from torch.utils.data import default_collate

from .head_experiment import physical
from .kernel_experiment import load as load_parent
from .pair_experiment import predictions
from .pooled_residual import CONTRACT, initialize
from .readout_experiment import absolute_scores, coefficient_loss
from .train import move_batch, seed_everything

ARMS = ("control", "detail")
SEEDS = (42, 43, 44)


def freeze(parent, external, output):
    """@entry Freeze the complete paired budget and validation-only selection."""
    if output.exists():
        raise FileExistsError("Use a fresh continuation directory")
    p, data, features = load_parent(parent)
    train = [c for c in data["cases"] if c["split"] == "train"]
    mask = np.array([c["split"] == "train" for c in data["cases"]])
    mean = features["pooled"][mask].mean(0, dtype=np.float64).astype(np.float32)
    std = np.maximum(features["pooled"][mask].std(0, dtype=np.float64), 0.05).astype(
        np.float32
    )
    output.mkdir(parents=True)
    files = [
        Path(__file__),
        Path(__file__).with_name("pooled_residual.py"),
        Path(__file__).with_name("end_to_end_evaluation.py"),
        external,
    ]
    for seed in SEEDS:
        draws = (
            torch.multinomial(
                sampling_weights(train),
                12 * len(train),
                True,
                generator=torch.Generator().manual_seed(seed + 3),
            )
            .reshape(12, -1)
            .numpy()
            .astype(np.int32)
        )
        np.save(output / f"draws-{seed}.npy", draws)
        files.append(output / f"draws-{seed}.npy")
    p = {
        "checkpoint": p["checkpoint"],
        "prepared": p["prepared"],
        "normalization": p["normalization"],
        "parent": str(parent.resolve()),
        "external": str(external.resolve()),
        "detail_mean": mean.tolist(),
        "detail_std": std.tolist(),
        "arms": ARMS,
        "seeds": SEEDS,
        "epochs": 12,
        "batch_size": 32,
        "learning_rate": 1e-4,
        "baseline_validation": p["baseline_validation"],
        "initial_function": "Same complete baseline weights; detail correction last layer zero. Both arms update every neural parameter.",
        "augmentation": "None in both arms; isolate the residual feature path.",
        "loss": "Observed-label Cd MAE + Cl MAE in original coefficient units; no ratio, pair or ranking term.",
        "selection": "Per-run minimum validation source-macro Cd MAE+Cl MAE with neither above the fixed baseline; epoch0 fallback. Compare all three seeds; no best-test seed selection.",
        "deployment_candidate": "Select the equal-weight three-seed arm ensemble by validation sum with both validation MAEs no worse than baseline; baseline fallback. Freeze before reading test/external errors.",
        "external_scope": "Same92 Rim queries and three training references; score absolute anchored Cd. External Cl/training exposure/live identity unverified; no generic architecture-superiority claim.",
        "sha256": {
            **p["sha256"],
            **{
                str(f.resolve()): hashlib.sha256(f.read_bytes()).hexdigest()
                for f in files
            },
        },
    }
    (output / "protocol.json").write_text(json.dumps(p, indent=2) + "\n")
    print(
        json.dumps(
            {
                "runs": 6,
                "epochs_each": 12,
                "train": len(train),
                "baseline": p["checkpoint"],
            }
        )
    )


def load(experiment):
    p = json.loads((experiment / "protocol.json").read_text())
    for path, sha in p["sha256"].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != sha:
            raise ValueError(f"Frozen experiment input changed: {path}")
    data = json.loads((Path(p["prepared"]) / "dataset.json").read_text())
    return p, data


def save(model, source, path, p, arm, seed, epoch, metrics):
    ck = dict(source)
    for key in ("pair_arm", "migration", "validation_selection_score"):
        ck.pop(key, None)
    ck.update(
        input_contract=CONTRACT,
        detail_path=arm == "detail",
        seed=seed,
        epoch=epoch,
        model_state={
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        },
        validation_absolute_mae=metrics,
        training_mode="full_network_continuation",
        source_checkpoint=p["checkpoint"],
        source_checkpoint_sha256=p["sha256"][p["checkpoint"]],
        provenance={
            "end_to_end_protocol": str((path.parent.parent / "protocol.json").resolve())
        },
    )
    torch.save(ck, path)


def train(experiment, arm, seed, device_name="mps"):
    """@entry Match source draws, original-unit loss and optimizer budget across arms."""
    p, data = load(experiment)
    if arm not in p["arms"] or seed not in p["seeds"]:
        raise ValueError("Unknown paired run")
    out = experiment / f"{arm}-{seed}"
    out.mkdir()
    seed_everything(seed)
    torch.set_num_threads(2)
    device = torch.device(device_name)
    source = torch.load(p["checkpoint"], map_location="cpu", weights_only=True)
    model = initialize(source, arm == "detail", p["detail_mean"], p["detail_std"]).to(
        device
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=p["learning_rate"])
    cases = [c for c in data["cases"] if c["split"] == "train"]
    values = SceneCases(Path(p["prepared"]), cases, p["normalization"])
    val_cases = [c for c in data["cases"] if c["split"] == "val"]
    validation = SceneCases(Path(p["prepared"]), val_cases, p["normalization"])
    draws = np.load(experiment / f"draws-{seed}.npy", allow_pickle=False)
    std = torch.tensor(
        p["normalization"]["target_std"], dtype=torch.float32, device=device
    )
    limits = p["baseline_validation"]["macro"]
    start = time.monotonic()
    initial = predictions(model, validation, device)
    initial_metrics = absolute_scores(val_cases, physical(initial, p["normalization"]))[
        "macro"
    ]
    if any(abs(initial_metrics[k] - limits[k]) > 2e-6 for k in limits):
        raise ValueError("Initial prediction changed")
    save(model, source, out / "model.pt", p, arm, seed, 0, limits)
    best = sum(limits.values())
    history, steps = [], 0
    exposures = np.zeros(len(cases), int)
    trace = hashlib.sha256()
    parameters = list(model.parameters())
    for epoch, row in enumerate(draws, 1):
        model.train()
        loss_sum = 0.0
        for offset in range(0, len(row), p["batch_size"]):
            selected = row[offset : offset + p["batch_size"]]
            trace.update(selected.tobytes())
            exposures += np.bincount(selected, minlength=len(cases))
            batch = default_collate([values[int(i)] for i in selected])
            views, params, target = move_batch(batch[:3], device)
            optimizer.zero_grad(set_to_none=True)
            loss, _, _ = coefficient_loss(
                model(views, params), target, std, parameters, False
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite full-network loss")
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach())
            steps += 1
        pred = predictions(model, validation, device)
        scores = absolute_scores(val_cases, physical(pred, p["normalization"]))
        metrics = scores["macro"]
        eligible = all(metrics[k] <= limits[k] for k in limits)
        if eligible and sum(metrics.values()) < best:
            best = sum(metrics.values())
            save(model, source, out / "model.pt", p, arm, seed, epoch, metrics)
        record = {
            "epoch": epoch,
            "validation": scores,
            "eligible": eligible,
            "mean_training_loss": loss_sum
            / ((len(row) + p["batch_size"] - 1) // p["batch_size"]),
            "elapsed_seconds": time.monotonic() - start,
        }
        history.append(record)
        report = {
            "history": history,
            "optimizer_steps": steps,
            "exposures": exposures.tolist(),
            "draw_trace_sha256": trace.hexdigest(),
            "initial_validation": initial_metrics,
        }
        (out / "training_report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "arm": arm,
                    "seed": seed,
                    "epoch": epoch,
                    **metrics,
                    "seconds": record["elapsed_seconds"],
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("freeze")
    p.add_argument("parent", type=Path)
    p.add_argument("external", type=Path)
    p.add_argument("output", type=Path)
    p = sub.add_parser("train")
    p.add_argument("experiment", type=Path)
    p.add_argument("arm", choices=ARMS)
    p.add_argument("seed", type=int)
    p.add_argument("--device", default="mps")
    a = parser.parse_args()
    if a.action == "freeze":
        freeze(a.parent, a.external, a.output)
    else:
        train(a.experiment, a.arm, a.seed, a.device)
