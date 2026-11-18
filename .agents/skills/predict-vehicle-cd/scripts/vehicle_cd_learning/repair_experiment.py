"""Paired source-gradient correction and parameter-averaging vehicle experiments."""

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from aircraft_learning.scene_experiment import SceneCases, sampling_weights
from torch.utils.data import default_collate

from .end_to_end_experiment import load as load_parent
from .head_experiment import physical
from .pair_experiment import predictions
from .pooled_residual import load_model
from .readout_experiment import absolute_scores
from .source_gradients import backward, source_losses, update_average
from .train import move_batch, seed_everything

ARMS = ("control", "projected")
SEEDS = (42, 43, 44)


def freeze(parent, output):
    """@entry Freeze paired continuations and the later integrated-correction protocol."""
    if output.exists():
        raise FileExistsError("Use a fresh repair experiment directory")
    old, data = load_parent(parent)
    prior = json.loads((parent / "comparison.json").read_text())
    train = [c for c in data["cases"] if c["split"] == "train"]
    output.mkdir(parents=True)
    inputs = [
        parent / "comparison.json",
        parent / "ensemble.json",
        Path(__file__),
        Path(__file__).with_name("source_gradients.py"),
        Path(__file__).with_name("repair_selection.py"),
    ]
    initial = {
        str(seed): str((parent / f"detail-{seed}/model.pt").resolve()) for seed in SEEDS
    }
    inputs += [Path(path) for path in initial.values()]
    for seed in SEEDS:
        row = torch.multinomial(
            sampling_weights(train),
            8 * len(train),
            True,
            generator=torch.Generator().manual_seed(seed + 13),
        )
        np.save(
            output / f"draws-{seed}.npy", row.reshape(8, -1).numpy().astype(np.int32)
        )
        inputs.append(output / f"draws-{seed}.npy")
    p = {
        "parent": str(parent.resolve()),
        "prepared": old["prepared"],
        "normalization": old["normalization"],
        "external": old["external"],
        "feature_experiment": old["parent"],
        "initial_checkpoints": initial,
        "prior_validation": prior["models"]["detail-ensemble"]["scores"]["val"],
        "prior_ensemble": str((parent / "ensemble.json").resolve()),
        "arms": ARMS,
        "seeds": SEEDS,
        "epochs": 8,
        "batch_size": 32,
        "learning_rate": 5e-5,
        "ema_decay": 0.99,
        "source_names": sorted({c["source"] for c in train}),
        "loss": "The unchanged observed-label Cd MAE + Cl MAE in original units; source contributions use whole-batch target counts.",
        "gradient_rule": "One uniformly sampled pair of present verified CFD sources per batch. If their CNN gradients conflict, project both against the original other gradient. Keep other parameter/source gradients unchanged. Builtin labels still train normally.",
        "algorithm_scope": "Bounded pairwise PCGrad-style intervention on CNN parameters, not a full all-task PCGrad sweep or a guarantee of global conflict removal.",
        "augmentation": "None; raw and EMA weights from each identical training trajectory isolate averaging.",
        "selection": "Per seed/policy: val source-macro Cd MAE+Cl MAE minimum with neither worse than that seed's start; epoch0 fallback. Only equal-weight three-seed ensembles compete for the new base.",
        "base_selection": "Among prior/control-raw/control-ema/projected-raw/projected-ema, require global Cd/Cl plus Rim Cd and Windsor Cl not worse than prior ensemble, then minimize global Cd+Cl. Prior fallback.",
        "correction": {
            "scales": [0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
            "alphas": [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
            "selection": "Train-only residual correction to the chosen neural ensemble, shared geometry kernel with separately regularized output columns. Per target validation MAE selection; require Rim Cd/Windsor Cl no worse than the chosen base. Zero correction is the fallback.",
        },
        "test_policy": "Finish every run, freeze neural and integrated choices using validation only, then score historical test and fixed-reference Radiant; no ratio/ranking/external-teacher objective.",
        "sha256": {
            **old["sha256"],
            **{
                str(f.resolve()): hashlib.sha256(f.read_bytes()).hexdigest()
                for f in inputs
            },
        },
    }
    (output / "protocol.json").write_text(json.dumps(p, indent=2) + "\n")
    print(
        json.dumps(
            {"runs": 6, "epochs": 8, "train": len(train), "ema_decay": p["ema_decay"]}
        )
    )


def load(experiment):
    p = json.loads((experiment / "protocol.json").read_text())
    for path, sha in p["sha256"].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != sha:
            raise ValueError(f"Frozen repair input changed: {path}")
    data = json.loads((Path(p["prepared"]) / "dataset.json").read_text())
    return p, data


def save(model, original, path, p, arm, seed, epoch, policy, metrics):
    ck = dict(original)
    ck.update(
        model_state={
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        },
        epoch=epoch,
        seed=seed,
        training_mode="source_gradient_repair",
        repair_arm=arm,
        weight_policy=policy,
        validation_absolute_mae=metrics,
        source_checkpoint=p["initial_checkpoints"][str(seed)],
        source_checkpoint_sha256=p["sha256"][p["initial_checkpoints"][str(seed)]],
        provenance={
            "repair_protocol": str((path.parent.parent / "protocol.json").resolve())
        },
    )
    torch.save(ck, path)


def train(experiment, arm, seed):
    """@entry Same draws, masks and optimizer steps; only shared-CNN gradient projection differs."""
    p, data = load(experiment)
    if arm not in p["arms"] or seed not in p["seeds"]:
        raise ValueError("Unknown repair run")
    out = experiment / f"{arm}-{seed}"
    out.mkdir()
    seed_everything(seed)
    torch.set_num_threads(2)
    device = torch.device("mps")
    model, original = load_model(Path(p["initial_checkpoints"][str(seed)]))
    average, _ = load_model(Path(p["initial_checkpoints"][str(seed)]))
    model.to(device)
    average.to(device).eval().requires_grad_(False)
    optimizer = torch.optim.Adam(model.parameters(), lr=p["learning_rate"])
    shared = [
        v
        for name, v in model.named_parameters()
        if name.startswith(("image_encoder.", "bottom_encoder."))
        and ".project." not in name
    ]
    cases = [c for c in data["cases"] if c["split"] == "train"]
    val = [c for c in data["cases"] if c["split"] == "val"]
    values, validation = [
        SceneCases(Path(p["prepared"]), group, p["normalization"])
        for group in (cases, val)
    ]
    draws = np.load(experiment / f"draws-{seed}.npy", allow_pickle=False)
    source_ids = np.array([p["source_names"].index(c["source"]) for c in cases])
    builtin = p["source_names"].index("builtin_assumed")
    pair_rng = random.Random(seed + 17)
    std = torch.tensor(
        p["normalization"]["target_std"], device=device, dtype=torch.float32
    )
    initial = predictions(model, validation, device)
    limits = absolute_scores(val, physical(initial, p["normalization"]))["macro"]
    best = {policy: sum(limits.values()) for policy in ("raw", "ema")}
    for policy, instance in (("raw", model), ("ema", average)):
        save(instance, original, out / f"{policy}.pt", p, arm, seed, 0, policy, limits)
    history, steps, projection_count = [], 0, 0
    exposures, trace = np.zeros(len(cases), int), hashlib.sha256()
    started = time.monotonic()
    for epoch, row in enumerate(draws, 1):
        model.train()
        records = []
        for offset in range(0, len(row), p["batch_size"]):
            chosen = row[offset : offset + p["batch_size"]]
            trace.update(chosen.tobytes())
            exposures += np.bincount(chosen, minlength=len(cases))
            batch = default_collate([values[int(i)] for i in chosen])
            views, params, targets = move_batch(batch[:3], device)
            groups = torch.tensor(source_ids[chosen], device=device, dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            losses = source_losses(model(views, params), targets, std, groups)
            if not torch.isfinite(sum(losses.values())):
                raise FloatingPointError("Nonfinite coefficient loss")
            eligible = sorted(s for s in losses if s != builtin)
            pair = (
                pair_rng.sample(eligible, 2)
                if arm == "projected" and len(eligible) > 1
                else None
            )
            record = backward(losses, shared, pair)
            if record is not None:
                records.append(record)
                projection_count += int(record["projected"])
            optimizer.step()
            update_average(average, model, p["ema_decay"])
            steps += 1
        scores = {}
        for policy, instance in (("raw", model), ("ema", average)):
            pred = physical(
                predictions(instance, validation, device), p["normalization"]
            )
            score = absolute_scores(val, pred)
            metrics = score["macro"]
            allowed = all(metrics[k] <= limits[k] for k in limits)
            if allowed and sum(metrics.values()) < best[policy]:
                best[policy] = sum(metrics.values())
                save(
                    instance,
                    original,
                    out / f"{policy}.pt",
                    p,
                    arm,
                    seed,
                    epoch,
                    policy,
                    metrics,
                )
            scores[policy] = {"score": score, "eligible": allowed}
        history.append(
            {
                "epoch": epoch,
                "validation": scores,
                "projections": records,
                "elapsed_seconds": time.monotonic() - started,
            }
        )
        (out / "training_report.json").write_text(
            json.dumps(
                {
                    "history": history,
                    "optimizer_steps": steps,
                    "projection_count": projection_count,
                    "exposures": exposures.tolist(),
                    "draw_trace_sha256": trace.hexdigest(),
                    "initial_validation": limits,
                },
                indent=2,
            )
            + "\n"
        )
        print(
            json.dumps(
                {
                    "arm": arm,
                    "seed": seed,
                    "epoch": epoch,
                    "raw": scores["raw"]["score"]["macro"],
                    "ema": scores["ema"]["score"]["macro"],
                    "projections": projection_count,
                    "seconds": history[-1]["elapsed_seconds"],
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("freeze")
    p.add_argument("parent", type=Path)
    p.add_argument("output", type=Path)
    p = sub.add_parser("train")
    p.add_argument("experiment", type=Path)
    p.add_argument("arm", choices=ARMS)
    p.add_argument("seed", type=int)
    a = parser.parse_args()
    freeze(a.parent, a.output) if a.action == "freeze" else train(
        a.experiment, a.arm, a.seed
    )
