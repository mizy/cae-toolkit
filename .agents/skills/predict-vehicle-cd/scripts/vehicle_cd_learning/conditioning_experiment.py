"""Isolate dimensional inputs, representation jitter and cross-source gradients."""

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from aircraft_learning.scene_experiment import (
    SceneCases,
    normalization,
    sampling_weights,
    verify_data,
)
from aircraft_learning.width_experiment import initialization_hash
from torch.utils.data import DataLoader, default_collate

from .data import jitter_views
from .geometry_experiment import check_inputs as prior_inputs
from .geometry_experiment import select_cases, train_epoch
from .point_baseline import make_model
from .train import evaluate, move_batch

ROOT = Path(__file__).resolve().parents[5]
SCALE = ["log_length_m", "log_velocity_m_s"]
RUNS = [
    {
        "name": "full",
        "mask": [],
        "jitter": False,
        "focus": False,
        "local_targets": False,
        "fold": "primary",
    },
    {
        "name": "dimensionless",
        "mask": SCALE,
        "jitter": False,
        "focus": False,
        "local_targets": False,
        "fold": "primary",
    },
    {
        "name": "full-jitter",
        "mask": [],
        "jitter": True,
        "focus": False,
        "local_targets": False,
        "fold": "primary",
    },
    {
        "name": "dimensionless-jitter",
        "mask": SCALE,
        "jitter": True,
        "focus": False,
        "local_targets": False,
        "fold": "primary",
    },
    {
        "name": "holdout-no-velocity",
        "mask": ["log_velocity_m_s"],
        "jitter": False,
        "focus": False,
        "local_targets": False,
        "fold": "ahmed_holdout",
    },
    {
        "name": "rim-global",
        "mask": [],
        "jitter": False,
        "focus": True,
        "local_targets": False,
        "fold": "primary",
    },
    {
        "name": "rim-local",
        "mask": [],
        "jitter": False,
        "focus": True,
        "local_targets": True,
        "fold": "primary",
    },
    {
        "name": "rim-local-jitter",
        "mask": [],
        "jitter": True,
        "focus": True,
        "local_targets": True,
        "fold": "primary",
    },
]


def cases_for(data, run):
    cases = select_cases(data, {"fold": run["fold"], "physics": "declared"}, {})
    masked = {data["parameter_names"].index(k) for k in run["mask"]}
    return [
        dict(
            c,
            parameters=[
                None if i in masked else x for i, x in enumerate(c["parameters"])
            ],
        )
        for c in cases
    ]


def run_stats(cases, run):
    train = [c for c in cases if c["split"] == "train"]
    stats = normalization(train)
    selection_std = list(stats["target_std"])
    if run["local_targets"]:
        rim = normalization([c for c in train if c["source"] == "drivaerrim"])
        for key in ("target_mean", "target_std"):
            stats[key] = rim[key]
    return stats, selection_std


def freeze(experiment, previous):
    """@entry Freeze interventions before reading any new test predictions."""
    if experiment.exists():
        raise FileExistsError("Use a fresh experiment directory")
    old, data, _, _ = prior_inputs(previous)
    verify_data(Path(old["prepared"]), data)
    files = [
        Path(__file__),
        Path(__file__).with_name("conditioning_diagnostics.py"),
        Path(__file__).with_name("conditioning_report.py"),
        Path(__file__).with_name("detail_kernel.py"),
        Path(__file__).with_name("data.py"),
    ]
    preserved = dict(old["preserved_artifacts"])
    for filename in ("protocol.json", "comparison.json", "final-verification.json"):
        p = previous / filename
        preserved[str(p.resolve())] = hashlib.sha256(p.read_bytes()).hexdigest()
    for run in old["runs"]:
        for filename in ("model.pt", "training_report.json"):
            p = previous / run["name"] / filename
            preserved[str(p.resolve())] = hashlib.sha256(p.read_bytes()).hexdigest()
    protocol = {
        "previous": str(previous.resolve()),
        "prepared": old["prepared"],
        "seed": 42,
        "epochs": 40,
        "batch_size": 32,
        "learning_rate": 0.001,
        "runs": RUNS,
        "normalization": {},
        "selection_std": {},
        "code_sha256": {
            **old["code_sha256"],
            **{
                str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in files
            },
        },
        "input_sha256": dict(old["input_sha256"]),
        "preserved_artifacts": preserved,
        "selection": "Global/holdout minimum val MSE in original global target units; Rim controls use identical Rim val cases and the same global target-unit metric. Save final40 and Rim-selected global checkpoints for matched comparisons.",
        "focus": "Same full-cohort draw sequence, optimizer step count and Rim exposures. Only Rim contributes supervised gradients; loss denominator remains the full batch observed-target count. Global normalization uses full train moments; local target treatment changes target mean/std only.",
        "jitter": "Existing sampling_jitter_v1; apply to the complete mixed batch before Rim filtering, same independent seed+1 generator across matched arms; no crops.",
        "scope": "Single-seed development diagnosis, original historical geometry splits reused; no new CFD or pure unseen-family certification; shared Cd/Cl head and default A preserved.",
        "kernel": "Rim-only full-view variable-pixel linear/RBF ridge; train-only feature/target moments, alpha/scale chosen by val only; evaluate test only after all neural runs complete. Source-specialized diagnostic, not generic deployment.",
    }
    for run in RUNS:
        cases = cases_for(data, run)
        stats, selection = run_stats(cases, run)
        protocol["normalization"][run["name"]] = stats
        protocol["selection_std"][run["name"]] = selection
    assert protocol["normalization"]["full"] == old["normalization"]["primary/declared"]
    experiment.mkdir(parents=True)
    (experiment / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print(
        json.dumps({"frozen_runs": len(RUNS), "cases": len(data["cases"])}), flush=True
    )


def load(experiment):
    p = json.loads((experiment / "protocol.json").read_text())
    for name, expected in {**p["code_sha256"], **p["input_sha256"]}.items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen code/data changed: {name}")
    return p, json.loads((Path(p["prepared"]) / "dataset.json").read_text())


def supervised_loss(prediction, targets, original_count):
    observed = torch.isfinite(targets)
    return (prediction[observed] - targets[observed]).square().sum() / original_count


def intervened_epoch(
    model, values, cases, indices, optimizer, device, batch_size, run, jitter_rng
):
    """Preserve original batch denominators, steps and jitter before Rim filtering."""
    model.train()
    total, denominator, zero_steps = 0.0, 0, 0
    for start in range(0, len(indices), batch_size):
        ids = indices[start : start + batch_size]
        batch = default_collate([values[i] for i in ids])
        geometry, parameters, targets = move_batch(batch[:3], device)
        observed_count = int(torch.isfinite(targets).sum())
        if run["jitter"]:
            geometry = jitter_views(geometry, jitter_rng)
        selected = torch.tensor(
            [not run["focus"] or cases[i]["source"] == "drivaerrim" for i in ids],
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        if bool(selected.any()):
            prediction = model(
                {k: v[selected] for k, v in geometry.items()}, parameters[selected]
            )
            loss = supervised_loss(prediction, targets[selected], observed_count)
        else:
            # A zero-gradient step preserves AdamW momentum/decay timing.
            prediction = model({k: v[:1] for k, v in geometry.items()}, parameters[:1])
            loss = prediction.sum() * 0
            zero_steps += 1
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite intervention loss")
        loss.backward()
        optimizer.step()
        total += loss.item() * observed_count
        denominator += observed_count
    return total / denominator, zero_steps


def validation_scores(model, loader, device, stats, reference_std, val):
    _, errors, _ = evaluate(model, loader, device)
    physical = errors * stats["target_std"]
    normalized = (physical / reference_std) ** 2
    rim = np.array([c["source"] == "drivaerrim" for c in val])
    return (
        float(np.nanmean(normalized)),
        float(np.nanmean(normalized[rim])),
        {
            source: np.nanmean(
                abs(physical[[c["source"] == source for c in val]]), 0
            ).tolist()
            for source in sorted({c["source"] for c in val})
        },
    )


def train_run(experiment, name, device_name="mps"):
    p, data = load(experiment)
    run = next(r for r in p["runs"] if r["name"] == name)
    out = experiment / name
    if out.exists():
        raise FileExistsError(f"Run exists: {name}")
    out.mkdir()
    torch.set_num_threads(2)
    device, root = torch.device(device_name), Path(p["prepared"])
    cases = cases_for(data, run)
    train = [c for c in cases if c["split"] == "train"]
    val = [
        c
        for c in cases
        if c["split"] == "val" and (not run["focus"] or c["source"] == "drivaerrim")
    ]
    stats, reference_std = p["normalization"][name], p["selection_std"][name]
    training, validation = SceneCases(root, train, stats), SceneCases(root, val, stats)
    training = [training[i] for i in range(len(training))]
    validation = [validation[i] for i in range(len(validation))]
    model = make_model("views", p["seed"])
    initial = initialization_hash(model)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=p["learning_rate"])
    rng = torch.Generator().manual_seed(p["seed"] + 3)
    jitter_rng = torch.Generator(device=device).manual_seed(p["seed"] + 1)
    weights, exposures = sampling_weights(train), np.zeros(len(train), dtype=int)
    rim_trace, history = hashlib.sha256(), []
    loader = DataLoader(validation, batch_size=p["batch_size"])
    best, rim_best, steps = float("inf"), float("inf"), 0
    print(
        json.dumps(
            {
                "run": name,
                "sampled_train": len(train),
                "supervised_train": sum(
                    not run["focus"] or c["source"] == "drivaerrim" for c in train
                ),
            }
        ),
        flush=True,
    )
    for epoch in range(1, p["epochs"] + 1):
        started = time.perf_counter()
        indices = torch.multinomial(
            weights, len(train), replacement=True, generator=rng
        ).numpy()
        exposures += np.bincount(indices, minlength=len(train))
        for start in range(0, len(indices), p["batch_size"]):
            ids = indices[start : start + p["batch_size"]]
            rim_ids = np.asarray(
                [i for i in ids if train[i]["source"] == "drivaerrim"], dtype=np.int32
            )
            rim_trace.update(
                np.asarray([steps, len(rim_ids)], dtype=np.int32).tobytes()
            )
            rim_trace.update(rim_ids.tobytes())
            steps += 1
        if not run["jitter"] and not run["focus"]:
            loss = train_epoch(
                model,
                training,
                indices,
                np.ones((len(train), 2), np.float32),
                optimizer,
                device,
                p["batch_size"],
            )
            zero_steps = 0
        else:
            loss, zero_steps = intervened_epoch(
                model,
                training,
                train,
                indices,
                optimizer,
                device,
                p["batch_size"],
                run,
                jitter_rng,
            )
        val_loss, rim_loss, source_mae = validation_scores(
            model, loader, device, stats, reference_std, val
        )
        history.append(
            {
                "epoch": epoch,
                "train_objective": loss,
                "val_mse": val_loss,
                "rim_val_mse": rim_loss,
                "val_source_mae": source_mae,
                "zero_supervision_steps": zero_steps,
                "seconds": time.perf_counter() - started,
            }
        )

        save_best = val_loss < best
        save_rim = not run["focus"] and run["fold"] == "primary" and rim_loss < rim_best
        if save_best or save_rim or epoch == p["epochs"]:
            checkpoint = {
                "model_state": {
                    k: v.detach().cpu() for k, v in model.state_dict().items()
                },
                "epoch": epoch,
                "run": run,
                "normalization": stats,
                "initial_model_sha256": initial,
                "seed": p["seed"],
                "sampler_train_ids": [c["case_id"] for c in train],
                "supervised_train_ids": [
                    c["case_id"]
                    for c in train
                    if not run["focus"] or c["source"] == "drivaerrim"
                ],
                "dataset_sha256": p["input_sha256"][
                    str((root / "dataset.json").resolve())
                ],
            }

        if save_best:
            best = val_loss
            torch.save(checkpoint, out / "model.pt")
        if save_rim:
            rim_best = rim_loss
            torch.save(checkpoint, out / "rim-model.pt")
        if epoch == p["epochs"]:
            torch.save(checkpoint, out / "last.pt")
        report = {
            "run": run,
            "history": history,
            "best_epoch": int(np.argmin([h["val_mse"] for h in history])) + 1,
            "optimizer_steps": steps,
            "rim_trace_sha256": rim_trace.hexdigest(),
            "sampling_counts": dict(
                zip((c["case_id"] for c in train), exposures.tolist())
            ),
            "supervised_counts": {
                c["case_id"]: int(exposures[i])
                if not run["focus"] or c["source"] == "drivaerrim"
                else 0
                for i, c in enumerate(train)
            },
            "sampling_rng_sha256": hashlib.sha256(
                rng.get_state().numpy().tobytes()
            ).hexdigest(),
            "jitter_rng_sha256": hashlib.sha256(
                jitter_rng.get_state().cpu().numpy().tobytes()
            ).hexdigest(),
        }
        (out / "training_report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"run": name, **history[-1]}), flush=True)
    del model, training, validation, optimizer, loader
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "train", "run-all"))
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--run")
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    if args.action == "freeze":
        freeze(args.experiment, args.previous)
    elif args.action == "train":
        train_run(args.experiment, args.run, args.device)
    else:
        for run in json.loads((args.experiment / "protocol.json").read_text())["runs"]:
            train_run(args.experiment, run["name"], args.device)


if __name__ == "__main__":
    main()
