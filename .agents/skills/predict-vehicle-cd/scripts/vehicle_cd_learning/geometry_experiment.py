"""Freeze and train controlled geometry/condition/loss diagnostic arms."""

import argparse
import gc
import hashlib
import json
import time
from collections import Counter
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

from .point_baseline import PointCases, make_model, shared_initial_hash
from .train import evaluate, masked_mse_loss, move_batch

ROOT = Path(__file__).resolve().parents[5]
RUNS = [
    {"name": "views-legacy", "mode": "views", "fold": "primary", "physics": "legacy"},
    {"name": "views", "mode": "views", "fold": "primary", "physics": "declared"},
    {"name": "points", "mode": "points", "fold": "primary", "physics": "declared"},
    {
        "name": "parameters",
        "mode": "parameters",
        "fold": "primary",
        "physics": "declared",
    },
    {
        "name": "views-source-loss",
        "mode": "views",
        "fold": "primary",
        "physics": "declared",
    },
    {
        "name": "views-ahmed-holdout",
        "mode": "views",
        "fold": "ahmed_holdout",
        "physics": "declared",
    },
    {
        "name": "points-ahmed-holdout",
        "mode": "points",
        "fold": "ahmed_holdout",
        "physics": "declared",
    },
    {
        "name": "views-ahmed-no-scale",
        "mode": "views",
        "fold": "ahmed_holdout",
        "physics": "dimensionless",
    },
]


def select_cases(data, run, legacy):
    cases = data["cases"]
    if run["fold"] == "ahmed_holdout":
        cases = [
            c
            for c in cases
            if (c["split"] == "test" and c["source"] == "ahmedml")
            or (
                c["split"] != "test"
                and c["source"] not in ("ahmedml", "builtin_assumed")
            )
        ]
    if run["physics"] == "legacy":
        old = {c["case_id"]: c for c in legacy["cases"]}
        cases = [
            dict(c, parameters=old[c["case_id"]]["parameters"])
            if c["source"] == "builtin_assumed"
            else c
            for c in cases
        ]
    if run["physics"] == "dimensionless":
        cases = [
            dict(
                c,
                parameters=[
                    None if i in (2, 3) else x for i, x in enumerate(c["parameters"])
                ],
            )
            for c in cases
        ]
    return cases


def datasets(root, cases, stats, points, mode):
    if mode == "views":
        return SceneCases(root, cases, stats)
    return PointCases(root, cases, stats, points, read_points=mode == "points")


def check_inputs(experiment):
    protocol = json.loads((experiment / "protocol.json").read_text())
    for filename, expected in {
        **protocol["code_sha256"],
        **protocol["input_sha256"],
    }.items():
        if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen experiment input changed: {filename}")
    root = Path(protocol["prepared"])
    return (
        protocol,
        json.loads((root / "dataset.json").read_text()),
        json.loads(Path(protocol["legacy_dataset"]).read_text()),
        json.loads((root / "point-manifest.json").read_text())["cases"],
    )


def freeze(experiment, prepared):
    """@entry Prespecify all runs, memberships, normalization and evaluation limits."""
    if (experiment / "protocol.json").exists():
        raise FileExistsError("Protocol already exists")
    data = json.loads((prepared / "dataset.json").read_text())
    verify_data(prepared, data)
    points = json.loads((prepared / "point-manifest.json").read_text())
    if not points["complete"] or len(points["cases"]) != len(data["cases"]):
        raise ValueError("Point preparation incomplete")
    for row in points["cases"].values():
        if (
            hashlib.sha256((prepared / row["path"]).read_bytes()).hexdigest()
            != row["sha256"]
        ):
            raise ValueError("Point cache changed")
        with np.load(prepared / row["path"], allow_pickle=False) as arrays:
            cloud = arrays["points"]
            if (
                cloud.shape != (points["points"], 3)
                or cloud.dtype != np.float32
                or not np.isfinite(cloud).all()
            ):
                raise ValueError("Invalid point tensor")
    legacy_path = ROOT / "outputs/aerodynamic-scene-abc-20261111/prepared/dataset.json"
    legacy = json.loads(legacy_path.read_text())
    code = [
        Path(__file__),
        Path(__file__).with_name("point_baseline.py"),
        Path(__file__).with_name("prepare_points.py"),
        Path(__file__).with_name("geometry_diagnostics.py"),
        Path(__file__).with_name("geometry_report.py"),
        Path(__file__).with_name("model.py"),
        Path(__file__).with_name("train.py"),
        ROOT
        / ".agents/skills/predict-aircraft-coefficients/scripts/aircraft_learning/scene_experiment.py",
        ROOT
        / ".agents/skills/predict-aircraft-coefficients/scripts/aircraft_learning/width_experiment.py",
        ROOT
        / ".agents/skills/predict-aircraft-coefficients/scripts/aircraft_learning/compare_scenes.py",
        Path(__file__).with_name("evaluate.py"),
        Path(__file__).with_name("render.py"),
    ]
    inputs = [prepared / "dataset.json", prepared / "point-manifest.json", legacy_path]
    before = json.loads(
        (
            ROOT
            / "outputs/aerodynamic-builtin-assumptions-20261112/preserved-before.json"
        ).read_text()
    )
    protocol = {
        "prepared": str(prepared.resolve()),
        "legacy_dataset": str(legacy_path),
        "epochs": 40,
        "batch_size": 32,
        "learning_rate": 0.001,
        "seed": 42,
        "runs": RUNS,
        "points": points["points"],
        "normalization": {},
        "split_counts": {},
        "code_sha256": {
            str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in code
        },
        "input_sha256": {
            str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs
        },
        "preserved_artifacts": before,
        "sampling": "Same source-balanced replacement sampling, builtin5% of vehicle budget; seed+3 independent RNG; same actual draw sequence for matched folds",
        "augmentation": "none in all matched diagnostic arms; differs from historical sampling-jitter training",
        "selection": "Minimum pooled observed-target standardized validation MSE, same within fold; no test selection",
        "source_loss_arm": "Only views-source-loss changes training loss to train-source variance scaling; validation stays identical",
        "scope": "Single-seed diagnostic on a vehicle-only slice, not a direct comparison with joint car/aircraft A; historical splits already examined",
        "holdout_limit": "Ahmed source and geometry family held out together, including its train/val labels; all assumed builtins excluded. Source/BC/Re/geometry effects confounded, not pure unseen-geometry causality or arbitrary-car validation",
        "scale_ablation": "views-ahmed-no-scale masks log_length_m and log_velocity_m_s in the same Ahmed holdout, retaining Re and dimensionless conditions/ratios; source facts stay unchanged in the dataset",
        "ranking": "Within-source heldout pairwise Cd ranking at absolute truth-delta thresholds0/0.001/0.003/0.005; not local-edit CFD validation",
        "overfit": "First32 prespecified Rim train cases, own train-only normalization,256 steps; numerical inputs constant, no validation/test tuning",
        "point_reference": "https://github.com/charlesq34/pointnet/blob/master/models/pointnet_cls_basic.py; reduced-width conditional regression adaptation, not an industry-score reproduction",
    }
    for run in RUNS:
        cases = select_cases(data, run, legacy)
        key = f"{run['fold']}/{run['physics']}"
        protocol["normalization"][key] = normalization(
            [c for c in cases if c["split"] == "train"]
        )
        protocol["split_counts"][key] = dict(Counter(c["split"] for c in cases))
    (experiment / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print(json.dumps(protocol["split_counts"]), flush=True)


def source_loss_weights(cases, stats, balanced):
    weights = {}
    for source in {c["source"] for c in cases}:
        y = np.asarray(
            [c["targets"] for c in cases if c["source"] == source], dtype=float
        )
        std = np.array(
            [np.std(a[np.isfinite(a)]) if np.isfinite(a).any() else 1 for a in y.T]
        )
        weights[source] = (
            (np.asarray(stats["target_std"]) / np.maximum(std, 1e-6)) ** 2
            if balanced
            else np.ones(2)
        )
    return np.asarray([weights[c["source"]] for c in cases], dtype=np.float32)


def train_epoch(model, values, indices, weights, optimizer, device, batch_size):
    model.train()
    total, count = 0.0, 0
    for start in range(0, len(indices), batch_size):
        ids = indices[start : start + batch_size]
        batch = default_collate([values[i] for i in ids])
        geometry, parameters, targets = move_batch(batch[:3], device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(geometry, parameters)
        observed = torch.isfinite(targets)
        if not torch.isfinite(prediction).all():
            raise FloatingPointError("Nonfinite diagnostic prediction")
        if weights is None:
            loss = masked_mse_loss(prediction, targets)
        else:
            w = torch.tensor(weights[ids], device=device)
            loss = (
                (prediction[observed] - targets[observed]).square() * w[observed]
            ).mean()
        loss.backward()
        optimizer.step()
        n = int(observed.sum())
        total += loss.item() * n
        count += n
    return total / count


def train_run(experiment, run_name, device_name="mps"):
    protocol, data, legacy, points = check_inputs(experiment)
    run = next(r for r in protocol["runs"] if r["name"] == run_name)
    folder = experiment / run_name
    if folder.exists():
        raise FileExistsError(f"Run already exists: {folder}")
    folder.mkdir()
    torch.set_num_threads(2)
    device, root = torch.device(device_name), Path(protocol["prepared"])
    cases = select_cases(data, run, legacy)
    train, val = (
        [c for c in cases if c["split"] == split] for split in ("train", "val")
    )
    stats = protocol["normalization"][f"{run['fold']}/{run['physics']}"]
    training = datasets(root, train, stats, points, run["mode"])
    validation = datasets(root, val, stats, points, run["mode"])
    training = [training[i] for i in range(len(training))]
    validation = [validation[i] for i in range(len(validation))]
    model = make_model(run["mode"], protocol["seed"])
    common_hash = shared_initial_hash(model)
    full_initial_hash = initialization_hash(model)
    parameter_count = sum(p.numel() for p in model.parameters())
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=protocol["learning_rate"])
    weights = sampling_weights(train)
    rng = torch.Generator().manual_seed(protocol["seed"] + 3)
    exposures = np.zeros(len(train), dtype=int)
    loss_weights = source_loss_weights(train, stats, run_name == "views-source-loss")
    history, best = [], float("inf")
    loader = DataLoader(validation, batch_size=protocol["batch_size"])
    print(
        json.dumps(
            {
                "run": run_name,
                "train": len(train),
                "val": len(val),
                "parameters": parameter_count,
            }
        ),
        flush=True,
    )
    for epoch in range(1, protocol["epochs"] + 1):
        started = time.perf_counter()
        indices = torch.multinomial(
            weights, len(train), replacement=True, generator=rng
        ).numpy()
        exposures += np.bincount(indices, minlength=len(train))
        loss = train_epoch(
            model,
            training,
            indices,
            loss_weights,
            optimizer,
            device,
            protocol["batch_size"],
        )
        val_loss, val_error, _ = evaluate(model, loader, device)
        val_sources = np.array([c["source"] for c in val])
        val_source_mae = {
            source: (abs(val_error[val_sources == source]) * stats["target_std"])
            .mean(0)
            .tolist()
            for source in sorted(set(val_sources))
        }
        history.append(
            {
                "epoch": epoch,
                "train_objective": loss,
                "val_mse": val_loss,
                "val_source_mae": val_source_mae,
                "seconds": time.perf_counter() - started,
            }
        )
        if val_loss < best:
            best = val_loss
            torch.save(
                {
                    "model_state": {
                        k: v.detach().cpu() for k, v in model.state_dict().items()
                    },
                    "run": run,
                    "epoch": epoch,
                    "normalization": stats,
                    "train_ids": [c["case_id"] for c in train],
                    "train_geometry_hashes": sorted(
                        {c["geometry_hash"] for c in train}
                    ),
                    "parameter_count": parameter_count,
                    "seed": protocol["seed"],
                    "shared_initial_sha256": common_hash,
                    "initial_model_sha256": full_initial_hash,
                    "dataset_sha256": protocol["input_sha256"][
                        str((root / "dataset.json").resolve())
                    ],
                },
                folder / "model.pt",
            )
        (folder / "training_report.json").write_text(
            json.dumps(
                {
                    "run": run,
                    "history": history,
                    "best_epoch": int(np.argmin([h["val_mse"] for h in history])) + 1,
                    "parameter_count": parameter_count,
                    "sampling_counts": dict(
                        zip((c["case_id"] for c in train), exposures.tolist())
                    ),
                    "sampling_rng_sha256": hashlib.sha256(
                        rng.get_state().numpy().tobytes()
                    ).hexdigest(),
                },
                indent=2,
            )
            + "\n"
        )
        print(json.dumps({"run": run_name, **history[-1]}), flush=True)
    if not np.all(exposures > 0):
        raise ValueError("Some training members were not sampled")
    del model, training, validation
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "train", "run-all"))
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--run")
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    if args.action == "freeze":
        freeze(args.experiment, args.prepared)
    elif args.action == "train":
        train_run(args.experiment, args.run, args.device)
    else:
        for run in json.loads((args.experiment / "protocol.json").read_text())["runs"]:
            train_run(args.experiment, run["name"], args.device)


if __name__ == "__main__":
    main()
