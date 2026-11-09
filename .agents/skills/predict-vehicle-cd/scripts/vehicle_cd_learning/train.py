"""Train a multi-view Cd/Cl predictor from a prepared dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .data import PARAMETER_NAMES, jitter_views
from .model import CoefficientModel


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch(batch, device):
    views, parameters, targets = batch
    return (
        {k: v.to(device, non_blocking=True) for k, v in views.items()},
        parameters.to(device, non_blocking=True),
        targets.to(device, non_blocking=True),
    )


def masked_mse_loss(prediction, targets):
    """Average squared error over observed targets only."""
    observed = torch.isfinite(targets)
    if not observed.any():
        raise ValueError("batch has no observed targets")
    return nn.functional.mse_loss(prediction[observed], targets[observed])


def evaluate(model, loader, device):
    model.eval()
    total, count, case_count, errors = 0.0, 0, 0, []
    target_counts = np.zeros(model.target_count, dtype=np.int64)
    with torch.no_grad():
        for batch in loader:
            views, parameters, targets = move_batch(batch, device)
            prediction = model(views, parameters)
            if not torch.isfinite(prediction).all():
                raise FloatingPointError("nonfinite model output during evaluation")
            observed = torch.isfinite(targets)
            count += int(observed.sum().item())
            case_count += targets.shape[0]
            target_counts += observed.sum(dim=0).cpu().numpy()
            total += (prediction[observed] - targets[observed]).square().sum().item()
            error = (prediction - targets).cpu().numpy()
            error[~observed.cpu().numpy()] = np.nan
            errors.append(error)
    if not count:
        reason = (
            "validation split is empty"
            if not case_count
            else "validation has no observed targets"
        )
        raise ValueError(reason)
    return total / count, np.concatenate(errors), target_counts


def training_normalization(parameters, targets, target_names):
    """Compute train-only numeric feature and observed-target statistics."""
    x = np.asarray(parameters, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("training parameters must be a finite 2D array")
    if y.ndim != 2 or y.shape[1] != len(target_names):
        raise ValueError("training targets must match target_names")
    counts = np.isfinite(y).sum(axis=0)
    missing = [name for name, count in zip(target_names, counts) if count == 0]
    if missing:
        raise ValueError(
            f"training targets have no observed values: {', '.join(missing)}"
        )
    target_mean = np.nanmean(y, axis=0)
    target_std = np.nanstd(y, axis=0)
    constant = [
        name
        for name, scale in zip(target_names, target_std)
        if not np.isfinite(scale) or scale < 1e-8
    ]
    if constant:
        raise ValueError(f"training targets need variation: {', '.join(constant)}")
    return {
        "parameter_mean": x.mean(axis=0).tolist(),
        "parameter_std": np.maximum(x.std(axis=0), 1e-8).tolist(),
        "target_mean": target_mean.tolist(),
        "target_std": target_std.tolist(),
        "target_counts": counts.astype(int).tolist(),
    }


def condition_profiles_with_train_ranges(condition_profiles, train_cases):
    """Copy source profiles and attach base-parameter ranges from training rows."""
    profiles = deepcopy(condition_profiles)
    for name, profile in profiles.items():
        values = np.asarray(
            [
                case["parameters"][: len(PARAMETER_NAMES)]
                for case in train_cases
                if case.get("condition_profile") == name
            ],
            dtype=np.float64,
        )
        if not len(values):
            raise ValueError(f"Condition profile has no training cases: {name}")
        profile["parameter_ranges"] = {
            parameter: [float(values[:, i].min()), float(values[:, i].max())]
            for i, parameter in enumerate(PARAMETER_NAMES)
        }
    return profiles


def train_model(
    train_set: Dataset,
    validation_set: Dataset,
    *,
    parameter_names: list[str],
    target_names: list[str] | None = None,
    normalization: dict,
    metadata: dict,
    output_dir: Path,
    variant: str = "five",
    epochs: int = 100,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    seed: int = 42,
    device_name: str = "cpu",
    augment: bool = False,
    model_size: str = "standard",
    encoder: str = "plain",
    fusion: str = "concat",
    workers: int = 0,
    preload: bool = False,
    encoder_checkpoint: Path | None = None,
    view_feature_size: int | None = None,
) -> Path:
    """Train normalized CaseDataset splits and save the best validation checkpoint."""
    if not len(train_set) or not len(validation_set):
        raise ValueError("train and validation splits must both be non-empty")
    if epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("epochs, batch_size, and learning_rate must be positive")
    if workers < 0:
        raise ValueError("workers must be nonnegative")
    if preload and workers:
        raise ValueError("preload uses the main process; set workers=0")
    target_names = target_names or ["cd", "cl"]
    if not target_names or any(name not in {"cd", "cl"} for name in target_names):
        raise ValueError("target_names must be cd, cl, or [cd, cl]")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "model.pt"
    if checkpoint_path.exists():
        raise FileExistsError(
            f"refusing to overwrite existing checkpoint: {checkpoint_path}"
        )
    seed_everything(seed)
    torch.set_num_threads(2)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    target_std = torch.tensor(normalization["target_std"], dtype=torch.float32)
    if not torch.isfinite(target_std).all() or torch.any(target_std < 1e-8):
        constant = [
            name
            for name, scale in zip(target_names, target_std)
            if not torch.isfinite(scale) or scale < 1e-8
        ]
        raise ValueError(f"training targets need variation: {', '.join(constant)}")
    preload_seconds = 0.0
    if preload:
        started = time.perf_counter()
        train_set = [train_set[index] for index in range(len(train_set))]
        validation_set = (
            train_set
            if metadata.get("fit_all")
            else [validation_set[index] for index in range(len(validation_set))]
        )
        preload_seconds = time.perf_counter() - started
        print(
            f"loaded train={len(train_set)} val={len(validation_set)} numerical cases in {preload_seconds:.1f}s",
            flush=True,
        )
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=workers,
        persistent_workers=workers > 0,
    )
    val_loader = DataLoader(
        validation_set,
        batch_size=batch_size,
        num_workers=workers,
        persistent_workers=workers > 0,
    )
    jitter_generator = torch.Generator(device=device).manual_seed(seed + 1)
    initial_rng = torch.get_rng_state()
    model = CoefficientModel(
        len(parameter_names),
        variant,
        len(target_names),
        size=model_size,
        encoder=encoder,
        fusion=fusion,
        view_feature_size=view_feature_size,
    ).to(device)
    if view_feature_size is not None:
        # Pair unchanged tensors across widths; only dimension-dependent layers differ.
        after_rng = torch.get_rng_state()
        torch.set_rng_state(initial_rng)
        reference = CoefficientModel(
            len(parameter_names),
            variant,
            len(target_names),
            size=model_size,
            encoder=encoder,
            fusion=fusion,
        )
        shared = {
            k: v
            for k, v in reference.state_dict().items()
            if model.state_dict()[k].shape == v.shape
        }
        model.load_state_dict(shared, strict=False)
        del reference
        torch.set_rng_state(after_rng)
    encoder_initialization = {"mode": "random", "loaded_modules": [], "loaded_keys": []}
    if encoder_checkpoint is not None:
        if variant != "five" or not encoder.startswith("pyramid"):
            raise ValueError(
                "encoder initialization requires a five-view pyramid model"
            )
        source = torch.load(encoder_checkpoint, map_location="cpu", weights_only=False)
        source_config = source.get("model_config", {})
        for name, expected in (
            ("variant", variant),
            ("size", model_size),
            ("encoder", encoder),
        ):
            if source_config.get(name) != expected:
                raise ValueError(
                    f"encoder checkpoint {name} must be {expected!r}; "
                    f"got {source_config.get(name)!r}"
                )
        source_state = source.get("model_state", {})
        model_state = model.state_dict()
        loaded_keys = []
        for prefix in ("image_encoder.", "bottom_encoder."):
            target_keys = {key for key in model_state if key.startswith(prefix)}
            source_keys = {key for key in source_state if key.startswith(prefix)}
            if not target_keys or source_keys != target_keys:
                raise ValueError(
                    f"encoder checkpoint has incompatible {prefix[:-1]} weights"
                )
            for key in target_keys:
                if view_feature_size is not None and ".project." in key:
                    continue
                if source_state[key].shape != model_state[key].shape:
                    raise ValueError(
                        f"encoder checkpoint tensor shape differs for {key}"
                    )
                model_state[key] = source_state[key]
                loaded_keys.append(key)
        model.load_state_dict(model_state)
        encoder_initialization = {
            "mode": "shared_pyramid_before_projection"
            if view_feature_size is not None
            else "shared_encoders_only",
            "checkpoint": str(encoder_checkpoint),
            "loaded_modules": ["image_encoder", "bottom_encoder"],
            "loaded_keys": sorted(loaded_keys),
        }
    width_initialization = {}
    if view_feature_size is not None:
        common_hash = hashlib.sha256()
        for key, value in sorted(model.state_dict().items()):
            if ".project." in key or key in {
                "head.0.weight",
                "view_attention.score.0.weight",
            }:
                continue
            common_hash.update(key.encode())
            common_hash.update(value.detach().cpu().contiguous().numpy().tobytes())
        width_initialization = {"common_initial_sha256": common_hash.hexdigest()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    best_loss, best_error, history = float("inf"), None, []

    for epoch in range(epochs):
        epoch_started = time.perf_counter()
        model.train()
        train_total, train_count = 0.0, 0
        train_target_counts = torch.zeros(len(target_names), dtype=torch.int64)
        for batch in train_loader:
            views, parameters, targets = move_batch(batch, device)
            if augment:
                views = jitter_views(views, jitter_generator)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(views, parameters)
            if not torch.isfinite(prediction).all():
                raise FloatingPointError(f"nonfinite model output at epoch {epoch + 1}")
            observed = torch.isfinite(targets)
            batch_count = int(observed.sum().item())
            train_target_counts += observed.sum(dim=0).cpu()
            loss = masked_mse_loss(prediction, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"nonfinite training loss at epoch {epoch + 1}"
                )
            loss.backward()
            optimizer.step()
            train_total += loss.item() * batch_count
            train_count += batch_count
        if torch.any(train_target_counts == 0):
            missing = [
                name
                for name, count in zip(target_names, train_target_counts.tolist())
                if count == 0
            ]
            raise ValueError(
                f"training targets have no observed values: {', '.join(missing)}"
            )
        train_loss = train_total / train_count
        val_loss, val_error, val_target_counts = evaluate(model, val_loader, device)
        if not np.isfinite(val_loss):
            raise FloatingPointError(f"nonfinite validation loss at epoch {epoch + 1}")
        history.append(
            {"epoch": epoch + 1, "train_mse": train_loss, "val_mse": val_loss}
        )
        if (
            view_feature_size is not None
            or (epoch + 1) % 10 == 0
            or epoch + 1 == epochs
        ):
            metric_name = "fit_mse" if metadata.get("fit_all") else "val_mse"
            print(
                f"epoch {epoch + 1}/{epochs} train_mse={train_loss:.6g} "
                f"{metric_name}={val_loss:.6g}",
                flush=True,
            )
            if view_feature_size is not None:
                print(
                    f"epoch_seconds={time.perf_counter() - epoch_started:.2f}",
                    flush=True,
                )
        if (epoch == epochs - 1) if metadata.get("fit_all") else (val_loss < best_loss):
            best_loss, best_error = val_loss, val_error
            checkpoint = {
                "model_state": {
                    k: v.detach().cpu() for k, v in model.state_dict().items()
                },
                "model_config": {
                    "parameter_count": len(parameter_names),
                    "variant": variant,
                    "target_count": len(target_names),
                    "size": model_size,
                    "encoder": encoder,
                    "fusion": fusion,
                    **(
                        {"view_feature_size": view_feature_size}
                        if view_feature_size is not None
                        else {}
                    ),
                },
                "device": device_name,
                "augmentation": "sampling_jitter_v1" if augment else "none",
                "encoder_initialization": encoder_initialization,
                **width_initialization,
                "parameter_names": parameter_names,
                "target_names": target_names,
                "parameter_mean": normalization["parameter_mean"],
                "parameter_std": normalization["parameter_std"],
                "target_mean": normalization["target_mean"],
                "target_std": normalization["target_std"],
                "render_config": metadata["render_config"],
                "conditions": metadata["conditions"],
                **(
                    {"condition_profiles": metadata["condition_profiles"]}
                    if "condition_profiles" in metadata
                    else {}
                ),
                "parameter_ranges": metadata["parameter_ranges"],
                "data_kind": metadata["data_kind"],
                "provenance": metadata["provenance"],
                "train_ids": metadata["train_ids"],
                **(
                    {"train_source_counts": metadata["train_source_counts"]}
                    if "train_source_counts" in metadata
                    else {}
                ),
                "train_family_ids": metadata.get("train_family_ids", []),
                "split_policy": metadata.get("split_policy", "family_disjoint"),
                "train_geometry_hashes": metadata["train_geometry_hashes"],
                "heldout_ids": metadata["heldout_ids"],
                "dataset_sha256": metadata["dataset_sha256"],
                "seed": seed,
                "epoch": epoch + 1,
                "validation_mse_standardized": None
                if metadata.get("fit_all")
                else val_loss,
                "training_mode": "all_samples_final_epoch"
                if metadata.get("fit_all")
                else "validation_selected",
            }
            torch.save(checkpoint, checkpoint_path)

    scale = target_std.numpy()
    report = {
        "variant": variant,
        "model_size": model_size,
        "encoder": encoder,
        "fusion": fusion,
        **(
            {"view_feature_size": view_feature_size}
            if view_feature_size is not None
            else {}
        ),
        "device": device_name,
        "workers": workers,
        "preload": preload,
        "preload_seconds": preload_seconds,
        "augmentation": "sampling_jitter_v1" if augment else "none",
        "encoder_initialization": encoder_initialization,
        **width_initialization,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "train_count": len(train_set),
        "validation_count": len(validation_set),
        "train_ids": metadata["train_ids"],
        "train_target_counts": dict(zip(target_names, train_target_counts.tolist())),
        "validation_target_counts": dict(zip(target_names, val_target_counts.tolist())),
        "best_epoch": int(np.argmin([row["val_mse"] for row in history])) + 1,
        "best_validation_mse_standardized": best_loss,
        "validation_mae": {},
        "validation_rmse": {},
        "history": history,
        "seed": seed,
    }
    if "train_source_counts" in metadata:
        report["train_source_counts"] = metadata["train_source_counts"]
    for i, name in enumerate(target_names):
        err = best_error[:, i] * scale[i]
        observed = np.isfinite(err)
        report["validation_mae"][name] = (
            float(np.nanmean(np.abs(err))) if observed.any() else None
        )
        report["validation_rmse"][name] = (
            float(np.sqrt(np.nanmean(err**2))) if observed.any() else None
        )
    if metadata.get("fit_all"):
        report["training_mode"] = "all_samples_final_epoch"
        report["validation_count"] = 0
        report["validation_target_counts"] = dict.fromkeys(target_names, 0)
        report["training_fit_target_counts"] = dict(
            zip(target_names, val_target_counts.tolist())
        )
        report["best_epoch"] = epochs
        report["training_fit_mse_standardized"] = report.pop(
            "best_validation_mse_standardized"
        )
        report["training_fit_mae"] = report.pop("validation_mae")
        report["training_fit_rmse"] = report.pop("validation_rmse")
        for row in report["history"]:
            row["training_fit_mse"] = row.pop("val_mse")
    (output_dir / "training_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    return checkpoint_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--variant", choices=("parameters", "four", "five"), default="five"
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--workers", type=int, default=0, help="Parallel numerical-view readers"
    )
    parser.add_argument(
        "--preload",
        action="store_true",
        help="Load numerical views once into RAM; requires workers=0",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Training-only coherent sampling jitter and small depth noise",
    )
    parser.add_argument(
        "--model-size", choices=("standard", "small"), default="standard"
    )
    parser.add_argument(
        "--fit-all",
        action="store_true",
        help="Fit every case; save final epoch; no independent validation claim",
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
    parser.add_argument(
        "--view-feature-size",
        type=int,
        help="Pyramid output width only; transfer pre-projection CNN weights for a matched width experiment",
    )
    parser.add_argument(
        "--encoder-checkpoint",
        type=Path,
        help="Initialize only the compatible five-view pyramid encoders",
    )
    args = parser.parse_args()
    from .data import CaseDataset, load_dataset

    manifest = load_dataset(args.prepared_dir)
    cases = manifest["cases"]
    train = cases if args.fit_all else [c for c in cases if c["split"] == "train"]
    validation = [c for c in cases if c["split"] == "val"]
    if args.fit_all:
        validation = train
    if not train or not validation:
        raise SystemExit("dataset.json must include non-empty train and val splits")
    target_names = manifest["target_names"]
    x = np.asarray([c["parameters"] for c in train], dtype=np.float64)
    y = np.asarray([c["targets"] for c in train], dtype=np.float64)
    try:
        stats = training_normalization(x, y, target_names)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    train_set = CaseDataset(
        args.prepared_dir,
        train,
        stats["parameter_mean"],
        stats["parameter_std"],
        stats["target_mean"],
        stats["target_std"],
        args.variant,
    )
    val_set = CaseDataset(
        args.prepared_dir,
        validation,
        stats["parameter_mean"],
        stats["parameter_std"],
        stats["target_mean"],
        stats["target_std"],
        args.variant,
    )
    source_counts = {}
    if any("condition_profile" in case for case in train):
        for case in train:
            profile = case.get("condition_profile", "unspecified")
            counts = source_counts.setdefault(
                profile, {"cases": 0, **dict.fromkeys(target_names, 0)}
            )
            counts["cases"] += 1
            for name, target in zip(
                target_names, np.asarray(case["targets"], dtype=np.float64)
            ):
                counts[name] += int(np.isfinite(target))
    condition_profiles = manifest.get("condition_profiles")
    if condition_profiles is not None:
        try:
            condition_profiles = condition_profiles_with_train_ranges(
                condition_profiles, train
            )
        except ValueError as error:
            raise SystemExit(str(error)) from error
    metadata = {
        "fit_all": args.fit_all,
        "render_config": manifest["render_config"],
        "conditions": manifest["conditions"],
        **(
            {"condition_profiles": condition_profiles}
            if condition_profiles is not None
            else {}
        ),
        **({"train_source_counts": source_counts} if source_counts else {}),
        "parameter_ranges": {
            n: [float(x[:, i].min()), float(x[:, i].max())]
            for i, n in enumerate(manifest["parameter_names"])
        },
        "data_kind": manifest.get("data_kind", "unspecified"),
        "provenance": manifest.get("provenance", {}),
        "train_ids": [c["case_id"] for c in train],
        "train_family_ids": sorted({c["family_id"] for c in train}),
        "split_policy": manifest.get("split_policy", "family_disjoint"),
        "train_geometry_hashes": sorted({c["geometry_hash"] for c in train}),
        "heldout_ids": []
        if args.fit_all
        else [c["case_id"] for c in cases if c["split"] in ("val", "test")],
        "dataset_sha256": hashlib.sha256(
            (args.prepared_dir / "dataset.json").read_bytes()
        ).hexdigest(),
    }
    path = train_model(
        train_set,
        val_set,
        parameter_names=manifest["parameter_names"],
        target_names=target_names,
        normalization=stats,
        metadata=metadata,
        output_dir=args.output_dir,
        variant=args.variant,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device_name=args.device,
        augment=args.augment,
        model_size=args.model_size,
        encoder=args.encoder,
        fusion=args.fusion,
        workers=args.workers,
        preload=args.preload,
        encoder_checkpoint=args.encoder_checkpoint,
        view_feature_size=args.view_feature_size,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
