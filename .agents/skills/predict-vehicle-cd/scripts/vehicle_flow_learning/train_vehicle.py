"""Train an experimental 3D decoder on existing vehicle coefficients.

All original train cases supervise the coefficient head. A bounded DrivAer
pilot additionally supervises integrated total traction and proxy mean-flow
physics. The box boundaries and constant eddy viscosity are explicit modeling
approximations, not verified source CFD settings or flow-field truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from vehicle_cd_learning.model import VIEWS, PyramidEncoder
from vehicle_cd_learning.predict import load_model
from vehicle_cd_learning.train import masked_mse_loss

from .prepare_vehicle import encode_cases, prepare_pilot
from .vehicle import VehicleField, force_coefficients, momentum_residual


def paired_field(feature_count, conditioning):
    """Initialize shared field weights identically for both conditioning layouts."""
    model = VehicleField(feature_count=feature_count + 2, width=48)
    feedback = model.condition[0].weight[:, -2:].detach().clone()
    if conditioning == "latent":
        state = torch.get_rng_state()
        layer = torch.nn.Linear(feature_count, 24)
        torch.set_rng_state(state)
        with torch.no_grad():
            layer.weight.copy_(model.condition[0].weight[:, :feature_count])
            layer.bias.copy_(model.condition[0].bias)
        model.condition[0] = layer
    return model, feedback


def boundary_loss(model, feature, inflow, count):
    # Explicit proxy domain, with full stationary floor and constant outer flow.
    outer = torch.rand(count, 3) * torch.tensor([9.0, 4.0, 3.0]) + torch.tensor(
        [-3.0, -2.0, 0.0]
    )
    sides = torch.randint(5, (count,))
    for side, (axis, value) in enumerate(((0, -3), (0, 6), (1, -2), (1, 2), (2, 3))):
        outer[sides == side, axis] = value
    ground = torch.rand(count, 3) * torch.tensor([9.0, 4.0, 0.0]) + torch.tensor(
        [-3.0, -2.0, 0.0]
    )
    outer_prediction = model(outer, feature, inflow)
    return (
        (outer_prediction[:, :3] - inflow).square().mean()
        + outer_prediction[:, 3].square().mean()
        + model(ground, feature, inflow)[:, :3].square().mean()
    )


def field_loss(
    model,
    sample,
    feature,
    count,
    eddy_viscosity,
    force_count=None,
    boundary_weight=1.0,
    cl_weight=1.0,
):
    case = sample["case"]
    viscosity = 1 / case["reynolds"] + eddy_viscosity
    inflow = feature.new_tensor([1.0, 0.0, 0.0])
    force_xyz, normals, weights = sample["xyz"], sample["normals"], sample["weights"]
    if force_count is not None:
        selected = torch.randperm(len(force_xyz))[:force_count]
        force_xyz, normals = force_xyz[selected], normals[selected]
        weights = weights[selected] * len(sample["xyz"]) / len(selected)
    coefficients = force_coefficients(
        model,
        force_xyz,
        normals,
        weights,
        feature,
        inflow,
        viscosity,
    )
    target = feature.new_tensor(case["targets"])
    coefficient_loss = (
        ((coefficients - target) / feature.new_tensor([0.04, 0.2])).square()
        * feature.new_tensor([1.0, cl_weight])
    ).mean()
    chosen = torch.randint(len(sample["fluid"]), (count,))
    residual = momentum_residual(
        model, sample["fluid"][chosen], feature, inflow, viscosity
    )
    wall = sample["xyz"][torch.randint(len(sample["xyz"]), (count,))]
    physics = residual.square().mean()
    boundary = (
        boundary_loss(model, feature, inflow, count)
        + model(wall, feature, inflow)[:, :3].square().mean()
    )
    return (
        coefficient_loss + physics + boundary_weight * boundary,
        coefficients,
        physics,
        boundary,
    )


def train(args):
    if (
        min(
            args.epochs,
            args.surface_count,
            args.collocation_count,
            args.batch_size,
            args.force_count,
        )
        < 1
    ):
        raise ValueError("Training sizes must be positive")
    if (
        not args.allow_proxy_physics
        or not np.isfinite([args.eddy_viscosity, args.lr, args.field_cl_weight]).all()
        or args.eddy_viscosity < 0
        or args.lr <= 0
        or args.field_cl_weight <= 0
    ):
        raise ValueError(
            "Require --allow-proxy-physics, nonnegative eddy viscosity and positive learning rate/Cl weight"
        )
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    audit_bytes = args.audit.read_bytes()
    audit = json.loads(audit_bytes)
    args.output_dir.mkdir(parents=True, exist_ok=args.resume or args.reuse_prepared)
    if (
        args.reuse_prepared
        and not args.resume
        and (args.output_dir / "model.pt").exists()
    ):
        raise ValueError("Prepared output already contains a model; use --resume")
    backbone, checkpoint = load_model(args.encoder)
    finetune = args.finetune_encoder
    if finetune and (
        backbone.variant != "five"
        or not isinstance(backbone.image_encoder, PyramidEncoder)
        or not backbone.image_encoder.detail_pooling
    ):
        raise ValueError(
            "Partial fine-tuning requires five-view detail pyramid encoders"
        )
    if not finetune and args.conditioning != "feedback":
        raise ValueError("Latent-only conditioning requires --finetune-encoder")
    encoder_sha = hashlib.sha256(args.encoder.read_bytes()).hexdigest()
    cache = args.output_dir / "encoded.npz"
    if cache.exists() and (args.resume or args.reuse_prepared):
        cache_identity = json.loads(
            (args.output_dir / "encoded-identity.json").read_text()
        )
        expected_cache = {
            "audit": hashlib.sha256(audit_bytes).hexdigest(),
            "encoder": encoder_sha,
            "cache": hashlib.sha256(cache.read_bytes()).hexdigest(),
        }
        if finetune:
            expected_cache["version"] = "pyramid-detail-pooled-v1"
            expected_cache["cases"] = hashlib.sha256(
                (args.output_dir / "encoded-cases.json").read_bytes()
            ).hexdigest()
        if cache_identity != expected_cache:
            raise ValueError("Encoded cache ownership or contents changed")
        cases = json.loads((args.output_dir / "encoded-cases.json").read_text())
        with np.load(cache) as data:
            features = torch.tensor(data["features"])
            parameters = torch.tensor(data["parameters"]) if finetune else None
    else:
        cases, features, parameters = encode_cases(
            audit, backbone, checkpoint, args.output_dir, pooled=finetune
        )
        cache_identity = {
            "audit": hashlib.sha256(audit_bytes).hexdigest(),
            "encoder": encoder_sha,
            "cache": hashlib.sha256(cache.read_bytes()).hexdigest(),
        }
        if finetune:
            cache_identity["version"] = "pyramid-detail-pooled-v1"
            cache_identity["cases"] = hashlib.sha256(
                (args.output_dir / "encoded-cases.json").read_bytes()
            ).hexdigest()
        (args.output_dir / "encoded-identity.json").write_text(
            json.dumps(cache_identity) + "\n"
        )
    pilot = prepare_pilot(
        args.audit,
        audit,
        args.output_dir,
        args.field_train_cases,
        args.field_val_cases,
        args.surface_count,
    )
    positions = {c["case_id"]: i for i, c in enumerate(cases)}
    train_ids = [i for i, c in enumerate(cases) if c["split"] == "train"]
    val_ids = [i for i, c in enumerate(cases) if c["split"] == "val"]
    field_train = [s for s in pilot if s["case"]["split"] == "train"]
    if not field_train:
        raise ValueError("At least one field train geometry required")
    target = torch.tensor(
        np.array([c["targets"] for c in cases], dtype=float), dtype=torch.float32
    )
    mean, scale = (
        torch.tensor(checkpoint["target_mean"]),
        torch.tensor(checkpoint["target_std"]),
    )
    normalized = (target - mean) / scale
    if finetune:
        backbone.requires_grad_(False)
        for module in (
            backbone.image_encoder.project,
            backbone.bottom_encoder.project,
            backbone.parameter_encoder,
            backbone.view_attention,
            backbone.head,
        ):
            if module is not None:
                module.requires_grad_(True)

    def latent(indices):
        if not finetune:
            return features[indices]
        pooled = features[indices]
        return backbone.encode_pooled(
            {name: pooled[:, i] for i, name in enumerate(VIEWS)}, parameters[indices]
        )

    feature_count = backbone.head[0].in_features
    model, feedback_initial = paired_field(feature_count, args.conditioning)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + [p for p in backbone.parameters() if p.requires_grad]
        if finetune
        else list(model.parameters()) + list(backbone.head.parameters()),
        lr=args.lr,
    )
    start_epoch, history = 0, []
    path = args.output_dir / "model.pt"
    identity = {
        "audit_sha256": hashlib.sha256(audit_bytes).hexdigest(),
        "encoder_sha256": encoder_sha,
        "surface_count": args.surface_count,
        "eddy_viscosity": args.eddy_viscosity,
        "batch_size": args.batch_size,
        "collocation_count": args.collocation_count,
        "lr": args.lr,
        "force_count": args.force_count,
        "boundary_weight": args.boundary_weight,
        "condition_layout": "encoder feature plus predicted standardized Cd/Cl",
        "field_train_ids": [s["case"]["case_id"] for s in field_train],
    }
    if args.field_cl_weight != 1.0:
        identity["field_cl_weight"] = args.field_cl_weight
    if finetune:
        initial_hash = hashlib.sha256()
        for name, value in model.state_dict().items():
            if name == "condition.0.weight":
                value = value[:, :feature_count]
            initial_hash.update(name.encode())
            initial_hash.update(value.detach().contiguous().numpy().tobytes())
        identity.update(
            {
                "conditioning": args.conditioning,
                "condition_layout": "encoder feature plus predicted standardized Cd/Cl"
                if args.conditioning == "feedback"
                else "encoder feature only",
                "encoder_training": "pyramid project, parameter encoder, view attention and head; pooled CNN frozen",
                "field_val_ids": [
                    s["case"]["case_id"] for s in pilot if s["case"]["split"] == "val"
                ],
                "seed": args.seed,
                "pooled_cache_sha256": cache_identity["cache"],
                "field_shared_initial_sha256": initial_hash.hexdigest(),
            }
        )
        surface_hashes = {}
        for sample in pilot:
            folder = Path(sample["surface_path"]).parent
            surface_hashes[sample["case"]["case_id"]] = {}
            for name in (
                "surface.json",
                "surface.npz",
                "surface.vtp",
                "fluid-pool.npy",
            ):
                with (folder / name).open("rb") as handle:
                    surface_hashes[sample["case"]["case_id"]][name] = (
                        hashlib.file_digest(handle, "sha256").hexdigest()
                    )
        identity["prepared_surface_sha256"] = surface_hashes
    if args.resume:
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved["identity"] != identity:
            raise ValueError("Resume data/physics differs")
        model.load_state_dict(saved["field_state"])
        if finetune:
            backbone.load_state_dict(saved["backbone_state"])
            feedback_initial = saved["feedback_initial"]
        else:
            backbone.head.load_state_dict(saved["coefficient_head"])
        optimizer.load_state_dict(saved["optimizer_state"])
        torch.set_rng_state(saved["torch_rng_state"])
        start_epoch, history = saved["epoch"], saved["history"]
        if args.epochs <= start_epoch:
            raise ValueError("Resume requires a later final epoch")
    seen = set(saved["train_ids"]) if args.resume else set()
    step = start_epoch * math.ceil(len(train_ids) / args.batch_size)
    for epoch in range(start_epoch, args.epochs):
        permutation = torch.tensor(train_ids)[torch.randperm(len(train_ids))]
        sums = []
        for batch in permutation.split(args.batch_size):
            scalar_loss = masked_mse_loss(
                backbone.head(latent(batch)), normalized[batch]
            )
            sample = field_train[step % len(field_train)]
            raw_feature = latent([positions[sample["case"]["case_id"]]])[0]
            feature = (
                torch.cat((raw_feature, backbone.head(raw_feature)))
                if args.conditioning == "feedback"
                else raw_feature
            )
            spatial_loss, _, residual, boundary = field_loss(
                model,
                sample,
                feature,
                args.collocation_count,
                args.eddy_viscosity,
                min(args.force_count, args.surface_count),
                args.boundary_weight,
                args.field_cl_weight,
            )
            loss = scalar_loss + spatial_loss
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite joint loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            seen.update(cases[i]["case_id"] for i in batch.tolist())
            step += 1
            sums.append(
                [
                    float(scalar_loss.detach()),
                    float(spatial_loss.detach()),
                    float(residual.detach()),
                    float(boundary.detach()),
                ]
            )
        with torch.no_grad():
            val_predictions = torch.cat(
                [
                    backbone.head(latent(batch))
                    for batch in torch.tensor(val_ids).split(args.batch_size)
                ]
            )
            error = (val_predictions * scale + mean) - target[val_ids]
        row = {
            "epoch": epoch + 1,
            "mean_losses_scalar_field_pde_boundary": np.mean(sums, 0).tolist(),
            "coefficient_head_val_mae": error.abs().mean(0).tolist(),
        }
        history.append(row)
        saved = {
            "field_state": model.state_dict(),
            "coefficient_head": backbone.head.state_dict(),
            "field_config": {
                "feature_count": model.condition[0].in_features,
                "width": 48,
            },
            "encoder_checkpoint": str(args.encoder.resolve()),
            "encoder_sha256": encoder_sha,
            "identity": identity,
            "epoch": epoch + 1,
            "history": history,
            "optimizer_state": optimizer.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "scope": "Experimental weak field supervision, constant effective-viscosity closure and proxy domain; no validated vehicle fields",
            "domain_l": [[-3, -2, 0], [6, 2, 3]],
            "train_ids": sorted(seen),
            "builtin_train_ids": sorted(
                c["case_id"]
                for c in cases
                if c["condition_profile"] == "builtin_assumed" and c["case_id"] in seen
            ),
        }
        if finetune:
            saved["backbone_state"] = backbone.state_dict()
            saved["feedback_initial"] = feedback_initial
        temporary = path.with_suffix(".tmp")
        torch.save(saved, temporary)
        temporary.replace(path)
        print(json.dumps(row), flush=True)
    rows = []
    feature_checkpoint_sha = (
        hashlib.sha256(path.read_bytes()).hexdigest() if finetune else None
    )
    for sample in pilot:
        case = sample["case"]
        with torch.no_grad():
            raw_feature = latent([positions[case["case_id"]]])[0]
            feature = (
                torch.cat((raw_feature, backbone.head(raw_feature)))
                if args.conditioning == "feedback"
                else raw_feature
            )
        _, force, residual, boundary = field_loss(
            model, sample, feature, args.collocation_count, args.eddy_viscosity
        )
        rows.append(
            {
                "case_id": case["case_id"],
                "split": case["split"],
                "force_cd_cl": force.detach().tolist(),
                "targets": case["targets"],
                "pde_mse": float(residual.detach()),
                "boundary_mse": float(boundary.detach()),
            }
        )
        np.savez(
            args.output_dir / f"{case['case_id']}-features.npz",
            feature=raw_feature.detach().numpy(),
            inflow=np.array([1.0, 0.0, 0.0]),
            profile="drivaerml",
            encoder_sha256=encoder_sha,
            **(
                {"feature_checkpoint_sha256": feature_checkpoint_sha}
                if finetune
                else {}
            ),
            surface_path=sample["surface_path"],
            length_m=sample["length_m"],
        )
    report = {
        "scope": saved["scope"],
        "cases_seen": len(seen),
        "builtin_seen": saved["builtin_train_ids"],
        "field_pilot": rows,
        "history": history,
        "test_used": False,
        "physics_approximation": "Constant nu_eff=1/Re+configured eddy viscosity; full stationary floor and outer flow on proxy box, not verified original source BC/closure",
        "field_labels_used": False,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit", type=Path)
    parser.add_argument("encoder", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--surface-count", type=int, default=512)
    parser.add_argument("--force-count", type=int, default=512)
    parser.add_argument("--boundary-weight", type=float, default=1.0)
    parser.add_argument("--field-cl-weight", type=float, default=1.0)
    parser.add_argument("--collocation-count", type=int, default=32)
    parser.add_argument("--field-train-cases", type=int, default=4)
    parser.add_argument("--field-val-cases", type=int, default=2)
    parser.add_argument("--eddy-viscosity", type=float, default=0.001)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--allow-proxy-physics", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--reuse-prepared",
        action="store_true",
        help="Reuse verified prepared files in an existing new output directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Training random seed; surface preparation remains fixed at 42",
    )
    parser.add_argument(
        "--finetune-encoder",
        action="store_true",
        help="Train pyramid projections and fusion from cached pooled features",
    )
    parser.add_argument(
        "--conditioning", choices=("feedback", "latent"), default="feedback"
    )
    train(parser.parse_args())


if __name__ == "__main__":
    main()
