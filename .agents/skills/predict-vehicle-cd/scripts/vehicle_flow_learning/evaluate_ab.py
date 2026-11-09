"""Evaluate completed paired vehicle field runs on fixed validation samples only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from vehicle_cd_learning.model import VIEWS

from .train_vehicle import boundary_loss
from .vehicle import field_derivatives, load_vehicle, momentum_residual


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def validate_run(folder, audit_hash, epoch):
    """Check the completed checkpoint and its exact cached training inputs."""
    report = json.loads((folder / "report.json").read_text())
    backbone, original, field, saved = load_vehicle(folder / "model.pt")
    identity = saved["identity"]
    if (
        saved["epoch"] != epoch
        or len(report["history"]) != epoch
        or report["history"] != saved["history"]
        or report["history"][-1]["epoch"] != epoch
    ):
        raise ValueError(
            f"{folder}: report and checkpoint must both complete epoch {epoch}"
        )
    if report["test_used"] is not False or report["field_labels_used"] is not False:
        raise ValueError("Report must confirm no test or field labels were used")
    if {row["case_id"] for row in report["field_pilot"]} != set(
        identity["field_train_ids"] + identity["field_val_ids"]
    ):
        raise ValueError("Report does not contain the complete field pilot")
    if identity["audit_sha256"] != audit_hash:
        raise ValueError("Audit identity differs")
    cache = folder / "encoded.npz"
    cases_path = folder / "encoded-cases.json"
    cache_id = json.loads((folder / "encoded-identity.json").read_text())
    expected = {
        "audit": audit_hash,
        "encoder": identity["encoder_sha256"],
        "cache": sha256(cache),
        "cases": sha256(cases_path),
        "version": "pyramid-detail-pooled-v1",
    }
    if cache_id != expected or cache_id["cache"] != identity["pooled_cache_sha256"]:
        raise ValueError("Pooled cache identity differs")
    cases = json.loads(cases_path.read_text())
    if any(case["split"] not in ("train", "val") for case in cases):
        raise ValueError("Only train/validation cases may occur in the pooled cache")
    if len({case["case_id"] for case in cases}) != len(cases):
        raise ValueError("Case IDs must be unique")
    field_ids = identity["field_train_ids"] + identity["field_val_ids"]
    if len(set(field_ids)) != len(field_ids) or set(field_ids) != set(
        identity["prepared_surface_sha256"]
    ):
        raise ValueError("Field case and surface membership differs")
    split_by_id = {case["case_id"]: case["split"] for case in cases}
    if any(
        split_by_id.get(case_id) != "train" for case_id in identity["field_train_ids"]
    ) or any(
        split_by_id.get(case_id) != "val" for case_id in identity["field_val_ids"]
    ):
        raise ValueError("Field training or validation split differs")
    for case_id, files in identity["prepared_surface_sha256"].items():
        if set(files) != {
            "surface.json",
            "surface.npz",
            "surface.vtp",
            "fluid-pool.npy",
        }:
            raise ValueError("Incomplete prepared surface identity")
        for name, digest in files.items():
            if sha256(folder / "surfaces" / case_id / name) != digest:
                raise ValueError(f"Prepared surface or pool changed: {case_id}/{name}")
    return backbone, original, field, saved, cases, cache_id


def validate_pair(a, b):
    ia, ib = a[3]["identity"], b[3]["identity"]
    if ia["conditioning"] != "feedback" or ib["conditioning"] != "latent":
        raise ValueError("Expected A=feedback and B=latent")
    ignored = {"conditioning", "condition_layout"}
    if {k: v for k, v in ia.items() if k not in ignored} != {
        k: v for k, v in ib.items() if k not in ignored
    }:
        raise ValueError("Paired training identities differ")
    if a[5] != b[5] or a[4] != b[4]:
        raise ValueError("Paired cases or cached inputs differ")
    if not torch.equal(a[3]["feedback_initial"], b[3]["feedback_initial"]):
        raise ValueError("Feedback initialization differs")
    if not torch.equal(a[3]["torch_rng_state"], b[3]["torch_rng_state"]):
        raise ValueError("Training RNG streams differ")
    if a[3]["train_ids"] != b[3]["train_ids"]:
        raise ValueError("Paired training membership differs")


def integrated_force(
    model, xyz, normals, weights, feature, inflow, viscosity, batch_size
):
    """Integrate traction in chunks with one global weighted stress mean."""
    weighted_stress = torch.zeros((3, 3), dtype=torch.float64)
    weighted_normal = torch.zeros(3, dtype=torch.float64)
    weighted_traction = torch.zeros(3, dtype=torch.float64)
    total_weight = 0.0
    for start in range(0, len(xyz), batch_size):
        sl = slice(start, start + batch_size)
        _, values, jacobian, _ = field_derivatives(model, xyz[sl], feature, inflow)
        stress = -values[:, 3, None, None] * torch.eye(3)
        stress = stress + 2 * viscosity * (jacobian + jacobian.transpose(1, 2))
        stress, normal, weight = (
            stress.detach().double(),
            normals[sl].double(),
            weights[sl].double(),
        )
        weighted_stress += (stress * weight[:, None, None]).sum(0)
        weighted_normal += (normal * weight[:, None]).sum(0)
        weighted_traction += (
            torch.einsum("nij,nj->ni", stress, normal) * weight[:, None]
        ).sum(0)
        total_weight += float(weight.sum())
    if total_weight <= 0:
        raise ValueError("Surface weights must have positive sum")
    force = weighted_traction - (weighted_stress / total_weight) @ weighted_normal
    return force[[0, 2]].tolist()


def scalar_results(backbone, original, pooled, parameters, cases, batch_size):
    ids = [i for i, c in enumerate(cases) if c["split"] == "val"]
    predictions = []
    with torch.no_grad():
        for start in range(0, len(ids), batch_size):
            indices = ids[start : start + batch_size]
            features = pooled[indices]
            latent = backbone.encode_pooled(
                {name: features[:, i] for i, name in enumerate(VIEWS)},
                parameters[indices],
            )
            predictions.append(
                backbone.head(latent) * torch.tensor(original["target_std"])
                + torch.tensor(original["target_mean"])
            )
    predictions = torch.cat(predictions).numpy()
    target = np.asarray([cases[i]["targets"] for i in ids], dtype=float)
    scores = {}
    for index, name in enumerate(original["target_names"]):
        valid = np.isfinite(target[:, index])
        errors = np.abs(predictions[valid, index] - target[valid, index])
        scores[name] = {
            "count": int(valid.sum()),
            "mae": float(errors.mean()),
            "p95_absolute_error": float(np.quantile(errors, 0.95)),
        }
    return scores, len(ids)


def read_sample(folder, case_id):
    directory = folder / "surfaces" / case_id
    metadata = json.loads((directory / "surface.json").read_text())
    with np.load(directory / "surface.npz", allow_pickle=False) as data:
        xyz = torch.tensor(
            data["points_m"] / metadata["normalization"]["length_m"],
            dtype=torch.float32,
        )
        normals = torch.tensor(data["normals"], dtype=torch.float32)
        weights = torch.tensor(
            data["weights_m2"] / metadata["reference_area_m2"], dtype=torch.float32
        )
    return (
        xyz,
        normals,
        weights,
        torch.tensor(np.load(directory / "fluid-pool.npy"), dtype=torch.float32),
    )


def evaluate_fields(folder, run, pooled, parameters, count, seed, batch_size):
    backbone, _, field, saved, cases, _ = run
    cases_by_id = {c["case_id"]: (i, c) for i, c in enumerate(cases)}
    rows = []
    for number, case_id in enumerate(saved["identity"]["field_val_ids"]):
        index, case = cases_by_id[case_id]
        if case["split"] != "val" or case["condition_profile"] != "drivaerml":
            raise ValueError("Field evaluation must use DrivAer validation cases")
        xyz, normals, weights, fluid = read_sample(folder, case_id)
        with torch.no_grad():
            f = pooled[index : index + 1]
            latent = backbone.encode_pooled(
                {name: f[:, i] for i, name in enumerate(VIEWS)},
                parameters[index : index + 1],
            )[0]
            feature = (
                torch.cat((latent, backbone.head(latent)))
                if saved["identity"]["conditioning"] == "feedback"
                else latent
            )
        inflow = torch.tensor([1.0, 0.0, 0.0])
        viscosity = 1 / case["reynolds"] + saved["identity"]["eddy_viscosity"]
        coefficients = integrated_force(
            field, xyz, normals, weights, feature, inflow, viscosity, batch_size
        )
        torch.manual_seed(seed + number)
        points = fluid[torch.randint(len(fluid), (count,))]
        wall = xyz[torch.randint(len(xyz), (count,))]
        residual = (
            momentum_residual(field, points, feature, inflow, viscosity).square().mean()
        )
        with torch.no_grad():
            boundary = (
                boundary_loss(field, feature, inflow, count)
                + field(wall, feature, inflow)[:, :3].square().mean()
            )
        target = case["targets"]
        if not np.isfinite(coefficients + target).all():
            raise ValueError("Non-finite coefficients")
        ratio_valid = abs(coefficients[0]) > 1e-6 and abs(target[0]) > 1e-6
        ratios = (
            [coefficients[1] / coefficients[0], target[1] / target[0]]
            if ratio_valid
            else None
        )
        rows.append(
            {
                "case_id": case_id,
                "predicted_cd_cl": coefficients,
                "target_cd_cl": target,
                "absolute_error_cd_cl": [
                    abs(coefficients[i] - target[i]) for i in range(2)
                ],
                "sign_error_cd_cl": [
                    bool(np.sign(coefficients[i]) != np.sign(target[i]))
                    for i in range(2)
                ],
                "predicted_target_lift_over_drag": ratios,
                "absolute_error_lift_over_drag": abs(ratios[0] - ratios[1])
                if ratios
                else None,
                "sign_error_lift_over_drag": bool(
                    np.sign(ratios[0]) != np.sign(ratios[1])
                )
                if ratios
                else None,
                "pde_mse": float(residual.detach()),
                "boundary_mse": float(boundary),
            }
        )
        print(
            json.dumps(
                {
                    "stage": "field_validation",
                    "conditioning": saved["identity"]["conditioning"],
                    "case": number + 1,
                    "total": len(saved["identity"]["field_val_ids"]),
                }
            ),
            flush=True,
        )
    return rows


def field_metrics(rows):
    predicted = np.asarray([r["predicted_cd_cl"] for r in rows])
    target = np.asarray([r["target_cd_cl"] for r in rows])
    scores = {}
    for index, name in enumerate(("cd", "cl")):
        error = np.abs(predicted[:, index] - target[:, index])
        scores[name] = {
            "mae": float(error.mean()),
            "p95_absolute_error": float(np.quantile(error, 0.95)),
            "sign_errors": int(
                (np.sign(predicted[:, index]) != np.sign(target[:, index])).sum()
            ),
        }
    valid = (np.abs(predicted[:, 0]) > 1e-6) & (np.abs(target[:, 0]) > 1e-6)
    if valid.any():
        error = np.abs(
            predicted[valid, 1] / predicted[valid, 0]
            - target[valid, 1] / target[valid, 0]
        )
        scores["lift_over_drag"] = {
            "count": int(valid.sum()),
            "excluded_near_zero_drag": int((~valid).sum()),
            "mae": float(error.mean()),
            "p95_absolute_error": float(np.quantile(error, 0.95)),
            "sign_errors": int(
                (
                    np.sign(predicted[valid, 1] / predicted[valid, 0])
                    != np.sign(target[valid, 1] / target[valid, 0])
                ).sum()
            ),
        }
    else:
        scores["lift_over_drag"] = {
            "count": 0,
            "excluded_near_zero_drag": len(rows),
            "mae": None,
            "p95_absolute_error": None,
            "sign_errors": 0,
        }
    for name in ("pde_mse", "boundary_mse"):
        values = np.asarray([r[name] for r in rows])
        scores[name] = {
            "mean": float(values.mean()),
            "p95": float(np.quantile(values, 0.95)),
        }
    return scores


def parameter_counts(backbone, field):
    size = lambda module: sum(p.numel() for p in module.parameters())
    shared = sum(
        size(m)
        for m in (
            backbone.image_encoder.project,
            backbone.bottom_encoder.project,
            backbone.parameter_encoder,
            backbone.view_attention,
        )
        if m is not None
    )
    return {
        "backbone_total_including_frozen_cnn": size(backbone),
        "trainable_shared_encoder": shared,
        "coefficient_head": size(backbone.head),
        "field": size(field),
        "trainable_total": shared + size(backbone.head) + size(field),
    }


def evaluate(args):
    if min(args.epoch, args.collocation_count, args.batch_size) < 1:
        raise ValueError("Epoch, collocation count and batch size must be positive")
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    audit_hash = sha256(args.audit)
    a = validate_run(args.a, audit_hash, args.epoch)
    b = validate_run(args.b, audit_hash, args.epoch)
    validate_pair(a, b)
    if len(a[3]["identity"]["field_val_ids"]) != args.field_val_count:
        raise ValueError("Unexpected validation field count")
    if len(a[3]["identity"]["field_train_ids"]) != args.field_train_count:
        raise ValueError("Unexpected training field count")
    train_ids = {c["case_id"] for c in a[4] if c["split"] == "train"}
    if len(train_ids) != args.scalar_train_count or set(a[3]["train_ids"]) != train_ids:
        raise ValueError("Unexpected scalar training membership")
    builtin_train_count = sum(
        c["split"] == "train" and c["condition_profile"] == "builtin_assumed"
        for c in a[4]
    )
    if builtin_train_count != args.builtin_train_count:
        raise ValueError("Unexpected built-in scalar training count")
    with np.load(args.a / "encoded.npz", allow_pickle=False) as data:
        pooled, parameters = (
            torch.tensor(data["features"]),
            torch.tensor(data["parameters"]),
        )
    if (
        len(pooled) != len(a[4])
        or len(parameters) != len(a[4])
        or not torch.isfinite(pooled).all()
        or not torch.isfinite(parameters).all()
    ):
        raise ValueError("Invalid pooled cache dimensions or values")
    result = {
        "scope": "Validation only; proxy physics and integrated force are not field truth.",
        "test_used": False,
        "audit_sha256": audit_hash,
        "training_seed": a[3]["identity"]["seed"],
        "evaluation_seed": args.seed,
        "collocation_count": args.collocation_count,
        "epoch": args.epoch,
        "scalar_training_cases": len(train_ids),
        "builtin_training_cases": builtin_train_count,
        "source_checkpoint_sha256": a[3]["identity"]["encoder_sha256"],
        "pooled_cache_sha256": a[3]["identity"]["pooled_cache_sha256"],
        "field_shared_initial_sha256": a[3]["identity"]["field_shared_initial_sha256"],
        "initial_encoder_evidence": "Same source checkpoint, seed and training configuration; no independent initial encoder snapshot was saved.",
        "training_rng_states_equal": True,
        "runs": {},
    }
    for name, folder, run in (("A", args.a, a), ("B", args.b, b)):
        scalar, scalar_count = scalar_results(
            run[0], run[1], pooled, parameters, run[4], args.batch_size
        )
        if scalar_count != args.scalar_val_count:
            raise ValueError("Unexpected validation scalar count")
        rows = evaluate_fields(
            folder,
            run,
            pooled,
            parameters,
            args.collocation_count,
            args.seed,
            args.batch_size,
        )
        result["runs"][name] = {
            "checkpoint_sha256": sha256(folder / "model.pt"),
            "parameters": parameter_counts(run[0], run[2]),
            "scalar_val_cases": scalar_count,
            "scalar_head": scalar,
            "field_val_cases": len(rows),
            "field_metrics": field_metrics(rows),
            "cases": rows,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "training_seed": result["training_seed"],
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("a", type=Path, help="Completed feedback run directory")
    parser.add_argument("b", type=Path, help="Completed latent-only run directory")
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epoch", type=int, default=31)
    parser.add_argument("--field-val-count", type=int, default=34)
    parser.add_argument("--field-train-count", type=int, default=32)
    parser.add_argument("--scalar-val-count", type=int, default=119)
    parser.add_argument("--scalar-train-count", type=int, default=1099)
    parser.add_argument("--builtin-train-count", type=int, default=26)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--collocation-count", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4096)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
