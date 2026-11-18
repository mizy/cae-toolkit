"""Prepare, train and predict vehicle coefficients without absolute U/L features."""

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from aircraft_learning.prepare_scenes import resample_views
from aircraft_learning.scene_experiment import (
    SceneCases,
    normalization,
    sampling_weights,
    verify_data,
)
from torch.utils.data import WeightedRandomSampler

from .data import validate_views
from .model import VIEWS, CoefficientModel
from .render import render_geometry
from .train import train_model

CONTRACT = "vehicle_dimensionless_v1"
PARAMETERS = [
    "width_over_length",
    "height_over_length",
    "log_re_body",
    "mach",
    "alpha_sin",
    "alpha_cos",
    "beta_sin",
    "beta_cos",
    "log_reference_area_over_length2",
    "ground_present",
    "clearance_over_length",
    "ground_speed_over_velocity",
    "wheels_present",
    "wheel_rotation_speed_over_velocity",
    "transition_enabled",
    "gravity_g_length_over_velocity2",
    "altitude_over_10000m",
    "wheel_rotating",
]
NAMES = PARAMETERS + [f"{name}_known" for name in PARAMETERS]
MODEL = {
    "parameter_count": len(NAMES),
    "variant": "five",
    "target_count": 2,
    "size": "small",
    "encoder": "pyramid_detail_attention",
    "fusion": "attention",
}
NETWORK_RENDER = {"ordinary_size": 256, "bottom_size": 256, "window_l": 1.3}
SOURCE_RENDER = {"ordinary_size": 512, "bottom_size": 512, "window_l": 1.3}


def parameters(dimensions, conditions):
    """Geometry supplies ratios; Re is supplied directly, never inferred from U/nu."""
    allowed = set(PARAMETERS) - {
        "width_over_length",
        "height_over_length",
        "log_re_body",
        "alpha_sin",
        "alpha_cos",
        "beta_sin",
        "beta_cos",
        "log_reference_area_over_length2",
    }
    allowed |= {"re_body", "alpha_deg", "beta_deg", "reference_area_over_length2"}
    extra = set(conditions) - allowed
    if extra:
        raise ValueError(
            f"Unsupported coefficient conditions {sorted(extra)}; supply re_body and dimensionless conditions"
        )
    length, width, height = (dimensions[k] for k in ("length_m", "width_m", "height_m"))
    if (
        not np.isfinite([length, width, height]).all()
        or min(length, width, height) <= 0
    ):
        raise ValueError("Geometry dimensions must be finite and positive")
    values = {k: v for k, v in conditions.items() if k in PARAMETERS}
    values.update(width_over_length=width / length, height_over_length=height / length)
    for key, target in (
        ("re_body", "log_re_body"),
        ("reference_area_over_length2", "log_reference_area_over_length2"),
    ):
        value = conditions.get(key)
        if value is not None:
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be finite and positive")
            values[target] = float(np.log(value))
    for axis in ("alpha", "beta"):
        value = conditions.get(axis + "_deg")
        if value is not None:
            if not np.isfinite(value) or not -180 <= value <= 180:
                raise ValueError(f"{axis}_deg must be finite and within [-180,180]")
            values[axis + "_sin"] = float(np.sin(np.deg2rad(value)))
            values[axis + "_cos"] = float(np.cos(np.deg2rad(value)))
    result = [values.get(k) for k in PARAMETERS]
    if not np.isfinite([v for v in result if v is not None]).all():
        raise ValueError("Nonfinite dimensionless condition")
    return result


def normalize_parameters(raw, stats):
    values = np.asarray(raw, dtype=float)
    if values.shape != (len(PARAMETERS),):
        raise ValueError("Unexpected dimensionless input length")
    known = np.isfinite(values)
    mean, std = (
        np.asarray(stats[k][: len(PARAMETERS)])
        for k in ("parameter_mean", "parameter_std")
    )
    return np.concatenate(
        (np.where(known, (values - mean) / std, 0), known.astype(float))
    ).astype(np.float32)


def prepare(source, output):
    """Write a new vehicle-only input contract; keep labels, groups and views intact."""
    if output.exists():
        raise FileExistsError("Use a fresh prepared directory")
    path = source / "dataset.json"
    data = json.loads(path.read_text())
    legacy = data["parameter_names"]
    if set(legacy) != set(PARAMETERS) | {"log_length_m", "log_velocity_m_s"}:
        raise ValueError("Expected the audited twenty-field source contract")
    data["cases"] = [c for c in data["cases"] if c["scene"] == 1]
    verify_data(source, data)
    keep = [legacy.index(name) for name in PARAMETERS]
    output.mkdir(parents=True)
    for c in data["cases"]:
        c["parameters"] = [c["parameters"][i] for i in keep]
        target = output / c["views"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / c["views"], target)
    data.update(
        input_contract=CONTRACT,
        parameter_names=PARAMETERS,
        source_prepared=str(path.resolve()),
        source_dataset_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    data["source_counts"] = {
        s: dict(Counter(c["split"] for c in data["cases"] if c["source"] == s))
        for s in sorted({c["source"] for c in data["cases"]})
    }
    data["unknown_counts"] = {
        s: {
            k: sum(
                c["parameters"][i] is None for c in data["cases"] if c["source"] == s
            )
            for i, k in enumerate(PARAMETERS)
        }
        for s in data["source_counts"]
    }
    (output / "dataset.json").write_text(
        json.dumps(data, indent=2, allow_nan=False) + "\n"
    )
    verify_data(output, data)


def read_prepared(root):
    data = json.loads((root / "dataset.json").read_text())
    if data.get("input_contract") != CONTRACT or data["parameter_names"] != PARAMETERS:
        raise ValueError("Use the vehicle dimensionless prepared data")
    if any(c["scene"] != 1 for c in data["cases"]):
        raise ValueError("This contract is vehicle-only")
    verify_data(root, data)
    return data


def parameter_ranges(cases):
    result = {}
    for i, name in enumerate(PARAMETERS):
        values = [c["parameters"][i] for c in cases if c["parameters"][i] is not None]
        result[name] = [float(min(values)), float(max(values))] if values else None
    return result


def export_checkpoint(source, prepared, output):
    """Remove only columns proven inactive in the trained dimensionless A candidate."""
    if output.exists():
        raise FileExistsError("Refusing to overwrite checkpoint")
    data = read_prepared(prepared)
    ck = torch.load(source, map_location="cpu", weights_only=True)
    if not {"log_length_m", "log_velocity_m_s"}.issubset(
        ck.get("run", {}).get("mask", [])
    ):
        raise ValueError(
            "Source was not trained with both absolute-scale features masked"
        )
    old_path = Path(data["source_prepared"])
    if (
        hashlib.sha256(old_path.read_bytes()).hexdigest()
        != data["source_dataset_sha256"]
    ):
        raise ValueError("Original prepared manifest changed")
    old_data = json.loads(old_path.read_text())
    if ck.get("dataset_sha256") != data["source_dataset_sha256"]:
        raise ValueError("Source checkpoint belongs to different data")
    old_names = old_data["parameter_names"]
    keep = [old_names.index(k) for k in PARAMETERS]
    columns = keep + [i + len(old_names) for i in keep]
    removed = sorted(set(range(2 * len(old_names))) - set(columns))
    stats = dict(ck["normalization"])
    if any(
        stats["parameter_mean"][i] != 0 or stats["parameter_std"][i] != 1
        for i in removed
    ):
        raise ValueError(
            "Removed inputs do not have the trained zero/missing normalization"
        )
    for key in ("parameter_mean", "parameter_std"):
        stats[key] = [stats[key][i] for i in columns]
    stats["parameter_observed_counts"] = [
        stats["parameter_observed_counts"][i] for i in keep
    ]
    train = [c for c in data["cases"] if c["split"] == "train"]
    if set(ck["supervised_train_ids"]) != {c["case_id"] for c in train}:
        raise ValueError(
            "Source candidate did not supervise the same vehicle training set"
        )
    if stats != normalization(train):
        raise ValueError(
            "Compact normalization differs from training-only data moments"
        )
    state = dict(ck["model_state"])
    original = CoefficientModel(**{**MODEL, "parameter_count": 40})
    original.load_state_dict(state, strict=True)
    state["parameter_encoder.0.weight"] = state["parameter_encoder.0.weight"][
        :, columns
    ].clone()
    model = CoefficientModel(**MODEL)
    model.load_state_dict(state, strict=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "input_contract": CONTRACT,
            "model_config": MODEL,
            "model_state": state,
            "raw_parameter_names": PARAMETERS,
            "parameter_names": NAMES,
            "target_names": ["cd", "cl"],
            **stats,
            "render_config": NETWORK_RENDER,
            "source_render_config": SOURCE_RENDER,
            "parameter_ranges": parameter_ranges(train),
            "train_ids": ck["supervised_train_ids"],
            "dataset_sha256": hashlib.sha256(
                (prepared / "dataset.json").read_bytes()
            ).hexdigest(),
            "source_checkpoint": str(source.resolve()),
            "source_checkpoint_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "epoch": ck["epoch"],
            "seed": ck["seed"],
            "migration": "Four trained-zero columns removed; no reoptimization or accuracy claim",
        },
        output,
    )


def load_model(path):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    if (
        ck.get("input_contract") != CONTRACT
        or ck.get("parameter_names") != NAMES
        or ck.get("raw_parameter_names") != PARAMETERS
    ):
        raise ValueError("Checkpoint is not the dimensionless vehicle contract")
    if ck["model_config"]["parameter_count"] != len(NAMES):
        raise ValueError("Checkpoint numeric input dimension differs")
    for key in ("parameter_mean", "parameter_std"):
        values = np.asarray(ck[key])
        if (
            values.shape != (len(NAMES),)
            or not np.isfinite(values).all()
            or (key.endswith("std") and np.any(values <= 0))
        ):
            raise ValueError("Invalid checkpoint normalization")
    model = CoefficientModel(**ck["model_config"])
    model.load_state_dict(ck["model_state"], strict=True)
    return model.eval(), ck


def predict_views(model, ck, views, raw):
    validate_views(views, "five", ck["render_config"])
    x = normalize_parameters(raw, ck)
    with torch.inference_mode():
        y = model(
            {
                k: torch.from_numpy(np.asarray(views[k], np.float32)[None])
                for k in VIEWS
            },
            torch.from_numpy(x[None]),
        ).numpy()[0]
    result = y * ck["target_std"] + ck["target_mean"]
    if not np.isfinite(result).all() or result[0] < 0:
        raise ValueError("Invalid passive-vehicle coefficient prediction")
    return result


def predict_case(checkpoint, case_path, output):
    torch.set_num_threads(2)
    model, ck = load_model(checkpoint)
    case = json.loads(case_path.read_text())
    g = case["geometry"]
    views, dimensions = render_geometry(
        (case_path.parent / g["path"]).resolve(), g, ck["source_render_config"]
    )
    arrays = resample_views(
        {k: np.stack((v.depth, v.mask)).astype(np.float32) for k, v in views.items()}
    )
    raw = parameters(dimensions, case["conditions"])
    result = predict_views(model, ck, arrays, raw)
    warnings = []
    for name, value in zip(PARAMETERS, raw):
        bounds = ck["parameter_ranges"][name]
        if value is not None and (
            bounds is None or value < bounds[0] - 1e-6 or value > bounds[1] + 1e-6
        ):
            warnings.append(f"{name} is outside observed training support")
    if case["conditions"].get("re_body") is None:
        warnings.append(
            "Re is unknown; this call does not establish a Reynolds response"
        )
    targets = (
        ["cd"]
        if case.get("condition_profile") == "builtin_assumed"
        else ck["target_names"]
    )
    payload = {
        "coefficients": {
            k: float(result[i])
            for i, k in enumerate(ck["target_names"])
            if k in targets
        },
        "input_contract": CONTRACT,
        "numeric_inputs": dict(zip(PARAMETERS, raw)),
        "warnings": warnings,
        "checkpoint": str(checkpoint.resolve()),
        "scope": "Experimental vehicle candidate; marginal support checks do not certify new conditions or vehicle families",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return payload


def train(prepared, output, epochs=40, device="mps"):
    data = read_prepared(prepared)
    cases = data["cases"]
    training, validation = (
        [c for c in cases if c["split"] == s] for s in ("train", "val")
    )
    stats = normalization(training)
    sampler = WeightedRandomSampler(
        sampling_weights(training),
        len(training),
        replacement=True,
        generator=torch.Generator().manual_seed(45),
    )
    metadata = {
        "render_config": NETWORK_RENDER,
        "conditions": {"input_contract": CONTRACT, "parameters": PARAMETERS},
        "parameter_ranges": {},
        "data_kind": "cfd_and_assumed_reference",
        "provenance": {"input_contract": CONTRACT},
        "train_ids": [c["case_id"] for c in training],
        "train_geometry_hashes": sorted({c["geometry_hash"] for c in training}),
        "heldout_ids": [c["case_id"] for c in cases if c["split"] != "train"],
        "train_family_ids": sorted({c["family_id"] for c in training}),
        "dataset_sha256": hashlib.sha256(
            (prepared / "dataset.json").read_bytes()
        ).hexdigest(),
        "train_source_counts": dict(Counter(c["source"] for c in training)),
        "split_policy": data["split_policy"],
    }
    path = train_model(
        SceneCases(prepared, training, stats),
        SceneCases(prepared, validation, stats),
        parameter_names=NAMES,
        normalization=stats,
        metadata=metadata,
        output_dir=output,
        epochs=epochs,
        batch_size=32,
        seed=42,
        device_name=device,
        augment=True,
        model_size="small",
        encoder="pyramid_detail_attention",
        fusion="attention",
        train_sampler=sampler,
    )
    ck = torch.load(path, map_location="cpu", weights_only=True)
    ck.update(
        input_contract=CONTRACT,
        raw_parameter_names=PARAMETERS,
        source_render_config=SOURCE_RENDER,
        parameter_ranges=parameter_ranges(training),
    )
    torch.save(ck, path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    a = sub.add_parser("prepare")
    a.add_argument("source", type=Path)
    a.add_argument("--output-dir", type=Path, required=True)
    b = sub.add_parser("export")
    b.add_argument("source", type=Path)
    b.add_argument("--prepared", type=Path, required=True)
    b.add_argument("--output", type=Path, required=True)
    c = sub.add_parser("predict")
    c.add_argument("checkpoint", type=Path)
    c.add_argument("case", type=Path)
    c.add_argument("--output", type=Path, required=True)
    d = sub.add_parser("train")
    d.add_argument("prepared", type=Path)
    d.add_argument("--output-dir", type=Path, required=True)
    d.add_argument("--epochs", type=int, default=40)
    d.add_argument("--device", default="mps")
    args = p.parse_args()
    if args.action == "prepare":
        prepare(args.source, args.output_dir)
    elif args.action == "export":
        export_checkpoint(args.source, args.prepared, args.output)
    elif args.action == "predict":
        print(
            json.dumps(predict_case(args.checkpoint, args.case, args.output), indent=2)
        )
    else:
        train(args.prepared, args.output_dir, args.epochs, args.device)


if __name__ == "__main__":
    main()
