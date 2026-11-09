"""Predict mean Cd/Cl using checkpoint-owned preprocessing and fixed conditions."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .data import (
    PARAMETER_NAMES,
    parameters,
    validate_parameter_names,
    validate_views,
)
from .model import CoefficientModel
from .render import render_geometry


def load_model(path: Path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    parameter_names = checkpoint["parameter_names"]
    condition_profiles = checkpoint.get("condition_profiles")
    validate_parameter_names(parameter_names, condition_profiles)
    if checkpoint["target_names"] not in (["cd"], ["cd", "cl"]):
        raise ValueError("Unsupported checkpoint target names")
    if condition_profiles is not None:
        global_ranges = checkpoint.get("parameter_ranges", {})
        for profile in condition_profiles.values():
            if (
                not isinstance(profile, dict)
                or not isinstance(profile.get("conditions"), dict)
                or not profile["conditions"]
                or not isinstance(profile.get("provenance"), dict)
                or not profile["provenance"]
                or profile.get("data_kind") not in ("cfd", "reference_assumed")
                or profile.get("labeled_targets") not in (["cd"], ["cd", "cl"])
                or not set(profile["labeled_targets"]).issubset(
                    checkpoint["target_names"]
                )
            ):
                raise ValueError("Invalid checkpoint condition profile metadata")
            profile_ranges = profile.get("parameter_ranges")
            if not isinstance(profile_ranges, dict) or set(profile_ranges) != set(
                PARAMETER_NAMES
            ):
                raise ValueError(
                    "Each checkpoint condition profile needs ranges for all base parameters"
                )
            for name, bounds in profile_ranges.items():
                value = np.asarray(bounds)
                if (
                    name not in global_ranges
                    or value.shape != (2,)
                    or not np.isfinite(value).all()
                    or value[0] > value[1]
                ):
                    raise ValueError("Invalid checkpoint condition profile range")
                global_bounds = np.asarray(global_ranges[name])
                if (
                    global_bounds.shape != (2,)
                    or not np.isfinite(global_bounds).all()
                    or global_bounds[0] > global_bounds[1]
                    or value[0] < global_bounds[0] - 1e-6
                    or value[1] > global_bounds[1] + 1e-6
                ):
                    raise ValueError(
                        f"Condition profile {name} range falls outside global range"
                    )
    for prefix, names in (
        ("parameter", parameter_names),
        ("target", checkpoint["target_names"]),
    ):
        for suffix in ("mean", "std"):
            value = np.asarray(checkpoint[f"{prefix}_{suffix}"])
            if value.shape != (len(names),) or not np.isfinite(value).all():
                raise ValueError(
                    f"Checkpoint {prefix} {suffix} must be a finite vector of length {len(names)}"
                )
            if suffix == "std" and np.any(value <= 0):
                raise ValueError(f"Checkpoint {prefix} scale must be positive")
    for name, bounds in checkpoint.get("parameter_ranges", {}).items():
        if (
            name not in parameter_names
            or np.asarray(bounds).shape != (2,)
            or not np.isfinite(bounds).all()
            or bounds[0] > bounds[1]
        ):
            raise ValueError("Invalid checkpoint parameter range")
    if any(
        not torch.isfinite(value).all() for value in checkpoint["model_state"].values()
    ):
        raise ValueError("Checkpoint model weights must be finite")
    config = checkpoint["model_config"]
    if config["parameter_count"] != len(parameter_names) or config.get(
        "target_count", 2
    ) != len(checkpoint["target_names"]):
        raise ValueError(
            "Checkpoint model dimensions differ from parameter/target names"
        )
    model = CoefficientModel(**config)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model.eval(), checkpoint


def _condition_profile_name(checkpoint, values):
    """Return the profile encoded by a valid checkpoint-order one-hot tail."""
    condition_profiles = checkpoint.get("condition_profiles")
    if not condition_profiles:
        return None
    profile_values = np.asarray(values, dtype=np.float64)[len(PARAMETER_NAMES) :]
    if profile_values.shape != (len(condition_profiles),):
        raise ValueError(
            "Prediction profile parameters do not match condition profiles"
        )
    selected = np.flatnonzero(profile_values == 1.0)
    if len(selected) != 1 or np.count_nonzero(profile_values) != 1:
        raise ValueError(
            "Prediction profile parameters must be an exact declared one-hot vector"
        )
    return list(condition_profiles)[int(selected[0])]


def _prediction_parameter_names(checkpoint):
    condition_profiles = checkpoint.get("condition_profiles")
    names = checkpoint.get("parameter_names")
    if condition_profiles is not None:
        if names is None:
            raise ValueError("Profile prediction requires checkpoint parameter_names")
        validate_parameter_names(names, condition_profiles)
    elif names is None:
        return PARAMETER_NAMES
    else:
        validate_parameter_names(names)
    return names


def predict(model, checkpoint, views, values, *, allow_geometry_extrapolation=False):
    values = np.asarray(values, dtype=np.float64)
    parameter_names = _prediction_parameter_names(checkpoint)
    condition_profiles = checkpoint.get("condition_profiles", {})
    if values.shape != (len(parameter_names),) or not np.isfinite(values).all():
        raise ValueError(
            "Prediction parameters must be a finite vector matching parameter_names"
        )
    profile_name = _condition_profile_name(checkpoint, values)
    profile_ranges = (
        condition_profiles[profile_name]["parameter_ranges"]
        if profile_name is not None
        else {}
    )
    for index, name in enumerate(parameter_names):
        bounds = profile_ranges.get(
            name, checkpoint.get("parameter_ranges", {}).get(name)
        )
        if bounds is None:
            continue
        lo, hi = bounds
        if not lo - 1e-6 <= values[index] <= hi + 1e-6:
            if name not in ("width_ratio", "height_ratio"):
                raise ValueError(
                    f"Unsupported condition: {name} outside trained range [{lo:g}, {hi:g}]"
                )
            if not allow_geometry_extrapolation:
                raise ValueError(
                    f"Unsupported geometry: {name} outside trained range [{lo:g}, {hi:g}]"
                )
    validate_views(views, model.variant, checkpoint.get("render_config"))
    x = (values - checkpoint["parameter_mean"]) / checkpoint["parameter_std"]
    fixed = np.asarray(checkpoint["parameter_std"]) <= 1e-8
    for index in np.flatnonzero(fixed):
        if allow_geometry_extrapolation and parameter_names[index] in (
            "width_ratio",
            "height_ratio",
        ):
            continue
        if abs(values[index] - checkpoint["parameter_mean"][index]) > 1e-6:
            raise ValueError(
                f"Unsupported fixed parameter: {parameter_names[index]} differs from training"
            )
    x[fixed] = 0.0
    required = (
        () if model.variant == "parameters" else ("front", "rear", "right", "top")
    )
    if model.variant == "five":
        required += ("bottom",)
    tensors = {
        name: torch.from_numpy(np.asarray(views[name], dtype=np.float32)[None])
        for name in required
    }
    with torch.inference_mode():
        y = model(tensors, torch.tensor(x[None], dtype=torch.float32)).numpy()[0]
    result = y * checkpoint["target_std"] + checkpoint["target_mean"]
    if (
        result.shape != (len(checkpoint["target_names"]),)
        or not np.isfinite(result).all()
    ):
        raise ValueError("Model produced invalid or nonfinite coefficients")
    if (
        "cd" in checkpoint["target_names"]
        and result[checkpoint["target_names"].index("cd")] < 0
    ):
        raise ValueError(
            "Model predicted negative drag; input is outside the supported passive-vehicle domain"
        )
    return result


def prediction_warnings(checkpoint, values, profile_name=None):
    """Report geometric extrapolation and source-label limitations without claiming confidence."""
    warnings = []
    condition_profiles = checkpoint.get("condition_profiles", {})
    parameter_names = _prediction_parameter_names(checkpoint)
    if profile_name is None and condition_profiles:
        profile_name = _condition_profile_name(checkpoint, values)
    profile_ranges = (
        condition_profiles[profile_name]["parameter_ranges"]
        if profile_name is not None
        else {}
    )
    for name, value in zip(parameter_names, values):
        bounds = profile_ranges.get(
            name, checkpoint.get("parameter_ranges", {}).get(name)
        )
        if bounds is not None:
            lo, hi = bounds
            if value < lo - 1e-6 or value > hi + 1e-6:
                warnings.append(
                    f"{name}={value:g} outside training range [{lo:g}, {hi:g}]"
                )
    if checkpoint.get("data_kind") == "synthetic":
        warnings.append(
            "Synthetic-label checkpoint: predictions have no validated aerodynamic meaning"
        )
    if checkpoint.get("data_kind") == "reference_assumed":
        warnings.append(
            "Mixed public Cd labels with assumed conditions: exploratory model, not validated CFD accuracy"
        )
    if checkpoint.get("data_kind") == "mixed_sources":
        warnings.append(
            "Mixed-source checkpoint: prediction support depends on the selected condition profile"
        )
    if profile_name is not None:
        profile = condition_profiles[profile_name]
        if profile["data_kind"] == "reference_assumed":
            warnings.append(
                f"Condition profile {profile_name} uses assumed conditions and public Cd labels"
            )
        if (
            "cl" not in profile["labeled_targets"]
            and "cl" in checkpoint["target_names"]
        ):
            warnings.append(
                f"Cl is unsupported for condition profile {profile_name}; its source has no Cl label"
            )
    if checkpoint.get("split_policy") == "geometry_disjoint":
        families = ", ".join(checkpoint.get("train_family_ids", []))
        warnings.append(
            "Geometry-disjoint split does not establish accuracy on unseen vehicle families; "
            f"training families: {families or 'see dataset provenance'}"
        )
    return warnings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "case", type=Path, help="JSON: geometry, reynolds, yaw_deg, conditions"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--allow-geometry-extrapolation",
        action="store_true",
        help="Diagnostic only: allow geometry ratios outside trained ranges; conditions remain strict",
    )
    args = parser.parse_args()
    torch.set_num_threads(2)
    total_start = time.perf_counter()
    model, checkpoint = load_model(args.checkpoint)
    model_load_seconds = time.perf_counter() - total_start
    case = json.loads(args.case.read_text())
    condition_profiles = checkpoint.get("condition_profiles")
    profile_name = None
    conditions = checkpoint.get("conditions")
    if condition_profiles:
        if "condition_profile" not in case:
            raise ValueError(
                "Mixed-source inference requires an explicit condition_profile"
            )
        profile_name = case["condition_profile"]
        if profile_name not in condition_profiles:
            raise ValueError(f"Unknown condition profile: {profile_name}")
        if case["conditions"] != condition_profiles[profile_name]["conditions"]:
            raise ValueError(
                f"Case conditions differ from condition profile {profile_name}"
            )
        conditions = condition_profiles[profile_name]["conditions"]
    elif case["conditions"] != checkpoint["conditions"]:
        raise ValueError("Case conditions differ from checkpoint fixed conditions")
    start = time.perf_counter()
    geometry = case["geometry"]
    views, dimensions = render_geometry(
        (args.case.parent / geometry["path"]).resolve(),
        geometry,
        checkpoint["render_config"],
    )
    arrays = {
        name: np.stack((value.depth, value.mask)).astype(np.float32)
        for name, value in views.items()
    }
    values = parameters(dimensions, case["reynolds"], case["yaw_deg"])
    if condition_profiles:
        values.extend(float(name == profile_name) for name in condition_profiles)
    render_seconds = time.perf_counter() - start
    start = time.perf_counter()
    result = predict(
        model,
        checkpoint,
        arrays,
        values,
        allow_geometry_extrapolation=args.allow_geometry_extrapolation,
    )
    inference_seconds = time.perf_counter() - start
    warnings = prediction_warnings(checkpoint, values, profile_name)
    target_names = checkpoint["target_names"]
    if profile_name is not None:
        labeled_targets = condition_profiles[profile_name]["labeled_targets"]
        unsupported_targets = [
            name for name in target_names if name not in labeled_targets
        ]
        reported_targets = [name for name in target_names if name in labeled_targets]
    else:
        unsupported_targets = []
        reported_targets = target_names
    output = {
        "coefficients": {
            name: float(result[target_names.index(name)]) for name in reported_targets
        },
        "checkpoint": str(args.checkpoint.resolve()),
        "dataset_sha256": checkpoint["dataset_sha256"],
        "conditions": conditions,
        "warnings": warnings,
        "prediction_mode": "experimental_geometry_extrapolation"
        if args.allow_geometry_extrapolation
        else "default",
        "render_seconds": render_seconds,
        "model_load_seconds": model_load_seconds,
        "inference_seconds": inference_seconds,
        "total_seconds": time.perf_counter() - total_start,
    }
    if profile_name is not None:
        output["condition_profile"] = profile_name
        output["unsupported_targets"] = unsupported_targets
    text = json.dumps(output, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
