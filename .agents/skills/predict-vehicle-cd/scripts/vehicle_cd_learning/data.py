"""Prepared case contract and train/inference feature construction."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

PARAMETER_NAMES = ["width_ratio", "height_ratio", "log_re", "yaw_sin", "yaw_cos"]
TARGET_NAMES = ["cd", "cl"]
VIEWS = ("front", "rear", "right", "top", "bottom")
SPLIT_POLICIES = ("family_disjoint", "geometry_disjoint")


def validate_parameter_names(names, condition_profiles=None):
    if condition_profiles is None:
        expected = PARAMETER_NAMES
    else:
        if not isinstance(condition_profiles, dict) or not condition_profiles:
            raise ValueError("condition_profiles must be a nonempty mapping")
        if any(not isinstance(name, str) or not name for name in condition_profiles):
            raise ValueError("Condition profile names must be nonempty strings")
        expected = PARAMETER_NAMES + [f"profile_{name}" for name in condition_profiles]
    if names != expected:
        raise ValueError("Unsupported dataset schema or parameter order")


def parameters(dimensions: dict, reynolds: float, yaw_deg: float) -> list[float]:
    values = [dimensions[k] for k in ("length_m", "width_m", "height_m")]
    if (
        not np.isfinite([*values, reynolds, yaw_deg]).all()
        or min(values) <= 0
        or reynolds <= 0
    ):
        raise ValueError(
            "Dimensions and Reynolds must be positive and all inputs finite"
        )
    if not -180 <= yaw_deg <= 180:
        raise ValueError("yaw_deg must be within [-180, 180]")
    length, width, height = values
    yaw = np.deg2rad(yaw_deg)
    return [
        width / length,
        height / length,
        float(np.log(reynolds)),
        float(np.sin(yaw)),
        float(np.cos(yaw)),
    ]


def validate_views(views, variant, render_config=None):
    """Check numerical depth/mask inputs at dataset and inference boundaries."""
    required = (
        () if variant == "parameters" else VIEWS[:4] if variant == "four" else VIEWS
    )
    for name in required:
        if name not in views:
            raise ValueError(f"Missing view: {name}")
        image = np.asarray(views[name])
        if image.ndim != 3 or image.shape[0] != 2 or not np.isfinite(image).all():
            raise ValueError(f"View {name} must be a finite (2,H,W) tensor")
        if render_config:
            size = render_config["bottom_size" if name == "bottom" else "ordinary_size"]
            if image.shape != (2, size, size):
                raise ValueError(
                    f"View {name} size differs from checkpoint render configuration"
                )
        if not np.isin(image[1], [0, 1]).all() or not image[1].any():
            raise ValueError(f"View {name} mask must be binary and nonempty")
        if np.any(image[0] < -1e-6) or np.any(image[0] > 1 + 1e-6):
            raise ValueError(f"View {name} depth must be within [0,1]")
        if np.any(image[0][image[1] == 0] != 0):
            raise ValueError(f"View {name} background depth must be zero")


def load_dataset(path: str | Path) -> dict:
    path = Path(path)
    if path.is_dir():
        path /= "dataset.json"
    dataset = json.loads(path.read_text())
    if dataset["schema_version"] != 1:
        raise ValueError("Unsupported dataset schema or parameter order")
    condition_profiles = dataset.get("condition_profiles")
    validate_parameter_names(dataset["parameter_names"], condition_profiles)
    parameter_names = dataset["parameter_names"]
    if condition_profiles is not None:
        for profile in condition_profiles.values():
            if (
                not isinstance(profile, dict)
                or not isinstance(profile.get("conditions"), dict)
                or not profile["conditions"]
                or not isinstance(profile.get("provenance"), dict)
                or not profile["provenance"]
                or profile.get("data_kind") not in ("cfd", "reference_assumed")
                or profile.get("labeled_targets") not in (["cd"], TARGET_NAMES)
            ):
                raise ValueError(
                    "Each profile needs conditions, provenance, data_kind and labeled_targets"
                )
    split_policy = dataset.get("split_policy", "family_disjoint")
    if split_policy not in SPLIT_POLICIES:
        raise ValueError(f"Unsupported split_policy: {split_policy}")
    if dataset["target_names"] not in (["cd"], TARGET_NAMES):
        raise ValueError("Supported targets: [cd] or [cd, cl]")
    if condition_profiles is not None and any(
        not set(profile["labeled_targets"]).issubset(dataset["target_names"])
        for profile in condition_profiles.values()
    ):
        raise ValueError("Profile labels exceed dataset target_names")
    cases = dataset["cases"]
    if not cases:
        raise ValueError("Dataset has no cases")
    ids, groups, checked_views = set(), {}, set()
    for case in cases:
        if case["case_id"] in ids:
            raise ValueError("Duplicate case_id")
        ids.add(case["case_id"])
        split = case["split"]
        if split not in ("train", "val", "test"):
            raise ValueError("split must be train, val or test")
        for key in ("geometry_id", "family_id", "geometry_hash"):
            if not case.get(key):
                raise ValueError(f"Missing {key}")
        split_keys = ("geometry_id", "geometry_hash")
        if split_policy == "family_disjoint":
            split_keys += ("family_id",)
        for key in split_keys:
            group = (key, case[key])
            if group in groups and groups[group] != split:
                raise ValueError(f"Split leakage: {key} {case[key]}")
            groups[group] = split
        parameters = np.asarray(case["parameters"], dtype=np.float64)
        if parameters.shape != (len(parameter_names),):
            raise ValueError("Incorrect parameter or target dimensions")
        if not np.isfinite(parameters).all():
            raise ValueError("Nonfinite parameters")
        if condition_profiles is not None:
            profile_name = case.get("condition_profile")
            if (
                not isinstance(profile_name, str)
                or profile_name not in condition_profiles
            ):
                raise ValueError("Missing or unsupported condition_profile")
            profile_values = parameters[len(PARAMETER_NAMES) :]
            expected_profile = list(condition_profiles).index(profile_name)
            if (
                not np.isin(profile_values, [0.0, 1.0]).all()
                or profile_values.sum() != 1.0
                or profile_values[expected_profile] != 1.0
            ):
                raise ValueError("Parameters do not match condition_profile one-hot")
        elif "condition_profile" in case:
            raise ValueError("condition_profile requires condition_profiles")
        targets = case["targets"]
        if len(targets) != len(dataset["target_names"]):
            raise ValueError("Incorrect parameter or target dimensions")
        if targets[0] is None:
            raise ValueError("Cd target is required")
        if condition_profiles is not None:
            profile_labels = condition_profiles[case["condition_profile"]][
                "labeled_targets"
            ]
            if (
                "cl" not in profile_labels
                and len(targets) == 2
                and targets[1] is not None
            ):
                raise ValueError(
                    "Case has a label not declared by its condition profile"
                )
        try:
            target_values = np.asarray(
                [np.nan if value is None else value for value in targets],
                dtype=np.float64,
            )
        except (TypeError, ValueError) as error:
            raise ValueError("Invalid target value") from error
        missing = np.asarray([value is None for value in targets])
        if (
            np.any(~np.isfinite(target_values) & ~missing)
            or (
                missing.any()
                and (
                    condition_profiles is None
                    or dataset["target_names"] != TARGET_NAMES
                )
            )
            or (missing[0])
            or (missing[1:].sum() != missing.sum())
        ):
            raise ValueError(
                "Targets must be finite, except missing Cl in profiled data"
            )
        view_path = (path.parent / case["views"]).resolve()
        if (
            not view_path.is_relative_to(path.parent.resolve())
            or not view_path.is_file()
        ):
            raise ValueError("Missing views or views outside dataset directory")
        if view_path not in checked_views:
            if (
                hashlib.sha256(view_path.read_bytes()).hexdigest()
                != case["views_sha256"]
            ):
                raise ValueError("Rendered views checksum mismatch")
            checked_views.add(view_path)
    return dataset


class CaseDataset:
    """Torch-compatible dataset; normalization belongs to training checkpoint."""

    def __init__(
        self,
        root,
        cases,
        parameter_mean,
        parameter_std,
        target_mean,
        target_std,
        variant="five",
    ):
        self.root, self.cases, self.variant = Path(root), cases, variant
        self.parameter_mean, self.parameter_std = (
            np.asarray(parameter_mean),
            np.asarray(parameter_std),
        )
        self.target_mean, self.target_std = (
            np.asarray(target_mean),
            np.asarray(target_std),
        )

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, index):
        import torch

        case = self.cases[index]
        views = {}
        if self.variant != "parameters":
            keys = VIEWS if self.variant == "five" else VIEWS[:4]
            with np.load(self.root / case["views"], allow_pickle=False) as arrays:
                arrays_dict = {name: arrays[name] for name in keys}
                validate_views(arrays_dict, self.variant)
                views = {
                    name: torch.from_numpy(image.astype(np.float32))
                    for name, image in arrays_dict.items()
                }
        x = (np.asarray(case["parameters"]) - self.parameter_mean) / self.parameter_std
        fixed = self.parameter_std <= 1e-8
        near_mean = np.abs(np.asarray(case["parameters"]) - self.parameter_mean) <= 1e-6
        x[fixed & near_mean] = 0.0
        targets = np.asarray(case["targets"], dtype=np.float64)
        y = (targets - self.target_mean) / self.target_std
        return (
            views,
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )


def jitter_views(views, generator):
    """Training-only representation noise; one 3D sampling offset shared by all views.

    Offset is at most one body-image pixel per canonical axis. Depth noise std
    is 1e-4 of the fixed depth window. Geometry, conditions and labels are unchanged.
    """
    import torch
    from torch.nn import functional as F

    from .render import VIEW_AXES

    if not views:
        return views
    first = views["front"]
    # Offsets are fractions of the common camera window (not physical ride height).
    offset = (
        torch.rand((first.shape[0], 3), device=first.device, generator=generator) * 2
        - 1
    ) / (first.shape[-1] - 8)
    result = {}
    for name, image in views.items():
        h_axis, v_axis, depth_axis, camera_sign = VIEW_AXES[name]
        batch, _, height, width = image.shape
        theta = torch.eye(2, 3, device=image.device).unsqueeze(0).repeat(batch, 1, 1)
        theta[:, 0, 2] = -2 * offset[:, h_axis] * (width - 8) / width
        theta[:, 1, 2] = 2 * offset[:, v_axis] * (height - 8) / height
        grid = F.affine_grid(theta, image.size(), align_corners=False)
        interpolated = F.grid_sample(image, grid, align_corners=False)
        mask = F.grid_sample(image[:, 1:2], grid, mode="nearest", align_corners=False)
        depth = interpolated[:, :1] / interpolated[:, 1:2].clamp_min(1e-6)
        depth -= camera_sign * offset[:, depth_axis, None, None, None]
        noise = (
            torch.randn(depth.shape, device=depth.device, generator=generator) * 1e-4
        )
        depth = (depth + noise).clamp(0, 1) * mask
        result[name] = torch.cat((depth, mask), dim=1)
    return result
