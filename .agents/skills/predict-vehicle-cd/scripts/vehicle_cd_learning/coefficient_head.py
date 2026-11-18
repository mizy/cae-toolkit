"""Fixed full-view encoder with interchangeable shared Cd/Cl correction heads."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from aircraft_learning.prepare_scenes import resample_views
from torch import nn

from . import dimensionless as dim
from .model import CoefficientModel
from .render import render_geometry

CONTRACT = "vehicle_coefficient_head_v1"
KINDS = ("original", "linear", "mlp", "wide", "normalized", "rbf")


class CoefficientHead(CoefficientModel):
    """Own the correction and its normalization at cached/full-view call sites."""

    def __init__(self, head_kind, reference_count=0, **config):
        super().__init__(**config)
        self.head_kind = head_kind
        size = self.head[0].in_features
        self.register_buffer("feature_mean", torch.zeros(size))
        self.register_buffer("feature_std", torch.ones(size))
        dtype = torch.float64 if head_kind == "rbf" else torch.float32
        self.register_buffer(
            "kernel_reference", torch.zeros(reference_count, size, dtype=dtype)
        )
        self.register_buffer(
            "kernel_dual", torch.zeros(reference_count, 2, dtype=dtype)
        )
        self.register_buffer("kernel_bias", torch.zeros(2, dtype=dtype))
        self.register_buffer("kernel_scale", torch.ones(1, dtype=dtype))
        self.correction = None
        if head_kind == "linear":
            self.correction = nn.Sequential(nn.Linear(size, 2))
        elif head_kind == "mlp":
            self.correction = nn.Sequential(
                nn.Linear(size, 64), nn.GELU(), nn.Linear(64, 2)
            )
        elif head_kind == "wide":
            self.correction = nn.Sequential(
                nn.Linear(size, 128),
                nn.GELU(),
                nn.Linear(128, 64),
                nn.GELU(),
                nn.Linear(64, 2),
            )
        elif head_kind == "normalized":
            self.correction = nn.Sequential(
                nn.Linear(size, 64), nn.LayerNorm(64), nn.GELU(), nn.Linear(64, 2)
            )
        elif head_kind not in ("original", "rbf"):
            raise ValueError("Unknown coefficient head")
        if self.correction is not None:
            nn.init.zeros_(self.correction[-1].weight)
            nn.init.zeros_(self.correction[-1].bias)

    def predict_features(self, features):
        base = self.head(features)
        if self.head_kind == "original":
            return base
        if self.head_kind == "rbf":
            x = (
                features.double() - self.feature_mean.double()
            ) / self.feature_std.double()
            if len(self.kernel_reference) == 0:
                return base
            ref = self.kernel_reference
            distance = (
                (x * x).mean(1)[:, None]
                + (ref * ref).mean(1)[None]
                - 2 * (x @ ref.T) / x.shape[1]
            ).clamp_min(0)
            residual = (
                torch.exp(-distance / (2 * self.kernel_scale.square()))
                @ self.kernel_dual
                + self.kernel_bias
            )
            return base + residual.to(base.dtype)
        return base + self.correction((features - self.feature_mean) / self.feature_std)

    def forward(self, views, parameters, scene=None):
        return self.predict_features(self.encode(views, parameters, scene))


def initialize(source, kind, mean, std):
    model = CoefficientHead(head_kind=kind, **source["model_config"])
    missing, extra = model.load_state_dict(source["model_state"], strict=False)
    if extra or any(
        not (
            k.startswith(("correction.", "kernel_"))
            or k in ("feature_mean", "feature_std")
        )
        for k in missing
    ):
        raise ValueError("Unexpected source state")
    model.feature_mean.copy_(torch.tensor(mean, dtype=torch.float32))
    model.feature_std.copy_(torch.tensor(std, dtype=torch.float32))
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    target = model.head if kind == "original" else model.correction
    if target is not None:
        for parameter in target.parameters():
            parameter.requires_grad_(True)
    return model


def load_model(path):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    if (
        ck.get("input_contract") != CONTRACT
        or ck["parameter_names"] != dim.NAMES
        or ck["raw_parameter_names"] != dim.PARAMETERS
    ):
        raise ValueError("Use the coefficient-head checkpoint loader")
    state = ck["model_state"]
    model = CoefficientHead(
        head_kind=ck["head_kind"],
        reference_count=len(state["kernel_reference"]),
        **ck["model_config"],
    )
    model.load_state_dict(state, strict=True)
    if (
        not torch.isfinite(model.feature_mean).all()
        or not torch.isfinite(model.feature_std).all()
        or (model.feature_std <= 0).any()
    ):
        raise ValueError("Invalid head feature normalization")
    return model.eval(), ck


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
    raw = dim.parameters(dimensions, case["conditions"])
    result = dim.predict_views(model, ck, arrays, raw)
    names = (
        ["cd"]
        if case.get("condition_profile") == "builtin_assumed"
        else ck["target_names"]
    )
    warnings = []
    for name, value in zip(dim.PARAMETERS, raw):
        bounds = ck["parameter_ranges"].get(name)
        if value is not None and (
            bounds is None or value < bounds[0] - 1e-6 or value > bounds[1] + 1e-6
        ):
            warnings.append(f"{name} outside observed support")
    payload = {
        "coefficients": {
            k: float(result[i]) for i, k in enumerate(ck["target_names"]) if k in names
        },
        "head_kind": ck["head_kind"],
        "input_contract": CONTRACT,
        "warnings": warnings,
        "scope": "Experimental vehicle coefficient head; historical known-family validation only",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("case", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(predict_case(a.checkpoint, a.case, a.output), indent=2))
