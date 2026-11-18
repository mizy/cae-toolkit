"""Shared absolute-coefficient kernel readout before or after view compression."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from aircraft_learning.prepare_scenes import resample_views

from . import dimensionless as dim
from .model import VIEWS, CoefficientModel
from .render import render_geometry

CONTRACT = "vehicle_kernel_readout_v1"


class KernelReadout(CoefficientModel):
    """Own one feature/kernel contract for cached training and full-view inference."""

    def __init__(self, stage, feature_count, reference_count=0, **config):
        super().__init__(**config)
        if stage not in ("pooled", "fused"):
            raise ValueError("Unknown readout stage")
        self.stage = stage
        for name, value in {
            "feature_mean": torch.zeros(feature_count),
            "feature_scale": torch.ones(feature_count),
            "kernel_reference": torch.zeros(reference_count, feature_count),
            "kernel_dual": torch.zeros(reference_count, 2),
            "kernel_bias": torch.zeros(2),
            "kernel_bandwidth": torch.ones(1),
            "base_weight": torch.ones(1),
        }.items():
            self.register_buffer(name, value.double())

    def features(self, pooled, parameters):
        fused = self.encode_pooled(pooled, parameters)
        geometry = (
            torch.cat([pooled[n].flatten(1) for n in VIEWS], dim=1)
            if self.stage == "pooled"
            else fused
        )
        return torch.cat((geometry, parameters), dim=1), self.head(fused)

    def predict_features(self, features, base):
        if len(self.kernel_reference) == 0:
            return base
        x = (features.double() - self.feature_mean) / self.feature_scale
        ref = self.kernel_reference
        distance = (
            (x * x).mean(1)[:, None]
            + (ref * ref).mean(1)[None]
            - 2 * (x @ ref.T) / x.shape[1]
        ).clamp_min(0)
        prediction = (
            torch.exp(-distance / (2 * self.kernel_bandwidth.square()))
            @ self.kernel_dual
            + self.kernel_bias
            + self.base_weight * base.double()
        )
        return prediction.to(base.dtype)

    def forward(self, views, parameters, scene=None):
        body = self.image_encoder.pooled_features(
            torch.cat([views[n] for n in VIEWS[:4]])
        )
        pooled = dict(zip(VIEWS[:4], body.chunk(4)))
        pooled["bottom"] = self.bottom_encoder.pooled_features(views["bottom"])
        return self.predict_features(*self.features(pooled, parameters))


def initialize(source, stage, feature_count):
    model = KernelReadout(stage, feature_count, **source["model_config"])
    missing, extra = model.load_state_dict(source["model_state"], strict=False)
    expected = {
        "feature_mean",
        "feature_scale",
        "kernel_reference",
        "kernel_dual",
        "kernel_bias",
        "kernel_bandwidth",
        "base_weight",
    }
    if extra or set(missing) != expected:
        raise ValueError("Unexpected baseline state")
    return model.eval().requires_grad_(False)


def load_model(path):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    if (
        ck.get("input_contract") != CONTRACT
        or ck["parameter_names"] != dim.NAMES
        or ck["raw_parameter_names"] != dim.PARAMETERS
    ):
        raise ValueError("Use the kernel-readout checkpoint loader")
    state = ck["model_state"]
    model = KernelReadout(
        ck["readout_stage"],
        len(state["feature_mean"]),
        len(state["kernel_reference"]),
        **ck["model_config"],
    )
    model.load_state_dict(state, strict=True)
    if not all(torch.isfinite(v).all() for v in state.values()):
        raise ValueError("Nonfinite kernel checkpoint")
    if (model.feature_scale <= 0).any() or (model.kernel_bandwidth <= 0).any():
        raise ValueError("Invalid kernel scale")
    return model.eval().requires_grad_(False), ck


def predict_case(checkpoint, case_path, output):
    """@entry Render the complete geometry, then predict absolute Cd/Cl once."""
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
    values = dim.predict_views(model, ck, arrays, raw)
    targets = (
        ["cd"] if case.get("condition_profile") == "builtin_assumed" else ["cd", "cl"]
    )
    warnings = []
    for name, value in zip(dim.PARAMETERS, raw):
        limits = ck["parameter_ranges"].get(name)
        if value is not None and (
            limits is None or value < limits[0] - 1e-6 or value > limits[1] + 1e-6
        ):
            warnings.append(f"{name} outside observed support")
    if case["conditions"].get("re_body") is None:
        warnings.append(
            "Re is unknown; this call does not establish a Reynolds response"
        )
    result = {
        "coefficients": {
            k: float(v) for k, v in zip(("cd", "cl"), values) if k in targets
        },
        "readout_stage": model.stage,
        "input_contract": CONTRACT,
        "warnings": warnings,
        "scope": "Known-family vehicle coefficient experiment; external-family accuracy unverified",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("case", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(predict_case(args.checkpoint, args.case, args.output), indent=2))
