"""Full-view coefficient inference with train-only pooled-feature normalization."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from aircraft_learning.prepare_scenes import resample_views

from . import dimensionless as dim
from .model import VIEWS, CoefficientModel
from .render import render_geometry

CONTRACT = "vehicle_dimensionless_readout_v1"
BUFFER_NAMES = ["body_mean", "body_std", "bottom_mean", "bottom_std"]


class ScaledReadout(CoefficientModel):
    """Keep the shared network; own feature statistics at both execution paths."""

    def __init__(self, **config):
        super().__init__(**config)
        for name in BUFFER_NAMES:
            value = (
                torch.ones(1, 16, 8, 8)
                if name.endswith("std")
                else torch.zeros(1, 16, 8, 8)
            )
            self.register_buffer(name, value)

    def encode_pooled(self, pooled, parameters, scene=None):
        normalized = {
            name: (
                pooled[name]
                - (self.bottom_mean if name == "bottom" else self.body_mean)
            )
            / (self.bottom_std if name == "bottom" else self.body_std)
            for name in VIEWS
        }
        return super().encode_pooled(normalized, parameters, scene)

    def encode(self, views, parameters, scene=None):
        body = self.image_encoder.pooled_features(
            torch.cat([views[n] for n in VIEWS[:4]])
        )
        pooled = dict(zip(VIEWS[:4], body.chunk(4)))
        pooled["bottom"] = self.bottom_encoder.pooled_features(views["bottom"])
        return self.encode_pooled(pooled, parameters, scene)


def initialize(source, statistics=None):
    """Change feature units while preserving the starting prediction function."""
    model = ScaledReadout(**source["model_config"])
    missing, unexpected = model.load_state_dict(source["model_state"], strict=False)
    if set(missing) != set(BUFFER_NAMES) or unexpected:
        raise ValueError("Unexpected source model state for readout continuation")
    if statistics is not None:
        with torch.no_grad():
            for prefix, encoder in (
                ("body", model.image_encoder),
                ("bottom", model.bottom_encoder),
            ):
                mean = torch.tensor(statistics[prefix + "_mean"], dtype=torch.float64)
                std = torch.tensor(statistics[prefix + "_std"], dtype=torch.float64)
                layer = encoder.project[1]
                old_weight = layer.weight.detach().double().clone()
                layer.bias.copy_(
                    (layer.bias.detach().double() + old_weight @ mean).float()
                )
                layer.weight.copy_((old_weight * std).float())
                getattr(model, prefix + "_mean").copy_(
                    mean.float().reshape(1, 16, 8, 8)
                )
                getattr(model, prefix + "_std").copy_(std.float().reshape(1, 16, 8, 8))
    for name, parameter in model.named_parameters():
        if (
            name.startswith(("image_encoder.", "bottom_encoder."))
            and ".project." not in name
        ):
            parameter.requires_grad_(False)
    return model


def load_model(path):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    if ck.get("input_contract") != CONTRACT or ck.get("parameter_names") != dim.NAMES:
        raise ValueError("Use a readout-normalized checkpoint and its loader")
    if ck.get("raw_parameter_names") != dim.PARAMETERS:
        raise ValueError("Unexpected dimensionless input fields")
    for name in ("parameter_mean", "parameter_std"):
        values = np.asarray(ck[name])
        if (
            values.shape != (len(dim.NAMES),)
            or not np.isfinite(values).all()
            or (name.endswith("std") and (values <= 0).any())
        ):
            raise ValueError("Invalid numeric input normalization")
    model = ScaledReadout(**ck["model_config"])
    model.load_state_dict(ck["model_state"], strict=True)
    for name in BUFFER_NAMES:
        values = getattr(model, name)
        if not torch.isfinite(values).all() or (
            name.endswith("std") and (values <= 0).any()
        ):
            raise ValueError("Invalid pooled-feature normalization")
    return model.eval(), ck


def write_prediction(model, ck, arrays, raw, profile, output):
    result = dim.predict_views(model, ck, arrays, raw)
    targets = ["cd"] if profile == "builtin_assumed" else ck["target_names"]
    warnings = []
    for name, value in zip(dim.PARAMETERS, raw):
        bounds = ck["parameter_ranges"].get(name)
        if value is not None and (
            bounds is None or value < bounds[0] - 1e-6 or value > bounds[1] + 1e-6
        ):
            warnings.append(f"{name} is outside observed training support")
    if raw[dim.PARAMETERS.index("log_re_body")] is None:
        warnings.append(
            "Re is unknown; this call does not establish a Reynolds response"
        )
    payload = {
        "coefficients": {
            k: float(result[i])
            for i, k in enumerate(ck["target_names"])
            if k in targets
        },
        "input_contract": CONTRACT,
        "numeric_inputs": dict(zip(dim.PARAMETERS, raw)),
        "warnings": warnings,
        "scope": "Experimental absolute-coefficient candidate; historical known-family validation only",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return payload


def predict_case(checkpoint, case_path, output):
    torch.set_num_threads(2)
    model, ck = load_model(checkpoint)
    case = json.loads(case_path.read_text())
    geometry = case["geometry"]
    views, dimensions = render_geometry(
        (case_path.parent / geometry["path"]).resolve(),
        geometry,
        ck["source_render_config"],
    )
    arrays = resample_views(
        {k: np.stack((v.depth, v.mask)).astype(np.float32) for k, v in views.items()}
    )
    raw = dim.parameters(dimensions, case["conditions"])
    return write_prediction(
        model, ck, arrays, raw, case.get("condition_profile"), output
    )


def predict_cached(checkpoint, prepared, identity, output):
    torch.set_num_threads(2)
    model, ck = load_model(checkpoint)
    data = dim.read_prepared(prepared)
    case = next(c for c in data["cases"] if c["case_id"] == identity)
    with np.load(prepared / case["views"], allow_pickle=False) as values:
        arrays = {k: values[k] for k in VIEWS}
    return write_prediction(
        model, ck, arrays, case["parameters"], case["source"], output
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("predict")
    p.add_argument("checkpoint", type=Path)
    p.add_argument("case", type=Path)
    p.add_argument("--output", type=Path, required=True)
    c = sub.add_parser("predict-cached")
    c.add_argument("checkpoint", type=Path)
    c.add_argument("prepared", type=Path)
    c.add_argument("case_id")
    c.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()
    result = (
        predict_case(a.checkpoint, a.case, a.output)
        if a.action == "predict"
        else predict_cached(a.checkpoint, a.prepared, a.case_id, a.output)
    )
    print(json.dumps(result, indent=2))
