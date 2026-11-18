"""One complete-geometry predictor combining a neural ensemble and its fitted detail residual."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from aircraft_learning.prepare_scenes import resample_views
from torch import nn

from . import dimensionless as dim
from .model import VIEWS
from .multiscale_readout import SCALES, MultiScaleReadout
from .pooled_residual import load_model as load_neural
from .render import render_geometry

CONTRACT = "vehicle_corrected_coefficients_v1"


class CorrectedCoefficients(nn.Module):
    """Supply the selected neural base to the shared multiscale residual, once per query."""

    def __init__(self, models, correction):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.correction = correction

    def forward(self, views, parameters, scene=None):
        base = torch.stack([m(views, parameters) for m in self.models]).mean(0)
        model = self.correction
        if len(model.kernel_reference) == 0:
            return base
        body = model.image_encoder.pooled_features(
            torch.cat([views[n] for n in VIEWS[:4]])
        )
        pooled = dict(zip(VIEWS[:4], body.chunk(4)))
        pooled["bottom"] = model.bottom_encoder.pooled_features(views["bottom"])
        features, _ = model.features(pooled, parameters)
        return model.predict_features(features, base)


def load_model(manifest):
    spec = json.loads(manifest.read_text())
    if spec.get("input_contract") != CONTRACT or tuple(spec["scales"]) != SCALES:
        raise ValueError("Use a corrected-coefficient manifest")
    for file, sha in spec["sha256"].items():
        if hashlib.sha256(Path(file).read_bytes()).hexdigest() != sha:
            raise ValueError(f"Corrected predictor component changed: {file}")
    loaded = [load_neural(Path(p)) for p in spec["base_checkpoints"]]
    _, feature = dim.load_model(Path(spec["feature_checkpoint"]))
    for _, ck in loaded:
        for key in (
            "parameter_names",
            "raw_parameter_names",
            "parameter_mean",
            "parameter_std",
            "target_mean",
            "target_std",
        ):
            if ck[key] != feature[key]:
                raise ValueError("Neural and correction normalization contracts differ")
    with np.load(spec["correction"], allow_pickle=False) as arrays:
        model = MultiScaleReadout(
            "pooled",
            len(arrays["mean"]),
            len(arrays["reference"]),
            **feature["model_config"],
        )
        missing, extra = model.load_state_dict(feature["model_state"], strict=False)
        if extra or set(missing) != {
            "feature_mean",
            "feature_scale",
            "kernel_reference",
            "kernel_dual",
            "kernel_bias",
            "kernel_bandwidth",
            "base_weight",
        }:
            raise ValueError("Unexpected feature encoder state")
        for key, name in (
            ("feature_mean", "mean"),
            ("feature_scale", "scale"),
            ("kernel_reference", "reference"),
            ("kernel_dual", "dual"),
            ("kernel_bias", "bias"),
        ):
            getattr(model, key).copy_(torch.from_numpy(arrays[name]))
    if (
        not all(torch.isfinite(v).all() for v in model.state_dict().values())
        or (model.feature_scale <= 0).any()
    ):
        raise ValueError("Invalid fitted residual")
    return CorrectedCoefficients([m for m, _ in loaded], model).eval().requires_grad_(
        False
    ), feature


def predict_case(manifest, case_path, output):
    """@entry Render once; predict absolute Cd/Cl without a user-supplied reference vehicle."""
    torch.set_num_threads(2)
    model, ck = load_model(manifest)
    case = json.loads(case_path.read_text())
    g = case["geometry"]
    views, dimensions = render_geometry(
        (case_path.parent / g["path"]).resolve(), g, ck["source_render_config"]
    )
    arrays = resample_views(
        {n: np.stack((v.depth, v.mask)).astype(np.float32) for n, v in views.items()}
    )
    raw = dim.parameters(dimensions, case["conditions"])
    values = dim.predict_views(model, ck, arrays, raw)
    names = (
        ["cd"] if case.get("condition_profile") == "builtin_assumed" else ["cd", "cl"]
    )
    warnings = []
    for name, value in zip(dim.PARAMETERS, raw):
        bounds = ck["parameter_ranges"].get(name)
        if value is not None and (
            bounds is None or value < bounds[0] - 1e-6 or value > bounds[1] + 1e-6
        ):
            warnings.append(f"{name} outside observed support")
    if case["conditions"].get("re_body") is None:
        warnings.append("Re unknown; no Reynolds-response validation")
    result = {
        "coefficients": {
            k: float(v) for k, v in zip(("cd", "cl"), values) if k in names
        },
        "input_contract": CONTRACT,
        "warnings": warnings,
        "scope": "Known-family vehicle development model; no independent-family guarantee",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("manifest", type=Path)
    p.add_argument("case", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    print(json.dumps(predict_case(a.manifest, a.case, a.output), indent=2))
