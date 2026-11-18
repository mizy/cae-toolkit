"""Trainable complete-view residual path before per-view compression."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from aircraft_learning.prepare_scenes import resample_views
from torch import nn

from . import dimensionless as dim
from .model import VIEWS, CoefficientModel
from .render import render_geometry

CONTRACT = "vehicle_pooled_residual_v1"


class PooledResidual(CoefficientModel):
    """Share pooled CNN features between the original head and a detail path."""

    def __init__(self, detail, **config):
        super().__init__(**config)
        self.detail = detail
        count = self.image_encoder.project[1].in_features * 5 + self.parameter_count
        self.register_buffer("detail_mean", torch.zeros(count))
        self.register_buffer("detail_std", torch.ones(count))
        self.correction = None
        if detail:
            self.correction = nn.Sequential(
                nn.Linear(count, 128),
                nn.LayerNorm(128),
                nn.GELU(),
                nn.Linear(128, 64),
                nn.GELU(),
                nn.Linear(64, 2),
            )
            nn.init.zeros_(self.correction[-1].weight)
            nn.init.zeros_(self.correction[-1].bias)

    def predict_pooled(self, pooled, parameters):
        prediction = self.head(self.encode_pooled(pooled, parameters))
        if self.correction is not None:
            features = torch.cat(
                [*[pooled[n].flatten(1) for n in VIEWS], parameters], dim=1
            )
            prediction = prediction + self.correction(
                (features - self.detail_mean) / self.detail_std
            )
        return prediction

    def forward(self, views, parameters, scene=None):
        body = self.image_encoder.pooled_features(
            torch.cat([views[n] for n in VIEWS[:4]])
        )
        pooled = dict(zip(VIEWS[:4], body.chunk(4)))
        pooled["bottom"] = self.bottom_encoder.pooled_features(views["bottom"])
        return self.predict_pooled(pooled, parameters)


def initialize(source, detail, mean, std):
    model = PooledResidual(detail, **source["model_config"])
    missing, extra = model.load_state_dict(source["model_state"], strict=False)
    if extra or any(
        k not in ("detail_mean", "detail_std") and not k.startswith("correction.")
        for k in missing
    ):
        raise ValueError("Unexpected source checkpoint")
    model.detail_mean.copy_(torch.tensor(mean, dtype=torch.float32))
    model.detail_std.copy_(torch.tensor(std, dtype=torch.float32))
    return model


def load_model(path):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    if ck.get("input_contract") != CONTRACT or ck["parameter_names"] != dim.NAMES:
        raise ValueError("Use the pooled-residual loader")
    model = PooledResidual(ck["detail_path"], **ck["model_config"])
    model.load_state_dict(ck["model_state"], strict=True)
    if (
        not all(torch.isfinite(v).all() for v in model.state_dict().values())
        or (model.detail_std <= 0).any()
    ):
        raise ValueError("Invalid pooled-residual state")
    return model.eval(), ck


def predict_case(checkpoint, case_path, output):
    """@entry Predict absolute coefficients from the original complete geometry."""
    torch.set_num_threads(2)
    paths = (
        [Path(p) for p in json.loads(checkpoint.read_text())["checkpoints"]]
        if checkpoint.suffix == ".json"
        else [checkpoint]
    )
    models = [load_model(path) for path in paths]
    model, ck = models[0]
    case = json.loads(case_path.read_text())
    g = case["geometry"]
    views, dimensions = render_geometry(
        (case_path.parent / g["path"]).resolve(), g, ck["source_render_config"]
    )
    arrays = resample_views(
        {k: np.stack((v.depth, v.mask)).astype(np.float32) for k, v in views.items()}
    )
    raw = dim.parameters(dimensions, case["conditions"])
    values = np.mean(
        [dim.predict_views(m, state, arrays, raw) for m, state in models], axis=0
    )
    names = (
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
            "Re unknown; this prediction does not validate Reynolds response"
        )
    result = {
        "coefficients": {
            k: float(v) for k, v in zip(("cd", "cl"), values) if k in names
        },
        "input_contract": CONTRACT,
        "detail_path": model.detail,
        "ensemble_size": len(models),
        "warnings": warnings,
        "scope": "Vehicle continuation on known-family development splits",
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
