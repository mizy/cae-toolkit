"""Predict aircraft Cd/Cl from a complete mesh and declared flight conditions."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from vehicle_cd_learning.model import CoefficientModel
from vehicle_cd_similarity_lib.mesh_io import read_mesh

from .prepare import PARAMETERS, parameters, render


def predict(
    checkpoint_path,
    mesh_path,
    unit_scale_to_m,
    alpha_deg,
    velocity_m_s,
    altitude_m,
    front_axis="-x",
    up_axis="+z",
    allow_extrapolation=False,
):
    started = time.perf_counter()
    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if ck["parameter_names"] != PARAMETERS or ck["provenance"]["mode"] != "views":
        raise ValueError("This entry point needs the aircraft complete-view checkpoint")
    model = CoefficientModel(**ck["model_config"])
    model.load_state_dict(ck["model_state"])
    model.eval()
    if (
        not np.isfinite([alpha_deg, velocity_m_s, altitude_m]).all()
        or velocity_m_s <= 0
    ):
        raise ValueError("Finite flight parameters and positive speed required")
    mesh = read_mesh(mesh_path)
    views, dimensions = render(
        mesh,
        ck["render_config"]["size"],
        unit_scale_to_m=unit_scale_to_m,
        front_axis=front_axis,
        up_axis=up_axis,
    )
    raw = parameters(
        dimensions,
        {
            "alpha_deg": alpha_deg,
            "beta_deg": 0,
            "velocity_magnitude_ms": velocity_m_s,
            "altitude_m": altitude_m,
        },
    )
    outside = {
        name: float(v)
        for name, v in zip(PARAMETERS, raw)
        if not ck["parameter_ranges"][name][0] - 1e-6
        <= v
        <= ck["parameter_ranges"][name][1] + 1e-6
    }
    if outside and not allow_extrapolation:
        raise ValueError(f"Outside observed training input range: {outside}")
    x = torch.tensor(
        (np.array(raw) - ck["parameter_mean"]) / ck["parameter_std"],
        dtype=torch.float32,
    )[None]
    images = {
        n: torch.tensor(np.stack((v.depth, v.mask)), dtype=torch.float32)[None]
        for n, v in views.items()
    }
    with torch.inference_mode():
        pred = model(images, x)[0].numpy() * ck["target_std"] + ck["target_mean"]
    if not np.isfinite(pred).all():
        raise ValueError("Nonfinite coefficient prediction")
    return {
        "coefficients": dict(zip(ck["target_names"], pred.tolist())),
        "conditions": {
            "alpha_deg": alpha_deg,
            "beta_deg": 0,
            "velocity_m_s": velocity_m_s,
            "altitude_m": altitude_m,
        },
        "dimensions": dimensions,
        "extrapolated_parameters": outside,
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "geometry_sha256": hashlib.sha256(mesh_path.read_bytes()).hexdigest(),
        "total_seconds": time.perf_counter() - started,
        "scope": "COCOANet CCA-family low-speed CFD surrogate; arbitrary aircraft families, propulsion and new conditions not validated; Reynolds/Mach are not inferred from unverified atmosphere conventions",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("mesh", type=Path)
    p.add_argument("--unit-scale-to-m", type=float, required=True)
    p.add_argument("--front-axis", default="-x")
    p.add_argument("--up-axis", default="+z")
    p.add_argument("--alpha-deg", type=float, required=True)
    p.add_argument("--velocity-m-s", type=float, required=True)
    p.add_argument("--altitude-m", type=float, required=True)
    p.add_argument("--allow-extrapolation", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    result = predict(
        args.checkpoint,
        args.mesh,
        args.unit_scale_to_m,
        args.alpha_deg,
        args.velocity_m_s,
        args.altitude_m,
        args.front_axis,
        args.up_axis,
        args.allow_extrapolation,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
