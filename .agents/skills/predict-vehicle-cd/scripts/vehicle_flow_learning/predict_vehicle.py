"""Experimental raw-vehicle field inference on the trained proxy domain."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pyvista as pv
import torch
from vehicle_cd_learning.data import parameters
from vehicle_cd_learning.predict import predict, prediction_warnings
from vehicle_cd_learning.render import render_five_views
from vehicle_cd_similarity_lib.mesh_io import read_mesh

from vehicle_flow_learning.surface import sample_surface
from vehicle_flow_learning.vehicle import force_coefficients, load_vehicle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("case", type=Path)
    parser.add_argument(
        "queries", type=Path, help="N x 3 NPY coordinates in car lengths"
    )
    parser.add_argument("--output", type=Path, required=True, help="Output NPY path")
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--surface-count", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sampling", choices=("iid", "systematic"), default="iid")
    parser.add_argument("--allow-geometry-extrapolation", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    total = time.perf_counter()
    start = time.perf_counter()
    backbone, original, field, checkpoint = load_vehicle(args.checkpoint)
    load_seconds = time.perf_counter() - start
    start = time.perf_counter()
    case = json.loads(args.case.read_text())
    queries = np.load(args.queries, mmap_mode="r", allow_pickle=False)
    if (
        queries.ndim != 2
        or queries.shape[1] != 3
        or not np.isfinite(queries).all()
        or args.batch_size < 1
    ):
        raise ValueError("Queries must be finite N x 3 and batch size positive")
    profiles = original.get("condition_profiles", {})
    if case.get("condition_profile") != "drivaerml" or "drivaerml" not in profiles:
        raise ValueError("Only the explicit drivaerml profile is supported")
    if case["conditions"] != profiles["drivaerml"]["conditions"]:
        raise ValueError("Case conditions differ from checkpoint drivaerml profile")
    aref = case.get("reference_area_m2")
    if aref is not None and (not np.isfinite(aref) or aref <= 0):
        raise ValueError("reference_area_m2 must be finite and positive")
    io_seconds = time.perf_counter() - start
    start = time.perf_counter()
    geometry = case["geometry"]
    mesh = read_mesh((args.case.parent / geometry["path"]).resolve())
    with (args.case.parent / geometry["path"]).resolve().open("rb") as handle:
        geometry_sha = hashlib.file_digest(handle, "sha256").hexdigest()
    samples, diagnostics, surface = sample_surface(
        mesh,
        geometry,
        args.surface_count,
        args.seed,
        return_mesh=True,
        sampling=args.sampling,
    )
    if (
        not diagnostics["closed_by_edge_incidence"]
        or diagnostics["inconsistent_winding_edges"]
        or diagnostics["signed_volume_m3"] <= 0
        or diagnostics["constant_pressure_force_residual_fraction"] >= 1e-6
    ):
        raise ValueError(
            "Signed-distance masking requires a closed, consistently outward oriented surface"
        )
    geometry_seconds = time.perf_counter() - start
    start = time.perf_counter()
    views, dimensions = render_five_views(
        mesh,
        unit_scale_to_m=geometry["unit_scale_to_m"],
        front_axis=geometry["front_axis"],
        up_axis=geometry["up_axis"],
        ground_clearance_m=geometry.get("ground_clearance_m", 0.0),
        **original["render_config"],
    )
    arrays = {
        name: np.stack((view.depth, view.mask)).astype(np.float32)
        for name, view in views.items()
    }
    values = parameters(dimensions, case["reynolds"], case["yaw_deg"])
    values.extend(float(name == "drivaerml") for name in profiles)
    render_seconds = time.perf_counter() - start
    start = time.perf_counter()
    encoded = []
    hook = backbone.head.register_forward_pre_hook(
        lambda _module, inputs: encoded.append(inputs[0])
    )
    try:
        baseline = predict(
            backbone,
            original,
            arrays,
            values,
            allow_geometry_extrapolation=args.allow_geometry_extrapolation,
        )
    finally:
        hook.remove()
    with torch.no_grad():
        feature = encoded[0][0].clone()
        expected = field.condition[0].in_features
        if expected == len(feature) + len(original["target_names"]):
            feature = torch.cat((feature, backbone.head(feature)))
        elif expected != len(feature):
            raise ValueError(
                "Field feature size does not match encoder or encoder plus head"
            )
    inflow = torch.tensor([1.0, 0.0, 0.0])
    encode_seconds = time.perf_counter() - start
    start = time.perf_counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output = np.lib.format.open_memmap(
        args.output, mode="w+", dtype="float32", shape=(len(queries), 4)
    )
    io_seconds += time.perf_counter() - start
    start = time.perf_counter()
    valid_count = 0
    with torch.inference_mode():
        for offset in range(0, len(queries), args.batch_size):
            points = np.asarray(queries[offset : offset + args.batch_size])
            distance = pv.PolyData(
                points * dimensions["length_m"]
            ).compute_implicit_distance(surface)["implicit_distance"]
            valid = (
                (distance >= 0)
                & (points >= [-3, -2, 0]).all(1)
                & (points <= [6, 2, 3]).all(1)
            )
            values_out = np.full((len(points), 4), np.nan, dtype=np.float32)
            if valid.any():
                values_out[valid] = field(
                    torch.tensor(points[valid], dtype=torch.float32), feature, inflow
                ).numpy()
            valid_count += int(valid.sum())
            output[offset : offset + len(points)] = values_out
    force = None
    if aref is not None:
        viscosity = 1 / case["reynolds"] + checkpoint["identity"]["eddy_viscosity"]
        xyz = torch.tensor(
            samples["points_m"] / dimensions["length_m"], dtype=torch.float32
        )
        normals = torch.tensor(samples["normals"], dtype=torch.float32)
        weights = torch.tensor(samples["weights_m2"] / aref, dtype=torch.float32)
        # Use the same closed-surface stress control variate as training.
        force = (
            force_coefficients(field, xyz, normals, weights, feature, inflow, viscosity)
            .detach()
            .numpy()
            .tolist()
        )
    decode_seconds = time.perf_counter() - start
    start = time.perf_counter()
    output.flush()
    io_seconds += time.perf_counter() - start
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "case_sha256": hashlib.sha256(args.case.read_bytes()).hexdigest(),
        "geometry_sha256": geometry_sha,
        "output": str(args.output.resolve()),
        "points": len(queries),
        "valid_points": valid_count,
        "fields": ["u/U", "v/U", "w/U", "Cp"],
        "masked_values": "NaN for solid or outside proxy box",
        "condition_profile": "drivaerml",
        "conditions": case["conditions"],
        "field_physics": {
            "domain_l": checkpoint["domain_l"],
            "eddy_viscosity": checkpoint["identity"]["eddy_viscosity"],
            "matches_original_cfd": False,
        },
        "backbone_head_baseline": {
            name: float(baseline[i])
            for i, name in enumerate(original["target_names"])
            if name in profiles["drivaerml"]["labeled_targets"]
        },
        "field_total_stress_cd_cl": force,
        "reference_area_m2": aref,
        "surface": diagnostics,
        "warnings": prediction_warnings(original, values, "drivaerml")
        + [
            "Experimental unvalidated field; proxy box and constant effective viscosity differ from source CFD. Surface self-intersections and component orientation are not independently verified."
        ],
        "timing_seconds": {
            "load": load_seconds,
            "geometry": geometry_seconds,
            "render": render_seconds,
            "encode_and_baseline": encode_seconds,
            "decode_and_force": decode_seconds,
            "io": io_seconds,
            "end_to_end": time.perf_counter() - total,
        },
        "timing_scope": "Checkpoint load through flushed field output; excludes interpreter startup and final JSON metadata write",
    }
    metadata = args.output.with_suffix(".json")
    metadata.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
