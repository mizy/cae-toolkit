"""Validate example inference; mixed-source Cd references do not establish CFD accuracy."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import numpy as np
import pyvista as pv
import torch
from vehicle_cd_similarity_lib.mesh_io import read_mesh

from .data import VIEWS, load_dataset, parameters, validate_views
from .predict import load_model, predict, prediction_warnings
from .render import render_five_views, render_geometry, save_preview


def image_arrays(views):
    arrays = {
        name: np.stack((view.depth, view.mask)).astype(np.float32)
        for name, view in views.items()
    }
    validate_views(arrays, "five")
    return arrays


def check_geometry_invariance(mesh, config):
    """Return view and model deltas after equivalent unit/axis/translation transforms."""
    baseline_views, dimensions = render_five_views(
        mesh,
        unit_scale_to_m=1.0,
        front_axis="-x",
        up_axis="+z",
        **config,
    )
    baseline = image_arrays(baseline_views)
    variant = mesh.copy(deep=True)
    points = np.asarray(mesh.points)
    # Proper 90-degree coordinate rotation, translation, and metre-to-mm change.
    variant.points = np.column_stack(
        (-points[:, 1], points[:, 0], points[:, 2])
    ) * 1000.0 + np.asarray((25000.0, -13000.0, 5200.0))
    altered_views, altered_dimensions = render_five_views(
        variant,
        unit_scale_to_m=0.001,
        front_axis="-y",
        up_axis="+z",
        **config,
    )
    altered = image_arrays(altered_views)
    view_deltas = {}
    for name in VIEWS:
        if not np.array_equal(baseline[name][1], altered[name][1]):
            raise AssertionError(f"Equivalent coordinate transform changed {name} mask")
        view_deltas[name] = float(np.max(np.abs(baseline[name][0] - altered[name][0])))
        if view_deltas[name] > 1e-5:
            raise AssertionError(
                f"Equivalent coordinate transform changed {name} depth"
            )
    if not np.allclose(
        [dimensions[k] for k in ("length_m", "width_m", "height_m")],
        [altered_dimensions[k] for k in ("length_m", "width_m", "height_m")],
        rtol=1e-6,
        atol=1e-7,
    ):
        raise AssertionError(
            "Equivalent coordinate transform changed vehicle dimensions"
        )
    return baseline_views, baseline, altered, view_deltas


def stable_prediction(
    model, checkpoint, arrays, values, repeats, *, allow_geometry_extrapolation=False
):
    first = predict(
        model,
        checkpoint,
        arrays,
        values,
        allow_geometry_extrapolation=allow_geometry_extrapolation,
    )
    durations = []
    outputs = []
    for _ in range(repeats):
        start = time.perf_counter()
        outputs.append(
            predict(
                model,
                checkpoint,
                arrays,
                values,
                allow_geometry_extrapolation=allow_geometry_extrapolation,
            )
        )
        durations.append(time.perf_counter() - start)
    stacked = np.asarray(outputs)
    spread = np.ptp(stacked, axis=0)
    if not np.isfinite(stacked).all() or np.max(spread) > 1e-7:
        raise AssertionError(f"Repeated prediction is unstable: range={spread}")
    return first, durations, spread.tolist()


def stable_nonvehicle_prediction(
    model, checkpoint, arrays, values, repeats, *, allow_geometry_extrapolation=False
):
    """Record repeatable passive-drag rejection for synthetic nonvehicle shapes."""
    try:
        prediction, durations, spread = stable_prediction(
            model,
            checkpoint,
            arrays,
            values,
            repeats,
            allow_geometry_extrapolation=allow_geometry_extrapolation,
        )
        return prediction, durations, spread, None
    except ValueError as error:
        if "negative drag" not in str(error):
            raise
        durations, reasons = [], []
        for _ in range(repeats):
            start = time.perf_counter()
            try:
                predict(
                    model,
                    checkpoint,
                    arrays,
                    values,
                    allow_geometry_extrapolation=allow_geometry_extrapolation,
                )
            except ValueError as repeat_error:
                reasons.append(str(repeat_error))
            else:
                reasons.append(None)
            durations.append(time.perf_counter() - start)
        if len(set(reasons)) != 1 or reasons[0] != str(error):
            raise AssertionError(
                "Nonvehicle rejection was not stable across repeated calls"
            )
        return None, durations, [], str(error)


def predict_case(model, checkpoint, case, arrays, repeats):
    start = time.perf_counter()
    values = case["parameters"]
    rejection = None
    try:
        result, durations, spread = stable_prediction(
            model, checkpoint, arrays, values, repeats
        )
    except ValueError as error:
        message = str(error)
        if not (
            message.startswith("Unsupported geometry:")
            and any(name in message for name in ("width_ratio", "height_ratio"))
        ):
            raise
        rejection = message
        result, durations, spread = stable_prediction(
            model,
            checkpoint,
            arrays,
            values,
            repeats,
            allow_geometry_extrapolation=True,
        )
    ranges = checkpoint["parameter_ranges"]
    out_of_range = {
        name: {"value": float(value), "training_range": ranges[name]}
        for name, value in zip(checkpoint["parameter_names"], values)
        if value < ranges[name][0] - 1e-6 or value > ranges[name][1] + 1e-6
    }
    return {
        "case_id": case["case_id"],
        "family_id": case["family_id"],
        "split": case["split"],
        "status": "diagnostic-only: geometry outside training range"
        if rejection
        else "accepted",
        "default_mode_rejection_reason": rejection,
        "source_reference_cd": case["targets"][0]
        if case["targets"] and checkpoint["target_names"] == ["cd"]
        else None,
        "source_label_note": case.get("cd_source"),
        "prediction": dict(zip(checkpoint["target_names"], result.tolist())),
        "reference_delta": float(result[0] - case["targets"][0])
        if case["targets"] and checkpoint["target_names"] == ["cd"]
        else None,
        "source_conditions_verified": False,
        "prediction_warnings": prediction_warnings(checkpoint, values),
        "parameter_values": dict(zip(checkpoint["parameter_names"], values)),
        "out_of_training_range": out_of_range,
        "repeat_spread": spread,
        "inference_seconds": {
            "median": float(statistics.median(durations)),
            "p95": float(np.quantile(durations, 0.95)),
        },
        "_inference_samples_seconds": durations,
        "view_checks": {
            name: {
                "shape": list(array.shape),
                "depth_finite": bool(np.isfinite(array[0]).all()),
                "mask_pixels": int(array[1].sum()),
            }
            for name, array in arrays.items()
        },
        "evaluation_seconds": time.perf_counter() - start,
    }


def predict_prepared_examples(dataset_path, model, checkpoint, cases, repeats):
    """Run checksum-verified cached cases with repeat and mask checks."""
    results, durations = [], []
    for case in cases:
        with np.load(dataset_path / case["views"], allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in VIEWS}
        validate_views(arrays, model.variant, checkpoint["render_config"])
        result = predict_case(model, checkpoint, case, arrays, repeats)
        results.append(result)
        durations.extend(result.pop("_inference_samples_seconds"))
    return results, durations


def check_fresh_geometry(
    model, checkpoint, source_manifest, query, cached_result, output_dir
):
    """Compare raw-geometry render with cached images and equivalent coordinates."""
    source_manifest = Path(source_manifest).expanduser().resolve()
    source = json.loads(source_manifest.read_text())
    if source["conditions"] != checkpoint["conditions"]:
        raise ValueError("Source manifest conditions differ from checkpoint")
    source_case = next(
        item for item in source["cases"] if item["case_id"] == query["case_id"]
    )
    mesh_path = (source_manifest.parent / source_case["geometry"]["path"]).resolve()
    mesh = read_mesh(mesh_path)
    config = checkpoint["render_config"]
    start = time.perf_counter()
    views, arrays, transformed, depth_deltas = check_geometry_invariance(mesh, config)
    render_seconds = time.perf_counter() - start
    save_preview(output_dir / "coordinate-invariance-preview.png", views)

    dimensions = render_geometry(mesh_path, source_case["geometry"], config)[1]
    values = parameters(dimensions, source_case["reynolds"], source_case["yaw_deg"])
    if not np.allclose(values, query["parameters"], rtol=1e-6, atol=1e-7):
        raise AssertionError("Fresh geometry preprocessing differs from prepared data")
    fresh = predict(model, checkpoint, arrays, values)
    cached = np.asarray(cached_result["prediction"]["cd"])
    if not np.allclose(fresh[0], cached, rtol=0, atol=1e-7):
        raise AssertionError("Fresh render prediction differs from cached inference")
    transformed_prediction = predict(model, checkpoint, transformed, values)
    return source_case, {
        "geometry": query["case_id"],
        "transform": "meters to millimeters, x/y 90-degree rotation, translation, explicit front -y",
        "max_view_depth_deltas": depth_deltas,
        "max_view_depth_delta_tolerance": 1e-5,
        "prediction_absolute_delta": float(
            np.max(np.abs(fresh - transformed_prediction))
        ),
        "cached_vs_fresh_cd_absolute_delta": float(abs(fresh[0] - cached)),
        "render_seconds": render_seconds,
        "preview": "coordinate-invariance-preview.png",
    }


def predict_unseen_vehicles(model, checkpoint, source_case, paths, output_dir, repeats):
    """Render and run documented canonical meshes outside the training manifest."""
    results, durations = [], []
    config = checkpoint["render_config"]
    for mesh_path in paths:
        mesh_path = mesh_path.expanduser().resolve()
        mesh = read_mesh(mesh_path)
        start = time.perf_counter()
        views, dimensions = render_five_views(
            mesh, unit_scale_to_m=1.0, front_axis="-x", up_axis="+z", **config
        )
        render_seconds = time.perf_counter() - start
        arrays = image_arrays(views)
        save_preview(output_dir / f"{mesh_path.stem}-preview.png", views)
        values = parameters(dimensions, source_case["reynolds"], source_case["yaw_deg"])
        case = {
            "case_id": mesh_path.stem,
            "family_id": mesh_path.stem,
            "split": "unseen_vehicle",
            "parameters": values,
            "targets": [],
        }
        result = predict_case(model, checkpoint, case, arrays, repeats)
        durations.extend(result.pop("_inference_samples_seconds"))
        result.update(
            mesh_path=str(mesh_path),
            mesh_render_seconds=render_seconds,
            prediction_warnings=prediction_warnings(checkpoint, values),
            source_reference_cd=None,
            reference_delta=None,
            source_conditions_verified=False,
        )
        results.append(result)
    return results, durations


def predict_nonvehicle_shapes(model, checkpoint, source_case, output_dir, repeats):
    """Stress image parsing with visibly different shapes outside the learned domain."""
    config = checkpoint["render_config"]
    records = {}
    shapes = (
        (
            "sphere",
            pv.Sphere(radius=0.45, theta_resolution=24, phi_resolution=16),
            "sphere, synthetic, outside vehicle aspect-ratio range",
        ),
        (
            "rectangular_prism",
            pv.Cube(bounds=(-2.25, 2.25, -0.9, 0.9, 0.0, 1.5)),
            "rectangular prism 4.5 x 1.8 x 1.5 m; aspect ratios fit, semantics do not",
        ),
    )
    for name, mesh, description in shapes:
        start = time.perf_counter()
        views, dimensions = render_five_views(
            mesh, unit_scale_to_m=1.0, front_axis="-x", up_axis="+z", **config
        )
        render_seconds = time.perf_counter() - start
        arrays = image_arrays(views)
        save_preview(output_dir / f"nonvehicle-{name}-preview.png", views)
        values = parameters(dimensions, source_case["reynolds"], source_case["yaw_deg"])
        result, durations, spread, rejection = stable_nonvehicle_prediction(
            model,
            checkpoint,
            arrays,
            values,
            repeats,
            allow_geometry_extrapolation=(name == "sphere"),
        )
        records[name] = {
            "status": "rejected" if rejection else "predicted-numerical-test-only",
            "rejection_reason": rejection,
            "shape": description,
            "applicability": (
                "Numeric input stress only; predictions have no physical meaning. "
                "Rectangular prism is semantically OOD even when its ratios are in range."
            ),
            "prediction_warnings": prediction_warnings(checkpoint, values)
            + ["Synthetic nonvehicle is outside the vehicle training domain"],
            "prediction": dict(zip(checkpoint["target_names"], result.tolist()))
            if result is not None
            else None,
            "source_reference_cd": None,
            "source_conditions_verified": False,
            "render_seconds": render_seconds,
            "inference_median_seconds": float(statistics.median(durations)),
            "inference_p95_seconds": float(np.quantile(durations, 0.95)),
            "repeat_spread": spread,
            "view_checks": {
                view: {"shape": list(image.shape), "mask_pixels": int(image[1].sum())}
                for view, image in arrays.items()
            },
        }
    return records


def run(
    dataset_path,
    checkpoint_path,
    source_manifest,
    output_dir,
    extra_vehicles,
    repeats=50,
):
    """@entry Load the frozen model and write a real-mesh stability report."""
    started = time.perf_counter()
    torch.set_num_threads(2)
    load_started = time.perf_counter()
    model, checkpoint = load_model(checkpoint_path)
    load_seconds = time.perf_counter() - load_started
    dataset = load_dataset(dataset_path)
    digest = hashlib.sha256((dataset_path / "dataset.json").read_bytes()).hexdigest()
    if (
        digest != checkpoint["dataset_sha256"]
        or dataset["data_kind"] != checkpoint["data_kind"]
    ):
        raise ValueError("Dataset does not match the checkpoint training data")
    cases = dataset["cases"]
    if not cases or {case["split"] for case in cases} != {"train", "val", "test"}:
        raise ValueError("Need prepared train, val, and test example families")
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    vehicles, vehicle_times = predict_prepared_examples(
        dataset_path, model, checkpoint, cases, repeats
    )
    query = cases[0]
    source_case, invariance = check_fresh_geometry(
        model, checkpoint, source_manifest, query, vehicles[0], output_dir
    )
    extra, extra_times = predict_unseen_vehicles(
        model, checkpoint, source_case, extra_vehicles, output_dir, repeats
    )
    nonvehicles = predict_nonvehicle_shapes(
        model, checkpoint, source_case, output_dir, repeats
    )
    report = {
        "schema_version": 1,
        "checkpoint": str(checkpoint_path.resolve()),
        "dataset_sha256": checkpoint["dataset_sha256"],
        "scope": "Operational stability check. Reference Cd values have mixed unverified conditions; synthetic shapes are outside vehicle scope.",
        "fixed_conditions": checkpoint["conditions"],
        "checkpoint_load_seconds": load_seconds,
        "repeat_count": repeats,
        "vehicle_count": len(vehicles),
        "vehicles": vehicles,
        "extra_vehicles": extra,
        "default_mode_summary": {
            "scope": "Exploratory model behavior. Diagnostic predictions never change default rejection.",
            "real_vehicle_cases": len(vehicles) + len(extra),
            "default_mode_accepted": sum(
                not case["default_mode_rejection_reason"] for case in vehicles + extra
            ),
            "default_mode_rejected_out_of_geometry_range": sum(
                bool(case["default_mode_rejection_reason"]) for case in vehicles + extra
            ),
            "experimental_diagnostic_predictions": sum(
                case["status"].startswith("diagnostic-only")
                for case in vehicles + extra
            ),
        },
        "vehicle_prediction_warnings": prediction_warnings(
            checkpoint, query["parameters"]
        ),
        "coordinate_invariance": invariance,
        "nonvehicle_robustness": {
            "scope": "Synthetic shapes stress numeric input. Predictions do not have physical meaning.",
            **nonvehicles,
        },
        "latency_seconds": {
            "cached_vehicle_inference_median": float(np.median(vehicle_times)),
            "cached_vehicle_inference_p95": float(np.quantile(vehicle_times, 0.95)),
            "fresh_vehicle_render_seconds": [
                case["mesh_render_seconds"] for case in extra
            ],
            "fresh_vehicle_inference_median": float(np.median(extra_times))
            if extra_times
            else None,
            "fresh_vehicle_inference_p95": float(np.quantile(extra_times, 0.95))
            if extra_times
            else None,
            "nonvehicle_render_seconds": {
                name: record["render_seconds"] for name, record in nonvehicles.items()
            },
        },
        "validation_elapsed_seconds": time.perf_counter() - started,
    }
    report_path = output_dir / "example-validation.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "source_manifest",
        type=Path,
        help="Prepared source cases with geometry metadata",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--extra-vehicle", type=Path, action="append", default=[])
    parser.add_argument("--repeats", type=int, default=50)
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error("--repeats must be at least 2")
    path = run(
        args.dataset,
        args.checkpoint,
        args.source_manifest,
        args.output_dir,
        args.extra_vehicle,
        args.repeats,
    )
    report = json.loads(path.read_text())
    print(
        json.dumps(
            {
                "report": str(path),
                "default_mode_summary": report["default_mode_summary"],
                "latency_seconds": report["latency_seconds"],
                "validation_elapsed_seconds": report["validation_elapsed_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
