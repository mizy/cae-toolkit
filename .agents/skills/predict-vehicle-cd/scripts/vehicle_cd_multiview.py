"""Build and query a cached whole-car five-view Cd reference library."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from vehicle_cd_learning.data import load_dataset
from vehicle_cd_learning.render import render_geometry
from vehicle_cd_similarity_lib.multiview import (
    SCALES,
    VIEWS,
    describe_views,
    predict_descriptors,
    validate_descriptors,
)

METHOD = "whole-view-mask-depth-v1"


def operating_point(dataset):
    yaw = np.asarray([c["yaw_deg"] for c in dataset["cases"]], dtype=float)
    reynolds = np.asarray([c["reynolds"] for c in dataset["cases"]], dtype=float)
    if (
        not np.isfinite(yaw).all()
        or not np.isfinite(reynolds).all()
        or np.any(reynolds <= 0)
        or not np.allclose(yaw, yaw[0], atol=1e-8, rtol=0)
    ):
        raise ValueError(
            "One reference bank must have a single yaw and valid Reynolds values"
        )
    conditions = dataset["conditions"]
    speed = conditions.get("speed", {})
    air = conditions.get("air", {})
    if (
        "value_kmh" in speed
        and {"density_kg_m3", "dynamic_viscosity_pa_s"} <= air.keys()
    ):
        values = [
            speed["value_kmh"],
            air["density_kg_m3"],
            air["dynamic_viscosity_pa_s"],
        ]
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError("Fixed speed/air profile must be positive and finite")
        return {
            "kind": "fixed_speed",
            "speed_kmh": float(values[0]),
            "density_kg_m3": float(values[1]),
            "dynamic_viscosity_pa_s": float(values[2]),
            "yaw_deg": float(yaw[0]),
        }
    if not np.allclose(reynolds, reynolds[0], rtol=1e-6):
        raise ValueError(
            "Variable Reynolds requires a declared fixed speed/air profile"
        )
    return {
        "kind": "fixed_reynolds",
        "reynolds": float(reynolds[0]),
        "yaw_deg": float(yaw[0]),
    }


def build_library(prepared_dir: Path, output_dir: Path) -> Path:
    """@entry Precompute reference descriptors once, preserving all case provenance."""
    started = time.perf_counter()
    dataset = load_dataset(prepared_dir)
    if dataset["target_names"] != ["cd"]:
        raise ValueError("Similarity library requires Cd-only labels")
    profile = operating_point(dataset)
    cases = dataset["cases"]
    descriptors = []
    records = []
    if output_dir.exists():
        raise FileExistsError("Use a new directory for the reference bank")
    for case in cases:
        with np.load(prepared_dir / case["views"], allow_pickle=False) as arrays:
            descriptors.append(describe_views({name: arrays[name] for name in VIEWS}))
        records.append(
            {
                "id": case["case_id"],
                "family_id": case["family_id"],
                "geometry_hash": case["geometry_hash"],
                "cd": case["targets"][0],
                "cd_source": case.get("cd_source", ""),
            }
        )
    bank = {
        key: np.stack([item[key] for item in descriptors]) for key in descriptors[0]
    }
    validate_descriptors(bank)
    output_dir.mkdir(parents=True)
    features = output_dir / "features.npz"
    # Uncompressed arrays favor load latency; no repeated PNG decoding in matching.
    np.savez(features, record_ids=np.asarray([r["id"] for r in records]), **bank)
    metadata = {
        "schema_version": 1,
        "method": METHOD,
        "scales": list(SCALES),
        "view_order": list(VIEWS),
        "source_dataset_sha256": hashlib.sha256(
            (prepared_dir / "dataset.json").read_bytes()
        ).hexdigest(),
        "render_config": dataset["render_config"],
        "conditions": dataset["conditions"],
        "operating_point": profile,
        "data_kind": dataset["data_kind"],
        "provenance": dataset["provenance"],
        "records": records,
        "features": "features.npz",
        "features_sha256": hashlib.sha256(features.read_bytes()).hexdigest(),
        "build_seconds": time.perf_counter() - started,
        "descriptor_bytes": sum(v.nbytes for v in bank.values()),
    }
    path = output_dir / "library.json"
    path.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    return path


def load_library(path: Path):
    """Load the cached arrays with ID/order, method and integrity checks."""
    path = Path(path).resolve()
    metadata = json.loads(path.read_text())
    if (
        metadata["schema_version"] != 1
        or metadata["method"] != METHOD
        or metadata["scales"] != list(SCALES)
        or metadata["view_order"] != list(VIEWS)
    ):
        raise ValueError("Unsupported reference descriptor contract")
    features = (path.parent / metadata["features"]).resolve()
    if (
        not features.is_relative_to(path.parent)
        or hashlib.sha256(features.read_bytes()).hexdigest()
        != metadata["features_sha256"]
    ):
        raise ValueError("Reference feature cache path or checksum mismatch")
    with np.load(features, allow_pickle=False) as arrays:
        ids = arrays["record_ids"].tolist()
        if ids != [r["id"] for r in metadata["records"]] or len(set(ids)) != len(ids):
            raise ValueError("Reference record order or IDs differ from feature bank")
        bank = {key: arrays[key] for key in arrays.files if key != "record_ids"}
    if validate_descriptors(bank) != len(ids):
        raise ValueError("Reference feature count differs from records")
    labels = np.asarray([r["cd"] for r in metadata["records"]], dtype=float)
    if not np.isfinite(labels).all() or np.any(labels <= 0):
        raise ValueError("Reference Cd labels must be positive finite values")
    return metadata, bank


def query_conditions(metadata, case, length_m):
    if "conditions" in case and case["conditions"] != metadata["conditions"]:
        raise ValueError("Query conditions differ from the fixed reference profile")
    point = metadata["operating_point"]
    yaw = point["yaw_deg"]
    reynolds = (
        point["density_kg_m3"]
        * (point["speed_kmh"] / 3.6)
        * length_m
        / point["dynamic_viscosity_pa_s"]
        if point["kind"] == "fixed_speed"
        else point["reynolds"]
    )
    for name, expected in (("yaw_deg", yaw), ("reynolds", reynolds)):
        if name in case and (
            not np.isfinite(case[name])
            or not np.isclose(case[name], expected, rtol=1e-6, atol=1e-8)
        ):
            raise ValueError(f"Query {name} differs from fixed reference profile")
    return {"reynolds": float(reynolds), "yaw_deg": float(yaw)}


def predict_case(library_json: Path, case_path: Path, exclude_ids=()):
    """@entry Render complete views and query the loaded reference bank."""
    started = time.perf_counter()
    metadata, bank = load_library(library_json)
    load_seconds = time.perf_counter() - started
    unknown = set(exclude_ids) - {record["id"] for record in metadata["records"]}
    if unknown:
        raise ValueError(f"Unknown reference exclusions: {sorted(unknown)}")
    case = json.loads(case_path.read_text())
    geometry = case["geometry"]
    if "conditions" in case and case["conditions"] != metadata["conditions"]:
        raise ValueError("Query conditions differ from the fixed reference profile")
    tick = time.perf_counter()
    views, dimensions = render_geometry(
        (case_path.parent / geometry["path"]).resolve(),
        geometry,
        metadata["render_config"],
    )
    render_seconds = time.perf_counter() - tick
    point = query_conditions(metadata, case, dimensions["length_m"])
    tick = time.perf_counter()
    descriptor = describe_views(
        {
            name: np.stack((view.depth, view.mask)).astype(np.float32)
            for name, view in views.items()
        }
    )
    descriptor_seconds = time.perf_counter() - tick
    tick = time.perf_counter()
    result = predict_descriptors(
        descriptor, bank, metadata["records"], exclude_ids=set(exclude_ids)
    )
    scoring_seconds = time.perf_counter() - tick
    return {
        **result,
        "method": METHOD,
        "library": str(library_json.resolve()),
        "geometry": str((case_path.parent / geometry["path"]).resolve()),
        "conditions": metadata["conditions"],
        "operating_point": point,
        "data_kind": metadata["data_kind"],
        "excluded_ids": sorted(exclude_ids),
        "warnings": [
            "Geometry similarity is not calibrated uncertainty or validated aerodynamic change.",
            "Reference Cd values retain their source conditions and assumptions; a matching self reference can dominate the estimate.",
        ],
        "timing_seconds": {
            "library_load": load_seconds,
            "render": render_seconds,
            "describe": descriptor_seconds,
            "score": scoring_seconds,
            "total": time.perf_counter() - started,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build-library")
    build.add_argument("prepared", type=Path)
    build.add_argument("--output-dir", type=Path, required=True)
    predict = sub.add_parser("predict")
    predict.add_argument("case", type=Path)
    predict.add_argument("--library", type=Path, required=True)
    predict.add_argument("--exclude-id", action="append", default=[])
    predict.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "build-library":
        print(build_library(args.prepared, args.output_dir))
        return
    result = predict_case(args.library, args.case, args.exclude_id)
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
