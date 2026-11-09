"""Join every built-in Cd vehicle with the frozen CFD training/validation pools."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path

from .data import PARAMETER_NAMES, load_dataset
from .prepare import prepare


def combine(cfd_root: Path, builtin_source: Path, library: Path, output: Path):
    """@entry Preserve CFD holdouts, train all built-ins, and leave unknown Cl null."""
    source = json.loads(builtin_source.read_text())
    references = json.loads(library.read_text())["cars"]
    expected = {car["id"]: float(car["cd"]) for car in references}
    if len(expected) != len(references) or {
        case["case_id"] for case in source["cases"]
    } != set(expected):
        raise ValueError("Builtin source must match every packaged labeled vehicle")
    if any(float(case["cd"]) != expected[case["case_id"]] for case in source["cases"]):
        raise ValueError("Builtin Cd differs from the authoritative reference library")
    output.mkdir(parents=True, exist_ok=False)
    render_config = {"ordinary_size": 512, "bottom_size": 512, "window_l": 1.3}
    builtin_root = output / "builtin-prepared"
    prepare(builtin_source, builtin_root, render_config)
    inputs = [
        (name, cfd_root / name / "prepared")
        for name in ("drivaerml", "ahmedml", "windsorml")
    ]
    inputs.append(("builtin_assumed", builtin_root))
    datasets = [(name, root, load_dataset(root)) for name, root in inputs]
    profiles = {}
    for name, root, dataset in datasets:
        if (
            dataset["parameter_names"] != PARAMETER_NAMES
            or dataset["render_config"] != render_config
        ):
            raise ValueError(
                "Sources must have the same five-view rendering and base parameter contract"
            )
        profiles[name] = {
            "conditions": copy.deepcopy(dataset["conditions"]),
            "data_kind": dataset["data_kind"],
            "labeled_targets": dataset["target_names"],
            "provenance": {
                "source_prepared": str(root.resolve()),
                "source_dataset_sha256": hashlib.sha256(
                    (root / "dataset.json").read_bytes()
                ).hexdigest(),
                "description": "Packaged Cd labels with user-authorized assumed conditions"
                if name == "builtin_assumed"
                else f"Frozen full {name} CFD source; original complete provenance in source_prepared",
            },
        }
    prepared = output / "prepared"
    (prepared / "views").mkdir(parents=True)
    cases, source_counts = [], {}
    for index, (name, root, dataset) in enumerate(datasets):
        source_counts[name] = {split: 0 for split in ("train", "val", "test")}
        for original in dataset["cases"]:
            case = copy.deepcopy(original)
            if name == "builtin_assumed":
                case["split"] = "train"
            source_counts[name][case["split"]] += 1
            filename = f"{name}__{case['geometry_id']}.npz"
            destination = prepared / "views" / filename
            if not destination.exists():
                os.link(root / case["views"], destination)
            case["views"] = f"views/{filename}"
            case["condition_profile"] = name
            case["parameters"] += [float(index == j) for j in range(len(profiles))]
            if len(case["targets"]) == 1:
                case["targets"].append(None)
            cases.append(case)
    merged = {
        "schema_version": 1,
        "parameter_names": PARAMETER_NAMES + [f"profile_{name}" for name in profiles],
        "target_names": ["cd", "cl"],
        "render_config": render_config,
        "conditions": {
            "selection": "Explicit condition_profile selects one declared source/operating profile"
        },
        "condition_profiles": profiles,
        "data_kind": "mixed_sources",
        "split_policy": "geometry_disjoint",
        "split_provenance": {
            "cfd": "Unchanged frozen source train/val/test IDs",
            "builtin": "Every packaged Cd case in train; training fit is not a heldout accuracy measurement",
        },
        "provenance": {
            "builtin_library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
            "builtin_source_sha256": hashlib.sha256(
                builtin_source.read_bytes()
            ).hexdigest(),
            "builtin_ids": sorted(expected),
            "source_counts": source_counts,
            "missing_cl": "null; masked out of loss, normalization and metrics; never filled with zero",
        },
        "cases": cases,
    }
    target = prepared / "dataset.json"
    target.write_text(json.dumps(merged, indent=2, allow_nan=False) + "\n")
    validated = load_dataset(target)
    training_builtin = {
        case["case_id"]
        for case in validated["cases"]
        if case["condition_profile"] == "builtin_assumed" and case["split"] == "train"
    }
    if training_builtin != set(expected):
        raise ValueError("A packaged builtin vehicle was lost from actual training")
    audit = {
        "builtin_count": len(expected),
        "builtin_training_ids": sorted(training_builtin),
        "total_cases": len(cases),
        "source_counts": source_counts,
        "split_counts": {
            split: sum(case["split"] == split for case in cases)
            for split in ("train", "val", "test")
        },
        "unique_geometry_hashes": len({case["geometry_hash"] for case in cases}),
        "prepared": str(prepared),
        "dataset_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
    }
    (output / "inclusion-audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cfd-root",
        type=Path,
        default=Path("outputs/predict-vehicle-cd/full-attention-20261030/run-v3"),
    )
    parser.add_argument(
        "--builtin-source",
        type=Path,
        default=Path(
            "outputs/predict-vehicle-cd/all-builtin-training/run26-upright/source.json"
        ),
    )
    parser.add_argument(
        "--library",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "assets/reference-library/reference_cars.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            combine(args.cfd_root, args.builtin_source, args.library, args.output_dir),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
