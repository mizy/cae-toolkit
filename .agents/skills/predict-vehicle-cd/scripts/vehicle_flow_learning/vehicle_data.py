"""Bind frozen vehicle views and coefficient labels to their original geometry.

This audit prepares the real vehicle input path for flow learning. It does not
create flow-field targets or claim that coefficient labels determine a field.
Pass the original source manifests explicitly; their hashes are checked against
the owning frozen run's provenance rather than inferred from case names.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

from vehicle_cd_learning.data import load_dataset


def audit_vehicle_data(prepared: Path, sources: dict[str, Path], output: Path):
    """@entry Preserve frozen splits/labels and bind each case to its source mesh."""
    prepared = prepared.resolve()
    manifest = prepared / "dataset.json"
    dataset = load_dataset(prepared)
    profiles = dataset["condition_profiles"]
    if set(sources) != set(profiles):
        raise ValueError("Provide exactly one source manifest per condition profile")
    bound, source_records = {}, {}
    for name, profile in profiles.items():
        source_path = sources[name].resolve()
        cases, source_hash = _source_cases(source_path, name, profile, dataset)
        bound[name] = (source_path, cases)
        source_records[name] = {"path": str(source_path), "sha256": source_hash}
    rows = [
        _bind_case(case, bound, dataset["target_names"]) for case in dataset["cases"]
    ]
    builtin_ids = sorted(
        row["case_id"] for row in rows if row["condition_profile"] == "builtin_assumed"
    )
    expected_ids = sorted(dataset["provenance"]["builtin_ids"])
    if builtin_ids != expected_ids or any(
        row["split"] != "train" or row["target_mask"] != [True, False]
        for row in rows
        if row["condition_profile"] == "builtin_assumed"
    ):
        raise ValueError("Every declared built-in must remain Cd-only in training")
    result = {
        "prepared_root": str(prepared),
        "dataset_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "parameter_names": dataset["parameter_names"],
        "target_names": dataset["target_names"],
        "render_config": dataset["render_config"],
        "condition_profiles": profiles,
        "source_manifests": source_records,
        "field_labels": {
            "available": False,
            "note": "Original input has views, geometry and Cd/Cl only; no velocity, pressure or shear labels are fabricated.",
        },
        "geometry_validation": "Paths exist; prepared geometry_hash is preserved, mesh contents and normals are not recomputed by this binding audit.",
        "counts": dict(Counter(row["split"] for row in rows)),
        "builtin_training_ids": builtin_ids,
        "cases": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return result


def _source_cases(path, name, profile, dataset):
    source_bytes = path.read_bytes()
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    source_prepared = Path(profile["provenance"]["source_prepared"])
    parent_bytes = (source_prepared / "dataset.json").read_bytes()
    if (
        hashlib.sha256(parent_bytes).hexdigest()
        != profile["provenance"]["source_dataset_sha256"]
    ):
        raise ValueError(f"Prepared source changed: {name}")
    if name == "builtin_assumed":
        expected_hash = dataset["provenance"]["builtin_source_sha256"]
    else:
        protocol = json.loads(
            (source_prepared.parent.parent / "protocol.json").read_text()
        )
        expected_hash = protocol["sources"][name]["sha256"]
    if source_hash != expected_hash:
        raise ValueError(f"Source manifest hash mismatch: {name}")
    source, parent = json.loads(source_bytes), json.loads(parent_bytes)
    if source["provenance"] != parent["provenance"]:
        raise ValueError(f"Source provenance differs from prepared data: {name}")
    cases = {case["case_id"]: case for case in source["cases"]}
    if len(cases) != len(source["cases"]):
        raise ValueError(f"Duplicate source case ID: {name}")
    return cases, source_hash


def _bind_case(case, bound, targets):
    source_path, source_cases = bound[case["condition_profile"]]
    source = source_cases[case["case_id"]]
    if any(
        source[key] != case[key]
        for key in ("geometry_id", "family_id", "reynolds", "yaw_deg")
    ):
        raise ValueError(f"Source identity/conditions mismatch: {case['case_id']}")
    if (
        case["condition_profile"] != "builtin_assumed"
        and source["split"] != case["split"]
    ):
        raise ValueError(f"Source split mismatch: {case['case_id']}")
    for name, target in zip(targets, case["targets"], strict=True):
        if target != source.get(name):
            raise ValueError(f"Source label mismatch: {case['case_id']} {name}")
    geometry = copy.deepcopy(source["geometry"])
    path = (source_path.parent / geometry["path"]).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Source geometry missing: {path}")
    geometry["path"] = str(path)
    return {
        **copy.deepcopy(case),
        "geometry": geometry,
        "target_mask": [value is not None for value in case["targets"]],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared", type=Path)
    parser.add_argument(
        "--source", action="append", required=True, metavar="PROFILE=MANIFEST"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sources = {}
    for value in args.source:
        name, path = value.split("=", 1)
        if name in sources:
            parser.error(f"Duplicate profile: {name}")
        sources[name] = Path(path)
    result = audit_vehicle_data(args.prepared, sources, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "counts": result["counts"],
                "builtin_train_count": len(result["builtin_training_ids"]),
            }
        )
    )


if __name__ == "__main__":
    main()
