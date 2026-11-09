"""Audit available reference-car geometry and source Cd labels for annotation."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

DEFAULT_LOCAL_CAR_MODELS = Path.home() / "Documents/models/cars"


def _explicit_local_mesh(item: dict[str, Any], local_model_dir: Path) -> Path | None:
    """Return an explicitly recorded local mesh source under the allowed tree."""
    value = item.get("mesh_source")
    if not isinstance(value, str) or not value:
        return None
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        return None
    try:
        candidate.resolve().relative_to(local_model_dir.resolve())
    except ValueError:
        return None
    return candidate.resolve()


def _resolve_mesh(
    item: dict[str, Any],
    manifest_dir: Path,
    local_model_dir: Path,
) -> tuple[str | None, bool, str | None]:
    raw = item.get("mesh")
    primary = None
    if isinstance(raw, str) and raw:
        path = Path(raw).expanduser()
        primary = (path if path.is_absolute() else manifest_dir / path).resolve()
        if primary.is_file():
            return str(primary), True, "mesh"

    alternative = _explicit_local_mesh(item, local_model_dir)
    if alternative is not None and alternative.is_file():
        return str(alternative), True, "mesh_source"
    missing_path = str(primary) if primary is not None else None
    return missing_path, False, None


def _source_cd(item: dict[str, Any]) -> float | None:
    """Read only documented source labels; never promote model output to truth."""
    # An explicit source label wins. Generic `cd` is accepted only in a seed
    # record carrying provenance and confidence, and only without prediction markers.
    source_cd = item.get("source_cd")
    if source_cd is None:
        prediction_markers = {
            "predicted_cd",
            "prediction",
            "model_output",
            "model_name",
        }
        if (
            prediction_markers.intersection(item)
            or not item.get("cd_source")
            or not item.get("cd_confidence")
        ):
            return None
        source_cd = item.get("cd")
    if isinstance(source_cd, bool) or not isinstance(source_cd, (int, float)):
        return None
    value = float(source_cd)
    return value if math.isfinite(value) else None


def _load_seed(manifest_path: Path) -> dict[str, Any]:
    manifest_path = manifest_path.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not isinstance(manifest.get("cars"), list):
        raise TypeError("Seed manifest must contain a cars array")
    return manifest


def _has_canonical_normalized_geometry(
    item: dict[str, Any], manifest_dir: Path
) -> tuple[Path, bool]:
    """Use only the explicit primary normalized VTP; never substitute raw meshes."""
    raw_mesh, note = item.get("mesh"), item.get("mesh_note", "")
    if not isinstance(raw_mesh, str) or not isinstance(note, str):
        return Path(), False
    path = Path(raw_mesh).expanduser()
    path = (path if path.is_absolute() else manifest_dir / path).resolve()
    canonical_note = "normalized" in note.lower() and "canonical x,y,z" in note.lower()
    nose_known = "front -x" in note.lower() or "front_as_normalized" in note.lower()
    dimensions = item.get("target_dimensions_m")
    valid_dimensions = (
        isinstance(dimensions, list)
        and len(dimensions) == 3
        and all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and float(value) > 0
            for value in dimensions
        )
    )
    eligible = (
        path.is_file()
        and path.suffix.lower() == ".vtp"
        and path.name.lower().endswith("-normalized.vtp")
        and canonical_note
        and nose_known
        and valid_dimensions
    )
    return path, eligible


def training_manifest(
    manifest_path: Path,
    output_dir: Path,
    *,
    split_seed: int = 42,
) -> dict[str, Any]:
    """Build a Cd-only, explicitly assumption-labeled manifest from known normalized meshes."""
    import numpy as np

    manifest_path = manifest_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    manifest = _load_seed(manifest_path)
    eligible = []
    for item in manifest["cars"]:
        if not isinstance(item, dict):
            continue
        mesh, geometry_ok = _has_canonical_normalized_geometry(
            item, manifest_path.parent
        )
        source_cd = _source_cd(item)
        if geometry_ok and source_cd is not None:
            eligible.append((item, mesh, source_cd))
    if len(eligible) < 3:
        raise ValueError(
            "At least three labeled canonical normalized meshes are required for family splits"
        )

    ids = [str(item.get("id", "")) for item, _, _ in eligible]
    if any(not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("Eligible reference cars need unique nonempty ids")
    rng = np.random.default_rng(split_seed)
    order = np.array(sorted(range(len(eligible)), key=lambda i: ids[i]))
    rng.shuffle(order)
    validation_count = max(1, round(len(eligible) * 0.15))
    test_count = max(1, round(len(eligible) * 0.15))
    train_count = len(eligible) - validation_count - test_count
    if train_count < 1:
        raise ValueError("Need at least one geometry family in each split")
    split_by_id = {}
    for index in order[:train_count]:
        split_by_id[ids[int(index)]] = "train"
    for index in order[train_count : train_count + validation_count]:
        split_by_id[ids[int(index)]] = "val"
    for index in order[train_count + validation_count :]:
        split_by_id[ids[int(index)]] = "test"

    cases = []
    for item, mesh, source_cd in eligible:
        geometry_id = str(item["id"])
        cases.append(
            {
                "case_id": geometry_id,
                "geometry_id": geometry_id,
                "family_id": geometry_id,
                "split": split_by_id[geometry_id],
                "geometry": {
                    "path": os.path.relpath(mesh, output_dir),
                    "unit_scale_to_m": 1.0,
                    "front_axis": "-x",
                    "up_axis": "+z",
                },
                "reynolds": 1.204
                * (120.0 / 3.6)
                * float(item["target_dimensions_m"][0])
                / 1.81e-5,
                "yaw_deg": 0.0,
                "cd": source_cd,
                "cd_source": item["cd_source"],
                "cd_confidence": item["cd_confidence"],
                "assumptions": {
                    "conditions_assumed": True,
                    "source_conditions_verified": False,
                },
            }
        )
    return {
        "schema_version": 1,
        "data_kind": "reference_assumed",
        "target_names": ["cd"],
        "conditions": {
            "speed": {"value_kmh": 120.0, "assumed": True},
            "air": {
                "density_kg_m3": 1.204,
                "dynamic_viscosity_pa_s": 1.81e-5,
                "assumed": True,
            },
            "ground": {"value": "moving_wall_at_freestream_speed", "assumed": True},
            "wheels": {"value": "rotating", "assumed": True},
            "turbulence": {"value": "SST", "assumed": True},
            "solver": {"value": "mixed_public_reference", "assumed": True},
            "averaging": {"value": "reported_mean", "assumed": True},
            "reference_area": {"value": "frontal_area", "assumed": True},
            "force_axes": {
                "drag_positive": "downstream along freestream",
                "lift_positive": "+Z",
                "assumed": True,
            },
        },
        "provenance": {
            "description": "Public and user-confirmed source Cd references with unverified, mixed test conditions.",
            "source_manifest": str(manifest_path),
            "cd_labels_are_source_references": True,
            "conditions_are_assumptions_not_recovered_source_conditions": True,
            "reynolds_definition": "rho * U * vehicle_length / dynamic_viscosity; all inputs assumed except recorded vehicle length",
            "split_method": "deterministic geometry-family shuffle",
            "split_seed": split_seed,
        },
        "cases": cases,
    }


def inventory_manifest(
    manifest_path: Path,
    *,
    local_model_dir: Path = DEFAULT_LOCAL_CAR_MODELS,
) -> dict[str, Any]:
    manifest_path = manifest_path.expanduser().resolve()
    manifest = _load_seed(manifest_path)

    records = [(item, "cars") for item in manifest["cars"]]
    pending = manifest.get("pending_cars", [])
    if not isinstance(pending, list):
        raise TypeError("pending_cars must be an array when present")
    records.extend((item, "pending_cars") for item in pending)

    rows = []
    for item, group in records:
        if not isinstance(item, dict):
            continue
        mesh, exists, selected_from = _resolve_mesh(
            item, manifest_path.parent, local_model_dir
        )
        source_cd = _source_cd(item)
        missing_fields = []
        if item.get("source_cl", item.get("cl")) is None:
            missing_fields.append("cl")
        if item.get("reynolds", item.get("re")) is None:
            missing_fields.append("reynolds")
        if item.get("conditions") is None:
            missing_fields.append("conditions")
        rows.append(
            {
                "id": str(item.get("id", "")),
                "display_name": item.get("display_name"),
                "category": item.get("category"),
                "cd": source_cd,
                "cd_kind": "source_reference"
                if source_cd is not None
                else "unverified_or_missing",
                "source": item.get("cd_source"),
                "cd_confidence": item.get("cd_confidence"),
                "mesh": mesh,
                "mesh_exists": exists,
                "mesh_selected_from": selected_from,
                "mesh_note": item.get("mesh_note"),
                "mesh_license": item.get("mesh_license"),
                "target_dimensions_m": item.get("target_dimensions_m"),
                "geometry_note": item.get("mesh_note"),
                "manifest_group": group,
                "missing_fields": missing_fields,
                "status": "needs_annotations",
            }
        )
    return {
        "schema_version": 1,
        "manifest": str(manifest_path),
        "notes": "cd is a source reference label from the seed manifest, never a generated model prediction.",
        "summary": {
            "records": len(rows),
            "meshes_found": sum(row["mesh_exists"] for row in rows),
            "source_cd_labels": sum(row["cd"] is not None for row in rows),
            "needs_annotations": sum(bool(row["missing_fields"]) for row in rows),
        },
        "cars": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seed_manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--local-model-dir", type=Path, default=DEFAULT_LOCAL_CAR_MODELS
    )
    parser.add_argument("--split-seed", type=int, default=42)
    args = parser.parse_args(argv)
    try:
        audit = inventory_manifest(
            args.seed_manifest, local_model_dir=args.local_model_dir
        )
        training = training_manifest(
            args.seed_manifest,
            args.output_dir,
            split_seed=args.split_seed,
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = args.output_dir / "reference_audit.json"
        output_path.write_text(
            json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        training_path = args.output_dir / "cd_training_manifest.json"
        training_path.write_text(
            json.dumps(training, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    except (OSError, json.JSONDecodeError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    split_counts = {
        name: sum(case["split"] == name for case in training["cases"])
        for name in ("train", "val", "test")
    }
    print(
        json.dumps(
            {
                "audit": str(output_path),
                "training_manifest": str(training_path),
                "eligible_cases": len(training["cases"]),
                "splits": split_counts,
                **audit["summary"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
