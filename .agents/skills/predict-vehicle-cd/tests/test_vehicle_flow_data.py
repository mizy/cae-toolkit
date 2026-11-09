import hashlib
import json
import sys
from pathlib import Path

import pytest

SKILL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL / "scripts"))

from vehicle_flow_learning import vehicle_data


@pytest.fixture
def frozen_sources(tmp_path, monkeypatch):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    profiles, paths, rows = {}, {}, []
    for name, split, targets in (
        ("cfd", "val", [0.3, -0.1]),
        ("builtin_assumed", "train", [0.2, None]),
    ):
        root = tmp_path / name
        root.mkdir()
        mesh = root / "geometry.stl"
        mesh.write_text(
            "mesh presence fixture; geometry is not parsed by binding audit"
        )
        case = {
            "case_id": name,
            "geometry_id": name,
            "family_id": name,
            "split": split,
            "reynolds": 100.0,
            "yaw_deg": 0.0,
        }
        source_case = {**case, "geometry": {"path": mesh.name}, "cd": targets[0]}
        if targets[1] is not None:
            source_case["cl"] = targets[1]
        provenance = {"description": name}
        path = root / "source.json"
        path.write_text(json.dumps({"cases": [source_case], "provenance": provenance}))
        paths[name] = path
        parent = root / "run" / name / "prepared"
        parent.mkdir(parents=True)
        parent_file = parent / "dataset.json"
        parent_file.write_text(json.dumps({"provenance": provenance}))
        source_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        (parent.parent.parent / "protocol.json").write_text(
            json.dumps({"sources": {name: {"sha256": source_sha}}})
        )
        profiles[name] = {
            "provenance": {
                "source_prepared": str(parent),
                "source_dataset_sha256": hashlib.sha256(
                    parent_file.read_bytes()
                ).hexdigest(),
            }
        }
        rows.append(
            {
                **case,
                "condition_profile": name,
                "targets": targets,
                "geometry_hash": f"frozen-{name}",
                "views": f"views/{name}.npz",
                "views_sha256": f"view-hash-{name}",
                "parameters": [1.0],
            }
        )
    dataset = {
        "condition_profiles": profiles,
        "cases": rows,
        "parameter_names": ["example"],
        "target_names": ["cd", "cl"],
        "render_config": {},
        "provenance": {
            "builtin_ids": ["builtin_assumed"],
            "builtin_source_sha256": hashlib.sha256(
                paths["builtin_assumed"].read_bytes()
            ).hexdigest(),
        },
    }
    (prepared / "dataset.json").write_text(json.dumps(dataset))
    # The existing loader's schema/view validation is tested in its own suite.
    monkeypatch.setattr(vehicle_data, "load_dataset", lambda _: dataset)
    return prepared, paths, dataset


def test_binding_retains_splits_masks_and_existing_hashes(frozen_sources, tmp_path):
    prepared, sources, dataset = frozen_sources
    result = vehicle_data.audit_vehicle_data(prepared, sources, tmp_path / "audit.json")
    assert result["counts"] == {"val": 1, "train": 1}
    assert result["builtin_training_ids"] == ["builtin_assumed"]
    assert result["field_labels"]["available"] is False
    for row, original in zip(result["cases"], dataset["cases"], strict=True):
        assert all(row[key] == original[key] for key in original)
        assert row["target_mask"] == [value is not None for value in row["targets"]]
        assert Path(row["geometry"]["path"]).is_file()


def test_binding_rejects_changed_frozen_source(frozen_sources, tmp_path):
    prepared, sources, _ = frozen_sources
    path = sources["cfd"]
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="Source manifest hash mismatch"):
        vehicle_data.audit_vehicle_data(prepared, sources, tmp_path / "audit.json")


def test_binding_rejects_coefficient_or_geometry_drift(frozen_sources, tmp_path):
    prepared, sources, dataset = frozen_sources
    case = dataset["cases"][0]
    case["targets"][0] += 0.1
    with pytest.raises(ValueError, match="Source label mismatch"):
        vehicle_data.audit_vehicle_data(prepared, sources, tmp_path / "audit.json")
    case["targets"][0] = 0.3
    (sources["cfd"].parent / "geometry.stl").unlink()
    with pytest.raises(FileNotFoundError, match="Source geometry missing"):
        vehicle_data.audit_vehicle_data(prepared, sources, tmp_path / "audit.json")


def test_binding_rejects_builtin_validation_placement(frozen_sources, tmp_path):
    prepared, sources, dataset = frozen_sources
    dataset["cases"][1]["split"] = "val"
    with pytest.raises(ValueError, match="Cd-only in training"):
        vehicle_data.audit_vehicle_data(prepared, sources, tmp_path / "audit.json")
