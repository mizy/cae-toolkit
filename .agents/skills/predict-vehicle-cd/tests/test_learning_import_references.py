from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from vehicle_cd_learning import import_references


def test_inventory_resolves_manifest_relative_mesh_and_marks_annotations(
    tmp_path: Path,
) -> None:
    mesh = tmp_path / "meshes" / "car.stl"
    mesh.parent.mkdir()
    mesh.write_bytes(b"fixture")
    manifest = tmp_path / "seed.json"
    manifest.write_text(
        json.dumps(
            {
                "cars": [
                    {
                        "id": "known-car",
                        "mesh": "meshes/car.stl",
                        "cd": 0.28,
                        "cd_source": "wind-tunnel reference",
                        "cd_confidence": "authoritative",
                    },
                    {
                        "id": "prediction-only",
                        "mesh": "absent.stl",
                        "cd": 0.31,
                        "prediction": {"model": "test"},
                    },
                ],
                "pending_cars": [
                    {"id": "pending", "cd": 0.2, "cd_source": "published"}
                ],
            }
        ),
        encoding="utf-8",
    )

    audit = import_references.inventory_manifest(manifest)
    known, predicted, pending = audit["cars"]
    assert known["mesh"] == str(mesh.resolve())
    assert known["mesh_exists"] is True
    assert known["cd"] == pytest.approx(0.28)
    assert known["cd_kind"] == "source_reference"
    assert known["missing_fields"] == ["cl", "reynolds", "conditions"]
    assert known["status"] == "needs_annotations"
    assert predicted["cd"] is None
    assert predicted["cd_kind"] == "unverified_or_missing"
    assert predicted["mesh_exists"] is False
    assert (
        pending["cd"] is None
    )  # Missing confidence prevents treating it as a supervised label.
    assert audit["summary"] == {
        "records": 3,
        "meshes_found": 1,
        "source_cd_labels": 1,
        "needs_annotations": 3,
    }


def test_only_explicit_local_mesh_source_can_be_used_as_fallback(
    tmp_path: Path,
) -> None:
    allowed = tmp_path / "cars"
    allowed.mkdir()
    alternative = allowed / "body.glb"
    alternative.write_bytes(b"fixture")
    outside = tmp_path / "outside.glb"
    outside.write_bytes(b"fixture")
    manifest = tmp_path / "seed.json"
    manifest.write_text(
        json.dumps(
            {
                "cars": [
                    {
                        "id": "allowed",
                        "mesh": "missing.vtp",
                        "mesh_source": str(alternative),
                    },
                    {
                        "id": "outside",
                        "mesh": "missing2.vtp",
                        "mesh_source": str(outside),
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    rows = import_references.inventory_manifest(manifest, local_model_dir=allowed)[
        "cars"
    ]
    assert rows[0]["mesh"] == str(alternative.resolve())
    assert rows[0]["mesh_selected_from"] == "mesh_source"
    assert rows[0]["mesh_exists"] is True
    assert rows[1]["mesh_exists"] is False
    assert rows[1]["mesh"] == str((tmp_path / "missing2.vtp").resolve())


def test_cli_writes_audit_under_required_output_directory(tmp_path: Path) -> None:
    cases = []
    for index in range(3):
        mesh = tmp_path / f"car-{index}-normalized.vtp"
        mesh.write_bytes(b"fixture")
        cases.append(
            {
                "id": f"car-{index}",
                "mesh": mesh.name,
                "mesh_note": "Normalized from test source. Axis order canonical X,Y,Z; front -X.",
                "target_dimensions_m": [4.5, 1.8, 1.5],
                "cd": 0.25 + index * 0.01,
                "cd_source": "public reference",
                "cd_confidence": "authoritative",
            }
        )
    raw_mesh = tmp_path / "raw.glb"
    raw_mesh.write_bytes(b"fixture")
    cases.append(
        {
            "id": "raw-only",
            "mesh": raw_mesh.name,
            "target_dimensions_m": [4.5, 1.8, 1.5],
            "cd": 0.29,
            "cd_source": "public reference",
            "cd_confidence": "authoritative",
        }
    )
    manifest = tmp_path / "seed.json"
    manifest.write_text(json.dumps({"cars": cases}), encoding="utf-8")
    output = tmp_path / "outputs" / "audit"
    assert import_references.main([str(manifest), "--output-dir", str(output)]) == 0
    audit = json.loads((output / "reference_audit.json").read_text(encoding="utf-8"))
    training = json.loads(
        (output / "cd_training_manifest.json").read_text(encoding="utf-8")
    )
    assert audit["summary"]["records"] == 4
    assert training["data_kind"] == "reference_assumed"
    assert training["target_names"] == ["cd"]
    assert all(training["conditions"][key]["assumed"] for key in training["conditions"])
    assert all(
        abs(case["reynolds"] - 1.204 * (120 / 3.6) * 4.5 / 1.81e-5) < 1
        for case in training["cases"]
    )
    assert {case["yaw_deg"] for case in training["cases"]} == {0}
    assert {case["split"] for case in training["cases"]} == {"train", "val", "test"}
    assert all(
        case["cd"] == cases[index]["cd"] for index, case in enumerate(training["cases"])
    )
    assert "raw-only" not in {case["case_id"] for case in training["cases"]}
    first_splits = {case["case_id"]: case["split"] for case in training["cases"]}
    repeated = import_references.training_manifest(manifest, output)
    assert {
        case["case_id"]: case["split"] for case in repeated["cases"]
    } == first_splits
