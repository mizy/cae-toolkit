from __future__ import annotations

import hashlib
import json
import struct
import sys
from pathlib import Path

import pytest

SKILL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL / "scripts"))

from vehicle_cd_learning.deduplicate_windsor import deduplicate


def _stl(header: bytes, vertex: float = 1.0) -> bytes:
    header = header.ljust(80, b"\0")[:80]
    facet = struct.pack(
        "<12fH",
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        vertex,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0,
    )
    return header + struct.pack("<I", 1) + facet


def _dataset(tmp_path: Path, members: list[dict]) -> tuple[Path, Path]:
    geometries = tmp_path / "geometries"
    geometries.mkdir()
    cases, assets, prepared_cases = [], [], []
    for member in members:
        case_id = member["case_id"]
        payload = _stl(case_id.encode(), member.get("vertex", 1.0))
        relative = f"geometries/{case_id}.stl"
        (tmp_path / relative).write_bytes(payload)
        geometry = {
            "path": relative,
            "unit_scale_to_m": member.get("unit_scale_to_m", 1.0),
            "front_axis": member.get("front_axis", "-x"),
            "up_axis": member.get("up_axis", "+y"),
            "ground_clearance_m": member.get("ground_clearance_m", 0.0),
        }
        case = {
            "case_id": case_id,
            "geometry_id": case_id,
            "family_id": "windsor-base",
            "split": member["split"],
            "geometry": geometry,
            "reynolds": member.get("reynolds", 2_900_000.0),
            "yaw_deg": member.get("yaw_deg", 2.5),
            "cd": member.get("cd", 0.3),
            "cl": member.get("cl", -0.1),
            "design_value": {"run": case_id, "clearance": member.get("clearance", 50)},
        }
        cases.append(case)
        assets.append(
            {
                "run": case_id,
                "path": relative,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
        prepared_case = {
            "case_id": case_id,
            "geometry_id": case_id,
            "split": member["split"],
            "geometry_hash": member["geometry_hash"],
            "reynolds": case["reynolds"],
            "yaw_deg": case["yaw_deg"],
        }
        prepared_cases.append(prepared_case)

    source_path = tmp_path / "full-source.json"
    prepared_path = tmp_path / "prepared" / "dataset.json"
    prepared_path.parent.mkdir()
    source_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "data_kind": "cfd",
                "target_names": ["cd", "cl"],
                "conditions": {"ground": "fixed"},
                "provenance": {
                    "repository": "neashton/windsorml",
                    "geometry_assets": assets,
                },
                "split_policy": "geometry_disjoint",
                "split_provenance": {
                    "counts": {"train": 2, "val": 1, "test": 2},
                    "excluded_runs": [{"run": 404, "reason": "mesh_not_published"}],
                },
                "cases": cases,
            }
        )
    )
    prepared_path.write_text(json.dumps({"cases": prepared_cases}))
    return source_path, prepared_path


def test_deduplicate_prefers_heldout_and_keeps_representative_labels(tmp_path):
    same_hash = "same-geometry"
    source_path, prepared_path = _dataset(
        tmp_path,
        [
            {
                "case_id": "alias-train",
                "split": "train",
                "geometry_hash": same_hash,
                "cd": 0.1,
            },
            {
                "case_id": "alias-val",
                "split": "val",
                "geometry_hash": same_hash,
                "cd": 0.2,
            },
            {
                "case_id": "alias-z-test",
                "split": "test",
                "geometry_hash": same_hash,
                "cd": 0.3,
            },
            {
                "case_id": "alias-a-test",
                "split": "test",
                "geometry_hash": same_hash,
                "cd": 0.4,
                "cl": -0.25,
            },
            {"case_id": "unique-train", "split": "train", "geometry_hash": "unique"},
        ],
    )
    source_before = source_path.read_bytes()
    prepared_before = prepared_path.read_bytes()
    output_path = tmp_path / "derived.json"

    result = deduplicate(source_path, prepared_path, output_path)

    derived = json.loads(output_path.read_text())
    assert result["case_count"] == 2
    assert result["counts"] == {"train": 1, "val": 0, "test": 1}
    assert {case["case_id"] for case in derived["cases"]} == {
        "alias-a-test",
        "unique-train",
    }
    representative = next(c for c in derived["cases"] if c["case_id"] == "alias-a-test")
    assert representative["cd"] == 0.4
    assert representative["cl"] == -0.25
    assert len(derived["provenance"]["geometry_assets"]) == 5
    assert {
        c["case_id"] for c in derived["split_provenance"]["excluded_duplicate_cases"]
    } == {
        "alias-train",
        "alias-val",
        "alias-z-test",
    }
    assert derived["split_provenance"]["excluded_runs"] == [
        {"run": 404, "reason": "mesh_not_published"}
    ]
    assert source_path.read_bytes() == source_before
    assert prepared_path.read_bytes() == prepared_before


def test_deduplicate_rejects_rounded_hash_collision(tmp_path):
    source_path, prepared_path = _dataset(
        tmp_path,
        [
            {"case_id": "collision-a", "split": "train", "geometry_hash": "same"},
            {
                "case_id": "collision-b",
                "split": "val",
                "geometry_hash": "same",
                "vertex": 2.0,
            },
        ],
    )

    with pytest.raises(ValueError, match="rounded collision"):
        deduplicate(source_path, prepared_path, tmp_path / "derived.json")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("reynolds", 2_900_001.0),
        ("yaw_deg", 3.0),
        ("unit_scale_to_m", 0.001),
        ("front_axis", "+x"),
        ("up_axis", "+z"),
        ("ground_clearance_m", 0.01),
    ],
)
def test_deduplicate_rejects_aliases_with_different_conditions(tmp_path, field, value):
    first = {"case_id": "same-a", "split": "train", "geometry_hash": "same"}
    second = {"case_id": "same-b", "split": "val", "geometry_hash": "same"}
    second[field] = value
    source_path, prepared_path = _dataset(tmp_path, [first, second])

    with pytest.raises(ValueError, match="differing flow or geometry conditions"):
        deduplicate(source_path, prepared_path, tmp_path / "derived.json")
