from __future__ import annotations

import sys
from pathlib import Path

import pytest

SKILL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL / "scripts"))

from validate_multiview import family_folds, training_reference_indices


def test_family_folds_match_seeded_three_way_protocol_without_splitting_families():
    cases = [
        {"case_id": f"case-{index}", "family_id": f"f{index}"} for index in range(9)
    ]

    folds = family_folds(cases)

    assert folds == [
        ("f3", "f0", "f7"),
        ("f2", "f4", "f6"),
        ("f1", "f5", "f8"),
    ]
    assert sorted(family for fold in folds for family in fold) == sorted(
        case["family_id"] for case in cases
    )
    assert family_folds(cases) == folds


def _case(case_id, family_id, geometry_hash):
    return {
        "case_id": case_id,
        "family_id": family_id,
        "geometry_hash": geometry_hash,
    }


def _record(case):
    return {
        "id": case["case_id"],
        "family_id": case["family_id"],
        "geometry_hash": case["geometry_hash"],
    }


def test_training_references_exclude_every_heldout_family_member():
    cases = [
        _case("heldout-a", "family-a", "geometry-a"),
        _case("heldout-b", "family-a", "geometry-b"),
        _case("train", "family-b", "geometry-c"),
    ]
    records = [_record(case) for case in cases]

    indices = training_reference_indices(records, cases, [cases[0], cases[1]])

    assert [records[index]["id"] for index in indices] == ["train"]
    assert not {records[index]["id"] for index in indices} & {
        cases[0]["case_id"],
        cases[1]["case_id"],
    }
    assert not {records[index]["geometry_hash"] for index in indices} & {
        cases[0]["geometry_hash"],
        cases[1]["geometry_hash"],
    }


def test_training_reference_selection_rejects_cross_family_geometry_leakage():
    cases = [
        _case("heldout", "family-a", "shared-geometry"),
        _case("train", "family-b", "shared-geometry"),
    ]

    with pytest.raises(ValueError, match="geometry hashes"):
        training_reference_indices([_record(case) for case in cases], cases, [cases[0]])
