from __future__ import annotations

import hashlib
import importlib
import json
import shutil
import sys
from pathlib import Path

import pytest

SKILL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_learning_fixture import write_fixture
from vehicle_cd_learning.data import load_dataset
from vehicle_cd_learning.prepare import prepare


@pytest.fixture(scope="module")
def prepared_geometry_split(tmp_path_factory):
    root = tmp_path_factory.mktemp("geometry-split")
    manifest_path = write_fixture(root / "source")
    manifest = json.loads(manifest_path.read_text())
    manifest["split_policy"] = "geometry_disjoint"
    manifest["split_provenance"] = {
        "method": "explicit geometry-level assignment",
        "seed": 7,
    }
    manifest_path.write_text(json.dumps(manifest))
    output = root / "prepared"
    prepare(
        manifest_path,
        output,
        {"ordinary_size": 32, "bottom_size": 64, "window_l": 1.3},
    )
    return output


def split_one_family_across_geometries(prepared, tmp_path):
    source = json.loads((prepared / "dataset.json").read_text())
    family = source["cases"][0]["family_id"]
    train_geometry = next(
        case["geometry_id"]
        for case in source["cases"]
        if case["family_id"] == family and case["split"] == "train"
    )
    heldout_geometry = next(
        case["geometry_id"]
        for case in source["cases"]
        if case["family_id"] == family
        and case["split"] == "train"
        and case["geometry_id"] != train_geometry
    )
    for case in source["cases"]:
        if case["geometry_id"] == heldout_geometry:
            case["split"] = "test"
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(json.dumps(source))
    shutil.copytree(prepared / "views", tmp_path / "views")
    return dataset_path, source, train_geometry, heldout_geometry


def test_shared_family_requires_explicit_geometry_policy(
    prepared_geometry_split, tmp_path
):
    dataset_path, source, _, _ = split_one_family_across_geometries(
        prepared_geometry_split, tmp_path
    )
    source.pop("split_policy")
    dataset_path.write_text(json.dumps(source))
    with pytest.raises(ValueError, match="Split leakage: family_id"):
        load_dataset(dataset_path)

    source["split_policy"] = "geometry_disjoint"
    dataset_path.write_text(json.dumps(source))
    assert load_dataset(dataset_path)["split_policy"] == "geometry_disjoint"

    source["cases"][0].pop("family_id")
    dataset_path.write_text(json.dumps(source))
    with pytest.raises(ValueError, match="Missing family_id"):
        load_dataset(dataset_path)


@pytest.mark.parametrize("leak_key", ["geometry_id", "geometry_hash"])
def test_geometry_policy_still_rejects_geometry_leakage(
    prepared_geometry_split, tmp_path, leak_key
):
    dataset_path, source, train_geometry, heldout_geometry = (
        split_one_family_across_geometries(prepared_geometry_split, tmp_path)
    )
    source["split_policy"] = "geometry_disjoint"
    train_value = next(
        case[leak_key]
        for case in source["cases"]
        if case["geometry_id"] == train_geometry
    )
    for case in source["cases"]:
        if case["geometry_id"] == heldout_geometry:
            case[leak_key] = train_value
    dataset_path.write_text(json.dumps(source))

    with pytest.raises(ValueError, match=f"Split leakage: {leak_key}"):
        load_dataset(dataset_path)


def test_prepare_preserves_split_policy_and_provenance(tmp_path):
    manifest_path = write_fixture(tmp_path / "source")
    manifest = json.loads(manifest_path.read_text())
    manifest["split_policy"] = "geometry_disjoint"
    manifest["split_provenance"] = {"method": "curated geometry groups"}
    manifest_path.write_text(json.dumps(manifest))
    output = tmp_path / "prepared"

    prepare(
        manifest_path,
        output,
        {"ordinary_size": 32, "bottom_size": 64, "window_l": 1.3},
    )

    prepared = load_dataset(output)
    assert prepared["split_policy"] == "geometry_disjoint"
    assert prepared["split_provenance"] == manifest["split_provenance"]


def test_evaluation_reports_seen_and_unseen_families(
    prepared_geometry_split, tmp_path, monkeypatch
):
    dataset_path, source, _, _ = split_one_family_across_geometries(
        prepared_geometry_split, tmp_path
    )
    source["split_policy"] = "geometry_disjoint"
    dataset_path.write_text(json.dumps(source))
    train_hashes = sorted(
        {case["geometry_hash"] for case in source["cases"] if case["split"] == "train"}
    )
    checkpoint = {
        "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "train_geometry_hashes": train_hashes,
        "target_names": source["target_names"],
        "target_mean": [0.0, 0.0],
        "target_std": [1.0, 1.0],
        "parameter_names": source["parameter_names"],
        "parameter_ranges": {},
    }
    module = importlib.import_module("vehicle_cd_learning.evaluate")
    monkeypatch.setattr(module, "load_model", lambda _: (None, checkpoint))
    monkeypatch.setattr(
        module,
        "predict",
        lambda *_args, **_kwargs: [0.0, 0.0],
    )

    report = module.evaluate(
        tmp_path / "unused-checkpoint.pt",
        tmp_path,
        tmp_path / "evaluation",
        split="test",
    )

    assert report["split_policy"] == "geometry_disjoint"
    assert "within-family geometry interpolation" in report["family_evaluation_scope"]
    assert report["seen_training_family_count"] == 1
    assert report["unseen_training_family_count"] == 2
    assert report["seen_training_family_metrics"]
    assert report["unseen_training_family_metrics"]
