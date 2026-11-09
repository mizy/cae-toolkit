from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from vehicle_cd_learning.data import PARAMETER_NAMES
from vehicle_cd_multiview import (
    build_library,
    load_library,
    operating_point,
    query_conditions,
)
from vehicle_cd_similarity_lib.multiview import (
    VIEWS,
    describe_views,
    predict_descriptors,
)


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    root = tmp_path_factory.mktemp("multiview-library")
    prepared = root / "prepared"
    (prepared / "views").mkdir(parents=True)
    cases = []
    for i in range(3):
        views = {}
        for name in VIEWS:
            image = np.zeros((2, 256, 256), dtype=np.float32)
            image[1, 20:220, 40:200] = 1
            image[0, 20:220, 40:200] = 0.2 + i * 0.03
            views[name] = image
        path = prepared / f"views/car-{i}.npz"
        np.savez(path, **views)
        cases.append(
            {
                "case_id": f"car-{i}",
                "geometry_id": f"car-{i}",
                "family_id": f"family-{i}",
                "geometry_hash": f"synthetic-{i}",
                "split": "train",
                "views": f"views/car-{i}.npz",
                "views_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "parameters": [0.4, 0.3, float(np.log(3e6)), 0.0, 1.0],
                "targets": [0.2 + i * 0.03],
                "reynolds": 3e6,
                "yaw_deg": 0.0,
                "cd_source": "synthetic fixture",
            }
        )
    source = {
        "schema_version": 1,
        "parameter_names": PARAMETER_NAMES,
        "target_names": ["cd"],
        "render_config": {"ordinary_size": 256, "bottom_size": 256, "window_l": 1.3},
        "conditions": {"ground": "synthetic"},
        "data_kind": "synthetic",
        "provenance": "synthetic fixture",
        "cases": cases,
    }
    (prepared / "dataset.json").write_text(json.dumps(source))
    return build_library(prepared, root / "library"), prepared


def test_cached_library_roundtrip_matches_fresh_descriptors(library):
    path, prepared = library
    metadata, bank = load_library(path)
    assert (
        metadata["source_dataset_sha256"]
        == hashlib.sha256((prepared / "dataset.json").read_bytes()).hexdigest()
    )
    assert [r["id"] for r in metadata["records"]] == ["car-0", "car-1", "car-2"]
    with np.load(prepared / "views/car-1.npz") as arrays:
        query = describe_views({k: arrays[k] for k in VIEWS})
    for key in query:
        np.testing.assert_array_equal(query[key], bank[key][1])
    assert predict_descriptors(query, bank, metadata["records"])["cd"] == pytest.approx(
        0.23
    )
    assert query_conditions(metadata, {"reynolds": 3e6, "yaw_deg": 0}, 4.5) == {
        "reynolds": 3e6,
        "yaw_deg": 0,
    }
    with pytest.raises(ValueError, match="reynolds"):
        query_conditions(metadata, {"reynolds": 1e6}, 4.5)


@pytest.mark.parametrize("tamper", ["bytes", "order"])
def test_cached_bank_rejects_tampering(library, tmp_path, tamper):
    original, _ = library
    shutil.copytree(original.parent, tmp_path / "library")
    path = tmp_path / "library/library.json"
    if tamper == "bytes":
        with (path.parent / "features.npz").open("ab") as file:
            file.write(b"changed")
    else:
        meta = json.loads(path.read_text())
        meta["records"].reverse()
        path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="checksum|order"):
        load_library(path)


def test_bank_conditions_are_fixed_and_speed_reynolds_is_derived():
    data = {
        "conditions": {
            "speed": {"value_kmh": 120},
            "air": {"density_kg_m3": 1.204, "dynamic_viscosity_pa_s": 1.81e-5},
        },
        "cases": [{"reynolds": 9e6, "yaw_deg": 0}, {"reynolds": 1e7, "yaw_deg": 0}],
    }
    profile = operating_point(data)
    metadata = {"operating_point": profile, "conditions": data["conditions"]}
    assert query_conditions(metadata, {}, 4.5)["reynolds"] == pytest.approx(
        1.204 * 120 / 3.6 * 4.5 / 1.81e-5
    )
    with pytest.raises(ValueError, match="conditions"):
        query_conditions(metadata, {"conditions": {}}, 4.5)
    data["cases"][1]["yaw_deg"] = 5
    with pytest.raises(ValueError, match="single yaw"):
        operating_point(data)


def test_unknown_exclusion_fails_before_loading_query_geometry(library, tmp_path):
    from vehicle_cd_multiview import predict_case

    path, _ = library
    with pytest.raises(ValueError, match="Unknown reference exclusions"):
        predict_case(
            path, tmp_path / "unused-query.json", exclude_ids=["misspelled-car"]
        )
