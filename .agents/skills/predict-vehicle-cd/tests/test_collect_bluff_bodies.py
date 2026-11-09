from __future__ import annotations

import csv
import hashlib
import importlib
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

SKILL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL / "scripts"))

from vehicle_cd_learning.collect_bluff_bodies import DATASETS, _source_manifest
from vehicle_cd_learning.render import canonical_points


def test_windsor_freestream_yaw_and_floor_referenced_mesh(tmp_path, monkeypatch):
    raw_basis = np.vstack((np.zeros(3), np.eye(3)))
    canonical, _ = canonical_points(
        raw_basis,
        unit_scale_to_m=1.0,
        front_axis="-x",
        up_axis="+y",
        ground_clearance_m=0.0,
    )
    assert np.allclose(canonical[1] - canonical[0], [1.0, 0.0, 0.0])
    assert np.allclose(canonical[2] - canonical[0], [0.0, 0.0, 1.0])
    assert np.allclose(canonical[3] - canonical[0], [0.0, -1.0, 0.0])

    beta = np.deg2rad(2.5)
    raw_body_frame_freestream = np.array([np.cos(beta), 0.0, -np.sin(beta)])
    canonical_freestream = np.array(
        [
            raw_body_frame_freestream[0],
            -raw_body_frame_freestream[2],
            raw_body_frame_freestream[1],
        ]
    )
    assert canonical_freestream[1] > 0
    assert DATASETS["windsorml"]["yaw_deg"] == 2.5

    catalogue = tmp_path / "catalogue"
    output = tmp_path / "output"
    catalogue.mkdir()
    output.mkdir()
    labels_path = catalogue / DATASETS["windsorml"]["label_file"]
    geometry_path = catalogue / "geo_parameters_all.csv"
    run_ids = list(range(24))
    with labels_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["run", "cd", "cl"])
        writer.writeheader()
        writer.writerows({"run": run, "cd": 0.3, "cl": 0.2} for run in run_ids)
    with geometry_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["run", "clearance"])
        writer.writeheader()
        writer.writerows({"run": run, "clearance": str(10 + run)} for run in run_ids)

    module = importlib.import_module("vehicle_cd_learning.collect_bluff_bodies")
    monkeypatch.setattr(
        module, "read_mesh", lambda _path: SimpleNamespace(points=raw_basis)
    )
    selection = {
        "train": [f"run_{run}" for run in range(16)],
        "val": [f"run_{run}" for run in range(16, 20)],
        "test": [f"run_{run}" for run in range(20, 24)],
    }
    source = _source_manifest(
        "windsorml",
        catalogue,
        output,
        selection,
        {"revision": "test"},
        {},
        {run: {"bytes": 1, "sha256": "a" * 64} for run in run_ids},
    )

    assert all(case["yaw_deg"] == 2.5 for case in source["cases"])
    assert all(
        case["geometry"]["ground_clearance_m"] == 0.0 for case in source["cases"]
    )
    assert all(
        float(case["design_value"]["clearance"]) / 1000
        == (10 + int(case["design_value"]["run"])) / 1000
        for case in source["cases"]
    )
    assert all(case["cd"] == 0.3 and case["cl"] == 0.2 for case in source["cases"])
    assert "not freestream-projected" in source["conditions"]["force_axes"]["value"]
    assert "body rotation -2.5 degrees" in source["provenance"]["source_setup"]


def test_fresh_catalogue_downloads_only_pinned_small_files_and_verifies_git_cache(
    tmp_path, monkeypatch
):
    module = importlib.import_module("vehicle_cd_learning.collect_bluff_bodies")
    spec = DATASETS["windsorml"]
    split_manifest = {
        "full_train": [f"run_{run}" for run in range(16)],
        "full_val": [f"run_{run}" for run in range(16, 20)],
        "full_test": [f"run_{run}" for run in range(20, 24)],
    }
    contents = {
        "README.md": b"dataset readme",
        "LICENSE.txt": b"dataset license",
        spec["label_file"]: b"run, cd, cl\n0,0.3,0.2\n",
        "geo_parameters_all.csv": b"run, clearance\n0,50\n",
        module.SPLIT_SOURCE_PATH: json.dumps(split_manifest).encode(),
    }
    expected = {}
    for relative_path, content in contents.items():
        url = (
            module._manifest_url(spec)
            if relative_path == module.SPLIT_SOURCE_PATH
            else module._source_file_url(spec, relative_path)
        )
        git_blob_sha1 = hashlib.sha1(
            f"blob {len(content)}\0".encode() + content
        ).hexdigest()
        expected[url] = {"bytes": len(content), "git_blob_sha1": git_blob_sha1}

    monkeypatch.setattr(
        module,
        "_pinned_file_metadata",
        lambda url, _revision: expected[url],
    )
    monkeypatch.setattr(
        module,
        "urlopen",
        lambda url, timeout: io.BytesIO(
            contents[next(name for name in contents if url.endswith("/" + name))]
        ),
    )

    catalogue = tmp_path / "fresh-catalogue" / "windsorml"
    files = module._ensure_catalogue(catalogue, "windsorml")
    selected, metadata = module._split_selection(catalogue, "windsorml")

    assert len(files) == 5
    assert [len(selected[key]) for key in ("train", "val", "test")] == [16, 4, 4]
    assert (
        metadata["git_blob_sha1"]
        == expected[module._manifest_url(spec)]["git_blob_sha1"]
    )
    readme_url = module._source_file_url(spec, "README.md")
    assert (
        module._download(readme_url, catalogue / "README.md", expected[readme_url])
        == expected[readme_url]
    )
    assert not list(catalogue.rglob("*.stl"))


def test_full_split_selection_keeps_every_official_id(tmp_path, monkeypatch):
    module = importlib.import_module("vehicle_cd_learning.collect_bluff_bodies")
    manifest = {
        "full_train": ["run_0", "run_2"],
        "full_val": ["run_1"],
        "full_test": ["run_3", "run_4"],
    }
    content = json.dumps(manifest).encode()
    path = tmp_path / module.SPLIT_MANIFEST_NAME
    path.write_bytes(content)
    expected = {
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }
    monkeypatch.setattr(
        module, "_pinned_file_metadata", lambda _url, _revision: expected
    )

    selected, metadata = module._split_selection(tmp_path, "ahmedml", full=True)

    assert selected == {
        "train": ["run_0", "run_2"],
        "val": ["run_1"],
        "test": ["run_3", "run_4"],
    }
    assert metadata["selection"].startswith("all full_train")


def test_full_selection_records_missing_rows_and_meshes(tmp_path, monkeypatch):
    module = importlib.import_module("vehicle_cd_learning.collect_bluff_bodies")
    catalogue = tmp_path / "catalogue"
    output = tmp_path / "output" / "ahmedml"
    catalogue.mkdir()
    labels = catalogue / DATASETS["ahmedml"]["label_file"]
    geometry = catalogue / "geo_parameters_all.csv"
    labels.write_text("run,cd,cl\n0,0.3,0.2\n1,0.4,0.1\n2,0.5,0.0\n")
    geometry.write_text("run,length\n0,1.0\n2,1.1\n3,1.2\n")
    selection = {
        "train": ["run_0", "run_1"],
        "val": ["run_2"],
        "test": ["run_3"],
    }

    def pinned_metadata(url, _revision):
        if "/run_2/" in url:
            raise module.SourceAssetMissing(url)
        return {"bytes": 1, "sha256": "a" * 64}

    monkeypatch.setattr(module, "_pinned_file_metadata", pinned_metadata)

    available, metadata, excluded = module._available_full_selection(
        "ahmedml", catalogue, output, selection, workers=2
    )

    assert available == {"train": ["run_0"], "val": [], "test": []}
    assert set(metadata) == {0}
    assert excluded == [
        {
            "split": "test",
            "run": 3,
            "reasons": ["missing_variable_reference_labels"],
        },
        {
            "split": "train",
            "run": 1,
            "reasons": ["missing_geometry_parameters"],
        },
        {
            "split": "val",
            "run": 2,
            "reasons": ["mesh_not_published"],
            "source_url": module._mesh_url(DATASETS["ahmedml"], 2),
        },
    ]


def test_full_selection_reuses_pilot_stl_without_changing_pilot_manifest(
    tmp_path, monkeypatch
):
    module = importlib.import_module("vehicle_cd_learning.collect_bluff_bodies")
    spec = DATASETS["ahmedml"]
    catalogue = tmp_path / "catalogue"
    output = tmp_path / "output" / "ahmedml"
    catalogue.mkdir()
    output.mkdir(parents=True)
    (catalogue / spec["label_file"]).write_text("run,cd,cl\n0,0.3,0.2\n")
    (catalogue / "geo_parameters_all.csv").write_text("run,length\n0,1.0\n")
    mesh = output / "geometries" / "ahmedml-run-0.stl"
    mesh.parent.mkdir()
    mesh_bytes = b"pilot mesh bytes"
    mesh.write_bytes(mesh_bytes)
    pilot = {
        "provenance": {
            "repository": spec["repo"],
            "revision": spec["revision"],
            "geometry_assets": [
                {
                    "run": 0,
                    "bytes": len(mesh_bytes),
                    "sha256": hashlib.sha256(mesh_bytes).hexdigest(),
                }
            ],
        }
    }
    pilot_path = output / "source.json"
    pilot_bytes = (json.dumps(pilot) + "\n").encode()
    pilot_path.write_bytes(pilot_bytes)
    monkeypatch.setattr(
        module,
        "_pinned_file_metadata",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("pilot cache should be reused")
        ),
    )

    available, metadata, excluded = module._available_full_selection(
        "ahmedml",
        catalogue,
        output,
        {"train": ["run_0"], "val": [], "test": []},
        workers=2,
    )

    assert available == {"train": ["run_0"], "val": [], "test": []}
    assert metadata == {
        0: {"bytes": len(mesh_bytes), "sha256": hashlib.sha256(mesh_bytes).hexdigest()}
    }
    assert excluded == []
    assert pilot_path.read_bytes() == pilot_bytes


def test_download_retries_truncated_response(tmp_path, monkeypatch):
    module = importlib.import_module("vehicle_cd_learning.collect_bluff_bodies")
    content = b"complete mesh bytes"
    expected = {
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }
    attempts = []

    class TruncatedResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _size):
            raise module.IncompleteRead(b"partial", 10)

    def open_source(_url, timeout):
        attempts.append(timeout)
        return TruncatedResponse() if len(attempts) == 1 else io.BytesIO(content)

    monkeypatch.setattr(module, "urlopen", open_source)
    monkeypatch.setattr(module.time, "sleep", lambda _delay: None)

    destination = tmp_path / "asset.stl"
    assert (
        module._download("https://source.test/asset.stl", destination, expected)
        == expected
    )
    assert attempts == [120, 120]
    assert destination.read_bytes() == content
