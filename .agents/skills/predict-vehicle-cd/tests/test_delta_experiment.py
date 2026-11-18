import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

SKILL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL / "scripts"))

from vehicle_cd_learning.delta_experiment import (
    dataset_for,
    lock512,
    pair_schedule,
    predict_anchors,
    sha256,
)
from vehicle_cd_learning.model import VIEWS
from vehicle_cd_learning.paired_delta import PairedDelta


def test_pair_schedule_has_exact_exposure_and_zero_mean_without_heldout():
    indices = np.arange(7, 727)
    left, right = pair_schedule(indices)
    assert left.shape == right.shape == (12, 720)
    assert np.array_equal(left, pair_schedule(indices)[0])
    assert np.all(left != right)
    for a, b in zip(left, right):
        assert np.array_equal(np.sort(a), indices)
        assert np.array_equal(np.sort(b), indices)
    _, counts = np.unique(
        np.concatenate((left.ravel(), right.ravel())), return_counts=True
    )
    assert np.all(counts == 24)
    target = np.random.default_rng(4).normal(size=(800, 2))
    assert np.allclose((target[right] - target[left]).mean((0, 1)), 0, atol=1e-14)


def fixture_data(tmp_path):
    cases = []
    image = np.stack(
        (np.full((256, 256), 0.5, np.float32), np.ones((256, 256), np.float32))
    )
    for i, split in enumerate(("train", "val", "test")):
        file = tmp_path / f"{i}.npz"
        np.savez_compressed(file, **{n: image for n in VIEWS})
        cases.append(
            {
                "case_id": str(i),
                "source": "drivaerrim",
                "split": split,
                "parameters": [float(i), None] + [1.0] * 16,
                "targets": [0.2 + i * 0.01, -0.1],
                "geometry_hash": str(i),
                "views": file.name,
                "views_sha256": sha256(file),
            }
        )
    p = {
        "prepared": str(tmp_path),
        "prepared512": str(tmp_path),
        "experiment": str(tmp_path),
        "delta_std": [0.02, 0.03],
        "cache_cases": {"256": 1, "512": 1},
        "parameter_mean": [0.0] * 36,
        "parameter_std": [2.0] * 36,
        "case_ids": [c["case_id"] for c in cases],
    }
    return p, cases


def test_dataset_bounded_cache_normalization_hash_and_validation_api(tmp_path):
    torch.set_num_threads(2)
    p, cases = fixture_data(tmp_path)
    dataset = dataset_for(p, cases, 256)
    first = dataset[0]
    assert first[1].shape == (36,)
    assert first[1][1] == 0 and first[1][19] == 0 and first[1][18] == 1
    assert dataset[0][0]["front"].data_ptr() == first[0]["front"].data_ptr()
    dataset[1]
    assert len(dataset.cache) == 1 and set(dataset.cache) == {1}
    model = PairedDelta(8).eval()
    torch.nn.init.zeros_(model.head[-1].weight)
    prediction = predict_anchors(
        model, dataset, cases, ["0"], "val", torch.device("cpu")
    )
    assert prediction == {"0": {"1": [0.0, 0.0]}}
    assert 2 not in dataset.verified
    (tmp_path / cases[2]["views"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA differs"):
        dataset[2]


def test_512_lock_rejects_incomplete_and_semantic_mismatch(tmp_path):
    p, cases = fixture_data(tmp_path)
    inventory = tmp_path / "inventory.json"
    inventory.write_text("{}")
    source = tmp_path / "source.json"
    source.write_text("{}")
    p["inventory"] = str(inventory)
    p["input_sha256"] = {str(tmp_path / "dataset.json"): sha256(source)}
    p["code_sha256"] = {}
    manifest = {
        "cases": copy.deepcopy(cases),
        "resolution_preparation": {
            "train_ready": False,
            "inventory_sha256": sha256(inventory),
            "source_dataset_sha256": sha256(source),
            "code_sha256": {},
        },
    }
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="not complete"):
        lock512(p, cases)
    manifest["resolution_preparation"]["train_ready"] = True
    manifest["cases"][0]["targets"][0] += 0.01
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="field differs: targets"):
        lock512(p, cases)
    manifest["cases"] = cases
    path.write_text(json.dumps(manifest))
    _, digest = lock512(p, cases)
    assert digest == sha256(tmp_path / "data512-lock.json")
    path.write_text(json.dumps({**manifest, "changed": True}))
    with pytest.raises(ValueError, match="manifest changed"):
        lock512(p, cases)
