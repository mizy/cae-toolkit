import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

SKILL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL / "scripts"))
sys.path.insert(0, str(SKILL.parent / "predict-aircraft-coefficients/scripts"))

from aircraft_learning.scene_experiment import SceneCases, normalization
from vehicle_cd_learning import dimensionless as dim
from vehicle_cd_learning.model import VIEWS, CoefficientModel


def fixture(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    names = (
        dim.PARAMETERS[:2] + ["log_length_m", "log_velocity_m_s"] + dim.PARAMETERS[2:]
    )
    cases = []
    for i, split in enumerate(("train", "train", "val", "test")):
        path = source / f"{i}.npz"
        image = np.stack(
            (
                np.full((256, 256), 0.3 + 0.01 * i, np.float32),
                np.ones((256, 256), np.float32),
            )
        )
        np.savez_compressed(path, **{k: image for k in VIEWS})
        values = [
            0.4 + 0.01 * i,
            0.3,
            np.log(4),
            np.log(30),
            np.log(5e6 + i * 1e5),
            None,
            0,
            1,
            0,
            1,
        ] + [None] * 10
        cases.append(
            {
                "case_id": str(i),
                "geometry_id": str(i),
                "geometry_hash": str(i),
                "source": "fixture",
                "family_id": "fixture",
                "scene": 1,
                "split": split,
                "parameters": values,
                "targets": [0.25 + 0.01 * i, None if i == 2 else 0.1 + 0.01 * i],
                "conditions": {"velocity_m_s": 30, "re_body": 5e6 + i * 1e5},
                "views": path.name,
                "views_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    data = {
        "parameter_names": names,
        "target_names": ["cd", "cl"],
        "cases": cases,
        "source_counts": {"fixture": {"train": 2, "val": 1, "test": 1}},
        "split_policy": "geometry_disjoint",
    }
    (source / "dataset.json").write_text(json.dumps(data))
    prepared = tmp_path / "prepared"
    dim.prepare(source, prepared)
    masked = [
        dict(
            c,
            parameters=[
                None if j in (2, 3) else v for j, v in enumerate(c["parameters"])
            ],
        )
        for c in cases
    ]
    stats = normalization(masked[:2])
    torch.manual_seed(42)
    model = CoefficientModel(**{**dim.MODEL, "parameter_count": 40})
    path = tmp_path / "old.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "normalization": stats,
            "run": {"mask": ["log_length_m", "log_velocity_m_s"]},
            "supervised_train_ids": ["0", "1"],
            "epoch": 1,
            "seed": 42,
            "dataset_sha256": hashlib.sha256(
                (source / "dataset.json").read_bytes()
            ).hexdigest(),
        },
        path,
    )
    return source, prepared, path, model, stats, masked


def test_re_input_needs_no_velocity_length_or_viscosity_conditions():
    d = {"length_m": 4, "width_m": 1.8, "height_m": 1.5}
    c = {
        "re_body": 1e7,
        "alpha_deg": 0,
        "beta_deg": 2.5,
        "ground_speed_over_velocity": 1,
        "wheel_rotation_speed_over_velocity": 1,
    }
    first = dim.parameters(d, c)
    scaled = dim.parameters({k: v * 10 for k, v in d.items()}, c)
    assert first == scaled
    assert first[dim.PARAMETERS.index("log_re_body")] == np.log(1e7)
    assert (
        "log_length_m" not in dim.PARAMETERS
        and "log_velocity_m_s" not in dim.PARAMETERS
    )
    assert len(first) == 18 and len(dim.NAMES) == 36
    with pytest.raises(ValueError, match="Unsupported coefficient conditions"):
        dim.parameters(d, {**c, "velocity_m_s": 30})


def test_unknown_re_stays_unknown_and_invalid_re_is_rejected():
    d = {"length_m": 4, "width_m": 1.8, "height_m": 1.5}
    assert dim.parameters(d, {})[2] is None
    for re in (0, -1, np.inf, np.nan):
        with pytest.raises(ValueError, match="re_body"):
            dim.parameters(d, {"re_body": re})


def test_preparation_preserves_labels_splits_views_and_source_metadata(tmp_path):
    source, prepared, _, _, _, _ = fixture(tmp_path)
    before = json.loads((source / "dataset.json").read_text())
    after = dim.read_prepared(prepared)
    assert after["parameter_names"] == dim.PARAMETERS
    for a, b in zip(before["cases"], after["cases"]):
        assert {k: v for k, v in a.items() if k != "parameters"} == {
            k: v for k, v in b.items() if k != "parameters"
        }
        assert len(b["parameters"]) == 18
    assert after["cases"][2]["targets"][1] is None


def test_compact_checkpoint_is_prediction_equivalent_and_train_infer_match(tmp_path):
    source, prepared, old, original, stats, masked = fixture(tmp_path)
    output = tmp_path / "new.pt"
    digest = hashlib.sha256(old.read_bytes()).hexdigest()
    dim.export_checkpoint(old, prepared, output)
    model, ck = dim.load_model(output)
    original.eval()
    data = dim.read_prepared(prepared)
    legacy = SceneCases(source, masked, stats)
    current = SceneCases(prepared, data["cases"], normalization(data["cases"][:2]))
    torch.set_num_threads(2)
    for i in range(4):
        images, x, y, _ = current[i]
        _, old_x, old_y, _ = legacy[i]
        np.testing.assert_array_equal(
            x.numpy(), dim.normalize_parameters(data["cases"][i]["parameters"], ck)
        )
        torch.testing.assert_close(y, old_y, equal_nan=True)
        inputs = {k: v[None] for k, v in images.items()}
        with torch.no_grad():
            torch.testing.assert_close(
                original(inputs, old_x[None]),
                model(inputs, x[None]),
                atol=1e-6,
                rtol=1e-6,
            )
    assert ck["model_config"]["parameter_count"] == 36
    assert hashlib.sha256(old.read_bytes()).hexdigest() == digest


def test_export_refuses_full_input_weights_and_overwrite(tmp_path):
    _, prepared, old, _, _, _ = fixture(tmp_path)
    ck = torch.load(old, weights_only=True)
    ck["run"]["mask"] = []
    torch.save(ck, old)
    with pytest.raises(ValueError, match="not trained"):
        dim.export_checkpoint(old, prepared, tmp_path / "invalid.pt")
    path = tmp_path / "exists.pt"
    path.write_text("preserve")
    with pytest.raises(FileExistsError):
        dim.export_checkpoint(old, prepared, path)
    assert path.read_text() == "preserve"


def test_new_training_uses_compact_contract_and_loads_for_inference(tmp_path):
    _, prepared, _, _, _, _ = fixture(tmp_path)
    output = tmp_path / "trained"
    dim.train(prepared, output, epochs=1, device="cpu")
    model, ck = dim.load_model(output / "model.pt")
    assert model.parameter_count == 36 and ck["parameter_names"] == dim.NAMES
    data = dim.read_prepared(prepared)
    with np.load(prepared / data["cases"][2]["views"]) as f:
        result = dim.predict_views(
            model, ck, {k: f[k] for k in VIEWS}, data["cases"][2]["parameters"]
        )
    assert np.isfinite(result).all()
