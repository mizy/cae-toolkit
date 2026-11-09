from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import pyvista as pv
import torch

SKILL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_learning_fixture import write_fixture
from vehicle_cd_learning.data import CaseDataset, load_dataset, parameters
from vehicle_cd_learning.evaluate import evaluate
from vehicle_cd_learning.model import VIEWS, CoefficientModel
from vehicle_cd_learning.predict import load_model, predict
from vehicle_cd_learning.prepare import prepare
from vehicle_cd_learning.train import train_model


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    root = tmp_path_factory.mktemp("analytic-cd-cl")
    manifest = write_fixture(root / "source")
    output = root / "prepared"
    prepare(manifest, output, {"ordinary_size": 32, "bottom_size": 64, "window_l": 1.3})
    return output


def test_parameters_are_scale_invariant_and_encode_signed_yaw():
    dimensions = {"length_m": 4.5, "width_m": 1.8, "height_m": 1.4}
    base = parameters(dimensions, 2e6, 5.0)
    scaled = parameters(
        {key: value * 1000 for key, value in dimensions.items()}, 2e6, 5.0
    )
    assert np.allclose(base, scaled)
    assert base[2] == pytest.approx(np.log(2e6))
    assert base[3] == pytest.approx(np.sin(np.deg2rad(5)))
    assert parameters(dimensions, 2e6, -5.0)[3] == pytest.approx(-base[3])
    with pytest.raises(ValueError):
        parameters(dimensions, 0, 0)


def test_fixture_contract_has_family_level_splits_and_expected_resolution(prepared):
    data = load_dataset(prepared)
    assert len(data["cases"]) == 48
    families = {}
    first_views = {}
    for case in data["cases"]:
        prior = families.setdefault(case["family_id"], case["split"])
        assert prior == case["split"]
        with np.load(prepared / case["views"], allow_pickle=False) as images:
            for name in VIEWS[:4]:
                assert images[name].shape == (2, 32, 32)
            assert images["bottom"].shape == (2, 64, 64)
            for image in images.values():
                assert np.isfinite(image).all()
                assert set(np.unique(image[1])).issubset({0.0, 1.0})
            first_views.setdefault(
                case["geometry_id"], {key: images[key].copy() for key in VIEWS}
            )
    assert {
        key: list(families.values()).count(key) for key in set(families.values())
    } == {"train": 8, "val": 2, "test": 2}
    for family in families:
        base = first_views[f"{family}-design-0"]
        modified = first_views[f"{family}-design-1"]
        assert not np.array_equal(base["bottom"], modified["bottom"])
        assert not np.array_equal(base["right"], modified["right"])


def test_dataset_rejects_geometry_family_split_leakage(prepared, tmp_path):
    source = json.loads((prepared / "dataset.json").read_text())
    source["cases"][0]["split"] = "test"
    bad = tmp_path / "dataset.json"
    bad.write_text(json.dumps(source))
    shutil.copytree(prepared / "views", tmp_path / "views")
    with pytest.raises(ValueError, match="Split leakage"):
        load_dataset(bad)


def test_dataset_rejects_changed_view_cache(prepared, tmp_path):
    shutil.copytree(prepared, tmp_path / "prepared")
    copied = tmp_path / "prepared"
    data = load_dataset(copied)
    view_path = copied / data["cases"][0]["views"]
    with np.load(view_path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays["front"] = arrays["front"].copy()
    arrays["front"][0, 0, 0] += 0.1
    np.savez_compressed(view_path, **arrays)
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_dataset(copied)


def test_prepare_blocks_renamed_geometry_clone_across_splits(tmp_path):
    source_root = tmp_path / "source"
    manifest_path = write_fixture(source_root)
    source = json.loads(manifest_path.read_text())
    train_case = next(case for case in source["cases"] if case["split"] == "train")
    duplicate = json.loads(json.dumps(train_case))
    original_path = source_root / train_case["geometry"]["path"]
    copied_path = source_root / "geometries/renamed-unit-translated-copy.vtp"
    copied_path.parent.mkdir(parents=True, exist_ok=True)
    mesh = pv.read(original_path)
    mesh.points = mesh.points * 1000.0 + np.array([12000.0, -8000.0, 3500.0])
    mesh.save(copied_path)
    duplicate.update(
        case_id="renamed-copy-heldout",
        geometry_id="renamed-copy",
        family_id="different-family",
        split="test",
    )
    duplicate["geometry"] = {
        "path": "geometries/renamed-unit-translated-copy.vtp",
        "unit_scale_to_m": 0.001,
        "front_axis": "-x",
        "up_axis": "+z",
    }
    manifest_path.write_text(json.dumps({**source, "cases": [train_case, duplicate]}))
    with pytest.raises(ValueError, match="Split leakage: geometry_hash"):
        prepare(
            manifest_path,
            tmp_path / "prepared-copy-check",
            {"ordinary_size": 32, "bottom_size": 64, "window_l": 1.3},
        )


@pytest.mark.parametrize("variant", ["parameters", "four", "five"])
def test_model_variants_shapes_finite_gradients_and_shared_view_weights(variant):
    torch.manual_seed(7)
    model = CoefficientModel(5, variant)
    views = {name: torch.rand(2, 2, 32, 32) for name in VIEWS}
    x = torch.rand(2, 5)
    output = model(views, x)
    assert output.shape == (2, 2)
    output.square().mean().backward()
    grads = [
        p.grad for p in model.parameters() if p.requires_grad and p.grad is not None
    ]
    assert grads and all(torch.isfinite(grad).all() for grad in grads)
    if variant != "parameters":
        assert model.image_encoder is not None
        calls = []
        handle = model.image_encoder.register_forward_pre_hook(
            lambda _module, inputs: calls.append(inputs[0].shape[0])
        )
        model(views, x)
        handle.remove()
        assert calls == [8]  # four B=2 views pass through one shared encoder call
    if variant == "five":
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.bottom_encoder.parameters()
        )


@pytest.mark.parametrize("preload", [False, True])
def test_training_checkpoint_roundtrip(prepared, tmp_path, preload):
    manifest = load_dataset(prepared)
    train = [case for case in manifest["cases"] if case["split"] == "train"]
    val = [case for case in manifest["cases"] if case["split"] == "val"]
    x = np.asarray([case["parameters"] for case in train])
    y = np.asarray([case["targets"] for case in train])
    stats = {
        "parameter_mean": x.mean(0).tolist(),
        "parameter_std": np.maximum(x.std(0), 1e-8).tolist(),
        "target_mean": y.mean(0).tolist(),
        "target_std": np.maximum(y.std(0), 1e-8).tolist(),
    }
    train_set = CaseDataset(
        prepared,
        train,
        stats["parameter_mean"],
        stats["parameter_std"],
        stats["target_mean"],
        stats["target_std"],
        "five",
    )
    val_set = CaseDataset(
        prepared,
        val,
        stats["parameter_mean"],
        stats["parameter_std"],
        stats["target_mean"],
        stats["target_std"],
        "five",
    )
    metadata = {
        "render_config": manifest["render_config"],
        "conditions": manifest["conditions"],
        "parameter_ranges": {},
        "data_kind": manifest["data_kind"],
        "provenance": manifest["provenance"],
        "train_ids": [c["case_id"] for c in train],
        "train_geometry_hashes": sorted({c["geometry_hash"] for c in train}),
        "heldout_ids": [c["case_id"] for c in val],
        "dataset_sha256": hashlib.sha256(
            (prepared / "dataset.json").read_bytes()
        ).hexdigest(),
    }
    checkpoint_path = train_model(
        train_set,
        val_set,
        parameter_names=manifest["parameter_names"],
        normalization=stats,
        metadata=metadata,
        output_dir=tmp_path / "model",
        variant="five",
        epochs=1,
        batch_size=8,
        seed=3,
        device_name="cpu",
        preload=preload,
    )
    restored, checkpoint = load_model(checkpoint_path)
    views, params, _ = val_set[0]
    arrays = {key: value.numpy() for key, value in views.items()}
    with torch.no_grad():
        normalized = restored(
            {key: value.unsqueeze(0) for key, value in views.items()},
            params.unsqueeze(0),
        ).numpy()[0]
    manual = normalized * checkpoint["target_std"] + checkpoint["target_mean"]
    inferred = predict(restored, checkpoint, arrays, val[0]["parameters"])
    assert np.allclose(manual, inferred, rtol=1e-6, atol=1e-7)
    assert np.isfinite(inferred).all()
    report = evaluate(checkpoint_path, prepared, tmp_path / "evaluation", split="test")
    assert report["count"] == 8
    assert report["data_kind"] == "synthetic"
    assert (tmp_path / "evaluation" / "scatter.png").is_file()
    report = json.loads((tmp_path / "model" / "training_report.json").read_text())
    assert report["train_count"] == len(train)
    assert report["validation_count"] == len(val)
    assert report["preload"] is preload


def test_cd_only_dataset_trains_predicts_and_evaluates_without_synthetic_cl(tmp_path):
    manifest_path = write_fixture(tmp_path / "cd-source", target_names=("cd",))
    source = json.loads(manifest_path.read_text())
    assert source["target_names"] == ["cd"]
    assert all("cl" not in case for case in source["cases"])
    prepared = tmp_path / "cd-prepared"
    prepare(
        manifest_path,
        prepared,
        {"ordinary_size": 32, "bottom_size": 64, "window_l": 1.3},
    )
    dataset = load_dataset(prepared)
    assert dataset["target_names"] == ["cd"]
    assert all(len(case["targets"]) == 1 for case in dataset["cases"])
    train = [case for case in dataset["cases"] if case["split"] == "train"]
    val = [case for case in dataset["cases"] if case["split"] == "val"]
    x = np.asarray([case["parameters"] for case in train])
    y = np.asarray([case["targets"] for case in train])
    stats = {
        "parameter_mean": x.mean(0).tolist(),
        "parameter_std": np.maximum(x.std(0), 1e-8).tolist(),
        "target_mean": y.mean(0).tolist(),
        "target_std": np.maximum(y.std(0), 1e-8).tolist(),
    }
    train_set = CaseDataset(
        prepared,
        train,
        stats["parameter_mean"],
        stats["parameter_std"],
        stats["target_mean"],
        stats["target_std"],
        "parameters",
    )
    val_set = CaseDataset(
        prepared,
        val,
        stats["parameter_mean"],
        stats["parameter_std"],
        stats["target_mean"],
        stats["target_std"],
        "parameters",
    )
    metadata = {
        "render_config": dataset["render_config"],
        "conditions": dataset["conditions"],
        "parameter_ranges": {},
        "data_kind": dataset["data_kind"],
        "provenance": dataset["provenance"],
        "train_ids": [c["case_id"] for c in train],
        "train_geometry_hashes": sorted({c["geometry_hash"] for c in train}),
        "heldout_ids": [c["case_id"] for c in val],
        "dataset_sha256": hashlib.sha256(
            (prepared / "dataset.json").read_bytes()
        ).hexdigest(),
    }
    checkpoint_path = train_model(
        train_set,
        val_set,
        parameter_names=dataset["parameter_names"],
        target_names=["cd"],
        normalization=stats,
        metadata=metadata,
        output_dir=tmp_path / "cd-model",
        variant="parameters",
        epochs=1,
        batch_size=8,
        seed=4,
        device_name="cpu",
    )
    model, checkpoint = load_model(checkpoint_path)
    assert checkpoint["target_names"] == ["cd"]
    assert model.target_count == 1
    result = predict(model, checkpoint, {}, val[0]["parameters"])
    assert result.shape == (1,) and np.isfinite(result).all()
    report = evaluate(
        checkpoint_path, prepared, tmp_path / "cd-evaluation", split="test"
    )
    assert set(report["metrics"]) == {"cd"}
    assert report["count"] == 8


@pytest.mark.parametrize("augment", [False, True])
@pytest.mark.parametrize("preload", [False, True])
def test_final_fit_uses_every_case_without_validation_claim(
    prepared, tmp_path, monkeypatch, augment, preload
):
    from vehicle_cd_learning.train import main

    output = tmp_path / "all-fit"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            str(prepared),
            "--output-dir",
            str(output),
            "--fit-all",
            "--epochs",
            "1",
            "--variant",
            "parameters",
        ],
    )
    import vehicle_cd_learning.train as training

    calls = []
    original = training.jitter_views

    def observe(views, generator):
        calls.append(len(next(iter(views.values()))))
        return original(views, generator)

    monkeypatch.setattr(training, "jitter_views", observe)
    if augment:
        sys.argv += ["--augment", "--model-size", "small"]
        # Use views in the augmented branch rather than the parameter-only model.
        sys.argv[sys.argv.index("--variant") + 1] = "five"
    if preload:
        sys.argv.append("--preload")
    assert main() == 0
    assert len(calls) == (3 if augment else 0)
    checkpoint = torch.load(output / "model.pt", weights_only=True)
    dataset = load_dataset(prepared)
    assert set(checkpoint["train_ids"]) == {
        case["case_id"] for case in dataset["cases"]
    }
    assert checkpoint["heldout_ids"] == []
    assert checkpoint["training_mode"] == "all_samples_final_epoch"
    assert checkpoint["validation_mse_standardized"] is None
    report = json.loads((output / "training_report.json").read_text())
    assert report["validation_count"] == 0
    assert "validation_mae" not in report
    assert "training_fit_mae" in report
