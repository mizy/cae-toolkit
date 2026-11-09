from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

SKILL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_learning_fixture import write_fixture
from vehicle_cd_learning import evaluate as evaluation_module
from vehicle_cd_learning.data import PARAMETER_NAMES, load_dataset
from vehicle_cd_learning.evaluate import evaluate
from vehicle_cd_learning.model import CoefficientModel
from vehicle_cd_learning.prepare import prepare


def _prepared(root: Path) -> Path:
    manifest = write_fixture(root / "source")
    prepared = root / "prepared"
    prepare(manifest, prepared, {"ordinary_size": 32, "bottom_size": 64})
    return prepared


def _checkpoint(prepared: Path, output: Path) -> Path:
    dataset = load_dataset(prepared)
    training = [case for case in dataset["cases"] if case["split"] == "train"]
    model = CoefficientModel(5, variant="five", target_count=1, size="small")
    checkpoint = {
        "model_config": {
            "parameter_count": 5,
            "variant": "five",
            "target_count": 1,
            "size": "small",
        },
        "model_state": model.state_dict(),
        "parameter_names": PARAMETER_NAMES,
        "target_names": ["cd"],
        "parameter_mean": [0.0] * 5,
        "parameter_std": [1.0] * 5,
        "target_mean": [10.0],
        "target_std": [0.1],
        "parameter_ranges": {
            "width_ratio": [0.0, 1.0],
            "height_ratio": [0.0, 1.0],
            "log_re": [float(np.log(1e6)), float(np.log(1e6))],
            "yaw_sin": [-1.0, 1.0],
            "yaw_cos": [-1.0, 1.0],
        },
        "render_config": dataset["render_config"],
        "train_geometry_hashes": sorted({case["geometry_hash"] for case in training}),
        "dataset_sha256": hashlib.sha256(
            (prepared / "dataset.json").read_bytes()
        ).hexdigest(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    return output


def test_diagnostic_evaluates_condition_extrapolations_and_records_strict_rejections(
    tmp_path,
):
    prepared = _prepared(tmp_path)
    checkpoint = _checkpoint(prepared, tmp_path / "checkpoint.pt")

    with pytest.raises(ValueError, match="Unsupported condition: log_re"):
        evaluate(checkpoint, prepared, tmp_path / "strict")

    report = evaluate(checkpoint, prepared, tmp_path / "diagnostic", diagnostic=True)
    assert report["prediction_mode"] == "raw_diagnostic"
    assert report["count"] == 8
    assert report["default_rejection_count"] == 4
    assert sum(case["default_rejection"] is not None for case in report["cases"]) == 4
    assert all(np.isfinite(case["prediction"]).all() for case in report["cases"])
    assert "raw model forward" in report["inference_timing_scope"]


def test_diagnostic_still_rejects_corrupt_view_data(tmp_path):
    prepared = _prepared(tmp_path)
    dataset = load_dataset(prepared)
    test_case = next(case for case in dataset["cases"] if case["split"] == "test")
    view_path = prepared / test_case["views"]
    with np.load(view_path, allow_pickle=False) as archive:
        views = {name: archive[name] for name in archive.files}
    views["front"] = views["front"].copy()
    views["front"][1, 0, 0] = 0.5
    np.savez_compressed(view_path, **views)
    digest = hashlib.sha256(view_path.read_bytes()).hexdigest()
    for case in dataset["cases"]:
        if case["views"] == test_case["views"]:
            case["views_sha256"] = digest
    (prepared / "dataset.json").write_text(json.dumps(dataset, indent=2) + "\n")
    checkpoint = _checkpoint(prepared, tmp_path / "checkpoint.pt")

    with pytest.raises(ValueError, match="mask must be binary"):
        evaluate(checkpoint, prepared, tmp_path / "diagnostic", diagnostic=True)


def test_diagnostic_rejects_nonfinite_raw_model_predictions(tmp_path, monkeypatch):
    prepared = _prepared(tmp_path)
    checkpoint_path = _checkpoint(prepared, tmp_path / "checkpoint.pt")
    model = CoefficientModel(5, variant="five", target_count=1, size="small")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.forward = lambda _views, _parameters: torch.full((1, 1), float("nan"))
    monkeypatch.setattr(
        evaluation_module, "load_model", lambda _path: (model.eval(), checkpoint)
    )

    with pytest.raises(ValueError, match="Nonfinite raw model output"):
        evaluate(checkpoint_path, prepared, tmp_path / "diagnostic", diagnostic=True)


def test_paired_evaluation_tolerates_stl_rounding_but_rejects_condition_changes(
    tmp_path,
    monkeypatch,
):
    prepared = _prepared(tmp_path)
    dataset = load_dataset(prepared)
    test_cases = [case for case in dataset["cases"] if case["split"] == "test"]
    first = test_cases[0]
    second = next(
        case for case in test_cases[1:] if case["pair_id"] == first["pair_id"]
    )
    initial_reynolds = second["reynolds"]

    def run(name):
        (prepared / "dataset.json").write_text(json.dumps(dataset, indent=2) + "\n")
        checkpoint = _checkpoint(prepared, tmp_path / f"{name}.pt")
        return evaluate(checkpoint, prepared, tmp_path / name, diagnostic=True)

    second["reynolds"] = initial_reynolds * (1 + 3e-8)
    second["parameters"][2] = float(np.log(second["reynolds"]))
    report = run("rounding")
    assert report["paired_changes"]["cd"]["observed_count"] == 4
    assert sum(v["cd"]["observed_count"] for v in report["per_condition"].values()) == 8

    second["reynolds"] = initial_reynolds * (1 + 1e-4)
    second["parameters"][2] = float(np.log(second["reynolds"]))
    with pytest.raises(ValueError, match="same conditions"):
        run("different-reynolds")

    second["reynolds"] = initial_reynolds
    second["parameters"][2] = float(np.log(initial_reynolds))
    second["yaw_deg"] = first["yaw_deg"] + 1
    with pytest.raises(ValueError, match="same conditions"):
        run("different-yaw")

    second["yaw_deg"] = first["yaw_deg"]
    (prepared / "dataset.json").write_text(json.dumps(dataset, indent=2) + "\n")
    checkpoint = _checkpoint(prepared, tmp_path / "profile.pt")
    second["condition_profile"] = "different-profile"
    monkeypatch.setattr(evaluation_module, "load_dataset", lambda _: dataset)
    with pytest.raises(ValueError, match="cannot cross condition profiles"):
        evaluate(checkpoint, prepared, tmp_path / "different-profile", diagnostic=True)
