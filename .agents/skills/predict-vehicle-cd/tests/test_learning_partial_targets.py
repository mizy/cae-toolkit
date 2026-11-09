from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

SKILL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL / "scripts"))

from vehicle_cd_learning.data import PARAMETER_NAMES, CaseDataset, load_dataset


def write_dataset(root: Path, *, profiled=True, targets=None, parameters=None):
    root.mkdir(parents=True, exist_ok=True)
    views = root / "views"
    views.mkdir()
    image = np.zeros((2, 4, 4), dtype=np.float32)
    image[0, 1, 1] = 0.5
    image[1, 1, 1] = 1.0
    view_path = views / "case.npz"
    np.savez_compressed(
        view_path,
        **{name: image for name in ("front", "rear", "right", "top", "bottom")},
    )

    profiles = {
        "built_in": {
            "conditions": {"ground": {"value": "stationary floor", "assumed": True}},
            "provenance": {"source": "built-in coefficient library", "assumed": True},
            "data_kind": "reference_assumed",
            "labeled_targets": ["cd"],
        },
        "published": {
            "conditions": {"ground": {"value": "wind tunnel floor", "assumed": False}},
            "provenance": {"source": "published CFD", "assumed": False},
            "data_kind": "cfd",
            "labeled_targets": ["cd", "cl"],
        },
    }
    parameter_names = PARAMETER_NAMES + [f"profile_{name}" for name in profiles]
    if parameters is None:
        parameters = [0.4, 0.3, 14.0, 0.0, 1.0, 1.0, 0.0]
    if targets is None:
        targets = [0.31, None]
    case = {
        "case_id": "case-1",
        "geometry_id": "geometry-1",
        "family_id": "family-1",
        "geometry_hash": "hash-1",
        "split": "train",
        "views": "views/case.npz",
        "views_sha256": hashlib.sha256(view_path.read_bytes()).hexdigest(),
        "parameters": parameters,
        "targets": targets,
    }
    dataset = {
        "schema_version": 1,
        "parameter_names": parameter_names if profiled else PARAMETER_NAMES,
        "target_names": ["cd", "cl"],
        "cases": [case],
    }
    if profiled:
        dataset["condition_profiles"] = profiles
        case["condition_profile"] = "built_in"
    (root / "dataset.json").write_text(json.dumps(dataset))
    return root


def test_profiled_case_preserves_cd_and_marks_missing_cl_nan(tmp_path):
    root = write_dataset(tmp_path / "profiled")
    dataset = load_dataset(root)
    case = dataset["cases"][0]
    torch_dataset = CaseDataset(
        root,
        [case],
        parameter_mean=np.zeros(len(dataset["parameter_names"])),
        parameter_std=np.ones(len(dataset["parameter_names"])),
        target_mean=np.zeros(2),
        target_std=np.ones(2),
        variant="parameters",
    )

    views, parameters, targets = torch_dataset[0]

    assert views == {}
    assert parameters.shape == (len(PARAMETER_NAMES) + 2,)
    assert parameters[-2:].tolist() == [1.0, 0.0]
    assert targets[0].item() == pytest.approx(0.31)
    assert np.isnan(targets[1].item())


def test_profile_one_hot_must_match_case_profile(tmp_path):
    root = write_dataset(
        tmp_path / "wrong-profile",
        parameters=[0.4, 0.3, 14.0, 0.0, 1.0, 0.0, 1.0],
    )

    with pytest.raises(ValueError, match="one-hot"):
        load_dataset(root)


def test_cfd_profile_allows_missing_cl_when_cl_is_declared(tmp_path):
    root = write_dataset(tmp_path / "partial-cfd")
    manifest_path = root / "dataset.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["cases"][0]["condition_profile"] = "published"
    manifest["cases"][0]["parameters"][-2:] = [0.0, 1.0]
    manifest_path.write_text(json.dumps(manifest))

    assert load_dataset(root)["cases"][0]["targets"] == [0.31, None]


def test_case_cannot_include_cl_when_profile_declares_cd_only(tmp_path):
    root = write_dataset(tmp_path / "undeclared-cl", targets=[0.31, 0.02])

    with pytest.raises(ValueError, match="not declared"):
        load_dataset(root)


@pytest.mark.parametrize("bad_cl", [float("nan"), float("inf")])
def test_nonfinite_cl_is_not_treated_as_missing_for_cfd_profile(tmp_path, bad_cl):
    root = write_dataset(tmp_path / "nonfinite-cfd-cl", targets=[0.31, bad_cl])
    manifest_path = root / "dataset.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["cases"][0]["condition_profile"] = "published"
    manifest["cases"][0]["parameters"][-2:] = [0.0, 1.0]
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="Targets must be finite"):
        load_dataset(root)


@pytest.mark.parametrize(
    ("profiled", "targets", "error"),
    [
        (True, [None, 0.02], "Cd target is required"),
        (False, [0.31, None], "missing Cl"),
    ],
)
def test_partial_targets_require_finite_cd_and_declared_profiles(
    tmp_path, profiled, targets, error
):
    root = write_dataset(
        tmp_path / f"dataset-{profiled}-{targets[0]}",
        profiled=profiled,
        targets=targets,
        parameters=(
            [0.4, 0.3, 14.0, 0.0, 1.0, 1.0, 0.0]
            if profiled
            else [0.4, 0.3, 14.0, 0.0, 1.0]
        ),
    )

    with pytest.raises(ValueError, match=error):
        load_dataset(root)
