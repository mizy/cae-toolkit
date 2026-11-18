import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from vehicle_cd_learning import sampling_experiment as sampling
from vehicle_cd_learning.model import VIEWS


def test_auxiliary_draws_are_shared_deterministic_training_only_and_distinct():
    selected = np.arange(32) * 2
    u, v, phase = sampling.auxiliary_schedule(selected)
    assert u.shape == v.shape == phase.shape == (6, 90)
    assert np.all(u != v)
    assert set(u.ravel()) == set(v.ravel()) == set(selected)
    assert set(phase.ravel()) == {0, 1, 2, 3}
    for first, second in zip((u, v, phase), sampling.auxiliary_schedule(selected)):
        np.testing.assert_array_equal(first, second)
    trace = hashlib.sha256()
    for values in zip(u.ravel(), v.ravel(), phase.ravel()):
        trace.update(np.array(values, np.int32).tobytes())
    assert trace.hexdigest() == sampling.trace_sha(u, v, phase)


class Difference(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def encode(self, views):
        return {"value": views["front"][:, :, 0, 0]}

    def predict_pair(self, a, b, parameters):
        return (b["value"] - a["value"]) * self.weight


def test_twenty_image_loss_matches_all_five_terms_and_has_finite_gradients():
    model = Difference()
    encoded = {"value": torch.arange(40, dtype=torch.float32).reshape(20, 2).requires_grad_()}
    targets = torch.arange(40, dtype=torch.float32).reshape(20, 2) * 0.1
    parameters, scale = torch.zeros(20, 36), torch.tensor([2.0, 3.0])
    teacher = torch.full((8, 2), 0.3)
    total, parts = sampling.train_objective(model, encoded, parameters, targets, scale, teacher)
    real = encoded["value"][8:16] - encoded["value"][:8]
    auxiliary = torch.stack(
        (
            encoded["value"][19] - encoded["value"][18],
            encoded["value"][19] - encoded["value"][16],
            encoded["value"][17] - encoded["value"][18],
        )
    )
    nominal = encoded["value"][17:18] - encoded["value"][16:17]
    self_pairs = encoded["value"][18:20] - encoded["value"][16:18]
    expected = {
        "real": ((real - (targets[8:16] - targets[:8]) / scale) ** 2).mean(),
        "aux_phase": ((auxiliary - (targets[17:18] - targets[16:17]) / scale) ** 2).mean(),
        "self": self_pairs.square().mean(),
        "consistency": (auxiliary - nominal).square().mean(),
        "teacher": (real - teacher).square().mean(),
    }
    assert (auxiliary[0] - nominal[0]).square().sum() == 0
    assert parts["consistency"] > 0  # Both mixed predictions contribute despite exact common-phase parity.
    for key, value in expected.items():
        torch.testing.assert_close(parts[key], value)
    torch.testing.assert_close(
        total,
        expected["real"] + expected["teacher"] + 0.5 * sum(expected[k] for k in ("aux_phase", "self", "consistency")),
    )
    total.backward()
    assert torch.isfinite(model.weight.grad)
    assert torch.isfinite(encoded["value"].grad).all()


def test_control_duplicates_have_exact_zero_self_and_consistency_loss():
    values = torch.randn(18, 2)
    encoded = {"value": torch.cat((values, values[16:18]))}
    _, parts = sampling.train_objective(
        Difference(), encoded, torch.zeros(20, 36), torch.zeros(20, 2), torch.ones(2), torch.zeros(8, 2)
    )
    assert parts["self"].item() == parts["consistency"].item() == 0


def test_teacher_encodes_only_unique_training_images_once(monkeypatch, tmp_path):
    cases = [{"case_id": f"case-{i}", "split": "train" if i < 720 else "test"} for i in range(722)]
    seen = []

    class Nominal:
        def __init__(self, *args):
            self.parameters = torch.zeros(722, 36)

        def __getitem__(self, index):
            seen.append(index)
            image = torch.tensor([[[float(index)]], [[float(index)]]])
            return {n: image for n in VIEWS}, self.parameters[index], torch.zeros(2)

    monkeypatch.setattr(sampling.delta, "DeltaCases", Nominal)
    monkeypatch.setattr(sampling, "initialize_from_checkpoint", lambda *a: Difference())
    left = np.tile(np.arange(720), (6, 1))
    right = np.roll(left, 1, axis=1)
    evidence = sampling.teacher_predictions({}, cases, {}, left, right, tmp_path / "teacher.npz")
    assert sorted(seen) == list(range(720))
    assert evidence["encoded_geometry_count"] == 720
    with np.load(tmp_path / "teacher.npz") as teacher:
        np.testing.assert_array_equal(teacher["encoded_train_indices"], np.arange(720))
        expected = np.repeat((right - left)[..., None], 2, axis=-1)
        np.testing.assert_array_equal(teacher["prediction"], expected)


def test_control_data_and_batch_numeric_inputs_ignore_camera_phase(monkeypatch, tmp_path):
    class Nominal:
        def __init__(self, *args):
            self.index = {"source": 0}
            self.parameters, self.targets, self.delta_std = (
                torch.arange(36)[None],
                torch.tensor([[0.2, 0.3]]),
                np.array([0.1, 0.2]),
            )

        def __getitem__(self, index):
            return {n: torch.zeros(2, 512, 512) for n in VIEWS}, self.parameters[index], self.targets[index]

    monkeypatch.setattr(sampling.delta, "load", lambda *a: ({}, []))
    monkeypatch.setattr(sampling.delta, "DeltaCases", Nominal)
    data = sampling.SamplingCases(
        {"parent_experiment": str(tmp_path), "phase_root": str(tmp_path / "unfinished")}, [{}], with_phases=False
    )
    views, numeric, targets = sampling.batch_views(data, [(0, None), (0, None)], "cpu")
    assert data.nominal.limit == 256 and not data.records
    assert views["front"].shape == (2, 2, 512, 512)
    torch.testing.assert_close(numeric, torch.arange(36)[None].expand(2, -1))
    torch.testing.assert_close(targets, torch.tensor([[0.2, 0.3], [0.2, 0.3]]))


def test_checkpoint_metadata_is_weights_only_safe_and_floor_is_separate(tmp_path):
    p = {
        "experiment": str(tmp_path),
        "arms": {"coverage": {"coverage_floor": 1 / 16, "filter_input": True}},
        "delta_std": [0.1, 0.2],
        "parameter_mean": [0.0] * 36,
        "parameter_std": [1.0] * 36,
        "source_sha256": {"source": "sha"},
    }
    (tmp_path / "protocol.json").write_text(json.dumps(p))
    model = Difference()
    data = SimpleNamespace(data_lock_sha256="phase", nominal=SimpleNamespace(data_lock_sha256="nominal"))
    sampling.save_checkpoint(tmp_path / "epoch-00.pt", model, p, "coverage", 0, {"score": 0.1}, data)
    checkpoint = torch.load(tmp_path / "epoch-00.pt", map_location="cpu", weights_only=True)
    assert checkpoint["model_config"] == {"pool_size": 8, "parameter_count": 36}
    assert checkpoint["coverage_floor"] == 1 / 16
    assert checkpoint["epoch"] == 0
    assert checkpoint["source_sha256"] == p["source_sha256"]


def test_freeze_requires_fresh_output(tmp_path):
    with pytest.raises(FileExistsError):
        sampling.freeze(tmp_path / "missing", tmp_path / "missing", tmp_path)


@pytest.mark.parametrize("complete", [False, True])
def test_phase_binding_rejects_partial_or_incomplete_membership(tmp_path, complete):
    dataset = tmp_path / "dataset.json"
    dataset.write_text(json.dumps({"cases": []}))
    manifest = {
        "stage_complete": complete,
        "train_ready": complete,
        "item_count": 358,
        "dataset_sha256": sampling.sha256(dataset),
        "plan_sha256": "plan",
        "source_sha256": {},
        "code_sha256": {},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    data = object.__new__(sampling.SamplingCases)
    data.root = tmp_path
    with pytest.raises(ValueError, match="Complete locked|completion manifest"):
        data.bind_phases({"source_sha256": {str(tmp_path / "plan.json"): "plan"}})
