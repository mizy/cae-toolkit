import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest
import torch

SKILL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL / "scripts"))
sys.path.insert(0, str(SKILL.parent / "predict-aircraft-coefficients/scripts"))

from vehicle_cd_learning import joint_sampling_train as training
from vehicle_cd_learning.joint_sampling import AbsoluteReadout
from vehicle_cd_learning.model import VIEWS
from vehicle_cd_learning.paired_delta import PairedDelta

WEIGHTS = {
    "absolute_nominal": 0.25,
    "absolute_augmented": 0.25,
    "absolute_consistency": 0.1,
    "rim_nominal": 1,
    "rim_augmented": 0.5,
    "rim_self": 0.25,
    "rim_consistency": 0.25,
    "rim_teacher": 1,
}


def test_absolute_loss_masks_real_labels_weights_cases_and_detaches_nominal():
    prediction = torch.tensor(
        [[1.0, 9.0], [2.0, 3.0], [4.0, 8.0], [5.0, 6.0]], requires_grad=True
    )
    target = torch.tensor([[2.0, float("nan")], [3.0, 4.0]])
    weights = torch.tensor([2.0, 0.5])
    parts = training.absolute_objective(prediction, target, weights)
    expected = {
        "absolute_nominal": torch.tensor((2 * 1 + 0.5 * 1) / 2),
        "absolute_augmented": torch.tensor((2 * 4 + 0.5 * 4) / 2),
        "absolute_consistency": torch.tensor((2 * 9 + 0.5 * 9) / 2),
    }
    for key, value in expected.items():
        torch.testing.assert_close(parts[key], value)
    parts["absolute_consistency"].backward()
    assert (prediction.grad[:2] == 0).all()
    assert prediction.grad[2, 1] == 0
    assert torch.isfinite(prediction.grad).all()


class Difference(torch.nn.Module):
    def predict_pair(self, a, b, parameters):
        return b["value"] - a["value"]


def test_rim_objective_matches_three_shifted_patterns_and_teacher_with_detach():
    values = torch.tensor(
        [[1.0, 2.0], [4.0, 8.0], [2.0, 4.0], [6.0, 12.0]], requires_grad=True
    )
    nominal = values[1:2] - values[:1]
    augmented = torch.cat(
        (values[3:] - values[2:3], values[3:] - values[:1], values[1:2] - values[2:3])
    )
    targets, scale, teacher = (
        torch.tensor([[0.1, 0.2], [0.4, 0.8]]),
        torch.tensor([0.1, 0.2]),
        torch.zeros(1, 2),
    )
    parts = training.rim_objective(
        Difference(), {"value": values}, torch.zeros(2, 36), targets, scale, teacher
    )
    target = (targets[1:] - targets[:1]) / scale
    expected = {
        "rim_nominal": (nominal - target).square().mean(),
        "rim_augmented": (augmented - target).square().mean(),
        "rim_self": (values[2:] - values[:2]).square().mean(),
        "rim_consistency": (augmented - nominal.detach()).square().mean(),
        "rim_teacher": nominal.square().mean(),
    }
    for key, value in expected.items():
        torch.testing.assert_close(parts[key], value)
    reference_grad = torch.autograd.grad(
        expected["rim_consistency"], values, retain_graph=True
    )[0]
    actual_grad = torch.autograd.grad(parts["rim_consistency"], values)[0]
    torch.testing.assert_close(actual_grad, reference_grad)


class TinyFull:
    reads: ClassVar[list] = []

    def __init__(self, p, cases):
        self.cases, self.cache = cases, {}
        self.index = {c["case_id"]: i for i, c in enumerate(cases)}
        self.parameters = torch.zeros(len(cases), 36)
        self.standardized_targets = torch.tensor([c["targets"] for c in cases])
        self.weights = torch.tensor(
            [p["source_case_weights"][c["source"]] for c in cases]
        )

    def get(self, index):
        self.reads.append(index)
        image = torch.zeros(2, 8, 8)
        image[0, 2:6, 2:6] = 0.2 + 0.01 * index
        image[1, 2:6, 2:6] = 1
        return (
            {n: image.clone() for n in VIEWS},
            self.parameters[index],
            self.standardized_targets[index],
        )


class TinyRim(TinyFull):
    def __init__(self, p, cases, with_phases=False):
        assert with_phases is False
        super().__init__(p, cases)

    def get(self, index, phase=None):
        assert phase is None
        return super().get(index)


def test_control_feature_reuse_matches_duplicate_cnn_losses_and_gradients():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    model, readout = PairedDelta(8, 36), AbsoluteReadout()
    assert sum(v.numel() for v in model.parameters()) == 54506
    duplicate, duplicate_readout = copy.deepcopy(model), copy.deepcopy(readout)
    cases = [
        {"case_id": str(i), "targets": [0.2 + 0.1 * i, -0.1], "source": "a"}
        for i in range(2)
    ]
    data = TinyFull({"source_case_weights": {"a": 1}}, cases)
    rim = TinyRim({"source_case_weights": {"a": 1}}, cases)
    results = []
    for pair, absolute, online in (
        (model, readout, False),
        (duplicate, duplicate_readout, True),
    ):
        parts = training.full_backward(
            pair,
            absolute,
            data,
            np.array([0, 1]),
            torch.zeros(2, 2),
            "cpu",
            WEIGHTS,
            online,
        )
        parts.update(
            training.rim_backward(
                pair,
                rim,
                np.array([0]),
                np.array([1]),
                np.zeros((1, 2, 2), np.float32),
                np.zeros((1, 2), np.float32),
                "cpu",
                {"delta_std": [0.1, 0.2], "loss_weights": WEIGHTS},
                online,
            )
        )
        results.append(parts)
    for key in results[0]:
        assert results[0][key] == pytest.approx(results[1][key], rel=1e-4, abs=1e-6)
    for one, two in zip(
        [*model.parameters(), *readout.parameters()],
        [*duplicate.parameters(), *duplicate_readout.parameters()],
    ):
        assert one.grad is not None and two.grad is not None
        torch.testing.assert_close(one.grad, two.grad, rtol=2e-4, atol=2e-6)


class TinyPair(torch.nn.Module):
    encoded: ClassVar[list] = []

    def __init__(self, *args):
        super().__init__()
        self.image_encoder, self.bottom_encoder = (
            torch.nn.Linear(2, 2),
            torch.nn.Linear(2, 2),
        )
        self.head = torch.nn.Linear(2, 2, bias=False)

    def encode(self, views):
        self.encoded.append(len(views["front"]))
        return {
            n: (self.bottom_encoder if n == "bottom" else self.image_encoder)(
                v.mean((2, 3))
            )
            for n, v in views.items()
        }

    def predict_pair(self, a, b, parameters):
        return self.head(b["front"] - a["front"])


class TinyReadout(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head = torch.nn.Linear(2, 2)

    def forward(self, pooled, parameters):
        return self.head(sum(pooled[n] for n in VIEWS))


@pytest.fixture
def frozen_tiny(monkeypatch, tmp_path):
    monkeypatch.setattr(training, "PairedDelta", TinyPair)
    monkeypatch.setattr(training, "AbsoluteReadout", TinyReadout)
    monkeypatch.setattr(training, "FullCases", TinyFull)
    monkeypatch.setattr(training, "SamplingCases", TinyRim)
    TinyFull.reads, TinyPair.encoded = [], []
    cases = [
        {
            "case_id": str(i),
            "targets": [0.2 + 0.1 * i, -0.1],
            "source": "a",
            "split": "train" if i < 3 else "val",
            "geometry_hash": "shared" if i < 2 else str(i),
            "views_sha256": "shared" if i < 2 else str(i),
        }
        for i in range(4)
    ]
    parent = tmp_path / "parent"
    parent.mkdir()
    order = np.tile(np.arange(3, dtype=np.int32), (6, 1))
    left = np.tile(np.arange(2, dtype=np.int32), (6, 1))
    right = left[:, ::-1].copy()
    np.savez(
        tmp_path / "draws.npz",
        full_order=order,
        full_offsets=np.full((6, 3, 2), 0.25, np.float32),
        left=left,
        right=right,
        rim_offsets=np.full((6, 2, 2, 2), -0.25, np.float32),
    )
    np.savez(parent / "draws.npz", left=left, right=right)
    np.savez(
        parent / "teacher.npz",
        prediction=np.zeros((6, 2, 2), np.float32),
        encoded_train_indices=np.arange(2),
    )
    pair, readout = TinyPair(), TinyReadout()
    checkpoint = tmp_path / "original.pt"
    torch.save(
        {"model_state": pair.state_dict(), "original_field": "preserved"}, checkpoint
    )
    p = {
        "experiment": str(tmp_path),
        "checkpoint": str(checkpoint),
        "sampling_parent": str(parent),
        "seed": 42,
        "epochs": 6,
        "warm_epochs": 20,
        "batch_size": 2,
        "training_steps": 12,
        "train_ids": ["0", "1", "2"],
        "arms": ["control", "online"],
        "source_case_weights": {"a": 1},
        "normalization": {"target_mean": [0, 0], "target_std": [1, 1]},
        "delta_std": [0.1, 0.2],
        "parameter_mean": [0] * 36,
        "parameter_std": [1] * 36,
        "source_sha256": {},
        "initial_tensor_sha256": training.tensor_sha256(pair),
        "initial_absolute_sha256": training.tensor_sha256(readout),
        "full_trace_sha256": hashlib.sha256(order.tobytes()).hexdigest(),
        "pair_trace_sha256": hashlib.sha256(
            np.stack((left, right), -1).tobytes()
        ).hexdigest(),
        "cnn_lr": 5e-5,
        "head_lr": 3e-4,
        "clip_norm": 10,
        "loss_weights": WEIGHTS,
    }
    training.save_checkpoint(
        tmp_path / "initial.pt",
        pair,
        readout,
        p,
        "original_B",
        0,
        torch.load(checkpoint, weights_only=True),
    )
    (tmp_path / "protocol.json").write_text(json.dumps(p))
    rim_cases = [
        {**c, "split": "train" if i < 2 else "test"} for i, c in enumerate(cases)
    ]
    monkeypatch.setattr(training, "load", lambda root: (p, cases))
    monkeypatch.setattr(training, "load_sampling", lambda root: (p, rim_cases))
    return tmp_path, p, cases, rim_cases


def test_warm_start_caches_unique_training_geometry_once_and_keeps_B(frozen_tiny):
    root, p, cases, _ = frozen_tiny
    (root / "initial.pt").unlink()
    training.warm_start(p, cases)
    warm = json.loads((root / "warmup.json").read_text())
    assert warm["encoded_unique_geometry_count"] == 2
    assert warm["train_record_count"] == 3 and len(warm["history"]) == 20
    assert sorted(TinyFull.reads) == [1, 2]  # One representative of the shared views.
    assert TinyPair.encoded == [2]
    initial = torch.load(root / "initial.pt", weights_only=True)
    model = TinyPair()
    model.load_state_dict(initial["model_state"])
    assert training.tensor_sha256(model) == p["initial_tensor_sha256"]
    assert initial["arm"] == "original_B" and initial["epoch"] == 0
    assert initial["original_field"] == "preserved"
    with pytest.raises(FileExistsError):
        training.warm_start(p, cases)


def test_both_arms_execute_exact_records_pairs_steps_and_separate_compute(frozen_tiny):
    root, p, _, _ = frozen_tiny
    reports = {arm: training.train(root, arm) for arm in p["arms"]}
    for arm, report in reports.items():
        assert report["optimizer_steps"] == 12
        assert len(report["history"]) == 6
        assert report["executed_full_trace_sha256"] == p["full_trace_sha256"]
        assert report["executed_pair_sha256"] == p["pair_trace_sha256"]
        assert report["full_case_exposures"] == {"0": 6, "1": 6, "2": 6}
        assert report["rim_endpoint_exposures"] == {"0": 12, "1": 12}
        assert report["perturbed_full_case_exposures"] == {
            i: 6 if arm == "online" else 0 for i in ("0", "1", "2")
        }
        assert report["rim_augmented_endpoint_exposures"] == {
            i: 12 if arm == "online" else 0 for i in ("0", "1")
        }
        assert report["logical_image_exposures"] == {"full": 36, "rim": 48}
        assert report["actual_cnn_image_encodings"] == (
            {"full": 36, "rim": 48} if arm == "online" else {"full": 18, "rim": 24}
        )
        assert len(list((root / arm).glob("epoch-*.pt"))) == 7
        final = torch.load(root / arm / "epoch-06.pt", weights_only=True)
        assert final["epoch"] == 6 and final["arm"] == arm
        assert final["protocol_sha256"] == training.sha256(root / "protocol.json")
        assert final["original_field"] == "preserved"
        with pytest.raises(FileExistsError):
            training.train(root, arm)
    assert (
        reports["control"]["actual_offset_trace_sha256"]
        != reports["online"]["actual_offset_trace_sha256"]
    )
    assert 3 not in TinyFull.reads


def test_schedule_rejects_teacher_pair_or_heldout_changes(frozen_tiny):
    root, p, cases, rim_cases = frozen_tiny
    draws, _ = training.training_draws(root, p, cases, rim_cases)
    draws["left"][0, 0] = 3
    np.savez(root / "draws.npz", **draws)
    with pytest.raises(ValueError, match="pair schedule changed"):
        training.training_draws(root, p, cases, rim_cases)
