import sys
from pathlib import Path

import numpy as np
import torch

SKILL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL / "scripts"))
sys.path.insert(0, str(SKILL.parent / "predict-aircraft-coefficients/scripts"))

from aircraft_learning.scene_experiment import SceneCases
from vehicle_cd_learning.geometry_diagnostics import pairwise_ranking
from vehicle_cd_learning.geometry_experiment import (
    select_cases,
    source_loss_weights,
    train_epoch,
)
from vehicle_cd_learning.point_baseline import (
    PointCases,
    make_model,
    shared_initial_hash,
)
from vehicle_cd_learning.prepare_points import sample_surface, surface_cdf


def test_surface_sampling_respects_area_and_preserves_component_locations():
    first = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32)
    second = np.array([[10, 0, 0], [12, 0, 0], [10, 2, 0]], np.float32)
    faces = np.array([[0, 1, 2]])
    parts = [surface_cdf(p, faces) for p in (first, second)]
    sample = sample_surface(parts, 8192)
    np.testing.assert_array_equal(sample, sample_surface(parts, 8192))
    assert abs(np.mean(sample[:, 0] < 2) - 0.2) < 0.001
    assert np.all(sample[:, 2] == 0)
    assert np.all(sample[sample[:, 0] < 2, :2].sum(1) <= 1.00001)
    assert np.all((sample[sample[:, 0] > 2, :2] - [10, 0]).sum(1) <= 2.00001)


def test_point_baseline_is_permutation_invariant_and_uses_matched_head():
    torch.set_num_threads(2)
    image, point = make_model("views"), make_model("points")
    assert shared_initial_hash(image) == shared_initial_hash(point)
    point.eval()
    x, conditions = torch.randn(2, 3, 73), torch.randn(2, 40)
    with torch.no_grad():
        first = point({"points": x}, conditions)
        second = point({"points": x[:, :, torch.randperm(73)]}, conditions)
    torch.testing.assert_close(first, second, atol=1e-6, rtol=1e-6)
    assert first.shape == (2, 2)


def test_point_and_view_cases_have_identical_condition_and_target_masks(tmp_path):
    arrays = {
        v: np.zeros((2, 16, 16), np.float32)
        for v in ("front", "rear", "right", "top", "bottom")
    }
    np.savez(tmp_path / "views.npz", **arrays)
    np.savez(tmp_path / "points.npz", points=np.zeros((20, 3), np.float32))
    case = {
        "case_id": "example",
        "views": "views.npz",
        "scene": 1,
        "parameters": [0.4, 0.3, 1.0] + [None] * 17,
        "targets": [0.25, None],
    }
    stats = {
        "parameter_mean": [0.0] * 40,
        "parameter_std": [1.0] * 40,
        "target_mean": [0.3, 0.1],
        "target_std": [0.1, 0.2],
    }
    view = SceneCases(tmp_path, [case], stats)[0]
    point = PointCases(tmp_path, [case], stats, {"example": {"path": "points.npz"}})[0]
    for a, b in zip(view[1:], point[1:]):
        torch.testing.assert_close(a, b, equal_nan=True)
    assert int(point[1][20:].sum()) == 3 and torch.isnan(point[2][1])


def test_source_variance_loss_uses_train_variation_without_missing_cl():
    cases = [
        {"source": s, "targets": [y, None]}
        for s, y in (("wide", 0.2), ("wide", 0.4), ("narrow", 0.299), ("narrow", 0.301))
    ]
    weights = source_loss_weights(cases, {"target_std": [0.1, 0.2]}, True)
    assert weights[2, 0] > 9000 * weights[0, 0]
    assert np.isfinite(weights).all()
    values = [
        ({}, torch.zeros(40), torch.tensor([y, float("nan")]), torch.tensor(1))
        for y in (0.1, 0.2, 0.3, 0.4)
    ]
    model = make_model("parameters")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    loss = train_epoch(
        model, values, np.arange(4), weights, optimizer, torch.device("cpu"), 4
    )
    assert np.isfinite(loss)
    assert all(torch.isfinite(p).all() for p in model.parameters())


def test_source_holdout_removes_all_ahmed_training_and_assumed_car_labels():
    cases = [
        {"case_id": f"{s}-{split}", "source": s, "split": split}
        for s in ("drivaerml", "ahmedml", "builtin_assumed")
        for split in ("train", "val", "test")
    ]
    held = select_cases(
        {"cases": cases}, {"fold": "ahmed_holdout", "physics": "declared"}, {}
    )
    assert {c["case_id"] for c in held} == {
        "drivaerml-train",
        "drivaerml-val",
        "ahmedml-test",
    }


def test_scale_ablation_masks_only_dimensional_inputs_without_changing_truth():
    original = {
        "case_id": "car",
        "source": "drivaerml",
        "split": "train",
        "parameters": list(range(20)),
        "targets": [0.3, 0.1],
        "conditions": {"velocity_m_s": 40},
    }
    result = select_cases(
        {"cases": [original]}, {"fold": "primary", "physics": "dimensionless"}, {}
    )[0]
    assert result["parameters"][2:4] == [None, None]
    assert result["parameters"][:2] == [0, 1] and result["parameters"][4:] == list(
        range(4, 20)
    )
    assert original["parameters"] == list(range(20))
    assert (
        result["targets"] == original["targets"]
        and result["conditions"] == original["conditions"]
    )


def test_pairwise_ordering_reports_wrong_directions_and_true_ties():
    rows = [
        {"source": "car", "split": "test", "truth": [a, 0], "prediction": [b, 0]}
        for a, b in ((0.2, 0.3), (0.21, 0.29), (0.22, 0.28))
    ]
    result = pairwise_ranking(rows)["car"]["0.003"]
    assert result["pairs"] == 3 and result["accuracy"] == 0
    assert result["ties"] == 0
    for row in rows:
        row["prediction"][0] = row["truth"][0]
    assert pairwise_ranking(rows)["car"]["0.003"]["accuracy"] == 1
