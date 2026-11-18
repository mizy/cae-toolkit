import itertools
import sys
from pathlib import Path

import numpy as np
import torch

SKILL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL / "scripts"))
sys.path.insert(0, str(SKILL.parent / "predict-aircraft-coefficients/scripts"))

from vehicle_cd_learning.absolute_view_audit import (
    detail_intervals,
    predict_grid,
    project,
    view_metrics,
)
from vehicle_cd_learning.joint_sampling import AbsoluteReadout
from vehicle_cd_learning.model import VIEWS
from vehicle_cd_learning.paired_delta import PairedDelta
from vehicle_cd_learning.rim_reference_coefficients import estimate


def test_independent_combinations_agree_with_full_five_view_model():
    torch.manual_seed(5)
    model, readout = PairedDelta(8).eval(), AbsoluteReadout().eval()
    views = [{n: torch.rand(2, 2, 48, 48) for n in VIEWS} for _ in range(3)]
    for variant in views:
        for image in variant.values():
            image[:, 1] = (image[:, 1] > 0.5).float()
            image[:, 0] *= image[:, 1]
    parameters = torch.randn(2, 36)
    grid = np.array([[2, 1, 0, 2, 0], [0, 2, 1, 0, 2], [0, 0, 0, 0, 0]])
    stats = {"target_mean": [0.25, -0.2], "target_std": [0.04, 0.17]}
    with torch.inference_mode():
        embeddings = torch.stack([project(model, readout, v) for v in views], dim=1)
        predictions = predict_grid(readout, embeddings, parameters, grid, stats)
        for index, pattern in enumerate(grid):
            mixed = {n: views[s][n] for n, s in zip(VIEWS, pattern)}
            expected = (
                readout(model.encode(mixed), parameters).numpy() * stats["target_std"]
                + stats["target_mean"]
            )
            np.testing.assert_allclose(
                predictions[:, index], expected, rtol=0, atol=5e-7
            )


def test_tied_phases_can_hide_opposing_individual_view_effects():
    grid = np.array(list(itertools.product(range(2), repeat=5)))
    output = (grid[:, 0] - grid[:, 1])[None, :, None] * np.array([1.0, 2.0])
    score = view_metrics(output, np.zeros((1, 2)), grid)
    assert score["mean_case_worst_common_drift"] == [0, 0]
    assert score["mean_case_worst_independent_drift"] == [1, 2]
    assert score["single_view"]["front"]["mean_case_worst_drift"] == [1, 2]


def test_ordering_interval_matches_all_joint_endpoint_choices_and_excludes_ties():
    y = np.array([[0.0, 0.0], [1.0, 0.4], [1.0, 0.7]])
    predictions = np.array(
        [[[0.1, 0.1], [0.8, 0.6]], [[0.25, 0.8], [0.30, 0.9]], [[1.5, 0.6], [1.4, 0.2]]]
    )
    pairs = list(itertools.combinations(range(3), 2))
    bound = detail_intervals(
        y, predictions[:, 0], predictions.min(1), predictions.max(1), pairs
    )
    for j, target in enumerate(("Cd", "Cl")):
        brute = []
        for i, k in pairs:
            signal = y[k, j] - y[i, j]
            samples = [
                v[j] - u[j]
                for u, v in itertools.product(predictions[i], predictions[k])
            ]
            brute.append((signal, np.array(samples)))
        assert bound[target]["always_correct"] == sum(
            s != 0 and np.all(d * np.sign(s) > 5e-6) for s, d in brute
        )
        assert bound[target]["can_reverse"] == sum(
            s != 0 and min(d) < -5e-6 and max(d) > 5e-6 for s, d in brute
        )
        assert np.isclose(
            bound[target]["worst_delta_mae"],
            np.mean([np.max(np.abs(d - s)) for s, d in brute]),
        )


def test_reference_offset_preserves_signed_change_even_if_absolute_bias_moves():
    refs = np.array([[0.25, 0.15], [0.35, -0.15]])
    delta = np.array([[[0.01, 0.02], [0.03, 0.04]], [[-0.05, 0.03], [-0.03, 0.05]]])
    nominal = estimate(delta, refs, 0)
    calibrated = estimate(delta, refs, [0.005, -0.02])
    np.testing.assert_allclose(
        np.diff(nominal, axis=0), np.diff(calibrated, axis=0), rtol=0, atol=1e-14
    )
    assert not np.allclose(calibrated, nominal)
