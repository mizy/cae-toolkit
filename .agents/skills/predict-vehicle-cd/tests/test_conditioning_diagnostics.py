import copy
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

SKILL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL / "scripts"))
sys.path.insert(0, str(SKILL.parent / "predict-aircraft-coefficients/scripts"))

from vehicle_cd_learning.conditioning_experiment import (
    cases_for,
    intervened_epoch,
    run_stats,
    supervised_loss,
    validation_scores,
)
from vehicle_cd_learning.detail_kernel import distances, kernel


class Toy(nn.Module):
    target_count = 2

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)

    def forward(self, geometry, parameters, scene=None):
        return self.linear(parameters)


def test_input_ablation_preserves_source_truth_and_reynolds():
    c = {
        "source": "drivaerml",
        "split": "train",
        "parameters": [0.4, 1.0, 3.0, 4.0, 5.0],
        "targets": [0.3, None],
        "conditions": {"velocity_m_s": 40},
    }
    d = {
        "cases": [c],
        "parameter_names": [
            "width",
            "height",
            "log_length_m",
            "log_velocity_m_s",
            "re",
        ],
    }
    r = cases_for(d, {"fold": "primary", "mask": ["log_velocity_m_s"]})[0]
    assert r["parameters"] == [0.4, 1.0, 3.0, None, 5.0]
    assert c["parameters"] == [0.4, 1.0, 3.0, 4.0, 5.0]
    assert r["conditions"] == c["conditions"] and r["targets"] == [0.3, None]


def test_local_target_scale_keeps_all_input_moments_and_selection_units():
    cases = [
        {"source": s, "split": "train", "parameters": [i, None], "targets": [cd, cl]}
        for i, (s, cd, cl) in enumerate(
            (
                ("drivaerrim", 0.24, -0.10),
                ("drivaerrim", 0.25, -0.14),
                ("other", 0.4, 0.5),
            )
        )
    ]
    global_stats, global_selection = run_stats(cases, {"local_targets": False})
    local, selection = run_stats(cases, {"local_targets": True})
    assert local["parameter_mean"] == global_stats["parameter_mean"]
    assert local["parameter_std"] == global_stats["parameter_std"]
    assert selection == global_selection
    np.testing.assert_allclose(local["target_mean"], [0.245, -0.12])
    np.testing.assert_allclose(local["target_std"], [0.005, 0.02])


def test_rim_gradient_matches_its_component_of_full_batch_loss():
    torch.manual_seed(7)
    model, expected = Toy(), Toy()
    expected.load_state_dict(model.state_dict())
    values = [
        ({}, torch.tensor([1.0, 0.0]), torch.tensor([0.2, 0.1]), torch.tensor(1)),
        (
            {},
            torch.tensor([0.0, 1.0]),
            torch.tensor([999.0, float("nan")]),
            torch.tensor(1),
        ),
    ]
    cases = [{"source": "drivaerrim"}, {"source": "other"}]
    run = {"jitter": False, "focus": True}
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    intervened_epoch(
        model, values, cases, np.arange(2), optimizer, torch.device("cpu"), 2, run, None
    )
    reference = torch.optim.SGD(expected.parameters(), lr=0.1)
    reference.zero_grad()
    loss = supervised_loss(expected({}, values[0][1][None]), values[0][2][None], 3)
    loss.backward()
    reference.step()
    for a, b in zip(model.parameters(), expected.parameters()):
        torch.testing.assert_close(a, b)


def test_non_rim_label_values_do_not_affect_focused_update():
    torch.manual_seed(42)
    a, b = Toy(), Toy()
    b.load_state_dict(a.state_dict())
    first = [
        ({}, torch.tensor([1.0, 0.0]), torch.tensor([0.2, 0.1]), torch.tensor(1)),
        ({}, torch.tensor([0.0, 1.0]), torch.tensor([999.0, -999.0]), torch.tensor(1)),
    ]
    second = copy.deepcopy(first)
    second[1][2].neg_()
    for model, values in ((a, first), (b, second)):
        intervened_epoch(
            model,
            values,
            [{"source": "drivaerrim"}, {"source": "other"}],
            np.arange(2),
            torch.optim.AdamW(model.parameters()),
            torch.device("cpu"),
            2,
            {"focus": True, "jitter": False},
            None,
        )
    for x, y in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(x, y)


def test_jitter_draws_match_before_rim_filtering():
    values = []
    for i in range(2):
        image = torch.stack((torch.full((16, 16), 0.3), torch.ones((16, 16))))
        geometry = {
            n: image.clone() for n in ("front", "rear", "right", "top", "bottom")
        }
        values.append(
            (
                geometry,
                torch.tensor([float(i), 1.0]),
                torch.tensor([0.2, 0.1]),
                torch.tensor(1),
            )
        )
    states = []
    for focus in (False, True):
        model, rng = Toy(), torch.Generator().manual_seed(43)
        intervened_epoch(
            model,
            values,
            [{"source": "drivaerrim"}, {"source": "other"}],
            np.arange(2),
            torch.optim.AdamW(model.parameters()),
            torch.device("cpu"),
            2,
            {"focus": focus, "jitter": True},
            rng,
        )
        states.append(rng.get_state())
    assert torch.equal(*states)


def test_validation_metric_uses_fixed_physical_units():
    class Identity(nn.Module):
        target_count = 2

        def forward(self, geometry, parameters, scene=None):
            return parameters

    raw_prediction, raw_truth = np.array([0.25, -0.10]), np.array([0.245, -0.12])
    scores = []
    for mean, std in (([0.3, 0.1], [0.04, 0.18]), ([0.24, -0.13], [0.003, 0.02])):
        stats = {"target_mean": mean, "target_std": std}
        value = (
            {},
            torch.tensor((raw_prediction - mean) / std, dtype=torch.float32),
            torch.tensor((raw_truth - mean) / std, dtype=torch.float32),
            torch.tensor(1),
        )
        scores.append(
            validation_scores(
                Identity(),
                DataLoader([value]),
                torch.device("cpu"),
                stats,
                np.array([0.04, 0.18]),
                [{"source": "drivaerrim"}],
            )[:2]
        )
    np.testing.assert_allclose(scores[0], scores[1], rtol=1e-5)


def test_kernel_distances_and_linear_similarity():
    x = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    d = distances(x, x)
    np.testing.assert_allclose(np.diag(d), 0)
    assert d[0, 2] > d[0, 1] > 0
    gram = kernel(x, x, "linear", 1)
    assert np.linalg.eigvalsh(gram).min() >= -1e-12
    rbf = kernel(x, x, "rbf", 1)
    np.testing.assert_allclose(np.diag(rbf), 1)
    assert rbf[0, 2] < rbf[0, 1] < rbf[0, 0]
