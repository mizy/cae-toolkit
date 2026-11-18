import sys
from pathlib import Path

import numpy as np
import pytest
import torch

SKILL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL / "scripts"))
sys.path.insert(0, str(SKILL.parent / "predict-aircraft-coefficients/scripts"))

from vehicle_cd_learning import dimensionless as dim
from vehicle_cd_learning.end_to_end_evaluation import radiant_scores
from vehicle_cd_learning.model import VIEWS, CoefficientModel
from vehicle_cd_learning.pooled_residual import CONTRACT, initialize, load_model


@pytest.mark.parametrize("detail", [False, True])
def test_initial_function_and_saved_model_replay(tmp_path, detail):
    torch.set_num_threads(2)
    torch.manual_seed(42)
    source = CoefficientModel(**dim.MODEL).eval()
    ck = {"model_config": dim.MODEL, "model_state": source.state_dict()}
    model = initialize(ck, detail, np.zeros(5156), np.ones(5156)).eval()
    views, params = {n: torch.randn(2, 2, 32, 32) for n in VIEWS}, torch.randn(2, 36)
    with torch.no_grad():
        torch.testing.assert_close(model(views, params), source(views, params))
    if detail:
        with torch.no_grad():
            model.correction[-1].weight.normal_(0, 0.02)
    file = tmp_path / "model.pt"
    torch.save(
        {
            **ck,
            "input_contract": CONTRACT,
            "parameter_names": dim.NAMES,
            "detail_path": detail,
            "model_state": model.state_dict(),
        },
        file,
    )
    loaded, _ = load_model(file)
    with torch.no_grad():
        together = loaded(views, params)
        separate = torch.cat(
            [
                loaded({k: v[i : i + 1] for k, v in views.items()}, params[i : i + 1])
                for i in range(2)
            ]
        )
        torch.testing.assert_close(together, separate, atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(model(views, params), together)


def test_detail_path_backpropagates_to_the_cnn_without_the_old_head():
    torch.manual_seed(43)
    source = CoefficientModel(**dim.MODEL)
    model = initialize(
        {"model_config": dim.MODEL, "model_state": source.state_dict()},
        True,
        np.zeros(5156),
        np.ones(5156),
    )
    with torch.no_grad():
        for p in model.head.parameters():
            p.zero_()
        model.correction[-1].weight.normal_(0, 0.1)
    views, params = {n: torch.randn(2, 2, 32, 32) for n in VIEWS}, torch.randn(2, 36)
    loss = model(views, params).square().sum()
    loss.backward()
    for encoder in (model.image_encoder, model.bottom_encoder):
        gradient = encoder.blocks[0][0].weight.grad
        assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0


def test_external_absolute_score_uses_the_same_reference_and_checks_query_labels():
    cases = [
        {
            "case_id": "ref",
            "split": "train",
            "source": "drivaerrim",
            "targets": [0.2, 0.1],
        },
        {
            "case_id": "a",
            "split": "test",
            "source": "drivaerrim",
            "targets": [0.3, 0.2],
        },
        {
            "case_id": "b",
            "split": "test",
            "source": "drivaerrim",
            "targets": [0.4, 0.3],
        },
    ]
    prediction = np.array([[0.1, 0.0], [0.2, 0.1], [0.3, 0.2]])
    external = {
        "anchors": {"ref": 0.2},
        "rows_by_anchor": {
            "ref": [
                {
                    "case_id": "a",
                    "truth": 0.3,
                    "prediction": 0.32,
                    "delta_prediction": 0.12,
                },
                {
                    "case_id": "b",
                    "truth": 0.4,
                    "prediction": 0.37,
                    "delta_prediction": 0.17,
                },
            ]
        },
    }
    result = radiant_scores(cases, prediction, external)
    assert result["our_cd_mae_mean_references"] == pytest.approx(0, abs=1e-12)
    assert result["radiant_cd_mae_mean_references"] == pytest.approx(0.025)
    external["rows_by_anchor"]["ref"][0]["truth"] = 0.31
    with pytest.raises(ValueError, match="labels differ"):
        radiant_scores(cases, prediction, external)
