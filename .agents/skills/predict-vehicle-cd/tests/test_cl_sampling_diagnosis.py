import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[2] / "predict-aircraft-coefficients/scripts"),
)
from vehicle_cd_learning.cl_sampling_diagnosis import (
    mix_views,
    parameter_changes,
    phase_distribution,
    predict,
)
from vehicle_cd_learning.model import VIEWS
from vehicle_cd_learning.paired_delta import PairedDelta


def test_cached_single_view_matches_raw_mixed_forward():
    torch.set_num_threads(2)
    torch.manual_seed(81)
    model = PairedDelta(8, 36).eval()
    nominal = {name: torch.rand(1, 2, 32, 32) for name in VIEWS}
    shifted = {name: torch.rand(1, 2, 32, 32) for name in VIEWS}
    params = torch.randn(1, 36)
    with torch.inference_mode():
        a, b = model.encode(nominal), model.encode(shifted)
        for view in (*VIEWS, "all5"):
            raw = {
                name: shifted[name] if view in (name, "all5") else nominal[name]
                for name in VIEWS
            }
            cached = mix_views(a, b, view)
            expected = model(nominal, raw, params)
            actual = model.predict_pair(a, cached, params)
            torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-5)
            cache = {
                (0, None): {n: v[0] for n, v in a.items()},
                (0, "phase"): {n: v[0] for n, v in cached.items()},
            }
            result = predict(
                model, cache, [((0, None), (0, "phase"))], params, np.array([0.1, 0.2])
            )
            np.testing.assert_allclose(result, expected.numpy() * [0.1, 0.2], atol=2e-8)


def test_stage_rows_and_phase_sign_groups_are_independent():
    base = {
        "head.2.weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        "image_encoder.weight": torch.tensor([2.0]),
    }
    changed = {name: value.clone() for name, value in base.items()}
    changed["head.2.weight"][1] *= 2
    result = parameter_changes({"B": base, "E5": changed})["B_to_E5"]
    assert result["head_final_Cd_row"]["relative_l2"] == 0
    assert result["head_final_Cl_row"]["relative_l2"] == 1
    assert result["image_encoder"]["relative_l2"] == 0
    rows = np.array([[[1.0, 2.0]], [[-1.0, -2.0]], [[3.0, 4.0]], [[-3.0, -4.0]]])
    summary = phase_distribution(rows, ["x+", "x-", "y+", "y-"])
    assert summary["pooled"]["Cl"]["signed_mean"] == 0
    assert summary["by_direction"]["positive"]["Cl"]["signed_mean"] == 3
    assert summary["by_direction"]["negative"]["Cl"]["signed_mean"] == -3
