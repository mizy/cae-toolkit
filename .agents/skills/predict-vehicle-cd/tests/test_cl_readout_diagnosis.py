import sys
from pathlib import Path

import torch

SKILL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL / "scripts"))
sys.path.insert(0, str(SKILL.parent / "predict-aircraft-coefficients/scripts"))
from vehicle_cd_learning.cl_readout_diagnosis import GROUPS, restored_state


def test_restore_exact_stage_ownership_keeps_encoders_and_Cd_row():
    state = {
        "image_encoder.weight": torch.tensor([1.0]),
        "bottom_encoder.weight": torch.tensor([2.0]),
        "adapter.0.weight": torch.tensor([3.0]),
        "parameter_encoder.0.weight": torch.tensor([4.0]),
        "head.0.weight": torch.tensor([5.0]),
        "head.0.bias": torch.tensor([6.0]),
        "head.2.weight": torch.tensor([[7.0, 8.0], [9.0, 10.0]]),
    }
    baseline = {name: value + 100 for name, value in state.items()}
    for group in GROUPS:
        restored = restored_state(state, baseline, [group])
        assert torch.equal(
            restored["image_encoder.weight"], state["image_encoder.weight"]
        )
        assert torch.equal(
            restored["bottom_encoder.weight"], state["bottom_encoder.weight"]
        )
        assert torch.equal(restored["head.2.weight"][0], state["head.2.weight"][0])
        expected = (
            baseline["head.2.weight"][1]
            if group == "Cl_row"
            else state["head.2.weight"][1]
        )
        assert torch.equal(restored["head.2.weight"][1], expected)
        prefix = {
            "adapter": "adapter.",
            "parameter_encoder": "parameter_encoder.",
            "head_hidden": "head.0.",
        }.get(group)
        for name, value in state.items():
            if name == "head.2.weight":
                continue
            assert torch.equal(
                restored[name],
                baseline[name] if prefix and name.startswith(prefix) else value,
            )
    full = restored_state(state, baseline, GROUPS, full_readout=True)
    assert torch.equal(full["head.2.weight"], baseline["head.2.weight"])
    assert torch.equal(state["head.2.weight"], torch.tensor([[7.0, 8.0], [9.0, 10.0]]))
