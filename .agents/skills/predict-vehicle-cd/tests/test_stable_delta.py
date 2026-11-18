import sys
from pathlib import Path

import pytest
import torch

SKILL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL / "scripts"))

from vehicle_cd_learning.model import VIEWS
from vehicle_cd_learning.paired_delta import PairedDelta
from vehicle_cd_learning.stable_delta import StableDelta, initialize_from_checkpoint


def checkpoint(pool_size=8):
    torch.set_num_threads(2)
    torch.manual_seed(17)
    model = PairedDelta(pool_size)
    return {
        "model_config": {"pool_size": pool_size, "parameter_count": 36},
        "model_state": model.state_dict(),
    }


def views(mask="random", size=64):
    result = {}
    for name in VIEWS:
        image = torch.rand(1, 2, size, size)
        if mask == "full":
            image[:, 1] = 1
        elif mask in ("empty", "tiny"):
            image[:, 1] = 0
            if mask == "tiny":
                image[:, 1, 0, 0] = 1e-6
        else:
            image[:, 1] = (image[:, 1] > 0.5).float()
        result[name] = image.requires_grad_()
    return result


@pytest.mark.parametrize("pool_size", [8, 16])
def test_zero_floor_exactly_replays_original(pool_size):
    ck = checkpoint(pool_size)
    original = PairedDelta(**ck["model_config"]).eval()
    original.load_state_dict(ck["model_state"])
    stable = initialize_from_checkpoint(ck, coverage_floor=0).eval()
    a, b, parameters = views("tiny"), views(), torch.randn(1, 36)
    for name, value in original.encode(a).items():
        assert torch.equal(value, stable.encode(a)[name])
    assert torch.equal(original(a, b, parameters), stable(a, b, parameters))


def test_strict_weight_copy_keeps_source_unchanged_and_all_weights_trainable():
    ck = checkpoint()
    source = {name: value.clone() for name, value in ck["model_state"].items()}
    stable = initialize_from_checkpoint(ck)
    assert sum(p.numel() for p in stable.parameters()) == 54506
    assert set(stable.state_dict()) == set(source)
    for name, value in stable.state_dict().items():
        assert torch.equal(value, source[name])
    stable(views(), views(), torch.randn(1, 36)).square().sum().backward()
    assert all(p.requires_grad and p.grad is not None for p in stable.parameters())
    with torch.no_grad():
        next(stable.parameters()).add_(1)
    for name, value in ck["model_state"].items():
        assert torch.equal(value, source[name])


@pytest.mark.parametrize("mask", ["tiny", "empty"])
def test_sparse_foreground_outputs_and_gradients_are_finite(mask):
    stable = initialize_from_checkpoint(checkpoint())
    a, b = views(mask), views("tiny")
    output = stable(a, b, torch.randn(1, 36))
    assert torch.isfinite(output).all()
    output.sum().backward()
    assert all(torch.isfinite(image.grad).all() for image in (*a.values(), *b.values()))
    assert all(torch.isfinite(p.grad).all() for p in stable.parameters())


def test_exact_zero_antisymmetry_and_full_foreground_parity():
    ck = checkpoint()
    original = PairedDelta(**ck["model_config"]).eval()
    original.load_state_dict(ck["model_state"])
    stable = initialize_from_checkpoint(ck).eval()
    a, b, parameters = views("full"), views("full"), torch.randn(1, 36)
    pa, pb = stable.encode(a), stable.encode(b)
    assert torch.equal(stable.predict_pair(pa, pa, parameters), torch.zeros(1, 2))
    forward = stable.predict_pair(pa, pb, parameters)
    assert torch.equal(stable.predict_pair(pb, pa, parameters), -forward)
    assert torch.equal(original(a, b, parameters), forward)


def test_sparse_encoder_features_are_attenuated_by_existing_coverage():
    ck = checkpoint()
    original = initialize_from_checkpoint(ck, coverage_floor=0)
    stable = initialize_from_checkpoint(ck)
    a = views("tiny")
    before, after = original.encode(a), stable.encode(a)
    for name in VIEWS:
        gate = (before[name + "_mask"] / stable.coverage_floor).clamp_max(1)
        assert torch.equal(after[name], before[name] * gate)
        assert after[name].abs().max() < before[name].abs().max()


@pytest.mark.parametrize("floor", [-1, 1.1, float("nan")])
def test_invalid_floor(floor):
    with pytest.raises(ValueError, match="coverage_floor"):
        StableDelta(coverage_floor=floor)


def test_filtered_input_preserves_constants_and_has_finite_sparse_gradients():
    ck = checkpoint()
    ordinary = initialize_from_checkpoint(ck, coverage_floor=0).eval()
    filtered = initialize_from_checkpoint(
        ck, coverage_floor=0, filter_input=True
    ).eval()
    constant = {
        n: torch.cat((torch.full((1, 1, 32, 32), 0.5), torch.ones(1, 1, 32, 32)), 1)
        for n in VIEWS
    }
    for name, value in ordinary.encode(constant).items():
        torch.testing.assert_close(
            value, filtered.encode(constant)[name], atol=0, rtol=0
        )
    stable = initialize_from_checkpoint(ck, coverage_floor=1 / 16, filter_input=True)
    out = stable(views("tiny"), views("full"), torch.randn(1, 36))
    out.square().sum().backward()
    assert torch.isfinite(out).all()
    assert all(
        torch.isfinite(p.grad).all() for p in stable.parameters() if p.grad is not None
    )
