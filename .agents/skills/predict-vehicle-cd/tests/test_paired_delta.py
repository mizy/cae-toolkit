import sys
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

SKILL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL / "scripts"))

from vehicle_cd_learning.model import VIEWS, CoefficientModel
from vehicle_cd_learning.paired_delta import PairedDelta, SpatialEncoder, initialize


def source_checkpoint():
    config = {
        "parameter_count": 36,
        "variant": "five",
        "target_count": 2,
        "size": "small",
        "encoder": "pyramid_detail_attention",
        "fusion": "attention",
    }
    torch.manual_seed(17)
    return {
        "model_config": config,
        "model_state": CoefficientModel(**config).state_dict(),
    }


def views(batch=2, size=32):
    return {
        n: torch.cat(
            (
                torch.rand(batch, 1, size, size),
                (torch.rand(batch, 1, size, size) > 0.3).float(),
            ),
            dim=1,
        )
        for n in VIEWS
    }


@pytest.mark.parametrize("pool_size", [8, 16])
def test_zero_antisymmetry_batch_independence_and_gradient(pool_size):
    torch.set_num_threads(2)
    model = initialize(source_checkpoint(), pool_size).eval()
    a, b, parameters = views(), views(), torch.randn(2, 36)
    assert model.head[-1].bias is None
    assert torch.equal(model(a, b, parameters), torch.zeros(2, 2))
    # One head update must unlock upstream CNN gradients from the zero fallback.
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    (model(a, b, parameters) - 1).square().sum().backward()
    assert model.head[-1].weight.grad.abs().sum() > 0
    assert model.image_encoder.blocks[0][0].weight.grad.abs().sum() == 0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    pa, pb = model.encode(a), model.encode(b)
    forward = model.predict_pair(pa, pb, parameters)
    assert torch.equal(
        model.predict_pair(pa, pa, parameters), torch.zeros_like(forward)
    )
    assert torch.equal(model.predict_pair(pb, pa, parameters), -forward)
    separate = torch.cat(
        [
            model(
                {k: v[i : i + 1] for k, v in a.items()},
                {k: v[i : i + 1] for k, v in b.items()},
                parameters[i : i + 1],
            )
            for i in range(2)
        ]
    )
    torch.testing.assert_close(forward, separate, atol=1e-7, rtol=1e-5)
    forward.square().sum().backward()
    for encoder in (model.image_encoder, model.bottom_encoder):
        gradient = encoder.blocks[0][0].weight.grad
        assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    assert all(p.requires_grad and p.grad is not None for p in model.parameters())


def test_source_tensors_and_initialization_match_across_pool_sizes():
    checkpoint = source_checkpoint()
    small, large = [initialize(checkpoint, size) for size in (8, 16)]
    assert set(small.state_dict()) == set(large.state_dict())
    for name, tensor in small.state_dict().items():
        assert torch.equal(tensor, large.state_dict()[name])
        if name.startswith(("image_encoder.", "bottom_encoder.")):
            assert torch.equal(tensor, checkpoint["model_state"][name])
    assert sum(p.numel() for p in small.parameters()) == sum(
        p.numel() for p in large.parameters()
    )
    reference = CoefficientModel(**checkpoint["model_config"]).image_encoder
    reference.load_state_dict(
        {
            k.removeprefix("image_encoder."): v
            for k, v in checkpoint["model_state"].items()
            if k.startswith("image_encoder.")
        }
    )
    image = views(batch=1)["right"]
    torch.testing.assert_close(
        small.image_encoder.pooled_features(image),
        reference.pooled_features(image),
        atol=0,
        rtol=0,
    )


def test_sixteen_pools_high_resolution_features_directly():
    torch.set_num_threads(2)
    encoder = SpatialEncoder(16).eval()
    image = views(batch=1, size=64)["right"]
    # A smooth hook retains native 32×32 features; default attention is unit scale.
    native = []
    smooth_hook = encoder.smooth.register_forward_hook(
        lambda module, args, output: native.append(output)
    )
    output = encoder.pooled_features(image)
    smooth_hook.remove()
    combined = native[0]  # Initial channel/spatial gates each equal one after ×2.
    mask = F.adaptive_max_pool2d(image[:, 1:2], combined.shape[-2:])
    mean = F.adaptive_avg_pool2d(combined * mask, 16) / F.adaptive_avg_pool2d(
        mask, 16
    ).clamp_min(1e-8)
    maximum = F.adaptive_max_pool2d(combined.masked_fill(mask <= 0, -torch.inf), 16)
    maximum = torch.where(F.adaptive_avg_pool2d(mask, 16) > 0, maximum, 0)
    torch.testing.assert_close(
        output, torch.cat((mean, maximum), dim=1), atol=0, rtol=0
    )
    assert output.shape == (1, 16, 16, 16)
    coarse = F.interpolate(F.adaptive_avg_pool2d(output, 8), size=16, mode="nearest")
    assert not torch.equal(output, coarse)


def test_all_empty_views_are_finite():
    model = PairedDelta(16).eval()
    a = {n: torch.zeros(1, 2, 32, 32) for n in VIEWS}
    assert torch.equal(model(a, a, torch.zeros(1, 36)), torch.zeros(1, 2))
