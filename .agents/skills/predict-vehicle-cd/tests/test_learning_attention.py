from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from vehicle_cd_learning.data import PARAMETER_NAMES, VIEWS
from vehicle_cd_learning.model import CoefficientModel, PyramidEncoder, ViewAttention
from vehicle_cd_learning.predict import load_model, predict


@pytest.mark.parametrize(
    "encoder",
    ["pyramid", "pyramid_attention", "pyramid_detail", "pyramid_detail_attention"],
)
def test_full_view_features_cover_all_positions_and_checkpoint_roundtrip(
    encoder, tmp_path
):
    torch.set_num_threads(2)
    torch.manual_seed(12)
    model = CoefficientModel(5, target_count=1, size="small", encoder=encoder)
    views = {name: torch.rand(1, 2, 64, 64, requires_grad=True) for name in VIEWS}
    parameters = torch.zeros(1, 5)
    maps = []
    if encoder in {"pyramid_attention", "pyramid_detail_attention"}:
        hook = model.image_encoder.spatial.register_forward_hook(
            lambda _m, _i, o: maps.append(o.detach())
        )
    model(views, parameters).sum().backward()
    if encoder in {"pyramid_attention", "pyramid_detail_attention"}:
        hook.remove()
        assert maps[0].shape == (4, 1, 32, 32)
        assert ((maps[0] >= 0) & (maps[0] <= 1)).all()
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.image_encoder.spatial.parameters()
        )
    if encoder in {"pyramid_detail", "pyramid_detail_attention"}:
        maps.clear()
        pool = model.image_encoder.project[0].register_forward_pre_hook(
            lambda _m, inputs: maps.append(inputs[0].detach())
        )
        model(views, parameters)
        pool.remove()
        assert maps[0].shape == (4, 16, 8, 8)
        assert torch.isfinite(maps[0]).all()
    for image in views.values():
        assert image.grad is not None and torch.isfinite(image.grad).all()
        for row, col in ((0, 0), (0, 32), (32, 0), (32, 32)):
            assert image.grad[:, :, row : row + 32, col : col + 32].abs().sum() > 0
    arrays = {
        name: np.stack(
            (
                np.full((64, 64), 0.5, dtype=np.float32),
                np.ones((64, 64), dtype=np.float32),
            )
        )
        for name in VIEWS
    }
    checkpoint = {
        "model_config": {
            "parameter_count": 5,
            "target_count": 1,
            "size": "small",
            "encoder": encoder,
        },
        "model_state": model.state_dict(),
        "parameter_names": PARAMETER_NAMES,
        "target_names": ["cd"],
        "parameter_mean": [0.0] * 5,
        "parameter_std": [1.0] * 5,
        "target_mean": [0.3],
        "target_std": [0.01],
        "parameter_ranges": {},
        "render_config": {"ordinary_size": 64, "bottom_size": 64},
    }
    path = tmp_path / "model.pt"
    torch.save(checkpoint, path)
    loaded, config = load_model(path)
    with torch.inference_mode():
        expected = (
            model(
                {k: torch.from_numpy(v[None]) for k, v in arrays.items()}, parameters
            ).numpy()[0]
            * 0.01
            + 0.3
        )
    np.testing.assert_allclose(
        predict(loaded, config, arrays, [0.0] * 5), expected, atol=1e-7
    )


def test_detail_pooling_keeps_small_foreground_features_and_zeros_empty_bins():
    encoder = PyramidEncoder(output_size=4, detail_pooling=True)
    image = torch.zeros(1, 2, 32, 32)
    image[0, 1, 4, 4] = 1
    pooled = []

    def expose_small_signal(_module, _inputs, output):
        features = torch.full_like(output, -1)
        features[:, 0, 2, 2] = 4
        return features

    smooth = encoder.smooth.register_forward_hook(expose_small_signal)
    project = encoder.project[0].register_forward_pre_hook(
        lambda _module, inputs: pooled.append(inputs[0].detach())
    )
    encoder(image)
    smooth.remove()
    project.remove()

    result = pooled[0]
    assert result.shape == (1, 16, 8, 8)
    assert result[0, 0, 1, 1] == 4
    assert result[0, 8, 1, 1] == 4
    assert result[0, :, 0, 0].eq(0).all()

    empty = PyramidEncoder(output_size=4, detail_pooling=True)
    empty_pooled = []
    handle = empty.project[0].register_forward_pre_hook(
        lambda _module, inputs: empty_pooled.append(inputs[0].detach())
    )
    assert torch.isfinite(empty(torch.zeros(1, 2, 32, 32))).all()
    handle.remove()
    assert empty_pooled[0].eq(0).all()


def test_view_attention_starts_as_identity_and_learns_per_case_weights():
    torch.manual_seed(31)
    attention = ViewAttention(view_count=5, feature_size=8, condition_size=2)
    features = [torch.randn(2, 8, requires_grad=True) for _ in range(5)]
    condition = torch.randn(2, 2, requires_grad=True)
    scaled, weights = attention(features, condition)
    torch.testing.assert_close(weights, torch.full((2, 5), 0.2))
    for before, after in zip(features, scaled):
        torch.testing.assert_close(before, after)

    loss = sum((index + 1) * view.sum() for index, view in enumerate(scaled))
    loss.backward()
    assert attention.score[-1].weight.grad is not None
    assert torch.isfinite(attention.score[-1].weight.grad).all()
    assert attention.score[-1].weight.grad.abs().sum() > 0

    with torch.no_grad():
        attention.view_identity.weight[:, 0] = torch.linspace(-2, 2, 5)
        attention.score[0].weight.zero_()
        attention.score[0].bias.zero_()
        attention.score[0].weight[0, 8] = 1
        attention.score[0].weight[0, 10] = 1
        attention.score[-1].weight.zero_()
        attention.score[-1].weight[0, 0] = 1
    case_features = [torch.zeros(2, 8) for _ in range(5)]
    case_condition = torch.tensor([[-2.0, 0.0], [2.0, 0.0]])
    _, case_weights = attention(case_features, case_condition)
    assert not torch.allclose(case_weights[0], case_weights[1])
    assert not torch.allclose(case_weights[0], torch.full((5,), 0.2))


def test_combined_spatial_attention_preserves_initial_detail_features():
    plain = PyramidEncoder(output_size=8, detail_pooling=True)
    attention = PyramidEncoder(output_size=8, attention=True, detail_pooling=True)
    missing, extra = attention.load_state_dict(plain.state_dict(), strict=False)
    assert missing and not extra
    assert all(key.startswith(("channel.", "spatial.")) for key in missing)
    image = torch.rand(2, 2, 64, 64)
    image[:, 1] = (image[:, 1] > 0.7).float()
    torch.testing.assert_close(plain(image), attention(image), rtol=1e-6, atol=1e-7)
    attention(image).square().mean().backward()
    assert attention.spatial[0].weight.grad.abs().sum() > 0
    assert attention.channel[-1].weight.grad.abs().sum() > 0


def test_view_attention_keeps_shared_initial_weights_paired():
    torch.manual_seed(42)
    plain = CoefficientModel(5, target_count=2, size="small", encoder="pyramid_detail")
    torch.manual_seed(42)
    attention = CoefficientModel(
        5, target_count=2, size="small", encoder="pyramid_detail", fusion="attention"
    )
    for name, value in plain.state_dict().items():
        torch.testing.assert_close(value, attention.state_dict()[name], rtol=0, atol=0)
    views = {name: torch.rand(2, 2, 64, 64) for name in VIEWS}
    parameters = torch.randn(2, 5)
    torch.testing.assert_close(
        plain(views, parameters), attention(views, parameters), rtol=0, atol=0
    )


def test_full_view_attention_checkpoint_roundtrip(tmp_path):
    model = CoefficientModel(
        5,
        target_count=1,
        size="small",
        encoder="pyramid_detail",
        fusion="attention",
    ).eval()
    checkpoint = {
        "model_config": {
            "parameter_count": 5,
            "target_count": 1,
            "size": "small",
            "encoder": "pyramid_detail",
            "fusion": "attention",
        },
        "model_state": model.state_dict(),
        "parameter_names": PARAMETER_NAMES,
        "target_names": ["cd"],
        "parameter_mean": [0.0] * 5,
        "parameter_std": [1.0] * 5,
        "target_mean": [0.3],
        "target_std": [0.01],
        "parameter_ranges": {},
        "render_config": {"ordinary_size": 64, "bottom_size": 64},
    }
    path = tmp_path / "attention.pt"
    torch.save(checkpoint, path)
    loaded, _ = load_model(path)
    views = {name: torch.rand(1, 2, 64, 64) for name in VIEWS}
    parameters = torch.randn(1, 5)
    with torch.inference_mode():
        torch.testing.assert_close(
            loaded(views, parameters), model(views, parameters), rtol=0, atol=0
        )
