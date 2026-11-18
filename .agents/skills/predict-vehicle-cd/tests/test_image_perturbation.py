import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from vehicle_cd_learning.data import VIEWS
from vehicle_cd_learning.image_perturbation import translate_views


def rectangle(size=32, batch=2, device="cpu"):
    image = torch.zeros(batch, 2, size, size, device=device)
    image[:, 0, 8:20, 6:22] = 0.7
    image[:, 1, 8:20, 6:22] = 1
    return {name: image.clone() for name in VIEWS}


def test_subpixel_edges_keep_depth_and_area_with_soft_aligned_coverage():
    views = rectangle()
    before = {name: image.clone() for name, image in views.items()}
    offsets = torch.tensor([[0.25, -0.5], [-0.5, 0.375]])
    result = translate_views(views, offsets)
    for name in VIEWS:
        depth, mask = result[name].unbind(1)
        assert torch.isfinite(result[name]).all()
        assert ((mask >= 0) & (mask <= 1)).all()
        assert ((mask > 0) & (mask < 1)).any()
        torch.testing.assert_close(
            depth[mask > 0], torch.full_like(depth[mask > 0], 0.7)
        )
        assert (depth[mask == 0] == 0).all()
        torch.testing.assert_close(mask.sum((1, 2)), views[name][:, 1].sum((1, 2)))
        torch.testing.assert_close(result[name], result["front"], rtol=0, atol=0)
        torch.testing.assert_close(views[name], before[name], rtol=0, atol=0)
    # Each invocation starts from the supplied originals, without accumulated blur.
    repeated = translate_views(views, offsets)
    for name in VIEWS:
        torch.testing.assert_close(repeated[name], result[name], rtol=0, atol=0)


def test_integer_offsets_move_each_sample_in_the_declared_direction():
    views = rectangle()
    result = translate_views(views, torch.tensor([[2.0, 3.0], [-2.0, -3.0]]))
    expected = torch.zeros_like(views["front"])
    expected[0, :, 3:, 2:] = views["front"][0, :, :-3, :-2]
    expected[1, :, :-3, :-2] = views["front"][1, :, 3:, 2:]
    for image in result.values():
        torch.testing.assert_close(image, expected, rtol=0, atol=0)


@pytest.mark.parametrize("size", [256, 512])
@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_zero_offset_preserves_actual_resolution_images_exactly(size, device):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    generator = torch.Generator().manual_seed(18)
    views = rectangle(size=size)
    for image in views.values():
        image[:, 0] = torch.rand(2, size, size, generator=generator) * image[:, 1]
    views = {name: image.to(device) for name, image in views.items()}
    result = translate_views(views, torch.zeros(2, 2))
    for name in VIEWS:
        torch.testing.assert_close(result[name], views[name], rtol=0, atol=0)


def test_depth_and_mask_sample_the_same_original_pixels():
    views = rectangle()
    for image in views.values():
        image[:, 0, 8, 6] = 0.2
        image[:, 0, 8, 7] = 0.4
    result = translate_views(views, torch.full((2, 2), 0.5))
    for image in result.values():
        torch.testing.assert_close(image[:, 0, 8, 6], torch.full((2,), 0.2))
        torch.testing.assert_close(image[:, 1, 8, 6], torch.full((2,), 0.25))
        torch.testing.assert_close(image[:, 0, 8, 7], torch.full((2,), 0.3))
        torch.testing.assert_close(image[:, 1, 8, 7], torch.full((2,), 0.5))


def test_offsets_use_pixels_for_each_resolution_without_resizing():
    views = rectangle()
    views["bottom"] = rectangle(size=64)["bottom"]
    result = translate_views(views, torch.tensor([[0.25, 0.5], [0.25, 0.5]]))
    torch.testing.assert_close(result["bottom"][:, :, :32, :32], result["front"])
    assert result["bottom"].shape == (2, 2, 64, 64)


def test_outside_canvas_is_zero_and_empty_views_remain_finite():
    views = rectangle()
    views["bottom"].zero_()
    result = translate_views(views, torch.tensor([[64.0, 0.0], [0.0, -64.0]]))
    for image in result.values():
        assert torch.isfinite(image).all()
        assert (image == 0).all()


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_forward_backward_keeps_finite_image_and_offset_gradients(device):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    views = {
        name: image.requires_grad_() for name, image in rectangle(device=device).items()
    }
    offsets = torch.tensor(
        [[0.25, -0.375], [-0.125, 0.5]], device=device, requires_grad=True
    )
    result = translate_views(views, offsets)
    sum(image.square().sum() for image in result.values()).backward()
    for image in views.values():
        assert image.grad is not None
        assert torch.isfinite(image.grad).all()
        assert image.grad[:, 0].abs().sum() > 0
        assert image.grad[:, 1].abs().sum() > 0
    assert offsets.grad is not None and torch.isfinite(offsets.grad).all()
    assert offsets.grad.abs().sum() > 0


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_offsets_are_rejected(invalid):
    with pytest.raises(ValueError, match="finite Bx2"):
        translate_views(rectangle(), torch.tensor([[invalid, 0.0], [0.0, 0.0]]))
