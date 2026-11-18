"""Online image-plane translation of complete depth/mask views."""

import torch
from torch.nn import functional as F

from .data import VIEWS


def translate_views(views: dict[str, torch.Tensor], offsets: torch.Tensor):
    """Translate original Bx2xHxW float32 views by per-sample (dx, dy) pixels.

    Positive offsets move content right/down. All five views share each sample's
    offset, including when their resolutions differ. Bilinear coverage remains
    soft; depth is normalized by coverage so background cannot dilute an edge.
    The canvas stays fixed and samples outside it are zero. Callers generate
    random offsets and pass the original views on every call.
    """
    if set(views) != set(VIEWS):
        raise ValueError("Translation requires all five complete views")
    if offsets.ndim != 2 or offsets.shape[1] != 2 or not torch.isfinite(offsets).all():
        raise ValueError("Offsets must be a finite Bx2 tensor of (dx, dy) pixels")
    translated = {}
    for name, image in views.items():
        if image.ndim != 4 or image.shape[:2] != (len(offsets), 2):
            raise ValueError(f"View {name} must have shape Bx2xHxW")
        if image.dtype != torch.float32:
            raise ValueError(f"View {name} must use float32")
        _, _, height, width = image.shape
        shift = offsets.to(device=image.device, dtype=image.dtype)
        y, x = torch.meshgrid(
            torch.arange(height, device=image.device, dtype=image.dtype),
            torch.arange(width, device=image.device, dtype=image.dtype),
            indexing="ij",
        )
        grid = torch.stack(
            (
                (x[None] + 0.5 - shift[:, None, None, 0]) * (2 / width) - 1,
                (y[None] + 0.5 - shift[:, None, None, 1]) * (2 / height) - 1,
            ),
            dim=-1,
        )
        mask = image[:, 1:2]
        sampled = F.grid_sample(
            torch.cat((image[:, :1] * mask, mask), dim=1),
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        coverage = sampled[:, 1:2].clamp(0, 1)
        depth = sampled[:, :1] / torch.where(coverage > 0, coverage, 1)
        translated[name] = torch.cat((depth, coverage), dim=1)
    return translated
