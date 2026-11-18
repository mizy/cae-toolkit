"""Compare complete-view spatial features before predicting standardized ΔCd/ΔCl.

The caller owns train-only delta scaling and physical decoding. Antisymmetry and
zero self-difference are architectural; cycle consistency is not guaranteed.
"""

import torch
from torch import nn
from torch.nn import functional as F

from .model import VIEWS, PyramidEncoder


class SpatialEncoder(PyramidEncoder):
    """Retain the pretrained pyramid and choose its masked spatial pool size."""

    def __init__(self, pool_size):
        super().__init__(output_size=32, attention=True, detail_pooling=True)
        self.pool_size = pool_size
        self.project = nn.Identity()  # Pair comparison replaces vector projection.

    def pooled_features(self, image):
        if self.pool_size == 8:
            return super().pooled_features(image)
        levels = []
        feature = image
        for block, lateral in zip(self.blocks, self.lateral):
            feature = block(feature)
            levels.append(lateral(feature))
        combined = self.smooth(
            levels[0]
            + F.interpolate(
                levels[1],
                size=levels[0].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            + F.interpolate(
                levels[2],
                size=levels[0].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        )
        channel = torch.sigmoid(
            self.channel(combined.mean((2, 3), keepdim=True))
            + self.channel(combined.amax((2, 3), keepdim=True))
        )
        combined = combined * channel * 2.0
        spatial = self.spatial(
            torch.cat(
                (combined.mean(1, keepdim=True), combined.amax(1, keepdim=True)), dim=1
            )
        )
        combined = combined * spatial * 2.0
        mask = F.adaptive_max_pool2d(image[:, 1:2], combined.shape[-2:])
        return _masked_pool(combined, mask, self.pool_size)


def _masked_pool(feature, mask, size):
    occupancy = F.adaptive_avg_pool2d(mask, size)
    mean = F.adaptive_avg_pool2d(feature * mask, size)
    mean = mean / occupancy.clamp_min(1e-8)
    maximum = F.adaptive_max_pool2d(feature.masked_fill(mask <= 0, -torch.inf), size)
    maximum = torch.where(occupancy > 0, maximum, 0)
    return torch.cat((mean, maximum), dim=1)


class PairedDelta(nn.Module):
    """Shared spatial pair adapter and two-output conditional delta head.

    Views are all five B×2×H×W depth/mask images. Parameters are the existing
    normalized B×36 dimensionless fields. All neural weights remain trainable.
    """

    def __init__(self, pool_size, parameter_count=36):
        super().__init__()
        if pool_size not in (8, 16):
            raise ValueError("pool_size must be 8 or 16")
        self.pool_size = pool_size
        self.parameter_count = parameter_count
        self.image_encoder = SpatialEncoder(pool_size)
        self.bottom_encoder = SpatialEncoder(pool_size)
        self.adapter = nn.Sequential(
            nn.Conv2d(32, 16, 1),
            nn.GELU(),
            nn.Conv2d(16, 16, 3, padding=1, groups=16),
            nn.GELU(),
        )
        self.parameter_encoder = nn.Sequential(
            nn.Linear(parameter_count, 16), nn.GELU()
        )
        self.head = nn.Sequential(
            nn.Linear(5 * 32 * 2 * 2 + 16, 64),
            nn.GELU(),
            nn.Linear(64, 2, bias=False),
        )

    def encode(self, views):
        """@entry Encode all views, retaining spatial features and foreground masks."""
        body = self.image_encoder.pooled_features(
            torch.cat([views[n] for n in VIEWS[:4]])
        )
        pooled = dict(zip(VIEWS[:4], body.chunk(4)))
        pooled["bottom"] = self.bottom_encoder.pooled_features(views["bottom"])
        for name in VIEWS:
            shape = (views[name].shape[-2] + 1) // 2, (views[name].shape[-1] + 1) // 2
            mask = F.adaptive_max_pool2d(views[name][:, 1:2], shape)
            pooled[name + "_mask"] = F.adaptive_avg_pool2d(mask, self.pool_size)
        return pooled

    def predict_pair(self, pooled_a, pooled_b, parameters):
        """@entry Return standardized B−A; compare spatial maps before compression."""
        context = torch.cat([(pooled_a[n] + pooled_b[n]) * 0.5 for n in VIEWS])
        difference = torch.cat([pooled_b[n] - pooled_a[n] for n in VIEWS])
        mask = torch.cat(
            [
                torch.maximum(pooled_a[n + "_mask"], pooled_b[n + "_mask"]) > 0
                for n in VIEWS
            ]
        ).to(context.dtype)
        features = self.adapter(
            torch.cat(
                (
                    torch.cat((context, difference), dim=1),
                    torch.cat((context, -difference), dim=1),
                )
            )
        )
        vectors = _masked_pool(features, mask.repeat(2, 1, 1, 1), 2)
        positive, negative = vectors.chunk(2)
        batch = parameters.shape[0]
        condition = self.parameter_encoder(parameters)

        def predict(value):
            per_view = value.reshape(5, batch, -1).transpose(0, 1).reshape(batch, -1)
            return self.head(torch.cat((per_view, condition), dim=1))

        return (predict(positive) - predict(negative)) * 0.5

    def forward(self, views_a, views_b, parameters):
        return self.predict_pair(self.encode(views_a), self.encode(views_b), parameters)


def initialize(source_ck, pool_size, seed=42):
    """@entry Copy pretrained CNN tensors exactly; seed only the new pair readout."""
    config = source_ck["model_config"]
    if config["encoder"] != "pyramid_detail_attention" or config["variant"] != "five":
        raise ValueError("Expected five-view pyramid_detail_attention source")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = PairedDelta(pool_size, config["parameter_count"])
    nn.init.zeros_(model.head[-1].weight)
    for name in ("image_encoder", "bottom_encoder"):
        encoder = getattr(model, name)
        state = {
            key: source_ck["model_state"][name + "." + key]
            for key in encoder.state_dict()
        }
        encoder.load_state_dict(state, strict=True)
    return model
