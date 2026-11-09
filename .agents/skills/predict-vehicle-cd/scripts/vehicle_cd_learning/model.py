"""Small multi-view image model for vehicle aerodynamic coefficients."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F

VIEWS = ("front", "rear", "right", "top", "bottom")


class ViewEncoder(nn.Module):
    """Encode one depth and mask image while retaining coarse spatial layout."""

    def __init__(self, small: bool = False) -> None:
        super().__init__()
        channels = (8, 16, 24, 32) if small else (32, 64, 96, 128)
        layers = []
        previous = 2
        for index, count in enumerate(channels):
            kernel = 5 if index == 0 else 3
            layers.extend(
                (
                    nn.Conv2d(previous, count, kernel, stride=2, padding=kernel // 2),
                    nn.GroupNorm(8, count),
                    nn.GELU(),
                )
            )
            previous = count
        pool = 2 if small else 4
        layers.append(nn.AdaptiveAvgPool2d((pool, pool)))
        self.features = nn.Sequential(*layers)
        self.project = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels[-1] * pool * pool, 32 if small else 128),
            nn.GELU(),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.project(self.features(image))


class PyramidEncoder(nn.Module):
    """Full-view feature pyramid; every image position follows the same path."""

    def __init__(self, output_size=32, attention=False, detail_pooling=False):
        super().__init__()
        channels = (8, 16, 24)
        previous = 2
        blocks = []
        for count in channels:
            blocks.append(
                nn.Sequential(
                    nn.Conv2d(previous, count, 3, stride=2, padding=1),
                    nn.GroupNorm(4, count),
                    nn.GELU(),
                )
            )
            previous = count
        self.blocks = nn.ModuleList(blocks)
        self.lateral = nn.ModuleList([nn.Conv2d(count, 8, 1) for count in channels])
        self.smooth = nn.Sequential(nn.Conv2d(8, 8, 3, padding=1, groups=8), nn.GELU())
        self.channel = (
            nn.Sequential(nn.Conv2d(8, 2, 1), nn.GELU(), nn.Conv2d(2, 8, 1))
            if attention
            else None
        )
        self.spatial = (
            nn.Sequential(nn.Conv2d(2, 1, 7, padding=3), nn.Sigmoid())
            if attention
            else None
        )
        self.detail_pooling = detail_pooling
        if attention and detail_pooling:
            # The new detail+attention mode starts with unit channel/spatial
            # scales so weak foreground features survive the first updates.
            nn.init.zeros_(self.channel[-1].weight)
            nn.init.zeros_(self.channel[-1].bias)
            nn.init.zeros_(self.spatial[0].weight)
            nn.init.zeros_(self.spatial[0].bias)
        if detail_pooling:
            self.project = nn.Sequential(
                nn.Flatten(),
                nn.Linear(16 * 64, output_size),
                nn.GELU(),
            )
        else:
            self.project = nn.Sequential(
                nn.AdaptiveAvgPool2d(8),
                nn.Flatten(),
                nn.Linear(8 * 64, output_size),
                nn.GELU(),
            )

    def pooled_features(self, image):
        levels = []
        feature = image
        for block, lateral in zip(self.blocks, self.lateral):
            feature = block(feature)
            levels.append(lateral(feature))
        combined = (
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
        combined = self.smooth(combined)
        if self.channel is not None:
            channel = torch.sigmoid(
                self.channel(combined.mean((2, 3), keepdim=True))
                + self.channel(combined.amax((2, 3), keepdim=True))
            )
            gate_scale = 2.0 if self.detail_pooling else 1.0
            combined = combined * channel * gate_scale
            spatial = self.spatial(
                torch.cat(
                    (combined.mean(1, keepdim=True), combined.amax(1, keepdim=True)),
                    dim=1,
                )
            )
            combined = combined * spatial * gate_scale
        if self.detail_pooling:
            mask = F.adaptive_max_pool2d(image[:, 1:2], combined.shape[-2:])
            pooled_mask = F.adaptive_avg_pool2d(mask, 8)
            masked_mean = F.adaptive_avg_pool2d(combined * mask, 8)
            masked_mean = masked_mean / pooled_mask.clamp_min(1e-8)

            masked_features = combined.masked_fill(mask <= 0, -torch.inf)
            masked_max = F.adaptive_max_pool2d(masked_features, 8)
            masked_max = torch.where(pooled_mask > 0, masked_max, 0)
            combined = torch.cat((masked_mean, masked_max), dim=1)
        return combined

    def forward(self, image):
        return self.project(self.pooled_features(image))


class ViewAttention(nn.Module):
    """Scale complete-view features using per-case geometry and conditions."""

    def __init__(self, view_count, feature_size, condition_size, hidden_size=None):
        super().__init__()
        embedding_size = 8
        hidden_size = hidden_size or max(16, feature_size // 4)
        self.view_identity = nn.Embedding(view_count, embedding_size)
        self.score = nn.Sequential(
            nn.Linear(feature_size + condition_size + embedding_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )
        nn.init.zeros_(self.score[-1].weight)
        nn.init.zeros_(self.score[-1].bias)

    def forward(self, features, condition):
        stacked = torch.stack(features, dim=1)
        batch, view_count, _ = stacked.shape
        identities = torch.arange(view_count, device=stacked.device)
        identities = self.view_identity(identities).unsqueeze(0).expand(batch, -1, -1)
        conditions = condition.unsqueeze(1).expand(-1, view_count, -1)
        scores = self.score(torch.cat((stacked, conditions, identities), dim=-1))
        weights = torch.softmax(scores.squeeze(-1), dim=1)
        scaled = stacked * (view_count * weights).unsqueeze(-1)
        return list(scaled.unbind(dim=1)), weights


class CoefficientModel(nn.Module):
    """Regress standardized aerodynamic coefficients from views and parameters.

    `parameters` is an ordered B×P tensor whose order is stored in the dataset
    manifest. Images are B×2×H×W float tensors with depth and mask channels.
    """

    def __init__(
        self,
        parameter_count: int,
        variant: str = "five",
        target_count: int = 2,
        size: str = "standard",
        encoder: str = "plain",
        fusion: str = "concat",
        view_feature_size: int | None = None,
    ) -> None:
        super().__init__()
        if parameter_count < 1:
            raise ValueError("parameter_count must be positive")
        if target_count < 1:
            raise ValueError("target_count must be positive")
        if variant not in {"parameters", "four", "five"}:
            raise ValueError("variant must be parameters, four, or five")
        if size not in {"standard", "small"}:
            raise ValueError("size must be standard or small")
        if encoder not in {
            "plain",
            "pyramid",
            "pyramid_attention",
            "pyramid_detail",
            "pyramid_detail_attention",
        }:
            raise ValueError(
                "encoder must be plain/pyramid/pyramid_attention/pyramid_detail/pyramid_detail_attention"
            )
        if fusion not in {"concat", "attention"}:
            raise ValueError("fusion must be concat or attention")
        if fusion == "attention" and variant == "parameters":
            raise ValueError("view attention requires four or five image views")
        small = size == "small"
        feature_size = view_feature_size or (32 if small else 128)
        if view_feature_size is not None and (
            view_feature_size < 1 or encoder == "plain" or variant == "parameters"
        ):
            raise ValueError(
                "view_feature_size requires positive width and pyramid images"
            )
        self.parameter_count = parameter_count
        self.variant = variant
        self.target_count = target_count
        self.fusion = fusion
        self.image_encoder = None
        self.bottom_encoder = None
        if variant != "parameters":
            if encoder == "plain":
                self.image_encoder = ViewEncoder(small)
            else:
                self.image_encoder = PyramidEncoder(
                    feature_size,
                    encoder in {"pyramid_attention", "pyramid_detail_attention"},
                    encoder in {"pyramid_detail", "pyramid_detail_attention"},
                )
        if variant == "five":
            if encoder == "plain":
                self.bottom_encoder = ViewEncoder(small)
            else:
                self.bottom_encoder = PyramidEncoder(
                    feature_size,
                    encoder in {"pyramid_attention", "pyramid_detail_attention"},
                    encoder in {"pyramid_detail", "pyramid_detail_attention"},
                )
        self.parameter_encoder = nn.Sequential(
            nn.Linear(parameter_count, 16 if small else 64),
            nn.GELU(),
            nn.Linear(16 if small else 64, 8 if small else 32),
            nn.GELU(),
        )
        n_views = {"parameters": 0, "four": 4, "five": 5}[variant]
        self.head = nn.Sequential(
            nn.Linear(
                n_views * feature_size + (8 if small else 32),
                64 if small else 256,
            ),
            nn.GELU(),
            nn.Linear(64 if small else 256, 16 if small else 64),
            nn.GELU(),
            nn.Linear(16 if small else 64, target_count),
        )
        # Keep shared encoder/head initialization paired for concat vs attention.
        self.view_attention = (
            ViewAttention(
                n_views,
                feature_size,
                8 if small else 32,
                hidden_size=max(16, (32 if small else 128) // 4),
            )
            if fusion == "attention"
            else None
        )

    def encode(
        self, views: Mapping[str, torch.Tensor], parameters: torch.Tensor
    ) -> torch.Tensor:
        """Encode complete views and conditions for coefficient and field heads."""
        if parameters.ndim != 2 or parameters.shape[1] != self.parameter_count:
            raise ValueError(
                f"parameters must have shape (batch, {self.parameter_count})"
            )
        features = []
        if self.variant != "parameters":
            required = VIEWS[:4] if self.variant == "four" else VIEWS
            missing = [name for name in required if name not in views]
            if missing:
                raise ValueError(f"missing view tensors: {', '.join(missing)}")
            # Shared weights across the four orthogonal body views; fixed order
            # in the concatenation lets the head learn view-specific effects.
            body = torch.cat([views[name] for name in VIEWS[:4]], dim=0)
            encoded = self.image_encoder(body).chunk(4, dim=0)
            features.extend(encoded)
            if self.variant == "five":
                features.append(self.bottom_encoder(views["bottom"]))
        condition = self.parameter_encoder(parameters)
        if self.view_attention is not None:
            features, _weights = self.view_attention(features, condition)
        features.append(condition)
        return torch.cat(features, dim=1)

    def forward(
        self, views: Mapping[str, torch.Tensor], parameters: torch.Tensor
    ) -> torch.Tensor:
        return self.head(self.encode(views, parameters))

    def encode_pooled(self, pooled, parameters):
        """Encode cached pre-projection pyramid features with trainable fusion."""
        if self.variant != "five" or not isinstance(self.image_encoder, PyramidEncoder):
            raise ValueError("Pooled encoding requires a five-view pyramid model")
        if parameters.ndim != 2 or parameters.shape[1] != self.parameter_count:
            raise ValueError("Unexpected parameter shape")
        body = self.image_encoder.project(torch.cat([pooled[n] for n in VIEWS[:4]]))
        features = list(body.chunk(4))
        features.append(self.bottom_encoder.project(pooled["bottom"]))
        condition = self.parameter_encoder(parameters)
        if self.view_attention is not None:
            features, _ = self.view_attention(features, condition)
        return torch.cat((*features, condition), dim=1)
