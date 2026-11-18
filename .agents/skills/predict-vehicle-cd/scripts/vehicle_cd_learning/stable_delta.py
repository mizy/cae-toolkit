"""Coverage-bounded pooling for the existing full-view paired delta model.

The floor attenuates sparsely occupied bins without changing neural tensors.
Rasterization and maximum selection can still introduce sampling sensitivity.
"""

import torch
from torch.nn import functional as F

from .model import VIEWS
from .paired_delta import PairedDelta


class StableDelta(PairedDelta):
    """Keep the paired model's weights and bound low-coverage contributions."""

    def __init__(
        self, pool_size=8, parameter_count=36, coverage_floor=1 / 16, filter_input=False
    ):
        super().__init__(pool_size, parameter_count)
        if not 0 <= coverage_floor <= 1:
            raise ValueError("coverage_floor must be between zero and one")
        self.coverage_floor = float(coverage_floor)
        self.filter_input = bool(filter_input)

    def encode(self, views):
        """@entry Attenuate the original spatial features by foreground coverage."""
        if self.filter_input:
            kernel = (
                views[VIEWS[0]].new_tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]])[
                    None, None
                ]
                / 16
            )
            views = {
                name: F.conv2d(
                    F.pad(
                        torch.cat((v[:, :1] * v[:, 1:2], v[:, 1:2]), dim=1),
                        (1, 1, 1, 1),
                        mode="replicate",
                    ),
                    kernel.expand(2, 1, 3, 3),
                    groups=2,
                )
                for name, v in views.items()
            }
        pooled = super().encode(views)
        if self.coverage_floor:
            for name in VIEWS:
                gate = (pooled[name + "_mask"] / self.coverage_floor).clamp_max(1)
                pooled[name] = pooled[name] * gate
        return pooled

    def predict_pair(self, pooled_a, pooled_b, parameters):
        """@entry Compare B−A using continuous coverage and a bounded pool mean."""
        if not self.coverage_floor:
            return super().predict_pair(pooled_a, pooled_b, parameters)
        context = torch.cat([(pooled_a[n] + pooled_b[n]) * 0.5 for n in VIEWS])
        difference = torch.cat([pooled_b[n] - pooled_a[n] for n in VIEWS])
        mask = torch.cat(
            [torch.maximum(pooled_a[n + "_mask"], pooled_b[n + "_mask"]) for n in VIEWS]
        ).repeat(2, 1, 1, 1)
        features = self.adapter(
            torch.cat(
                (
                    torch.cat((context, difference), dim=1),
                    torch.cat((context, -difference), dim=1),
                )
            )
        )
        coverage = F.adaptive_avg_pool2d(mask, 2)
        mean = F.adaptive_avg_pool2d(features * mask, 2)
        mean = mean / coverage.clamp_min(self.coverage_floor)
        maximum = F.adaptive_max_pool2d(features.masked_fill(mask <= 0, -torch.inf), 2)
        maximum = torch.where(coverage > 0, maximum, 0)
        maximum = maximum * (coverage / self.coverage_floor).clamp_max(1)
        positive, negative = torch.cat((mean, maximum), dim=1).chunk(2)
        batch = parameters.shape[0]
        condition = self.parameter_encoder(parameters)

        def predict(value):
            per_view = value.reshape(5, batch, -1).transpose(0, 1).reshape(batch, -1)
            return self.head(torch.cat((per_view, condition), dim=1))

        return (predict(positive) - predict(negative)) * 0.5


def initialize_from_checkpoint(checkpoint, coverage_floor=1 / 16, filter_input=False):
    """@entry Copy paired-model tensors strictly, preserving their trainability."""
    model = StableDelta(
        **checkpoint["model_config"],
        coverage_floor=coverage_floor,
        filter_input=filter_input,
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model
