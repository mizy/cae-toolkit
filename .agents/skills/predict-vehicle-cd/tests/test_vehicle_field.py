import sys
from pathlib import Path

import pytest
import torch

SKILL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL / "scripts"))

from vehicle_cd_learning.model import VIEWS, CoefficientModel
from vehicle_flow_learning import train_vehicle
from vehicle_flow_learning.vehicle import (
    VehicleField,
    field_derivatives,
    force_coefficients,
    momentum_residual,
)


@pytest.mark.parametrize(
    "variant,fusion", [("parameters", "concat"), ("five", "attention")]
)
def test_coefficient_forward_uses_encoded_features(variant, fusion):
    torch.manual_seed(2)
    model = CoefficientModel(3, variant=variant, size="small", fusion=fusion).eval()
    views = {name: torch.randn(2, 2, 32, 32) for name in VIEWS}
    parameters = torch.randn(2, 3)
    torch.testing.assert_close(
        model(views, parameters), model.head(model.encode(views, parameters))
    )


def test_pooled_fine_tuning_preserves_encoding_and_scalar_gradients():
    torch.manual_seed(7)
    model = CoefficientModel(
        3,
        variant="five",
        size="small",
        encoder="pyramid_detail_attention",
        fusion="attention",
    ).eval()
    views = {name: torch.rand(2, 2, 32, 32) for name in VIEWS}
    parameters = torch.randn(2, 3)
    with torch.no_grad():
        pooled = {
            name: (
                model.bottom_encoder if name == "bottom" else model.image_encoder
            ).pooled_features(views[name])
            for name in VIEWS
        }
        expected = model.encode(views, parameters)
    actual = model.encode_pooled(pooled, parameters)
    torch.testing.assert_close(actual, expected)
    model.head(actual).square().mean().backward()
    assert model.image_encoder.project[1].weight.grad.abs().sum() > 0
    assert model.bottom_encoder.project[1].weight.grad.abs().sum() > 0
    assert model.parameter_encoder[0].weight.grad.abs().sum() > 0
    assert model.view_attention.score[-1].weight.grad.abs().sum() > 0
    assert model.image_encoder.blocks[0][0].weight.grad is None


def test_paired_field_shared_initialization_and_random_stream():
    torch.manual_seed(42)
    feedback, extra_a = train_vehicle.paired_field(168, "feedback")
    next_a = torch.rand(5)
    torch.manual_seed(42)
    latent, extra_b = train_vehicle.paired_field(168, "latent")
    next_b = torch.rand(5)
    torch.testing.assert_close(
        feedback.condition[0].weight[:, :168], latent.condition[0].weight
    )
    torch.testing.assert_close(feedback.condition[0].bias, latent.condition[0].bias)
    torch.testing.assert_close(feedback.condition[0].weight[:, 168:], extra_a)
    torch.testing.assert_close(extra_a, extra_b)
    for a, b in zip(feedback.spatial.parameters(), latent.spatial.parameters()):
        torch.testing.assert_close(a, b)
    torch.testing.assert_close(next_a, next_b)


@pytest.mark.parametrize("conditioning", ["feedback", "latent"])
def test_field_and_auxiliary_gradients_reach_shared_encoder(conditioning):
    torch.manual_seed(33)
    backbone = CoefficientModel(
        3,
        variant="five",
        size="small",
        encoder="pyramid_detail_attention",
        fusion="attention",
    )
    pooled = {name: torch.randn(2, 16, 8, 8) for name in VIEWS}
    parameters = torch.randn(2, 3)
    latent = backbone.encode_pooled(pooled, parameters)
    backbone.head(latent).square().mean().backward()
    assert backbone.image_encoder.project[1].weight.grad.abs().sum() > 0
    assert backbone.view_attention.score[-1].weight.grad.abs().sum() > 0
    backbone.zero_grad(set_to_none=True)
    latent = backbone.encode_pooled(pooled, parameters)[0]
    field, _ = train_vehicle.paired_field(len(latent), conditioning)
    feature = (
        torch.cat((latent, backbone.head(latent)))
        if conditioning == "feedback"
        else latent
    )
    sample = {
        "case": {"reynolds": 1e7, "targets": [0.3, 0.1]},
        "xyz": torch.randn(8, 3),
        "normals": torch.randn(8, 3),
        "weights": torch.rand(8),
        "fluid": torch.randn(8, 3),
    }
    field_loss, _, _, _ = train_vehicle.field_loss(
        field, sample, feature, 3, 0.001, force_count=5, cl_weight=4
    )
    field_loss.backward()
    assert backbone.image_encoder.project[1].weight.grad.abs().sum() > 0
    assert backbone.view_attention.score[-1].weight.grad.abs().sum() > 0
    head_grad = backbone.head[0].weight.grad
    if conditioning == "feedback":
        assert head_grad is not None and head_grad.abs().sum() > 0
    else:
        assert head_grad is None


def test_chunked_force_uses_one_global_stress_mean():
    from vehicle_flow_learning.evaluate_ab import integrated_force

    torch.manual_seed(19)
    field = VehicleField(feature_count=8, width=12)
    xyz = torch.randn(17, 3)
    normals = torch.randn(17, 3)
    weights = torch.rand(17) + 0.1
    feature = torch.randn(8)
    inflow = torch.tensor([1.0, 0.0, 0.0])
    expected = force_coefficients(field, xyz, normals, weights, feature, inflow, 0.001)
    actual = integrated_force(field, xyz, normals, weights, feature, inflow, 0.001, 4)
    torch.testing.assert_close(
        torch.tensor(actual), expected.detach(), rtol=2e-5, atol=2e-6
    )


class PolynomialField(torch.nn.Module):
    def forward(self, xyz, feature, inflow):
        x, y, z = xyz.unbind(1)
        return torch.stack((1 + x**2, 2 + y**2, 3 + z**2, 2 * x - 4 * y + 6 * z), 1)


def test_momentum_residual_matches_analytic_3d_field():
    xyz = torch.tensor([[0.2, -0.3, 0.4], [-0.5, 0.6, -0.7]], dtype=torch.float64)
    viscosity = 0.3
    x, y, z = xyz.unbind(1)
    expected = torch.stack(
        (
            2 * x + 2 * y + 2 * z,
            2 * x * (1 + x * x) + 1 - 2 * viscosity,
            2 * y * (2 + y * y) - 2 - 2 * viscosity,
            2 * z * (3 + z * z) + 3 - 2 * viscosity,
        ),
        1,
    )
    actual = momentum_residual(PolynomialField(), xyz, None, None, viscosity)
    torch.testing.assert_close(actual, expected)


def test_all_field_outputs_have_true_coordinate_derivatives():
    torch.manual_seed(13)
    model = VehicleField(feature_count=5, width=12).double()
    xyz = torch.randn(4, 3, dtype=torch.float64)
    feature, inflow = (
        torch.randn(1, 5, dtype=torch.float64),
        torch.randn(3, dtype=torch.float64),
    )
    _, _, jacobian, pressure = field_derivatives(model, xyz, feature, inflow)
    actual = torch.cat((jacobian, pressure[:, None, :]), 1)
    epsilon = 1e-5
    finite = torch.stack(
        [
            (
                model(xyz + torch.eye(3, dtype=xyz.dtype)[i] * epsilon, feature, inflow)
                - model(
                    xyz - torch.eye(3, dtype=xyz.dtype)[i] * epsilon, feature, inflow
                )
            )
            / (2 * epsilon)
            for i in range(3)
        ],
        2,
    )
    torch.testing.assert_close(actual, finite, rtol=1e-5, atol=1e-7)
    assert (actual.abs().sum((0, 2)) > 1e-5).all()


def test_closed_box_pressure_and_constant_stress_force():
    class BoxField(torch.nn.Module):
        def __init__(self, gradient):
            super().__init__()
            self.gradient = gradient

        def forward(self, xyz, feature, inflow):
            x, y, z = xyz.unbind(1)
            return torch.stack((x + 2 * y, y + z, x - z, 5 + self.gradient * x), 1)

    # Box [-1,1] x [-2,2] x [-3,3], face centers and outward normals.
    xyz = torch.tensor(
        [[1, 0, 0], [-1, 0, 0], [0, 2, 0], [0, -2, 0], [0, 0, 3], [0, 0, -3]],
        dtype=torch.float64,
    )
    normals = torch.tensor(
        [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
        dtype=torch.float64,
    )
    areas = torch.tensor([24, 24, 12, 12, 8, 8], dtype=torch.float64)
    for gradient, expected in [(0, [0, 0]), (2, [-96, 0])]:
        force = force_coefficients(
            BoxField(gradient), xyz, normals, areas, None, None, 0.3
        )
        torch.testing.assert_close(
            force, torch.tensor(expected, dtype=xyz.dtype), atol=1e-12, rtol=0
        )


def test_chunked_query_and_field_loss_gradient():
    torch.manual_seed(17)
    model = VehicleField(feature_count=5, width=12).double()
    xyz = torch.randn(11, 3, dtype=torch.float64)
    feature, inflow = (
        torch.randn(1, 5, dtype=torch.float64),
        torch.randn(3, dtype=torch.float64),
    )
    full = model(xyz, feature, inflow)
    chunked = torch.cat([model(chunk, feature, inflow) for chunk in xyz.split(3)])
    torch.testing.assert_close(full, chunked)
    momentum_residual(model, xyz, feature, inflow, 0.1).square().mean().backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()
    )


def test_pressure_reference_does_not_create_sampled_surface_force():
    class Field(torch.nn.Module):
        def __init__(self, offset):
            super().__init__()
            self.offset = offset

        def forward(self, xyz, feature, inflow):
            x, y, z = xyz.unbind(1)
            return torch.stack((x * y, y * z, z * x, x * x + self.offset), 1)

    torch.manual_seed(3)
    xyz = torch.randn(19, 3, dtype=torch.float64)
    normals = torch.randn_like(xyz)
    normals = normals / normals.norm(dim=1, keepdim=True)
    weights = torch.ones(19, dtype=torch.float64)
    a = force_coefficients(Field(0), xyz, normals, weights, None, None, 0.1)
    b = force_coefficients(Field(100), xyz, normals, weights, None, None, 0.1)
    torch.testing.assert_close(a, b, atol=1e-12, rtol=0)


def test_coefficient_context_receives_field_training_gradient():
    torch.manual_seed(11)
    head = torch.nn.Linear(5, 2)
    model = VehicleField(feature_count=7, width=12)
    feature = torch.randn(5)
    condition = torch.cat((feature, head(feature)))
    residual = momentum_residual(
        model, torch.rand(8, 3), condition, torch.tensor([1.0, 0.0, 0.0]), 0.01
    )
    residual.square().mean().backward()
    assert head.weight.grad is not None and head.weight.grad.norm() > 0


def test_field_cl_weight_changes_only_the_lift_supervision_gradient(monkeypatch):
    class Field(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.coefficients = torch.nn.Parameter(torch.tensor([0.04, 0.2]))

        def forward(self, xyz, feature, inflow):
            return xyz.new_zeros((len(xyz), 4))

    model = Field()
    monkeypatch.setattr(
        train_vehicle, "force_coefficients", lambda *args: model.coefficients
    )
    monkeypatch.setattr(
        train_vehicle,
        "momentum_residual",
        lambda model, xyz, *args: xyz.new_zeros((len(xyz), 4)),
    )
    monkeypatch.setattr(train_vehicle, "boundary_loss", lambda *args: torch.tensor(0.0))
    sample = {
        "case": {"reynolds": 10, "targets": [0.0, 0.0]},
        "xyz": torch.rand(4, 3),
        "normals": torch.ones(4, 3),
        "weights": torch.ones(4),
        "fluid": torch.rand(4, 3),
    }
    feature = torch.ones(2)
    a = train_vehicle.field_loss(model, sample, feature, 2, 0.001)[0]
    b = train_vehicle.field_loss(model, sample, feature, 2, 0.001, cl_weight=4)[0]
    ga = torch.autograd.grad(a, model.coefficients)[0]
    gb = torch.autograd.grad(b, model.coefficients)[0]
    torch.testing.assert_close(gb, ga * torch.tensor([1.0, 4.0]))
