import numpy as np
import torch
from vehicle_flow_learning.diagnose_vehicle import boundary_diagnostics, force_parts


def test_uniform_flow_boundary_diagnostic_and_flux():
    class Uniform(torch.nn.Module):
        def forward(self, xyz, feature, inflow):
            return torch.cat((torch.ones_like(xyz[:, :1]), torch.zeros_like(xyz)), 1)

    xyz = torch.zeros(8, 3)
    normals = torch.tensor([[1.0, 0.0, 0.0]]).expand(8, -1)
    result = boundary_diagnostics(
        Uniform(), None, xyz, normals, np.random.default_rng(42)
    )
    assert result["wall"]["rms_speed_over_U"] == 1
    assert result["wall"]["rms_normal_speed_over_U"] == 1
    assert result["wall"]["fraction_abs_normal_speed_gt_0_05U"] == 1
    assert result["outer_box_net_flux_over_inflow"] == 0
    for name, values in result["outer_faces"].items():
        assert values["rms_velocity_error_over_U"] == (1 if name == "ground" else 0)


def test_pressure_and_viscosity_force_decomposition():
    class Field(torch.nn.Module):
        def forward(self, xyz, feature, inflow):
            x, y, z = xyz.unbind(1)
            return torch.stack((x.square(), y * 0, z * 0, 2 * x + 5), 1)

    normals = torch.cat((torch.eye(3), -torch.eye(3))).double()
    weights = torch.full((6,), 4.0, dtype=torch.float64)
    forces = force_parts(Field(), normals, normals, weights, None, 0.1)
    np.testing.assert_allclose(forces[0], [-16, 0, 0], atol=1e-12)
    np.testing.assert_allclose(forces[1], [6.4, 0, 0], atol=1e-12)
