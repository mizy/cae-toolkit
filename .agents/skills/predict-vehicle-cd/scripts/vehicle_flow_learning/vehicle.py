"""Experimental vehicle-conditioned 3D fields and explicit force integration.

Mean momentum uses a prescribed constant effective viscosity. This is a closed
regularized approximation, not a reproduction of the source hybrid RANS/LES.
No trustworthy vehicle field accuracy follows from coefficient supervision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from vehicle_cd_learning.predict import load_model


class VehicleField(nn.Module):
    """Decode spatial fields from shared full-view geometry/condition features."""

    def __init__(self, feature_count=168, width=48):
        super().__init__()
        self.condition = nn.Sequential(nn.Linear(feature_count, 24), nn.Tanh())
        self.spatial = nn.Sequential(
            nn.Linear(45, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, 4),
        )

    def forward(self, xyz, feature, inflow):
        frequencies = xyz.new_tensor([1.0, 2.0, 4.0])
        phase = xyz[:, :, None] * frequencies * torch.pi
        position = torch.cat((xyz, phase.sin().flatten(1), phase.cos().flatten(1)), 1)
        condition = self.condition(feature).expand(len(xyz), -1)
        correction = self.spatial(torch.cat((position, condition), 1))
        return correction + torch.cat((inflow, inflow.new_zeros(1)))[None]


def field_derivatives(model, xyz, feature, inflow):
    """Full coordinate derivatives, including every spatial encoding branch."""
    xyz = xyz.detach().requires_grad_(True)
    values = model(xyz, feature, inflow)
    derivatives = [
        torch.autograd.grad(values[:, k].sum(), xyz, create_graph=True)[0]
        for k in range(4)
    ]
    return xyz, values, torch.stack(derivatives[:3], 1), derivatives[3]


def momentum_residual(model, xyz, feature, inflow, viscosity):
    xyz, values, jacobian, pressure_gradient = field_derivatives(
        model, xyz, feature, inflow
    )
    laplacian = []
    for i in range(3):
        terms = [
            torch.autograd.grad(jacobian[:, i, j].sum(), xyz, create_graph=True)[0][
                :, j
            ]
            for j in range(3)
        ]
        laplacian.append(sum(terms))
    divergence = jacobian.diagonal(dim1=1, dim2=2).sum(1)
    convection = torch.einsum("nij,nj->ni", jacobian, values[:, :3])
    momentum = (
        convection + 0.5 * pressure_gradient - viscosity * torch.stack(laplacian, 1)
    )
    return torch.cat((divergence[:, None], momentum), 1)


def force_coefficients(
    model, xyz, normals, weights_over_aref, feature, inflow, viscosity
):
    """Total traction with Cp=p/(0.5*rho*U²); normals point into the fluid."""
    _, values, jacobian, _ = field_derivatives(model, xyz, feature, inflow)
    stress = -values[:, 3, None, None] * torch.eye(3, device=xyz.device)
    stress = stress + 2 * viscosity * (jacobian + jacobian.transpose(1, 2))
    # Constant stress has zero net force on a closed surface. Remove its
    # quadrature contribution to avoid force from Monte Carlo normal imbalance.
    mean_stress = (stress * weights_over_aref[:, None, None]).sum(
        0
    ) / weights_over_aref.sum()
    traction = torch.einsum("nij,nj->ni", stress - mean_stress, normals)
    force = (traction * weights_over_aref[:, None]).sum(0)
    return force[[0, 2]]


def load_vehicle(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    source = Path(checkpoint["encoder_checkpoint"])
    if hashlib.sha256(source.read_bytes()).hexdigest() != checkpoint["encoder_sha256"]:
        raise ValueError("Encoder checkpoint differs from trained feature owner")
    backbone, original = load_model(source)
    if "backbone_state" in checkpoint:
        backbone.load_state_dict(checkpoint["backbone_state"])
    else:
        backbone.head.load_state_dict(checkpoint["coefficient_head"])
    field = VehicleField(**checkpoint["field_config"])
    field.load_state_dict(checkpoint["field_state"])
    return backbone.eval(), original, field.eval(), checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "features", type=Path, help="NPZ with feature and inflow for one case"
    )
    parser.add_argument(
        "queries", type=Path, help="N x 3 NPY coordinates in car lengths"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=65536)
    args = parser.parse_args()
    torch.set_num_threads(2)
    backbone, _, model, checkpoint = load_vehicle(args.checkpoint)
    with np.load(args.features, allow_pickle=False) as data:
        feature = torch.as_tensor(data["feature"], dtype=torch.float32)
        inflow = torch.as_tensor(data["inflow"], dtype=torch.float32)
        if str(data["profile"]) != "drivaerml":
            raise ValueError("Field training currently supports only the DrivAer proxy")
        if str(data["encoder_sha256"]) != checkpoint["encoder_sha256"]:
            raise ValueError("Feature encoder differs from field checkpoint")
        if "backbone_state" in checkpoint and (
            "feature_checkpoint_sha256" not in data
            or str(data["feature_checkpoint_sha256"])
            != hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
        ):
            raise ValueError("Feature differs from fine-tuned field checkpoint")
        surface_path = Path(str(data["surface_path"]))
        length = float(data["length_m"])
    if model.condition[0].in_features == len(feature) + 2:
        with torch.inference_mode():
            feature = torch.cat((feature, backbone.head(feature)))
    if feature.shape != (model.condition[0].in_features,) or inflow.shape != (3,):
        raise ValueError("Feature and inflow shapes differ from trained model")
    if not torch.isfinite(feature).all() or not torch.isfinite(inflow).all():
        raise ValueError("Features and inflow must be finite")
    import pyvista as pv

    surface = pv.read(surface_path)
    xyz = np.load(args.queries, mmap_mode="r", allow_pickle=False)
    if (
        xyz.ndim != 2
        or xyz.shape[1] != 3
        or not np.isfinite(xyz).all()
        or args.batch_size < 1
    ):
        raise ValueError("Queries must be finite N x 3 and batch size positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output = np.lib.format.open_memmap(
        args.output, mode="w+", dtype="float32", shape=(len(xyz), 4)
    )
    with torch.inference_mode():
        for start in range(0, len(xyz), args.batch_size):
            points = np.asarray(xyz[start : start + args.batch_size])
            distance = pv.PolyData(points * length).compute_implicit_distance(surface)[
                "implicit_distance"
            ]
            valid = (
                (distance >= 0)
                & (points >= [-3, -2, 0]).all(1)
                & (points <= [6, 2, 3]).all(1)
            )
            values = np.full((len(points), 4), np.nan, dtype=np.float32)
            values[valid] = model(
                torch.tensor(points[valid], dtype=torch.float32), feature, inflow
            ).numpy()
            output[start : start + len(points)] = values
    output.flush()
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "points": len(xyz),
                "fields": ["u/U", "v/U", "w/U", "Cp"],
                "scope": "experimental proxy physics; no field accuracy validation; solid/outside masked NaN",
            }
        )
    )


if __name__ == "__main__":
    main()
