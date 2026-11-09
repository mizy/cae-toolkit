"""Audit retained A models on their fixed vehicle validation geometries.

Boundary and conservation diagnostics check the declared proxy problem. They
do not supply CFD field accuracy or new-family generalization evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from vehicle_cd_learning.model import VIEWS

from .evaluate_ab import read_sample, sha256, validate_run
from .vehicle import field_derivatives, momentum_residual


def force_parts(model, xyz, normals, weights, feature, viscosity):
    inflow = xyz.new_tensor([1.0, 0.0, 0.0])
    stress_sum = torch.zeros(2, 3, 3, dtype=torch.float64)
    traction_sum = torch.zeros(2, 3, dtype=torch.float64)
    normal_sum = torch.zeros(3, dtype=torch.float64)
    weight_sum = 0.0
    for start in range(0, len(xyz), 2048):
        part = slice(start, start + 2048)
        _, value, jacobian, _ = field_derivatives(model, xyz[part], feature, inflow)
        stress = torch.stack(
            (
                -value[:, 3, None, None] * torch.eye(3),
                2 * viscosity * (jacobian + jacobian.transpose(1, 2)),
            )
        )
        stress = stress.detach().double()
        n, w = normals[part].double(), weights[part].double()
        stress_sum += (stress * w[None, :, None, None]).sum(1)
        traction_sum += (
            torch.einsum("bnij,nj->bni", stress, n) * w[None, :, None]
        ).sum(1)
        normal_sum += (n * w[:, None]).sum(0)
        weight_sum += float(w.sum())
    return (
        traction_sum - torch.einsum("bij,j->bi", stress_sum / weight_sum, normal_sum)
    ).numpy()


def outer_faces(size=24):
    limits = ((-3.0, 6.0), (-2.0, 2.0), (0.0, 3.0))
    fractions = (np.arange(size) + 0.5) / size
    a, b = np.meshgrid(fractions, fractions)
    faces = []
    for axis in range(3):
        others = [i for i in range(3) if i != axis]
        for side in range(2):
            points = np.empty((size * size, 3))
            points[:, axis] = limits[axis][side]
            area = 1.0
            for j, values in zip(others, (a, b)):
                length = limits[j][1] - limits[j][0]
                points[:, j] = limits[j][0] + length * values.ravel()
                area *= length
            normal = np.eye(3)[axis] * (1 if side else -1)
            name = ("x_min", "x_max", "y_min", "y_max", "ground", "top")[
                axis * 2 + side
            ]
            faces.append(
                (name, torch.tensor(points, dtype=torch.float32), normal, area)
            )
    return faces


def boundary_diagnostics(model, feature, xyz, normals, rng):
    inflow = xyz.new_tensor([1.0, 0.0, 0.0])
    indices = rng.choice(len(xyz), min(4096, len(xyz)), replace=False)
    with torch.no_grad():
        values = model(xyz[indices], feature, inflow).numpy()
        wall_speed = np.linalg.norm(values[:, :3], axis=1)
        penetration = np.sum(values[:, :3] * normals[indices].numpy(), axis=1)
        wall = {
            "rms_speed_over_U": float(np.sqrt(np.mean(wall_speed**2))),
            "p95_speed_over_U": float(np.quantile(wall_speed, 0.95)),
            "rms_normal_speed_over_U": float(np.sqrt(np.mean(penetration**2))),
            "fraction_abs_normal_speed_gt_0_05U": float(
                np.mean(abs(penetration) > 0.05)
            ),
        }
        faces, net_flux, incoming = {}, 0.0, 0.0
        for name, points, normal, area in outer_faces():
            result = model(points, feature, inflow).numpy()
            target = np.zeros(3) if name == "ground" else np.array([1.0, 0.0, 0.0])
            diff = result[:, :3] - target
            flux = result[:, :3] @ normal
            net_flux += float(flux.mean() * area)
            incoming += float(np.maximum(-flux, 0).mean() * area)
            faces[name] = {
                "rms_velocity_error_over_U": float(np.sqrt((diff**2).sum(1).mean())),
                "mean_abs_cp": float(abs(result[:, 3]).mean()),
            }
    return {
        "wall": wall,
        "outer_faces": faces,
        "outer_box_net_flux_over_inflow": net_flux / incoming,
        "flux_note": "Outer faces only; nonzero solid-wall penetration is reported separately. Face quadrature is a diagnostic, not a certified conservation bound.",
    }


def region_diagnostics(model, feature, case, fluid, rng):
    count = 512
    # All wake points are beyond the canonical body end x=0.5.
    wake = rng.uniform([0.6, -0.4, 0.02], [3.0, 0.4, 0.7], (count, 3))
    bulk = rng.uniform([-3, -2, 0], [6, 2, 3], (2 * count, 3))
    width, height = case["parameters"][:2]
    outside_box = (
        (abs(bulk[:, 0]) > 0.5) | (abs(bulk[:, 1]) > width / 2) | (bulk[:, 2] > height)
    )
    bulk = bulk[outside_box][:count]
    # Existing pool was filtered against the actual oriented geometry.
    indices = rng.choice(len(fluid), count, replace=False)
    points = {
        "prepared_fluid_pool": fluid[indices],
        "wake": torch.tensor(wake, dtype=torch.float32),
        "bulk_outside_body_bbox": torch.tensor(bulk, dtype=torch.float32),
    }
    result = {}
    for region, x in points.items():
        residual = (
            momentum_residual(
                model,
                x,
                feature,
                torch.tensor([1.0, 0.0, 0.0]),
                1 / case["reynolds"] + 0.001,
            )
            .detach()
            .numpy()
        )
        result[region] = {
            "rms_divergence_and_momentum": np.sqrt((residual**2).mean(0)).tolist()
        }
    return result


def audit_run(folder, output):
    saved = torch.load(folder / "model.pt", map_location="cpu", weights_only=True)
    backbone, original, field, saved, cases, _ = validate_run(
        folder, saved["identity"]["audit_sha256"], saved["epoch"]
    )
    if (
        saved["identity"]["conditioning"] != "feedback"
        or saved["identity"]["eddy_viscosity"] != 0.001
    ):
        raise ValueError(
            "This audit is for retained A and its current viscosity convention"
        )
    cache = np.load(folder / "encoded.npz", allow_pickle=False)
    pooled, parameters = (
        torch.tensor(cache["features"]),
        torch.tensor(cache["parameters"]),
    )
    lookup = {c["case_id"]: (i, c) for i, c in enumerate(cases)}
    rows = []
    for number, cid in enumerate(saved["identity"]["field_val_ids"]):
        i, case = lookup[cid]
        with torch.no_grad():
            h = backbone.encode_pooled(
                {n: pooled[i : i + 1, j] for j, n in enumerate(VIEWS)},
                parameters[i : i + 1],
            )[0]
            predicted = backbone.head(h)
            feature = torch.cat((h, predicted))
            scalar = (
                predicted * torch.tensor(original["target_std"])
                + torch.tensor(original["target_mean"])
            ).numpy()
        xyz, normals, weights, fluid = read_sample(folder, cid)
        force = force_parts(
            field, xyz, normals, weights, feature, 1 / case["reynolds"] + 0.001
        )
        rng = np.random.default_rng(20261105 + number)
        row = {
            "case_id": cid,
            "target_cd_cl": case["targets"],
            "head_cd_cl": scalar.tolist(),
            "field_cd_cl": force.sum(0)[[0, 2]].tolist(),
            "pressure_force_xyz": force[0].tolist(),
            "viscous_force_xyz": force[1].tolist(),
            "boundary": boundary_diagnostics(field, feature, xyz, normals, rng),
            "regions": region_diagnostics(field, feature, case, fluid, rng),
        }
        rows.append(row)
        print(
            json.dumps(
                {
                    "run": folder.name,
                    "case": number + 1,
                    "total": len(saved["identity"]["field_val_ids"]),
                }
            ),
            flush=True,
        )
    target = np.array([r["target_cd_cl"] for r in rows])
    summary = {}
    for name, key in [("head", "head_cd_cl"), ("field", "field_cd_cl")]:
        pred = np.array([r[key] for r in rows])
        err = abs(pred - target)
        summary[name] = {
            "mae_cd_cl": err.mean(0).tolist(),
            "mae_signed_lift_drag": float(
                abs(pred[:, 1] / pred[:, 0] - target[:, 1] / target[:, 0]).mean()
            ),
            "cl_sign_errors": int((np.sign(pred[:, 1]) != np.sign(target[:, 1])).sum()),
        }
    summary["mean_wall_metrics"] = {
        k: float(np.mean([r["boundary"]["wall"][k] for r in rows]))
        for k in rows[0]["boundary"]["wall"]
    }
    summary["mean_outer_face_metrics"] = {
        name: {
            k: float(np.mean([r["boundary"]["outer_faces"][name][k] for r in rows]))
            for k in rows[0]["boundary"]["outer_faces"][name]
        }
        for name in rows[0]["boundary"]["outer_faces"]
    }
    summary["mean_regions"] = {
        name: np.mean(
            [r["regions"][name]["rms_divergence_and_momentum"] for r in rows], axis=0
        ).tolist()
        for name in rows[0]["regions"]
    }
    summary["negative_viscous_drag_cases"] = sum(
        r["viscous_force_xyz"][0] < 0 for r in rows
    )
    summary["outer_box_relative_net_flux_mean_abs"] = float(
        np.mean([abs(r["boundary"]["outer_box_net_flux_over_inflow"]) for r in rows])
    )
    result = {
        "checkpoint": str((folder / "model.pt").resolve()),
        "checkpoint_sha256": sha256(folder / "model.pt"),
        "epoch": saved["epoch"],
        "seed": saved["identity"]["seed"],
        "test_used": False,
        "scope": "Fixed development-validation geometries and weights; boundary diagnostics for proxy no-slip/outer problem; no CFD field truth. Pressure/viscous decomposition uses identical full-surface correction.",
        "summary": summary,
        "cases": rows,
    }
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    for folder in args.runs:
        audit_run(folder, args.output_dir / f"{folder.name}.json")


if __name__ == "__main__":
    main()
