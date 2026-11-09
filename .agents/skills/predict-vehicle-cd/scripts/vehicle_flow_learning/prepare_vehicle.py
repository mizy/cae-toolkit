"""Prepare cached encoder inputs and sampled vehicle surfaces for training."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pyvista as pv
import torch
from vehicle_cd_learning.data import CaseDataset, load_dataset
from vehicle_cd_learning.model import VIEWS

from .surface import prepare


def encode_cases(audit, backbone, checkpoint, output, pooled=False):
    """Use the existing five-view encoder and preserve every frozen split."""
    root = Path(audit["prepared_root"])
    dataset = load_dataset(root)
    if (
        hashlib.sha256((root / "dataset.json").read_bytes()).hexdigest()
        != audit["dataset_sha256"]
    ):
        raise ValueError("Prepared data differs from vehicle audit")
    if checkpoint["dataset_sha256"] != audit["dataset_sha256"]:
        raise ValueError("Encoder and audited data differ")
    cases = [c for c in dataset["cases"] if c["split"] != "test"]
    loader = CaseDataset(
        root,
        cases,
        checkpoint["parameter_mean"],
        checkpoint["parameter_std"],
        checkpoint["target_mean"],
        checkpoint["target_std"],
    )
    features = []
    parameters_list = []
    with torch.inference_mode():
        for index in range(len(cases)):
            views, parameters, _ = loader[index]
            if pooled:
                encoders = [backbone.image_encoder] * 4 + [backbone.bottom_encoder]
                features.append(
                    torch.stack(
                        [
                            e.pooled_features(views[n][None])[0]
                            for n, e in zip(VIEWS, encoders)
                        ]
                    )
                )
                parameters_list.append(parameters)
            else:
                features.append(
                    backbone.encode(
                        {k: v[None] for k, v in views.items()}, parameters[None]
                    )[0]
                )
    array = torch.stack(features).numpy()
    np.savez_compressed(
        output / "encoded.npz",
        features=array,
        parameters=torch.stack(parameters_list).numpy() if pooled else np.empty((0,)),
        targets=np.asarray([c["targets"] for c in cases], dtype=float),
    )
    (output / "encoded-cases.json").write_text(json.dumps(cases, indent=2) + "\n")
    return (
        cases,
        torch.from_numpy(array),
        (torch.stack(parameters_list) if pooled else None),
    )


def prepare_pilot(audit_path, audit, output, train_count, val_count, surface_count):
    """Bind only real DrivAer train/val geometry with known force reference area."""
    selected = []
    for split, count in (("train", train_count), ("val", val_count)):
        selected.extend(
            [
                c
                for c in audit["cases"]
                if c["split"] == split and c["condition_profile"] == "drivaerml"
            ][:count]
        )
    samples = []
    for index, case in enumerate(selected):
        print(
            json.dumps(
                {
                    "stage": "prepare_geometry",
                    "case_id": case["case_id"],
                    "index": index + 1,
                    "total": len(selected),
                }
            ),
            flush=True,
        )
        folder = output / "surfaces" / case["case_id"]
        if not folder.exists():
            prepare(audit_path, case["case_id"], folder, surface_count, 42)
        metadata = json.loads((folder / "surface.json").read_text())
        with Path(metadata["geometry_path"]).open("rb") as handle:
            geometry_sha = hashlib.file_digest(handle, "sha256").hexdigest()
        manifest = metadata["source_manifest"]
        with Path(manifest["path"]).open("rb") as handle:
            manifest_sha = hashlib.file_digest(handle, "sha256").hexdigest()
        if (
            metadata["case_id"] != case["case_id"]
            or metadata["audit_sha256"]
            != hashlib.sha256(audit_path.read_bytes()).hexdigest()
            or metadata["geometry_convention"] != case["geometry"]
            or metadata["geometry_sha256"] != geometry_sha
            or manifest["sha256"] != manifest_sha
            or metadata["count"] != surface_count
            or metadata["seed"] != 42
        ):
            raise ValueError(f"Prepared surface identity changed: {case['case_id']}")
        if not metadata.get("experimental_force_quadrature"):
            raise ValueError(
                f"Surface cannot support experimental force quadrature: {case['case_id']}"
            )
        with np.load(folder / "surface.npz") as arrays:
            xyz = arrays["points_m"] / metadata["normalization"]["length_m"]
            normals, weights = (
                arrays["normals"],
                arrays["weights_m2"] / metadata["reference_area_m2"],
            )
        pool_path = folder / "fluid-pool.npy"
        if pool_path.exists():
            pool = np.load(pool_path)
        else:
            rng = np.random.default_rng(42 + index)
            pool = rng.uniform([-3, -2, 0], [6, 2, 3], (4096, 3))
            near = xyz + normals * rng.uniform(0.005, 0.05, (len(xyz), 1))
            pool = np.concatenate((pool, near))
            mesh = pv.read(folder / "surface.vtp")
            points_m = pool * metadata["normalization"]["length_m"]
            distance = pv.PolyData(points_m).compute_implicit_distance(mesh)[
                "implicit_distance"
            ]
            pool = pool[(distance > 1e-5) & (pool[:, 2] > 0)]
            np.save(pool_path, pool)
        if len(pool) < 128:
            raise ValueError("Insufficient exterior fluid samples")
        samples.append(
            {
                "case": case,
                "surface_path": str((folder / "surface.vtp").resolve()),
                "length_m": metadata["normalization"]["length_m"],
                "xyz": torch.tensor(xyz, dtype=torch.float32),
                "normals": torch.tensor(normals, dtype=torch.float32),
                "weights": torch.tensor(weights, dtype=torch.float32),
                "fluid": torch.tensor(pool, dtype=torch.float32),
            }
        )
    return samples
