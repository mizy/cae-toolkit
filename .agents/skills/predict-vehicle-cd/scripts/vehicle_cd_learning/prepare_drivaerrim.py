"""Reconstruct complete DrivAerRim views from shared body and mirrored wheel STL."""

import argparse
import csv
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyvista as pv

from .render import VIEW_AXES, _rasterize, save_five_views, save_preview


def merge_depth(first, second):
    """Compose component z-buffers in the same full-car camera frame."""
    for name in VIEW_AXES:
        a, b = first[name], second[name]
        use = b.mask & (~a.mask | (b.depth < a.depth))
        a.depth[use] = b.depth[use]
        a.mask[:] |= b.mask
    return first


def full_frame(mesh, origin, length):
    points = ((mesh.points - origin) / length).astype(np.float32)
    if np.max(np.abs(points[:, :2])) > 0.65 or points[:, 2].min() < -1e-5:
        raise ValueError("Part falls outside the declared full-car frame")
    faces = np.asarray(mesh.faces).reshape(-1, 4)[:, 1:]
    return {
        k: _rasterize(points, faces, view=k, size=256, window_l=1.3) for k in VIEW_AXES
    }


def wheel_views(gid, raw, origin, dimensions, entries):
    digest = hashlib.sha256()
    views = None
    for pos in ("FL", "RL"):
        relative = f"Geometry/Wheel_{pos}/Wheel_{pos}_{gid}.stl"
        path = raw / relative
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != entries[relative]["sha256"]:
            raise ValueError(f"Wheel source changed: {relative}")
        digest.update(actual.encode())
        mesh = pv.read(path)
        if not np.isclose(mesh.bounds[4], origin[2], atol=1e-5):
            raise ValueError("Unexpected tyre contact height")
        for mirror in (False, True):
            if mirror:
                mesh.points[:, 1] *= -1
            current = full_frame(mesh, origin, dimensions[0])
            views = current if views is None else merge_depth(views, current)
    # These are full-car camera buffers; no local crop is saved or fed to the CNN.
    return gid, views, digest.hexdigest()


def main():
    """@entry Verify all geometry/labels and preserve official wheel-ID splits."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("raw", type=Path)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--workers", type=int, default=2)
    args = p.parse_args()
    root = args.output_dir
    if root.exists():
        raise FileExistsError("Use a fresh output directory")
    inventory = json.loads((args.raw / "inventory-geometry.json").read_text())
    if not inventory["complete"]:
        raise ValueError("Source download/integrity verification incomplete")
    entries = {r["path"]: r for r in inventory["files"]}
    rows = list(
        csv.DictReader((args.raw / "NumData/Force_Coefficients_Combined.csv").open())
    )
    if len(rows) != 904 or {int(r["Rim"]) for r in rows} != set(range(1, 905)):
        raise ValueError("Expected all 904 uniquely labeled rims")
    splits = {}
    for split, filename in (
        ("train", "case_01_train_720/train_ids.txt"),
        ("val", "validation_ids.txt"),
        ("test", "test_ids.txt"),
    ):
        for line in (args.raw / "Dataset_Splits" / filename).read_text().splitlines():
            gid = line.strip().rsplit("_", 1)[-1]
            if gid in splits:
                raise ValueError("Wheel ID crosses source splits")
            splits[gid] = split
    if len(splits) != 904:
        raise ValueError("Incomplete source split")
    parts = []
    for part in (
        "Car_Body",
        "WheelHouse_FL",
        "WheelHouse_RL",
        "WheelSupport_FL",
        "WheelSupport_RL",
    ):
        relative = f"Geometry/{part}/{part}.stl"
        path = args.raw / relative
        if hashlib.sha256(path.read_bytes()).hexdigest() != entries[relative]["sha256"]:
            raise ValueError("Shared geometry changed")
        mesh = pv.read(path)
        parts.append(mesh)
        if part != "Car_Body":
            mirrored = mesh.copy()
            mirrored.points[:, 1] *= -1
            parts.append(mirrored)
    body = parts[0].append_polydata(*parts[1:])
    tyre = pv.read(args.raw / "Geometry/Wheel_FL/Wheel_FL_0001.stl")
    bounds = np.array(body.bounds).reshape(3, 2)
    bounds[2, 0] = tyre.bounds[4]
    dimensions = np.diff(bounds, axis=1).ravel()
    origin = np.array([bounds[0].mean(), bounds[1].mean(), bounds[2, 0]])
    body_views = full_frame(body, origin, dimensions[0])
    del body, parts
    (root / "views").mkdir(parents=True)
    (root / "previews").mkdir()
    geometries = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = [
            pool.submit(wheel_views, gid, args.raw, origin, dimensions, entries)
            for gid in splits
        ]
        for future in as_completed(jobs):
            gid, views, signature = future.result()
            # Wheel buffers are disposable, so compose the shared body into them.
            merge_depth(views, body_views)
            save_five_views(root / "views" / f"{gid}.npz", views)
            if len(geometries) < 12:
                save_preview(root / "previews" / f"{gid}.png", views)
            geometries[gid] = {
                "sha256": signature,
                "dimensions": dimensions.tolist(),
                "views_sha256": hashlib.sha256(
                    (root / "views" / f"{gid}.npz").read_bytes()
                ).hexdigest(),
            }
            if len(geometries) % 20 == 0:
                print(
                    json.dumps({"prepared": len(geometries), "total": 904}), flush=True
                )
    cases, hashes = [], {}
    for row in rows:
        gid = f"{int(row['Rim']):04d}"
        targets = [float(row["Cd"]), float(row["Cl"])]
        if not np.isfinite(targets).all() or targets[0] <= 0:
            raise ValueError("Invalid coefficient")
        signature = geometries[gid]["sha256"]
        if signature in hashes and hashes[signature] != splits[gid]:
            raise ValueError("Identical wheel geometry across splits")
        hashes[signature] = splits[gid]
        cases.append(
            {
                "case_id": f"rim-{gid}",
                "geometry_id": gid,
                "family_id": "drivaerrim-estateback",
                "split": splits[gid],
                "parameters": [
                    float(dimensions[1] / dimensions[0]),
                    float(dimensions[2] / dimensions[0]),
                    140 / 3.6,
                ],
                "condition_parameters": [140 / 3.6],
                "targets": targets,
                "conditions": {
                    "velocity_m_s": 140 / 3.6,
                    "yaw_deg": 0,
                    "reference_area_m2": 2.16,
                },
            }
        )
    data = {
        "dataset": inventory["repo"],
        "revision": inventory["revision"],
        "license": "CC BY-NC 4.0",
        "parameter_names": ["width_over_length", "height_over_length", "velocity_m_s"],
        "condition_names": ["velocity_m_s"],
        "geometry": geometries,
        "cases": cases,
        "render_config": {"size": 256},
        "geometry_convention": "metres, nose -X, up +Z, mirror left components across Y=0; complete-car buffers",
        "training_conditions": {
            "velocity_m_s": 140 / 3.6,
            "yaw_deg": 0,
            "density_kg_m3": 1.204,
            "reference_area_m2": 2.16,
            "ground": "moving no-slip wall",
            "wheels": "MRF rims and rotating tyre walls",
            "solver": "STAR-CCM+ steady RANS realizable k-epsilon",
            "averaging": "final 500 steady iterations",
            "reynolds": None,
            "reynolds_note": "Source viscosity not verified; fixed-condition model does not invent Re",
        },
        "scope": "Unseen rim designs on one fixed DrivAer Estateback at 140 km/h; no speed or vehicle-family generalization claim",
    }
    (root / "dataset.json").write_text(
        json.dumps(data, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
