"""Audit and display real complete-car Rim variants without altering model inputs."""

import argparse
import base64
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pyvista as pv
from PIL import Image, ImageDraw

from .model import VIEWS
from .pair_experiment import load
from .prepare_drivaerrim import full_frame, merge_depth, wheel_views


def display_mesh(mesh, origin, triangles):
    """Decimate only the interactive preview; preserve raw geometry and cached inputs."""
    original = mesh.n_cells
    preview = mesh.decimate_pro(
        max(0, 1 - triangles / original), preserve_topology=False
    ).triangulate()
    points = (np.asarray(preview.points, np.float32) - origin).astype(np.float32)
    faces = np.asarray(preview.faces, np.uint32)
    return {
        "positions": base64.b64encode(points.tobytes()).decode(),
        "polys": base64.b64encode(faces.tobytes()).decode(),
        "triangles": preview.n_cells,
        "source_triangles": original,
        "display_only": True,
    }


def change_image(root, base, query, destination):
    image = Image.new("RGB", (5 * 256, 3 * 280 + 28), "#fafafa")
    draw = ImageDraw.Draw(image)
    counts = {}
    with (
        np.load(root / base["views"], allow_pickle=False) as a,
        np.load(root / query["views"], allow_pickle=False) as b,
    ):
        for i, name in enumerate(VIEWS):
            aa, bb = a[name], b[name]
            changed = np.any(abs(aa - bb) > 1e-7, axis=0)
            counts[name] = {
                "changed_pixels": int(changed.sum()),
                "total_pixels": 256 * 256,
                "silhouette_changed_pixels": int(np.count_nonzero(aa[1] != bb[1])),
                "max_depth_difference": float(abs(aa[0] - bb[0]).max()),
            }
            for row, array in enumerate((aa, bb)):
                gray = np.where(array[1] > 0, 45 + 190 * (1 - array[0]), 250).astype(
                    np.uint8
                )
                image.paste(
                    Image.fromarray(np.repeat(gray[:, :, None], 3, axis=2)),
                    (i * 256, row * 280 + 28),
                )
            diff = np.full((256, 256, 3), 250, np.uint8)
            diff[(aa[1] > 0) | (bb[1] > 0)] = [212, 216, 223]
            diff[changed] = [20, 109, 226]
            image.paste(Image.fromarray(diff), (i * 256, 2 * 280 + 28))
            draw.text((i * 256 + 8, 7), name, fill="#222222")
    image.save(destination)
    return counts


def prepare(experiment, raw):
    """@entry Verify source mesh hashes and replay every displayed full-view input."""
    p, data = load(experiment)
    out = experiment / "demo"
    out.mkdir(exist_ok=True)
    if (out / "geometry.json").exists():
        raise FileExistsError("Display geometry already prepared")
    cases = {c["case_id"]: c for c in data["cases"]}
    ids = [p["anchor_id"]] + p["display_query_ids"]
    inventory = json.loads((raw / "inventory-geometry.json").read_text())
    entries = {r["path"]: r for r in inventory["files"]}
    rows = {
        f"drivaerrim__rim-{int(r['Rim']):04d}": r
        for r in csv.DictReader(
            (raw / "NumData/Force_Coefficients_Combined.csv").open()
        )
    }
    body_parts = []
    source_hashes = {}
    for name in (
        "Car_Body",
        "WheelHouse_FL",
        "WheelHouse_RL",
        "WheelSupport_FL",
        "WheelSupport_RL",
    ):
        relative = f"Geometry/{name}/{name}.stl"
        path = raw / relative
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if sha != entries[relative]["sha256"]:
            raise ValueError("Shared geometry hash mismatch")
        source_hashes[relative] = sha
        mesh = pv.read(path)
        body_parts.append(mesh)
        if name != "Car_Body":
            mirrored = mesh.copy()
            mirrored.points[:, 1] *= -1
            body_parts.append(mirrored)
    body = body_parts[0].append_polydata(*body_parts[1:])
    tyre = pv.read(raw / "Geometry/Wheel_FL/Wheel_FL_0001.stl")
    bounds = np.array(body.bounds).reshape(3, 2)
    bounds[2, 0] = tyre.bounds[4]
    dimensions = np.diff(bounds, axis=1).ravel()
    origin = np.array([bounds[0].mean(), bounds[1].mean(), bounds[2, 0]])
    del body_parts, tyre
    started = time.monotonic()
    body_views = full_frame(body, origin, dimensions[0])
    geometry = {
        "body": display_mesh(body, origin, 60000),
        "wheels": {},
        "origin_m": origin.tolist(),
    }
    del body
    audits = {}
    source_root = Path(p["prepared"])
    for identity in ids:
        case = cases[identity]
        gid = identity.rsplit("-", 1)[-1]
        if case["targets"] != [
            float(rows[identity]["Cd"]),
            float(rows[identity]["Cl"]),
        ]:
            raise ValueError("CFD row is mismatched")
        started_case = time.monotonic()
        _, views, signature = wheel_views(gid, raw, origin, dimensions, entries)
        merge_depth(views, body_views)
        differences = {}
        with np.load(source_root / case["views"], allow_pickle=False) as saved:
            for name in VIEWS:
                current = np.stack((views[name].depth, views[name].mask)).astype(
                    np.float32
                )
                differences[name] = float(abs(current - saved[name]).max())
        if signature != case["geometry_hash"] or max(differences.values()) > 1e-7:
            raise ValueError(
                "Raw source geometry does not replay the frozen model input"
            )
        wheels = {}
        for pos in ("FL", "RL"):
            relative = f"Geometry/Wheel_{pos}/Wheel_{pos}_{gid}.stl"
            path = raw / relative
            source_hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
            wheels[pos] = display_mesh(pv.read(path), origin, 12000)
        geometry["wheels"][identity] = wheels
        audits[identity] = {
            "geometry_hash": signature,
            "views_sha256": case["views_sha256"],
            "raw_source_replay_max_abs": differences,
            "wheel_render_and_preview_seconds": time.monotonic() - started_case,
            "truth": case["targets"],
            "split": case["split"],
        }
        if identity != p["anchor_id"]:
            path = out / f"{gid}-input-change.png"
            audits[identity]["input_changes"] = change_image(
                source_root, cases[p["anchor_id"]], case, path
            )
            audits[identity]["input_image"] = path.name
        print(
            json.dumps(
                {
                    "geometry_verified": identity,
                    "replay_max_abs": max(differences.values()),
                }
            ),
            flush=True,
        )
    report = {
        "source_revision": inventory["revision"],
        "source_hashes": source_hashes,
        "cases": audits,
        "dimensions_m": dimensions.tolist(),
        "shared_geometry_unchanged": True,
        "whole_car_reconstruction": "Shared body/wheelhouses/supports plus both left wheel assemblies mirrored acrossY=0; all four wheels change together.",
        "preprocessing_total_seconds": time.monotonic() - started,
        "display_limit": "Preview body/wheels are decimated only for interaction; raw source and full256 five-view model inputs remain unchanged.",
        "source_conditions": {
            "velocity_kmh": 140,
            "alpha_deg": 0,
            "beta_deg": 0,
            "reference_area_m2": 2.16,
            "ground": "moving",
            "wheels": "MRF rims and rotating tyre walls",
            "re_body_estimate": cases[ids[0]]["conditions"]["re_body"],
            "re_source_status": "estimated air properties; not verified source CFD Re",
        },
        "source_coefficient_limit": "Released Cd/Cl rounded to4 decimals; no CFD discretization/noise confidence supplied; this is not wind-tunnel validation.",
    }
    (out / "geometry.json").write_text(
        json.dumps(geometry, separators=(",", ":")) + "\n"
    )
    (out / "geometry-audit.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("raw", type=Path)
    args = parser.parse_args()
    prepare(args.experiment, args.raw)
