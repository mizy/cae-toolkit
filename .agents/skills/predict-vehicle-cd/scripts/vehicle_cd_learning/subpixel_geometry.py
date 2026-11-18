"""Collect retained-model features from real, independently sampled depth buffers."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pyvista as pv
import torch

from .absolute_view_renders import PHASES
from .delta_experiment import sha256, write_json
from .joint_sampling import load, load_checkpoint
from .model import VIEWS
from .phase_decoder_data import OFF_GRID
from .phase_views import checked_mesh
from .render import canonical_points
from .rim_resolution import SHARED
from .subpixel_sampling import (
    NATIVE,
    acquisition_offset,
    area_average,
    capture,
    project_vertices,
    resolve,
)

STATES = (*PHASES, *OFF_GRID)
METHODS = ("point", "single_inverse", "area4", "fine_inverse", "registered")


def rim_triangles(raw, entries, specifications, frame):
    parts = []
    for path, mirror in specifications:
        mesh = checked_mesh(raw, path, entries, {})
        faces = np.ascontiguousarray(mesh.faces.reshape(-1, 4)[:, 1:], np.int32)
        for reflected in range(2 if mirror else 1):
            if reflected:
                mesh.points[:, 1] *= -1
            points = (
                (mesh.points - frame["origin_m"]) / frame["dimensions_m"][0]
            ).astype(np.float32)
            parts.append((points, faces))
    return parts


def aircraft_triangles(case):
    path = Path(case["source_evidence"]["source_mesh"])
    if sha256(path) != case["geometry_hash"]:
        raise ValueError("Aircraft source hash differs")
    mesh = pv.read(path)
    hilift = case["source"] == "hilift"
    if hilift:
        other = mesh.copy()
        other.points[:, 1] *= -1
        mesh = mesh.append_polydata(other)
    points, dim = canonical_points(
        mesh.points, unit_scale_to_m=0.0254 if hilift else 0.001
    )
    points /= max(dim["normalized_extents_l"])
    faces = np.ascontiguousarray(
        mesh.triangulate().faces.reshape(-1, 4)[:, 1:], np.int32
    )
    return [(points, faces)]


def acquire(parts, size, phase, shift, base=None):
    buffers = {}
    for view in VIEWS:
        z = None if base is None else base[view].copy()
        for xyz, faces in parts:
            z = capture(xyz[view], faces, size, 4, phase, shift, z)
        buffers[view] = z
    return buffers


def projections(parts, size):
    return [
        ({n: project_vertices(points, n, size) for n in VIEWS}, faces)
        for points, faces in parts
    ]


def configurations(cases, joint, raw_root, decoder_design):
    raw = json.loads((raw_root / "manifest.json").read_text())["geometries"]
    selected = {
        g: {"case_ids": v["case_ids"], "split": "val", "source": v["source"]}
        for g, v in raw.items()
    }
    by_id = {c["case_id"]: c for c in cases}
    additional = list(decoder_design["off_grid_test_ids"]) + joint["anchors"]
    for cid in additional:
        c = by_id[cid]
        selected[c["geometry_id"]] = {
            "case_ids": [cid],
            "split": c["split"],
            "source": c["source"],
        }
    rng = np.random.default_rng(20261118)
    for source in ("cocoanet", "hilift"):
        groups = {}
        for c in cases:
            if c["source"] == source and c["split"] == "test":
                groups.setdefault(c["geometry_id"], []).append(c["case_id"])
        for gid in rng.choice(sorted(groups), 3, replace=False):
            selected[str(gid)] = {
                "case_ids": groups[gid],
                "split": "test",
                "source": source,
            }
    return selected


def feature_record(model, parts, body, size, nominal, device, sample_file=None):
    values = {}
    max_error = {m: [] for m in METHODS}
    mask_changes = []
    wrong_phase = []
    for i, phase in enumerate(STATES):
        offset = acquisition_offset(phase, 4)
        fine = acquire(parts, size, phase, (0.0, 0.0), body.get((i, False)))
        locked = (
            acquire(parts, size, phase, offset, body.get((i, True)))
            if np.any(offset)
            else fine
        )
        arrays = {n: [] for n in VIEWS}
        errors = {m: 0.0 for m in METHODS}
        raw_count = 0
        wrong = 0.0
        for n in VIEWS:
            d = fine[n]
            measured = np.asarray(phase)
            lo = d[2::4, 2::4]
            images = (
                resolve(d, size, 4),
                resolve(lo, size, 1, measured),
                area_average(d, size, 4),
                resolve(d, size, 4, measured),
                resolve(locked[n], size, 4, measured - offset),
            )
            for method, image in zip(METHODS, images):
                errors[method] = max(
                    errors[method], float(np.abs(image - nominal[n]).max())
                )
            raw_count += int(np.count_nonzero(images[0][1] != nominal[n][1]))
            wrong = max(
                wrong, float(np.abs(resolve(locked[n], size, 4) - nominal[n]).max())
            )
            if sample_file is not None and n == "bottom" and i == 5:
                np.savez_compressed(
                    sample_file,
                    captured_depth=locked[n],
                    source_phase=phase,
                    programmed_offset=offset,
                    centre_read_without_registration=resolve(locked[n], size, 4),
                    registered=images[-1],
                    original=nominal[n],
                )
            arrays[n] = torch.from_numpy(np.stack(images)).to(device)
        if i == 0 and errors["point"] > 2e-6:
            raise ValueError(
                f"Native rasterizer failed to replay original inputs: {errors}"
            )
        with torch.inference_mode():
            encoded = model.encode(arrays)
        for k, t in encoded.items():
            values.setdefault(k, []).append(t.cpu().numpy())
        for m in METHODS:
            max_error[m].append(errors[m])
        mask_changes.append(raw_count)
        wrong_phase.append(wrong)
        del fine, locked
    return {k: np.stack(v, axis=1) for k, v in values.items()}, {
        "max_input_difference": max_error,
        "raw_mask_pixels_changed": mask_changes,
        "wrong_registration_max_difference": wrong_phase,
    }


def prepare(experiment, audit, decoder_root, output):
    """@entry Compare five capture/resolve contracts on development and held-out shapes."""
    p, cases = load(experiment)
    by_id = {c["case_id"]: c for c in cases}
    retained = json.loads((experiment / "retention.json").read_text())
    if sha256(retained["checkpoint"]) != retained["sha256"]:
        raise ValueError("Retained checkpoint changed")
    parent = json.loads((Path(p["parent_experiment"]) / "protocol.json").read_text())
    raw_root = Path(parent["inventory"]).parent
    inventory = Path(parent["inventory"])
    entries = {f["path"]: f for f in json.loads(inventory.read_text())["files"]}
    if sha256(inventory) != parent["input_sha256"][str(inventory)]:
        raise ValueError("Rim inventory changed")
    nominal_root = Path(parent["prepared512"])
    nominal_data = json.loads((nominal_root / "dataset.json").read_text())
    frame = nominal_data["resolution_preparation"]
    rim = {c["geometry_id"]: c for c in nominal_data["cases"]}
    groups = configurations(
        cases, p, audit / "raw", json.loads((decoder_root / "plan.json").read_text())
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "features").mkdir(exist_ok=True)
    plan = {
        "methods": list(METHODS),
        "states": list(map(list, STATES)),
        "fine_factor": 4,
        "groups": groups,
        "checkpoint": retained["checkpoint"],
        "checkpoint_sha256": retained["sha256"],
        "sources": {
            str(f.resolve()): sha256(f)
            for f in [
                Path(__file__),
                Path(__file__).with_name("subpixel_sampling.py"),
                NATIVE,
                inventory,
                experiment / "protocol.json",
                audit / "raw/manifest.json",
                decoder_root / "plan.json",
            ]
        },
        "engineering_mae_max": (np.asarray(p["delta_std"]) * 0.05).tolist(),
        "engineering_p95_max": (np.asarray(p["delta_std"]) * 0.1).tolist(),
        "preservation": {
            "nominal_mae_max_ratio": 1.05,
            "direction_max_drop": 0.03,
            "slope_min_original_fraction": 0.95,
        },
        "sampling_score": "Per record take worst independent view phase drift first; then mean and p95 by source. Paired paths additionally measure self, drift and sign flips.",
        "scope": "Original 256 absolute and 512 paired camera contracts. Same CFD conditions; no model fitting or selection from test.",
    }
    if (output / "plan.json").exists() and json.loads(
        (output / "plan.json").read_text()
    ) != plan:
        raise ValueError("Incompatible partial acquisition")
    write_json(output / "plan.json", plan)
    torch.set_num_threads(2)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model, _ = load_checkpoint(retained["checkpoint"], device)
    model.eval()
    started = time.monotonic()
    records = {}
    for size in (256, 512):
        specs = [(f"Geometry/{n}/{n}.stl", n != "Car_Body") for n in SHARED]
        projected = projections(rim_triangles(raw_root, entries, specs, frame), size)
        body = {}
        for i, phase in enumerate(STATES):
            body[i, False] = acquire(projected, size, phase, (0.0, 0.0))
            offset = acquisition_offset(phase, 4)
            if np.any(offset):
                body[i, True] = acquire(projected, size, phase, offset)
        del projected
        for gid, g in groups.items():
            is_rim = g["source"] == "drivaerrim"
            if (size == 512 and not is_rim) or (size == 256 and g["split"] == "train"):
                continue
            key = f"{size}-{gid}"
            path = output / "features" / f"{key}.npz"
            audit_path = path.with_suffix(".json")
            if audit_path.exists():
                record = json.loads(audit_path.read_text())
                if sha256(path) != record["feature_sha256"]:
                    raise ValueError("Partial geometry features changed")
                records[key] = record
                continue
            c = by_id[g["case_ids"][0]]
            source = c if size == 256 else rim[gid]
            source_path = (
                Path(p["joint_prepared"]) if size == 256 else nominal_root
            ) / source["views"]
            if sha256(source_path) != source["views_sha256"]:
                raise ValueError("Nominal input hash changed")
            with np.load(source_path, allow_pickle=False) as f:
                nominal = {n: f[n] for n in VIEWS}
            if is_rim:
                number = c["case_id"].rsplit("-", 1)[-1]
                specs = [
                    (f"Geometry/Wheel_{pos}/Wheel_{pos}_{number}.stl", True)
                    for pos in ("FL", "RL")
                ]
                parts = rim_triangles(raw_root, entries, specs, frame)
            else:
                parts = aircraft_triangles(c)
            sample = (
                output / "off-grid-buffer.npz"
                if size == 256 and gid == next(iter(groups))
                else None
            )
            features, record = feature_record(
                model,
                projections(parts, size),
                body if is_rim else {},
                size,
                nominal,
                device,
                sample,
            )
            np.savez_compressed(path, **features)
            record.update(
                feature_sha256=sha256(path),
                source_geometry_hash=c["geometry_hash"],
                nominal_sha256=source["views_sha256"],
            )
            write_json(audit_path, record)
            records[key] = record
            print(
                key,
                "max registered diff",
                max(record["max_input_difference"]["registered"]),
                "elapsed",
                round(time.monotonic() - started, 1),
                flush=True,
            )
        del body
    write_json(
        output / "manifest.json",
        {
            "records": records,
            "elapsed_s": time.monotonic() - started,
            "plan_sha256": sha256(output / "plan.json"),
        },
    )


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for arg in ("experiment", "audit", "decoder_phases", "output"):
        cli.add_argument(arg, type=Path)
    a = cli.parse_args()
    prepare(a.experiment, a.audit, a.decoder_phases, a.output)
