"""Replay numerical 256-pixel inputs and render independent view phases for validation."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyvista as pv

from .delta_experiment import sha256, write_json
from .joint_sampling import load
from .phase_views import cache_difference, camera_views, checked_mesh, phase_id
from .prepare_drivaerrim import merge_depth
from .render import canonical_points, save_five_views
from .rim_resolution import SHARED

PHASES = ((0.0, 0.0), (0.25, 0.0), (-0.25, 0.0), (0.0, 0.25), (0.0, -0.25))


def validation_geometry(cases, rim_ids):
    rows = {c["case_id"]: c for c in cases if c["split"] == "val"}
    selected = [rows[cid]["geometry_id"] for cid in rim_ids]
    rng = np.random.default_rng(42)
    for source in ("cocoanet", "hilift"):
        geometry = sorted(
            {c["geometry_id"] for c in rows.values() if c["source"] == source}
        )
        selected.extend(sorted(rng.choice(geometry, 6, replace=False).tolist()))
    return {
        gid: [c for c in rows.values() if c["geometry_id"] == gid] for gid in selected
    }


def rim_buffers(parts, raw, entries, origin, length, hashes):
    buffers = {}
    for relative, mirror in parts:
        mesh = checked_mesh(raw, relative, entries, hashes)
        for reflected in range(2 if mirror else 1):
            if reflected:
                mesh.points[:, 1] *= -1
            for phase in PHASES:
                current = camera_views(mesh, origin, length, 256, phase)
                if phase in buffers:
                    merge_depth(buffers[phase], current)
                else:
                    buffers[phase] = current
    return buffers


def aircraft_buffers(case):
    path = Path(case["source_evidence"]["source_mesh"])
    if sha256(path) != case["geometry_hash"]:
        raise ValueError(f"Aircraft geometry hash changed: {path}")
    mesh = pv.read(path)
    hilift = case["source"] == "hilift"
    if hilift:
        mirrored = mesh.copy()
        mirrored.points[:, 1] *= -1
        mesh = mesh.append_polydata(mirrored)
    # Same units, center and maximum-extent camera as aircraft_learning.prepare.render.
    points, dims = canonical_points(
        mesh.points, unit_scale_to_m=0.0254 if hilift else 0.001
    )
    points /= max(dims["normalized_extents_l"])
    triangles = mesh.triangulate()
    if len(triangles.points) != len(points):
        raise ValueError(
            "Unexpected change in aircraft point ordering during triangulation"
        )
    triangles.points = points
    return {
        phase: camera_views(triangles, np.zeros(3), 1.0, 256, phase) for phase in PHASES
    }


def prepare(experiment, output):
    """@entry Render from source, accepting a geometry only after replaying its five input arrays."""
    p, cases = load(experiment)
    phase_root = Path(
        json.loads((Path(p["sampling_parent"]) / "protocol.json").read_text())[
            "phase_root"
        ]
    )
    phase_plan = json.loads((phase_root / "plan.json").read_text())
    groups = validation_geometry(cases, phase_plan["selection"]["val"])
    parent = json.loads((Path(p["parent_experiment"]) / "protocol.json").read_text())
    inventory = Path(parent["inventory"])
    if sha256(inventory) != parent["input_sha256"][str(inventory)]:
        raise ValueError("Original Rim inventory changed")
    entries = {e["path"]: e for e in json.loads(inventory.read_text())["files"]}
    raw, frame = inventory.parent, phase_plan["source_frame"]
    origin, length = np.array(frame["origin_m"]), frame["dimensions_m"][0]
    output.mkdir(parents=True, exist_ok=False)
    write_json(
        output / "plan.json",
        {
            "case_ids": {
                gid: [c["case_id"] for c in group] for gid, group in groups.items()
            },
            "phases_pixels": PHASES,
            "frame_rim": frame,
            "size": 256,
            "inputs": {
                str(f): sha256(f)
                for f in (
                    Path(__file__),
                    phase_root / "plan.json",
                    inventory,
                    Path(p["joint_prepared"]) / "dataset.json",
                )
            },
        },
    )
    hashes, records = {}, {}
    shared = rim_buffers(
        [(f"Geometry/{n}/{n}.stl", n != "Car_Body") for n in SHARED],
        raw,
        entries,
        origin,
        length,
        hashes,
    )
    for gid, group in groups.items():
        case = group[0]
        if len({(c["views_sha256"], c["geometry_hash"]) for c in group}) != 1:
            raise ValueError(f"Inconsistent views across conditions for {gid}")
        if case["source"] == "drivaerrim":
            number = case["case_id"].rsplit("-", 1)[-1]
            parts = [
                (f"Geometry/Wheel_{pos}/Wheel_{pos}_{number}.stl", True)
                for pos in ("FL", "RL")
            ]
            buffers = rim_buffers(parts, raw, entries, origin, length, hashes)
            signature = hashlib.sha256(
                "".join(hashes[name] for name, _ in parts).encode()
            ).hexdigest()
            if signature != case["geometry_hash"]:
                raise ValueError(f"Wheel geometry mismatch: {gid}")
            for phase in PHASES:
                merge_depth(buffers[phase], shared[phase])
        else:
            buffers = aircraft_buffers(case)
        nominal = Path(p["joint_prepared"]) / case["views"]
        if sha256(nominal) != case["views_sha256"]:
            raise ValueError(f"Nominal absolute-head input changed: {gid}")
        replay = cache_difference(buffers[PHASES[0]], nominal)
        caches = [{"path": str(nominal.resolve()), "sha256": case["views_sha256"]}]
        for phase in PHASES[1:]:
            path = output / "views" / f"{gid}-{phase_id(phase)}.npz"
            save_five_views(path, buffers[phase])
            caches.append({"path": str(path.resolve()), "sha256": sha256(path)})
        records[gid] = {
            "case_ids": [c["case_id"] for c in group],
            "source": case["source"],
            "source_geometry_hash": case["geometry_hash"],
            "views": caches,
            "nominal_replay_max_abs_by_view": replay,
        }
        print(gid, len(group), max(replay.values()), flush=True)
    write_json(
        output / "manifest.json",
        {
            "phases_pixels": PHASES,
            "geometries": records,
            "rim_part_hashes": hashes,
            "plan_sha256": sha256(output / "plan.json"),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    prepare(args.experiment, args.output)
