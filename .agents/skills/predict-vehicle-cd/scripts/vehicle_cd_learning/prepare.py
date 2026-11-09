"""Validate one coherent labeled dataset and render reusable numerical five-views."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from vehicle_cd_similarity_lib.mesh_io import read_mesh

from .data import (
    PARAMETER_NAMES,
    SPLIT_POLICIES,
    TARGET_NAMES,
    load_dataset,
    parameters,
)
from .render import canonical_points, render_five_views, save_five_views, save_preview

CONDITION_KEYS = (
    "ground",
    "wheels",
    "turbulence",
    "solver",
    "averaging",
    "reference_area",
    "force_axes",
)


def prepare(manifest: Path, output: Path, render_config: dict) -> Path:
    """@entry Render explicitly oriented geometry and preserve supplied family splits."""
    source = json.loads(manifest.read_text())
    if source.get("data_kind") not in (
        "cfd",
        "experiment",
        "synthetic",
        "reference_assumed",
    ):
        raise ValueError(
            "Declare data_kind: cfd, experiment, synthetic or reference_assumed"
        )
    conditions = source["conditions"]
    if any(
        key not in conditions or conditions[key] is None or conditions[key] == ""
        for key in CONDITION_KEYS
    ):
        raise ValueError(f"Dataset must declare common conditions: {CONDITION_KEYS}")
    if not source.get("provenance"):
        raise ValueError("Dataset provenance is required")
    split_policy = source.get("split_policy", "family_disjoint")
    if split_policy not in SPLIT_POLICIES:
        raise ValueError(f"Unsupported split_policy: {split_policy}")
    if output.exists():
        raise ValueError("Use a new output directory to avoid stale render caches")
    target_names = source.get("target_names", TARGET_NAMES)
    if target_names not in (["cd"], TARGET_NAMES):
        raise ValueError("Supported targets: [cd] or [cd, cl]")
    prepared = []
    geometry_records = {}
    view_hashes = {}
    output.mkdir(parents=True)
    (output / "views").mkdir()
    (output / "previews").mkdir()
    for case in source["cases"]:
        geometry = case["geometry"]
        gid = case["geometry_id"]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", gid):
            raise ValueError("geometry_id must be a safe filename identifier")
        path = (manifest.parent / geometry["path"]).resolve()
        signature = {**geometry, "path": str(path)}
        if gid in geometry_records:
            previous, dimensions, geometry_hash = geometry_records[gid]
            if previous != signature:
                raise ValueError(
                    "One geometry_id refers to different geometry definitions"
                )
        else:
            mesh = read_mesh(path)
            views, dimensions = render_five_views(
                mesh,
                unit_scale_to_m=geometry["unit_scale_to_m"],
                front_axis=geometry["front_axis"],
                up_axis=geometry["up_axis"],
                ground_clearance_m=geometry.get("ground_clearance_m", 0.0),
                **render_config,
            )
            # Hash actual geometry data and declarations, not names, to detect renamed duplicates.
            canonical, _ = canonical_points(
                mesh.points,
                unit_scale_to_m=geometry["unit_scale_to_m"],
                front_axis=geometry["front_axis"],
                up_axis=geometry["up_axis"],
            )
            # Ignore vertex numbering, units, translation and triangle winding.
            triangles = np.round(
                canonical[np.asarray(mesh.triangulate().faces).reshape(-1, 4)[:, 1:]], 6
            )
            triangles = np.sort(triangles.view("V12").reshape(-1, 3), axis=1)
            canonical_bytes = np.sort(triangles.reshape(-1).view("V36")).tobytes()
            geometry_hash = hashlib.sha256(canonical_bytes).hexdigest()
            save_five_views(output / "views" / f"{gid}.npz", views)
            save_preview(output / "previews" / f"{gid}.png", views)
            view_hashes[gid] = hashlib.sha256(
                (output / "views" / f"{gid}.npz").read_bytes()
            ).hexdigest()
            geometry_records[gid] = signature, dimensions, geometry_hash
        row = {
            key: case[key] for key in ("case_id", "geometry_id", "family_id", "split")
        }
        row.update(
            geometry_hash=geometry_hash,
            views=f"views/{gid}.npz",
            views_sha256=view_hashes[gid],
            parameters=parameters(dimensions, case["reynolds"], case["yaw_deg"]),
            targets=[float(case[key]) for key in target_names],
            reynolds=float(case["reynolds"]),
            yaw_deg=float(case["yaw_deg"]),
        )
        for key in (
            "pair_id",
            "design_value",
            "cd_source",
            "cd_confidence",
            "assumptions",
        ):
            if key in case:
                row[key] = case[key]
        prepared.append(row)
    result = {
        "schema_version": 1,
        "parameter_names": PARAMETER_NAMES,
        "target_names": target_names,
        "render_config": render_config,
        "conditions": conditions,
        "data_kind": source["data_kind"],
        "provenance": source["provenance"],
        "split_policy": split_policy,
        "cases": prepared,
    }
    if "split_provenance" in source:
        result["split_provenance"] = source["split_provenance"]
    target = output / "dataset.json"
    target.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    load_dataset(target)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--ordinary-size", type=int, default=256)
    parser.add_argument("--bottom-size", type=int, default=512)
    parser.add_argument("--window-l", type=float, default=1.3)
    args = parser.parse_args()
    path = prepare(
        args.manifest,
        args.output_dir,
        {
            "ordinary_size": args.ordinary_size,
            "bottom_size": args.bottom_size,
            "window_l": args.window_l,
        },
    )
    print(path)


if __name__ == "__main__":
    main()
