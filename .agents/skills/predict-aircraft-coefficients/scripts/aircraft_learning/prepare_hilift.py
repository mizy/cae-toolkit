"""Prepare complete mirrored CRM-HL views with source geometry-disjoint splits."""

import argparse
import csv
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyvista as pv
from vehicle_cd_learning.render import save_five_views, save_preview

from .prepare import render

PARAMETERS = [
    "span_over_length",
    "height_over_length",
    "log_re_mac",
    "mach",
    "alpha_sin",
    "alpha_cos",
]


def geometry(entry, raw, output):
    path = raw / entry["path"]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != entry["sha256"]:
        raise ValueError(f"Source STL changed: {path}")
    gid = path.parent.name.split("_")[1]
    mesh = pv.read(path)
    if abs(mesh.bounds[2]) > 1e-5 or mesh.bounds[3] < 1000:
        raise ValueError("Expected source half aircraft on positive Y, in inches")
    # Mirror geometry, without copying coefficients or doubling their values.
    other = mesh.copy()
    other.points[:, 1] *= -1
    mesh = mesh.append_polydata(other)
    views, dimensions = render(mesh, 256, unit_scale_to_m=0.0254)
    save_five_views(output / "views" / f"{gid}.npz", views)
    save_preview(output / "previews" / f"{gid}.png", views)
    return gid, {
        "mesh": str(path.resolve()),
        "sha256": digest,
        "dimensions": dimensions,
        "geometry_transform": "mirror Y=0; inches to metres; nose -X; up +Z",
        "views_sha256": hashlib.sha256(
            (output / "views" / f"{gid}.npz").read_bytes()
        ).hexdigest(),
    }


def main():
    """@entry Validate source pairing and render all 180 complete aircraft."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("raw", type=Path)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--workers", type=int, default=3)
    args = p.parse_args()
    root = args.output_dir
    if root.exists():
        raise FileExistsError("Use a fresh prepared directory")
    inventory = json.loads((args.raw / "inventory-geometry.json").read_text())
    if not inventory["complete"]:
        raise ValueError("Download must finish successfully before preparation")
    stls = [f for f in inventory["files"] if f["path"].endswith(".stl")]
    if len(stls) != 180:
        raise ValueError("Expected 180 distinct source STL files")
    rows = list(csv.DictReader((args.raw / "force_mom_all.csv").open()))
    designs = list(
        csv.DictReader((args.raw / "geo_values_all.csv").open(encoding="utf-8-sig"))
    )
    design_names = list(designs[0])[1:]
    design_map = {r["GeoID"]: [float(r[k]) for k in design_names] for r in designs}
    split = json.loads((args.raw / "splits/manifest.json").read_text())
    assignment = {}
    for s in ("train", "val", "test"):
        for cid in split[f"geometry_{s}"]:
            gid = cid.split("_")[1]
            if gid in assignment and assignment[gid] != s:
                raise ValueError("Source geometry split leaks")
            assignment[gid] = s
    if len(rows) != 1800 or len(assignment) != 180 or len(design_map) != 180:
        raise ValueError("Incomplete source pairing")
    for gid in assignment:
        ref = next(
            csv.DictReader(
                (args.raw / f"geo_{gid}_AoA_4/ref_values_geo_{gid}_AoA_4.csv").open()
            )
        )
        if not np.allclose(
            [float(ref[k]) for k in ("areaRef", "chordRef", "spanRef")],
            [297360, 275.8, 1156.75],
        ):
            raise ValueError("Reference definitions differ from source paper")
    (root / "views").mkdir(parents=True)
    (root / "previews").mkdir()
    geometries = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed(
            [pool.submit(geometry, e, args.raw, root) for e in stls]
        ):
            gid, record = future.result()
            geometries[gid] = record
            print(
                json.dumps({"prepared": len(geometries), "total": 180, "last": gid}),
                flush=True,
            )
    cases, identities, hash_splits, designs_splits = [], set(), {}, {}
    for row in rows:
        gid, alpha = row["GeoID"], float(row["AoA"])
        cid = f"geo_{gid}_AoA_{int(alpha)}"
        targets = [float(row["cd"]), float(row["cl"])]
        if cid in identities or not np.isfinite(targets).all() or targets[0] <= 0:
            raise ValueError("Duplicate ID or invalid targets")
        identities.add(cid)
        g, s = geometries[gid], assignment[gid]
        for owners, signature in (
            (hash_splits, g["sha256"]),
            (designs_splits, tuple(design_map[gid])),
        ):
            if signature in owners and owners[signature] != s:
                raise ValueError("Duplicate geometry/design crosses splits")
            owners[signature] = s
        if not np.allclose(
            [
                float(row["cdp"]) + float(row["cdv"]),
                float(row["clp"]) + float(row["clv"]),
            ],
            targets,
            atol=2e-6,
        ):
            raise ValueError("Pressure/viscous components do not sum to total")
        angle = np.deg2rad(alpha)
        condition = [
            float(np.log(1.6e6)),
            0.2,
            float(np.sin(angle)),
            float(np.cos(angle)),
        ]
        d = g["dimensions"]
        cases.append(
            {
                "case_id": cid,
                "geometry_id": gid,
                "family_id": "crm-hl",
                "split": s,
                "parameters": [
                    d["width_m"] / d["length_m"],
                    d["height_m"] / d["length_m"],
                    *condition,
                ],
                "condition_parameters": condition,
                "design_parameters": design_map[gid],
                "targets": targets,
                "conditions": {
                    "alpha_deg": alpha,
                    "beta_deg": 0,
                    "re_mac": 1.6e6,
                    "mach": 0.2,
                },
            }
        )
    data = {
        "dataset": inventory["repo"],
        "revision": inventory["revision"],
        "license": "CC BY 4.0",
        "parameter_names": PARAMETERS,
        "condition_names": PARAMETERS[2:],
        "design_names": design_names,
        "geometry": geometries,
        "cases": cases,
        "render_config": {"size": 256},
        "geometry_convention": "Full mirrored airframe from half-span source; inches; nose -X; up +Z",
        "training_conditions": {
            "mach": 0.2,
            "re_mac": 1.6e6,
            "beta_deg": 0,
            "source_area_half_m2": 297360 * 0.0254**2,
            "full_reference_area_m2": 2 * 297360 * 0.0254**2,
            "source_reference_chord_m": 275.8 * 0.0254,
            "coefficient_transform": "Source wind-axis Cd/Cl unchanged when mirroring symmetric half model",
            "solver": "Fidelity Charles WMLES; mean source coefficients, including six documented surface overrides",
        },
        "scope": "Geometry-disjoint CRM-HL high-lift family at Ma=0.2 and Re_MAC=1.6e6; no arbitrary-aircraft claim",
    }
    (root / "dataset.json").write_text(
        json.dumps(data, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
