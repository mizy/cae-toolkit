"""Download pinned COCOANet geometry and render complete aircraft views."""

import argparse
import csv
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyvista as pv
import requests
from vehicle_cd_learning.render import (
    VIEW_AXES,
    _rasterize,
    canonical_points,
    save_five_views,
    save_preview,
)

REVISION = "210068b62a4678a3bcfb9a87926bcec44e6416f9"
BASE = f"https://huggingface.co/datasets/ShapeBench/COCOANet/resolve/{REVISION}/"
DESIGN = [
    "Dihedral angle",
    "Max Wing Blend",
    "Inlet Angle 1",
    "Inlet Angle 2",
    "Wing Position",
    "Rear Point X",
    "Inlet Location",
    "NACA 4-digit code",
    "Fore Top Angle",
    "Aft Top Angle",
    "Top Height Aft",
    "Bottom Height Aft",
    "Wing span",
    "Rear tail offset",
    "Root Chord",
    "Tail Root chord",
]
PARAMETERS = [
    "span_over_length",
    "height_over_length",
    "log_length_m",
    "log_velocity_m_s",
    "altitude_over_10000m",
    "alpha_sin",
    "alpha_cos",
]


def render(mesh, size=256, *, unit_scale_to_m=0.001, front_axis="-x", up_axis="+z"):
    points, dimensions = canonical_points(
        mesh.points,
        unit_scale_to_m=unit_scale_to_m,
        front_axis=front_axis,
        up_axis=up_axis,
    )
    scale = max(dimensions["normalized_extents_l"])
    points /= scale
    if (
        np.max(np.abs(points[:, :2])) > 0.65
        or points[:, 2].min() < 0
        or points[:, 2].max() > 1.3
    ):
        raise ValueError("Aircraft exceeds full-view framing")
    faces = np.asarray(mesh.triangulate().faces).reshape(-1, 4)[:, 1:]
    views = {
        name: _rasterize(points, faces, view=name, size=size, window_l=1.3)
        for name in VIEW_AXES
    }
    return views, dimensions


def parameters(dimensions, row):
    length = dimensions["length_m"]
    alpha = np.deg2rad(float(row["alpha_deg"]))
    if float(row["beta_deg"]) != 0:
        raise ValueError("Pilot supports zero sideslip only")
    return [
        dimensions["width_m"] / length,
        dimensions["height_m"] / length,
        np.log(length),
        np.log(float(row["velocity_magnitude_ms"])),
        float(row["altitude_m"]) / 10000,
        np.sin(alpha),
        np.cos(alpha),
    ]


def fetch(root, relative):
    path = root / relative
    if not path.exists():
        response = requests.get(BASE + relative, timeout=120)
        response.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(response.content)
        temporary.replace(path)
    return path


def prepare_geometry(gid, root, size):
    mesh_path = fetch(root, f"case_{gid}/design_{gid}.stl")
    area = float(
        fetch(root, f"case_{gid}/reference_area/reference_area_m2.txt").read_text()
    )
    chord = float(fetch(root, f"case_{gid}/reference_area/mac.txt").read_text())
    if not np.isfinite([area, chord]).all() or min(area, chord) <= 0:
        raise ValueError("Invalid source area/chord")
    mesh = pv.read(mesh_path)
    views, dimensions = render(mesh, size)
    save_five_views(root / "views" / f"{gid}.npz", views)
    save_preview(root / "previews" / f"{gid}.png", views)
    # Keep area/chord and mesh declarations, including the raw source scale.
    return gid, {
        "dimensions": dimensions,
        "reference_area_m2": area,
        "reference_chord_m": chord,
        "mesh": str(mesh_path.resolve()),
        "sha256": hashlib.sha256(mesh_path.read_bytes()).hexdigest(),
        "views_sha256": hashlib.sha256(
            (root / "views" / f"{gid}.npz").read_bytes()
        ).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--size", type=int, default=256)
    args = parser.parse_args()
    root = args.output_dir
    root.mkdir(parents=True, exist_ok=True)
    if (root / "dataset.json").exists():
        raise FileExistsError("Prepared dataset exists")
    (root / "views").mkdir(exist_ok=True)
    (root / "previews").mkdir(exist_ok=True)
    assert (
        hashlib.sha256(args.csv.read_bytes()).hexdigest()
        == "ac33332ae3128c171d73bafa5a0f864574fe020d99029069966618f167169162"
    )
    rows = list(csv.DictReader(args.csv.open()))
    selected = [r for r in rows if float(r["velocity_magnitude_ms"]) <= 80]
    gids = sorted({r["case_id"] for r in selected}, key=int)
    designs = {
        gid: {tuple(r[k] for k in DESIGN) for r in rows if r["case_id"] == gid}
        for gid in gids
    }
    if any(len(x) != 1 for x in designs.values()):
        raise ValueError("Conflicting per-geometry design parameters")
    # Exact design duplicates are grouped before splitting; one plane cannot cross splits.
    signatures = sorted({next(iter(designs[g])) for g in gids})
    rng = np.random.default_rng(20261109)
    ordered = [signatures[i] for i in rng.permutation(len(signatures))]
    n = len(ordered)
    assignment = {
        k: ("train" if i < int(n * 0.7) else "val" if i < int(n * 0.85) else "test")
        for i, k in enumerate(ordered)
    }
    geometry = {}
    failures = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(prepare_geometry, g, root, args.size): g for g in gids}
        for future in as_completed(pending):
            gid = pending[future]
            try:
                _, geometry[gid] = future.result()
            except (
                requests.RequestException,
                OSError,
                ValueError,
                RuntimeError,
            ) as error:
                failures[gid] = f"{type(error).__name__}: {str(error)[:250]}"
            print(
                json.dumps(
                    {
                        "prepared": len(geometry),
                        "failed": len(failures),
                        "total": len(gids),
                        "last": gid,
                    }
                ),
                flush=True,
            )
    (root / "failures.json").write_text(json.dumps(failures, indent=2) + "\n")
    if failures:
        raise RuntimeError(
            "Geometry preparation failures; preserve files and retry before training"
        )
    # Byte-identical source meshes must not cross groups even with different parameter IDs.
    owners = {}
    cases = []
    for row in selected:
        gid = row["case_id"]
        g = geometry[gid]
        split = assignment[next(iter(designs[gid]))]
        if g["sha256"] in owners and owners[g["sha256"]] != split:
            raise ValueError("Identical mesh across splits")
        owners[g["sha256"]] = split
        alpha = np.deg2rad(float(row["alpha_deg"]))
        fx, fz = float(row["CFx"]), float(row["CFz"])
        if not np.allclose(
            [
                fx * np.cos(alpha) + fz * np.sin(alpha),
                -fx * np.sin(alpha) + fz * np.cos(alpha),
            ],
            [float(row["CD"]), float(row["CL"])],
            atol=1e-7,
            rtol=0,
        ):
            raise ValueError("Source body/wind-axis coefficient conversion differs")
        cases.append(
            {
                "case_id": f"{gid}-{row['run_id']}",
                "geometry_id": gid,
                "family_id": "cocoanet-cca",
                "split": split,
                "parameters": parameters(g["dimensions"], row),
                "design_parameters": [float(row[k]) for k in DESIGN],
                "targets": [float(row["CD"]), float(row["CL"])],
                "conditions": {
                    k: float(row[k])
                    for k in (
                        "alpha_deg",
                        "beta_deg",
                        "velocity_magnitude_ms",
                        "altitude_m",
                    )
                },
            }
        )
    data = {
        "dataset": "ShapeBench/COCOANet",
        "revision": REVISION,
        "csv_sha256": hashlib.sha256(args.csv.read_bytes()).hexdigest(),
        "license": "CC BY 4.0",
        "scope": "Low-speed U<=80m/s; one parametric CCA family; geometry/design-group disjoint; coefficients are source corrected wind-axis values; no field supervision",
        "geometry_convention": "Source nTop geometry mm; nose -X (rear-point X defines tail), span Y, up Z; translate before rendering, uniform max-extent scaling shared by all full views",
        "parameter_names": PARAMETERS,
        "design_names": DESIGN,
        "render_config": {
            "size": args.size,
            "window_l": 1.3,
            "scale": "maximum_extent",
        },
        "split_seed": 20261109,
        "geometry": geometry,
        "cases": cases,
    }
    (root / "dataset.json").write_text(
        json.dumps(data, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "cases": len(cases),
                "geometries": len(geometry),
                "splits": {
                    s: sum(c["split"] == s for c in cases)
                    for s in ("train", "val", "test")
                },
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
