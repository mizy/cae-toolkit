"""Prepare every built-in labeled car using explicit local mesh resolutions."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from vehicle_cd_similarity_lib.artifacts import load_artifacts
from vehicle_cd_similarity_lib.mesh_io import read_mesh
from vehicle_cd_similarity_lib.normalization import normalize_vehicle_mesh

from .prepare import prepare
from .render import render_five_views


def silhouette(mask):
    rows, cols = np.nonzero(mask)
    image = Image.fromarray(
        mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]
    )
    return np.asarray(image.resize((128, 64), Image.Resampling.NEAREST), dtype=bool)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("library", type=Path)
    parser.add_argument("seed", type=Path)
    parser.add_argument("conditions_manifest", type=Path)
    parser.add_argument(
        "resolutions", type=Path, help="JSON mapping missing IDs to local mesh paths"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    library = json.loads(args.library.read_text())
    seed = {c["id"]: c for c in json.loads(args.seed.read_text())["cars"]}
    overrides = json.loads(args.resolutions.read_text())
    source = json.loads(args.conditions_manifest.read_text())
    source["cases"] = []
    out = args.output_dir.resolve()
    (out / "meshes").mkdir(parents=True, exist_ok=True)
    audit = []
    for car in library["cars"]:
        cid = car["id"]
        item = seed[cid]
        path = (
            Path(
                overrides[cid]["path"]
                if isinstance(overrides[cid], dict)
                else overrides[cid]
            )
            if cid in overrides
            else (args.seed.parent / item["mesh"]).resolve()
        )
        mesh = read_mesh(path)
        if (
            "normalized" not in path.name.lower()
            and "negx-zup" not in path.name.lower()
        ):
            normalized = normalize_vehicle_mesh(mesh)
            mesh = normalized.mesh
        reference = silhouette(
            load_artifacts(args.library.parent, car["artifacts"]).mask
        )
        best = None
        for flip_x, flip_z in (
            (False, False),
            (True, False),
            (False, True),
            (True, True),
        ):
            candidate = mesh.copy()
            points = np.asarray(candidate.points).copy()
            if flip_x:
                points[:, 0] *= -1
                points[:, 1] *= -1
            if flip_z:
                points[:, 2] *= -1
                points[:, 1] *= -1
            candidate.points = points
            views, dimensions = render_five_views(
                candidate, unit_scale_to_m=1, ordinary_size=64, bottom_size=64
            )
            mask = silhouette(views["right"].mask)
            score = float(
                np.logical_and(mask, reference).sum()
                / np.logical_or(mask, reference).sum()
            )
            if best is None or score > best[0]:
                best = (score, candidate, dimensions, flip_x, flip_z)
        score, mesh, dimensions, flip_x, flip_z = best
        if isinstance(overrides.get(cid), dict) and overrides[cid].get(
            "invert_z_after_matching"
        ):
            mesh.points[:, 2] *= -1
            mesh.points[:, 1] *= -1
            flip_z = not flip_z
        target = out / "meshes" / f"{cid}.vtp"
        mesh.save(target)
        source["cases"].append(
            {
                "case_id": cid,
                "geometry_id": cid,
                "family_id": cid,
                "split": "train",
                "geometry": {
                    "path": str(target),
                    "unit_scale_to_m": 1.0,
                    "front_axis": "-x",
                    "up_axis": "+z",
                },
                "cd": car["cd"],
                "cd_source": car["cd_source"],
                "reynolds": 1.204 * (120 / 3.6) * dimensions["length_m"] / 1.81e-5,
                "yaw_deg": 0.0,
                "assumptions": {
                    "orientation": "matched to built-in silhouette, without using Cd",
                    "source_conditions_verified": False,
                },
            }
        )
        audit.append(
            {
                "id": cid,
                "source_mesh": str(path),
                "side_mask_iou": score,
                "flip_x": flip_x,
                "flip_z": flip_z,
                "dimensions": dimensions,
            }
        )
        print(cid, round(score, 3), flush=True)
    (out / "orientation-audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    source["provenance"]["all_builtin_ids"] = [c["id"] for c in library["cars"]]
    manifest = out / "source.json"
    manifest.write_text(json.dumps(source, indent=2) + "\n")
    prepare(
        manifest,
        out / "prepared",
        {"ordinary_size": 256, "bottom_size": 512, "window_l": 1.3},
    )


if __name__ == "__main__":
    main()
