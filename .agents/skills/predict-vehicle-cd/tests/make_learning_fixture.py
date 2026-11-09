"""Create a tiny analytic, explicitly non-aerodynamic learning fixture."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyvista as pv


def _box(bounds):
    x0, x1, y0, y1, z0, z1 = bounds
    points = np.array(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ]
    )
    quads = [
        (0, 3, 2, 1),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    ]
    faces = []
    for a, b, c, d in quads:
        faces.extend((3, a, b, c, 3, a, c, d))
    return pv.PolyData(points, np.asarray(faces))


def _geometry(
    path: Path, family: int, design: int
) -> tuple[float, float, float, float]:
    """Write a blocky vehicle; return width, height, roof, underbody factors."""
    width = 1.72 + 0.008 * family + 0.012 * design
    height = 1.32 + 0.006 * family + 0.018 * design
    roof = 0.48 + 0.025 * ((family + design) % 5)
    underbody = 0.0 if design == 0 else 0.025 + 0.005 * (family % 3)
    body_top = height * 0.66
    body = _box((-2.25, 2.25, -width / 2, width / 2, 0.25, body_top))
    cabin = _box(
        (
            -1.35 * roof / 0.5,
            0.75 * roof / 0.5,
            -width * 0.37,
            width * 0.37,
            body_top - 0.03,
            height,
        )
    )
    parts = [body, cabin]
    # Identical wheel blocks set a stable ground plane and expose the undertray.
    for x in (-1.45, 1.45):
        for y in (-width * 0.34, width * 0.34):
            parts.append(_box((x - 0.20, x + 0.20, y - 0.09, y + 0.09, 0.0, 0.30)))
    if underbody:
        # Lower the central undertray locally; bottom view sees the recess/rib.
        parts.append(_box((-1.7, 1.8, -0.10, 0.10, 0.25 - underbody, 0.25)))
    pv.MultiBlock(parts).combine().extract_surface(algorithm="dataset_surface").save(
        path
    )
    return width, height, roof, underbody


def write_fixture(root: Path, target_names=("cd", "cl")) -> Path:
    """Write 12 family groups (8/2/2 splits), two designs and two conditions."""
    root.mkdir(parents=True, exist_ok=True)
    cases = []
    splits = {
        **{i: "train" for i in (0, 1, 3, 5, 6, 8, 10, 11)},
        **{i: "val" for i in (2, 9)},
        **{i: "test" for i in (4, 7)},
    }
    for family, split in splits.items():
        for design in range(2):
            geometry_id = f"family-{family:02d}-design-{design}"
            relpath = f"geometries/{geometry_id}.vtp"
            geom_path = root / relpath
            geom_path.parent.mkdir(parents=True, exist_ok=True)
            width, height, roof, underbody = _geometry(geom_path, family, design)
            for condition, (reynolds, yaw) in enumerate(((1.0e6, 0.0), (2.0e6, 5.0))):
                # Synthetic regression target. This formula is a software test signal,
                # not a physical/aerodynamic model or a claimed CFD label.
                yaw_rad = np.deg2rad(yaw)
                cd = (
                    0.2
                    + 0.07 * (width / 4.5)
                    + 0.03 * roof
                    + 0.2 * underbody
                    + 1e-3 * np.log(reynolds / 1e6)
                    + 0.01 * yaw_rad
                )
                cl = (
                    -0.04
                    + 0.02 * (height / 4.5)
                    - 0.03 * roof
                    + 0.1 * underbody
                    + 0.004 * yaw
                )
                case_id = f"{geometry_id}-condition-{condition}"
                case = {
                    "case_id": case_id,
                    "geometry_id": geometry_id,
                    "family_id": f"family-{family:02d}",
                    "split": split,
                    "geometry": {
                        "path": relpath,
                        "unit_scale_to_m": 1.0,
                        "front_axis": "-x",
                        "up_axis": "+z",
                    },
                    "reynolds": reynolds,
                    "yaw_deg": yaw,
                    "pair_id": f"family-{family:02d}-condition-{condition}",
                    "design_value": design,
                }
                if "cd" in target_names:
                    case["cd"] = float(cd)
                if "cl" in target_names:
                    case["cl"] = float(cl)
                cases.append(case)
    manifest = {
        "data_kind": "synthetic",
        "target_names": list(target_names),
        "provenance": "Analytic software fixture only; labels are synthetic and are not aerodynamic or CFD data.",
        "conditions": {
            key: value
            for key, value in {
                "ground": "fixed synthetic convention",
                "wheels": "omitted synthetic convention",
                "turbulence": "not applicable to analytic fixture",
                "solver": "analytic fixture generator",
                "averaging": "not applicable",
                "reference_area": "synthetic convention",
                "force_axes": "synthetic convention",
            }.items()
        },
        "cases": cases,
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    print(write_fixture(parser.parse_args().output))
