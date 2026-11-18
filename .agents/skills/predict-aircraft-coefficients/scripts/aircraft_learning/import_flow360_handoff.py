"""Verify a local user-authorized Flow360 aircraft handoff without network calls."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pyvista as pv
from vehicle_cd_learning.render import save_five_views, save_preview

from .prepare import render


def main():
    """@entry Stage exact solver surfaces and independently check coefficient tails."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("handoff", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.handoff.read_text())
    root = args.output_dir
    if root.exists():
        raise FileExistsError("Use a fresh output directory")
    (root / "geometry").mkdir(parents=True)
    geometries = {}
    for group in source["groups"]:
        surface = Path(group["actual_surface_ugrid"])
        if (
            hashlib.sha256(surface.read_bytes()).hexdigest()
            != group["actual_surface_sha256"]
        ):
            raise ValueError("Actual solver surface checksum changed")
        original = group["actual_surface_npz"]
        path = Path(original["path"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != original["sha256"]:
            raise ValueError("Handoff surface checksum changed")
        with np.load(path, allow_pickle=False) as data:
            points, triangles = data["vertices_mm"], data["triangles_zero_based"]
        if (
            not np.isfinite(points).all()
            or triangles.min() < 0
            or triangles.max() >= len(points)
        ):
            raise ValueError("Invalid solver surface coordinates/connectivity")
        gid = group["group_id"]
        shutil.copy2(path, root / "geometry" / path.name)
        faces = np.column_stack([np.full(len(triangles), 3), triangles]).ravel()
        views, dimensions = render(pv.PolyData(points, faces), 256)
        save_five_views(root / "views" / f"{gid}.npz", views)
        save_preview(root / "previews" / f"{gid}.png", views)
        geometries[gid] = {
            "source_surface_sha256": group["actual_surface_sha256"],
            "geometry": f"geometry/{path.name}",
            "sha256": original["sha256"],
            "dimensions": dimensions,
            "caveat": group["caveat"],
        }
    records = []
    for case in source["cases"]:
        if (
            case["surface_sha256"]
            != geometries[case["group_id"]]["source_surface_sha256"]
        ):
            raise ValueError("Case refers to a different solver surface")
        force_path = Path(case["files"]["force_csv"])
        residual_path = Path(case["files"]["residual_csv"])
        forces = np.genfromtxt(force_path, delimiter=",", names=True)
        residuals = np.genfromtxt(residual_path, delimiter=",", names=True)
        end = float(forces["pseudo_step"].max())
        if (
            end != float(residuals["pseudo_step"].max())
            or end != case["convergence"]["end_step"]
        ):
            raise ValueError("Force/residual end steps differ")
        tail = forces[forces["pseudo_step"] >= 0.9 * end]
        if not np.all(forces["physical_step"] == 0):
            raise ValueError("This first handoff expects steady cases only")
        truth = [float(tail[n].mean()) for n in ("CD", "CL")]
        if not np.isfinite(truth).all() or truth[0] <= 0:
            raise ValueError("Invalid source force means")
        if not np.allclose(
            truth, [case["labels"][n] for n in ("CD", "CL")], atol=1e-10, rtol=0
        ):
            raise ValueError("Recomputed labels differ from handoff")
        sim = json.loads(Path(case["files"]["simulation_json"]).read_text())
        solver = json.loads(Path(case["files"]["translated_solver_json"]).read_text())
        ref, op = sim["reference_geometry"], case["operating"]
        angle = sim["operating_condition"]["alpha"]["value"]
        rotated = [
            tail["CFx"].mean() * np.cos(angle) + tail["CFz"].mean() * np.sin(angle),
            -tail["CFx"].mean() * np.sin(angle) + tail["CFz"].mean() * np.cos(angle),
        ]
        if not np.allclose(rotated, truth, atol=1e-8, rtol=0):
            raise ValueError("Wind-axis coefficient conversion differs")
        if ref["area"]["units"] != "mm**2" or not np.isclose(
            ref["area"]["value"] * 1e-6, case["reference"]["area_m2"]
        ):
            raise ValueError("Reference area units differ")
        re_solver = (
            solver["freestream"]["Mach"]
            * solver["geometry"]["momentLength"][1]
            / solver["freestream"]["muRef"]
        )
        if not np.isclose(re_solver, op["Re_reference_derived"], rtol=1e-8):
            raise ValueError("Physical and translated Reynolds numbers differ")
        warnings = []
        if len(tail) < 20:
            warnings.append(
                "Short restart tail: fewer than 20 saved samples; join parent history before high-confidence use"
            )
        fluid = next(m for m in sim["models"] if m["type"] == "Fluid")
        records.append(
            {
                "case_id": case["case_id"],
                "geometry_id": case["group_id"],
                "environment": case["environment"],
                "usage": case["usage"],
                "source_train_candidate": case["train_eligible"],
                "warnings": warnings,
                "targets": truth,
                "conditions": op,
                "reference": case["reference"],
                "gravity": fluid.get("gravity"),
                "transition": fluid.get("transition_model_solver"),
                "turbulence": fluid.get("turbulence_model_solver"),
                "tail_samples": len(tail),
                "tail_relative_std_cd_cl": [
                    float(tail[n].std() / abs(tail[n].mean())) for n in ("CD", "CL")
                ],
                "tail_relative_drift_cd_cl": [
                    float((tail[n][-1] - tail[n][0]) / abs(tail[n].mean()))
                    for n in ("CD", "CL")
                ],
                "force_csv_sha256": hashlib.sha256(force_path.read_bytes()).hexdigest(),
                "residual_csv_sha256": hashlib.sha256(
                    residual_path.read_bytes()
                ).hexdigest(),
            }
        )
    result = {
        "source_handoff_sha256": hashlib.sha256(args.handoff.read_bytes()).hexdigest(),
        "ownership": source["ownership"],
        "geometry": geometries,
        "cases": records,
        "status": "verified local candidate pool, not mixed into existing model training",
        "training_gate": "Only three related geometries; source SA plus transition and gravity differ from current source contracts. Keep sensitivity cases separate; do not claim an independent low-Re generalization test.",
        "scope": "Source numerical CFD, not experimental truth; original CAD STL is not substituted for the actual meshed surface",
    }
    (root / "pool.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "geometry_count": len(geometries),
                "records": len(records),
                "source_candidates": sum(c["source_train_candidate"] for c in records),
                "sensitivity_records": sum(
                    not c["source_train_candidate"] for c in records
                ),
                "short_restart_candidates": sum(
                    c["source_train_candidate"] and bool(c["warnings"]) for c in records
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
