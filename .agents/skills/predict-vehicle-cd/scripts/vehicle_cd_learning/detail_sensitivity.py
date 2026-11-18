"""Test full-view coefficient sensitivity to a controlled pair of mirror edits."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pyvista as pv
import torch

from .model import VIEWS, CoefficientModel
from .render import _rasterize, canonical_points


def smooth_window(value, core, padding):
    lower = np.clip((value - core[0] + padding) / padding, 0, 1)
    upper = np.clip((core[1] + padding - value) / padding, 0, 1)
    return lower**2 * (3 - 2 * lower) * upper**2 * (3 - 2 * upper)


def edit_mirrors(points, scale, selection):
    """Shorten lateral mirror protrusion with a monotone, fixed-root Y mapping."""
    gate = smooth_window(points[:, 0], selection["x_core"], selection["padding_m"])
    gate *= smooth_window(points[:, 2], selection["z_core"], selection["padding_m"])
    extension = np.maximum(abs(points[:, 1]) - selection["root_abs_y_m"], 0)
    result = points.copy()
    result[:, 1] -= (1 - scale) * np.sign(points[:, 1]) * extension * gate
    return result


def geometry_metrics(mesh, source_points=None):
    from mesh_metrics import signed_volume, triangle_quality

    points = np.asarray(mesh.points, dtype=float)
    faces = np.asarray(mesh.faces).reshape(-1, 4)[:, 1:]
    boundary = mesh.extract_feature_edges(
        boundary_edges=True,
        non_manifold_edges=False,
        manifold_edges=False,
        feature_edges=False,
    ).n_cells
    nonmanifold = mesh.extract_feature_edges(
        boundary_edges=False,
        non_manifold_edges=True,
        manifold_edges=False,
        feature_edges=False,
    ).n_cells
    triangles = points[faces]
    normals = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    areas = np.linalg.norm(normals, axis=1) / 2
    result = {
        "points": mesh.n_points,
        "triangles": mesh.n_cells,
        "boundary_edges": boundary,
        "non_manifold_edges": nonmanifold,
        "components": int(mesh.connectivity("all").cell_data["RegionId"].max()) + 1,
        "degenerate_faces_area_le_1e_14_m2": int((areas <= 1e-14).sum()),
        "surface_area_m2": float(areas.sum()),
        "signed_volume_m3": signed_volume(points, faces),
        "dimensions_m": np.ptp(points, axis=0).tolist(),
        "triangle_quality": triangle_quality(points, faces),
    }
    if source_points is not None:
        original = source_points[faces]
        old_normals = np.cross(
            original[:, 1] - original[:, 0], original[:, 2] - original[:, 0]
        )
        nondegenerate = (np.linalg.norm(old_normals, axis=1) > 2e-14) & (areas > 1e-14)
        result["faces_rotated_at_least_90deg"] = int(
            ((normals * old_normals).sum(1)[nondegenerate] <= 0).sum()
        )
        displacement = np.linalg.norm(points - source_points, axis=1)
        result.update(
            moved_points=int((displacement > 1e-9).sum()),
            max_displacement_m=float(displacement.max()),
        )
    return result


def render_fixed(points, faces, original_points, original_canonical, length, offset):
    from aircraft_learning.prepare_scenes import resample_views

    # Both variants use the ORIGINAL camera, length, origin and depth convention.
    canonical = (
        original_canonical + (points - original_points).astype(np.float32) / length
    )
    shift = np.asarray([offset[0], offset[1], 0], dtype=np.float32) * (1.3 / (256 - 8))
    canonical = canonical + shift
    source = {
        v: np.stack((a.depth, a.mask)).astype(np.float32)
        for v in VIEWS
        for a in [_rasterize(canonical, faces, view=v, size=512, window_l=1.3)]
    }
    return resample_views(source)


def prepare(args):
    from source_projected_validation import exact_self_intersection_report

    if args.output_dir.exists():
        raise FileExistsError("Use a fresh geometry output directory")
    data = json.loads((args.prepared / "dataset.json").read_text())
    case = next(c for c in data["cases"] if c["case_id"] == args.case_id)
    if case["source"] != "drivaerml" or case["split"] != "test":
        raise ValueError("This probe expects a held-out DrivAerML complete car")
    mesh = pv.read(args.source_mesh).triangulate()
    points = np.asarray(mesh.points, dtype=float)
    faces = np.asarray(mesh.faces).reshape(-1, 4)[:, 1:]
    canonical, dimensions = canonical_points(
        points, unit_scale_to_m=1, front_axis="-x", up_axis="+z"
    )
    if not np.isclose(dimensions["length_m"], np.exp(case["parameters"][2]), atol=2e-6):
        raise ValueError("Source mesh dimensions do not match the selected record")
    # Lateral protrusions above the wheel/fender belt identify two source mirror seeds.
    seeds = points[(abs(points[:, 1]) > 1.0) & (points[:, 2] > 0.2)]
    if len(seeds) < 100 or np.ptp(seeds[:, 0]) > 0.25 or np.ptp(seeds[:, 2]) > 0.25:
        raise ValueError("Mirror seed region is ambiguous; inspect before editing")
    selection = {
        "source_sha256": hashlib.sha256(args.source_mesh.read_bytes()).hexdigest(),
        "source_region_id": 0,
        "root_abs_y_m": 0.93,
        "padding_m": 0.04,
        "x_core": [float(seeds[:, 0].min()) - 0.005, float(seeds[:, 0].max()) + 0.005],
        "z_core": [float(seeds[:, 2].min()) - 0.005, float(seeds[:, 2].max()) + 0.005],
        "semantics": "Both A-pillar exterior mirror housings/stems, confirmed through full-car marked-face previews; shorten Y protrusion beyond fixed roots, do not remove body or wheels",
        "mapping": "x,z unchanged; y'=y-(1-scale)*sign(y)*max(abs(y)-root,0)*smooth_x*smooth_z; derivative dy'/dy>=scale>0",
    }
    root = args.output_dir
    root.mkdir(parents=True)
    offsets = [(0, 0), (0.1, 0), (-0.1, 0), (0, 0.1), (0, -0.1)]
    report = {
        "source_mesh": str(args.source_mesh.resolve()),
        "case": case,
        "selection": selection,
        "camera_offsets_pixels": offsets,
        "variants": [],
        "input_kind": "mesh",
        "output_kind": "mesh_design_variants",
        "purpose": "External-aerodynamics coefficient sensitivity, not a CFD solve or repair",
        "conditions": "Same speed/Re/alpha/beta/ground/wheels/Sref for all variants; camera from original fixed",
        "reference_area_policy": "Keep original Sref fixed so coefficient changes are not area-renormalization changes",
    }
    for label, scale in (
        ("original", 1.0),
        ("zero_control", 1.0),
        ("mirrors_minus15", 0.85),
        ("mirrors_minus30", 0.7),
    ):
        changed = edit_mirrors(points, scale, selection)
        output = mesh.copy()
        output.points = changed.astype(mesh.points.dtype)
        changed = np.asarray(output.points, dtype=float)
        moved = np.linalg.norm(changed - points, axis=1) > 1e-9
        output.point_data["modified_mirror_region"] = moved.astype(np.uint8)
        output.compute_normals(
            cell_normals=False,
            point_normals=True,
            split_vertices=False,
            consistent_normals=False,
            auto_orient_normals=False,
            inplace=True,
        )
        path = root / f"{label}.vtp"
        output.save(path)
        readback = pv.read(path)
        if not np.array_equal(readback.points, output.points) or not np.array_equal(
            readback.faces, mesh.faces
        ):
            raise ValueError("VTP serialization changed points or connectivity")
        metrics = geometry_metrics(readback, points)
        if label == "zero_control":
            if moved.any():
                raise ValueError("Zero-change control moved geometry")
            exact = report["variants"][0]["self_intersection"]
        else:
            exact = exact_self_intersection_report(changed, faces, first_only=True)
        views = {}
        for index, offset in enumerate(offsets):
            arrays = render_fixed(
                changed, faces, points, canonical, dimensions["length_m"], offset
            )
            for value in arrays.values():
                if (
                    not np.isfinite(value).all()
                    or (value[0] < 0).any()
                    or (value[0] > 1).any()
                ):
                    raise ValueError("Invalid complete-view depth")
            rel = f"views/{label}-offset{index}.npz"
            destination = root / rel
            destination.parent.mkdir(exist_ok=True)
            np.savez_compressed(destination, **arrays)
            views[str(index)] = rel
        numeric = list(case["parameters"])
        numeric[0] = float(np.ptp(changed, axis=0)[1] / dimensions["length_m"])
        numeric[1] = float(np.ptp(changed, axis=0)[2] / dimensions["length_m"])
        record = {
            "label": label,
            "outboard_extension_scale": scale,
            "mesh": path.name,
            "mesh_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "geometry_metrics": metrics,
            "self_intersection": exact,
            "parameters_actual_dimensions": numeric,
            "views": views,
        }
        report["variants"].append(record)
        print(
            json.dumps(
                {
                    "prepared_variant": label,
                    "moved_points": metrics["moved_points"],
                    "self_intersection": exact["status"],
                    "pairs": exact["intersection_pairs"],
                }
            ),
            flush=True,
        )
    active = edit_mirrors(points, 0.7, selection) != points
    np.savez_compressed(
        root / "source_region_ids.npz",
        point_ids=np.flatnonzero(active.any(1)),
        face_ids=np.flatnonzero(active.any(1)[faces].any(1)),
    )
    # Capture numerical sensitivity at the actual model input, not only a 3D screenshot.
    with np.load(root / report["variants"][0]["views"]["0"]) as a:
        original_views = {n: a[n].copy() for n in VIEWS}
    with np.load(args.prepared / case["views"]) as f:
        report["source_cache_replay"] = {
            n: {
                "max_abs_difference": float(abs(original_views[n] - f[n]).max()),
                "different_values": int((original_views[n] != f[n]).sum()),
            }
            for n in VIEWS
        }
    for variant in report["variants"]:
        with np.load(root / variant["views"]["0"]) as f:
            variant["input_changes"] = {
                n: {
                    "mask_pixels": int((f[n][1] != original_views[n][1]).sum()),
                    "depth_pixels": int(
                        (abs(f[n][0] - original_views[n][0]) > 1e-7).sum()
                    ),
                    "depth_mean_abs": float(abs(f[n][0] - original_views[n][0]).mean()),
                }
                for n in VIEWS
            }
    (root / "geometry-report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )


def predict(args):
    from aircraft_learning.scene_experiment import SceneCases

    root = args.output_dir
    geometry = json.loads((root / "geometry-report.json").read_text())
    width = json.loads((args.width_experiment / "protocol.json").read_text())
    results = {
        "scope": "Sensitivity only; no paired CFD labels for mirror variants. Numerical changes are not established aerodynamic contributions or validated design improvements.",
        "offsets_scope": "Five deterministic ±0.1-pixel full-camera shifts; robustness range, not statistical confidence intervals",
        "models": {},
    }
    torch.set_num_threads(2)
    for size in (32, 128):
        path = (
            Path(width["baseline_checkpoint"])
            if size == 32
            else args.width_experiment / "width-128-seed-42/model.pt"
        )
        ck = torch.load(path, map_location="cpu", weights_only=True)
        model = CoefficientModel(**ck["model_config"])
        model.load_state_dict(ck["model_state"])
        model.eval()
        output, features = [], {}
        for variant in geometry["variants"]:
            for offset, views in variant["views"].items():
                for mode in ("actual_dimensions", "image_only", "numeric_only"):
                    case = dict(geometry["case"])
                    case["views"] = (
                        geometry["variants"][0]["views"][offset]
                        if mode == "numeric_only"
                        else views
                    )
                    case["parameters"] = (
                        geometry["case"]["parameters"]
                        if mode == "image_only"
                        else variant["parameters_actual_dimensions"]
                    )
                    images, values, _, scene = SceneCases(root, [case], ck)[0]
                    with torch.inference_mode():
                        fused = model.encode(
                            {k: v[None] for k, v in images.items()},
                            values[None],
                            scene[None],
                        )
                        features[(variant["label"], int(offset), mode)] = fused[
                            0
                        ].numpy()
                        pred = model.head(fused)[0].numpy()
                        pred = pred * np.asarray(ck["target_std"]) + ck["target_mean"]
                    output.append(
                        {
                            "variant": variant["label"],
                            "offset": int(offset),
                            "mode": mode,
                            "cd": float(pred[0]),
                            "cl": float(pred[1]),
                            "cl_over_cd": float(pred[1] / pred[0])
                            if pred[0] > 1e-6
                            else None,
                        }
                    )
        for row in output:
            baseline = next(
                r
                for r in output
                if r["variant"] == "original"
                and r["offset"] == row["offset"]
                and r["mode"] == row["mode"]
            )
            row["delta_cd"] = row["cd"] - baseline["cd"]
            row["delta_cl"] = row["cl"] - baseline["cl"]
            f = features[(row["variant"], row["offset"], row["mode"])]
            original = features[("original", row["offset"], row["mode"])]
            row["fused_feature_relative_l2"] = float(
                np.linalg.norm(f - original) / max(np.linalg.norm(original), 1e-12)
            )
            if (
                row["variant"] == "zero_control"
                and max(abs(row["delta_cd"]), abs(row["delta_cl"])) > 1e-10
            ):
                raise ValueError("Zero-change control changed model coefficients")
        results["models"][str(size)] = {
            "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "predictions": output,
        }
    (root / "sensitivity.json").write_text(
        json.dumps(results, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                k: [
                    r
                    for r in v["predictions"]
                    if r["offset"] == 0 and r["variant"] == "mirrors_minus30"
                ]
                for k, v in results["models"].items()
            },
            indent=2,
        )
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    a = sub.add_parser("prepare")
    a.add_argument("--prepared", type=Path, required=True)
    a.add_argument("--case-id", required=True)
    a.add_argument("--source-mesh", type=Path, required=True)
    a.add_argument("--output-dir", type=Path, required=True)
    b = sub.add_parser("predict")
    b.add_argument("--output-dir", type=Path, required=True)
    b.add_argument("--width-experiment", type=Path, required=True)
    args = p.parse_args()
    sys.path.insert(
        0, str(Path(__file__).resolve().parents[3] / "mesh-watertight-repair/scripts")
    )
    if args.action == "prepare":
        prepare(args)
    else:
        predict(args)


if __name__ == "__main__":
    main()
