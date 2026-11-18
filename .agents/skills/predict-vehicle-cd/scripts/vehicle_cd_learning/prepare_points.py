"""Pair frozen vehicle cases with area-sampled complete source surfaces."""

import argparse
import hashlib
import json
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyvista as pv

from .render import canonical_points

ROOT = Path(__file__).resolve().parents[5]


def surface_cdf(points, faces):
    """Precompute area once for shared complete-car components."""
    triangles = np.asarray(points, dtype=np.float32)[faces]
    area = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    if not np.isfinite(area).all() or area.sum() <= 0:
        raise ValueError("Source has no finite positive surface area")
    return triangles, np.cumsum(area, dtype=np.float64)


def sample_surface(parts, count, seed=42):
    """Sample complete surface area, independently of mesh vertex density."""
    rng = np.random.default_rng(seed)
    # Stratification reduces sampling variance; the same random numbers are used
    # across variants. No labels or held-out targets enter sampling.
    limits = np.cumsum([cdf[-1] for _, cdf in parts])
    area_samples = (np.arange(count) + rng.random(count)) / count * limits[-1]
    part_ids = np.searchsorted(limits, area_samples)
    uv = rng.random((count, 2))
    uv[uv.sum(axis=1) > 1] = 1 - uv[uv.sum(axis=1) > 1]
    selected = np.empty((count, 3, 3), dtype=np.float32)
    for i, (triangles, cdf) in enumerate(parts):
        mask = part_ids == i
        offsets = area_samples[mask] - (limits[i - 1] if i else 0)
        selected[mask] = triangles[np.searchsorted(cdf, offsets)]
    result = selected[:, 0] + (selected[:, 1] - selected[:, 0]) * uv[:, :1]
    result += (selected[:, 2] - selected[:, 0]) * uv[:, 1:]
    return result.astype(np.float32)


def source_specs():
    specs, hashes = {}, {}
    for source, relative in (
        ("drivaerml", "public-data/drivaerml/full-training_manifest.json"),
        ("ahmedml", "public-data/ahmedml/full-source.json"),
        ("windsorml", "public-data/windsorml/full-source-deduplicated.json"),
        ("builtin_assumed", "all-builtin-training/run26-upright/source.json"),
    ):
        path = ROOT / "outputs/predict-vehicle-cd" / relative
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        for c in json.loads(path.read_text())["cases"]:
            g = dict(c["geometry"])
            g["path"] = str((path.parent / g["path"]).resolve())
            specs[f"{source}__{c['case_id']}"] = g
    return specs, hashes


def rim_body(raw):
    """Reuse the complete body/assembly contract from prepare_drivaerrim."""
    inventory = json.loads((raw / "inventory-geometry.json").read_text())
    expected = {x["path"]: x["sha256"] for x in inventory["files"] if "sha256" in x}
    parts, sources = [], []
    for name in (
        "Car_Body",
        "WheelHouse_FL",
        "WheelHouse_RL",
        "WheelSupport_FL",
        "WheelSupport_RL",
    ):
        path = raw / f"Geometry/{name}/{name}.stl"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected[str(path.relative_to(raw))]:
            raise ValueError(f"Pinned shared rim geometry changed: {path}")
        sources.append({"path": str(path), "sha256": digest})
        mesh = pv.read(path)
        parts.append(mesh)
        if name != "Car_Body":
            other = mesh.copy()
            other.points[:, 1] *= -1
            parts.append(other)
    mesh = parts[0].append_polydata(*parts[1:])
    bounds = np.array(mesh.bounds).reshape(3, 2)
    bounds[2, 0] = pv.read(raw / "Geometry/Wheel_FL/Wheel_FL_0001.stl").bounds[4]
    dimensions = np.diff(bounds, axis=1).ravel()
    origin = np.array([bounds[0].mean(), bounds[1].mean(), bounds[2, 0]])
    faces = np.asarray(mesh.faces).reshape(-1, 4)[:, 1:]
    surface = surface_cdf((mesh.points - origin) / dimensions[0], faces)
    return (surface, origin, dimensions), sources, expected


def prepare_case(case, spec, output, count, shared, shared_sources, rim_expected):
    if case["source"] == "drivaerrim":
        raw = ROOT / "outputs/aerodynamic-expansion-20261109/raw/drivaerrim"
        gid = case["case_id"].rsplit("-", 1)[-1]
        surface, origin, dimensions = shared
        surfaces, sources = [surface], list(shared_sources)
        for pos in ("FL", "RL"):
            path = raw / f"Geometry/Wheel_{pos}/Wheel_{pos}_{gid}.stl"
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != rim_expected[str(path.relative_to(raw))]:
                raise ValueError(f"Pinned wheel source changed: {path}")
            sources.append({"path": str(path), "sha256": digest})
            mesh = pv.read(path)
            points = ((mesh.points - origin) / dimensions[0]).astype(np.float32)
            if points[:, 2].min() < -1e-5 or np.max(np.abs(points[:, :2])) > 0.65:
                raise ValueError("Wheel leaves the original complete-car frame")
            faces = np.asarray(mesh.faces).reshape(-1, 4)[:, 1:]
            part = surface_cdf(points, faces)
            surfaces.append(part)
            reflected = part[0].copy()
            reflected[:, :, 1] *= -1
            surfaces.append((reflected, part[1]))
        actual = dimensions.tolist()
    else:
        path = Path(spec["path"])
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        sidecar = Path(f"{path}.source.json")
        if sidecar.exists():
            source = json.loads(sidecar.read_text())
            if digest != source.get("source_hash", source.get("sha256", digest)):
                raise ValueError(f"Pinned geometry changed: {path}")
        sources = [{"path": str(path), "sha256": digest}]
        mesh = pv.read(path)
        points, dimensions = canonical_points(
            mesh.points,
            **{k: spec[k] for k in ("unit_scale_to_m", "front_axis", "up_axis")},
            ground_clearance_m=spec.get("ground_clearance_m", 0.0),
        )
        faces = np.asarray(mesh.triangulate().faces).reshape(-1, 4)[:, 1:]
        surfaces = [surface_cdf(points, faces)]
        actual = [dimensions[k] for k in ("length_m", "width_m", "height_m")]
    expected = [
        np.exp(case["parameters"][2]),
        case["parameters"][0] * np.exp(case["parameters"][2]),
        case["parameters"][1] * np.exp(case["parameters"][2]),
    ]
    if not np.allclose(actual, expected, rtol=2e-5, atol=2e-5):
        raise ValueError(f"Source frame differs from CNN geometry: {case['case_id']}")
    cloud = sample_surface(surfaces, count)
    path = output / "points" / f"{case['geometry_id']}.npz"
    np.savez_compressed(path, points=cloud)
    return case["case_id"], {
        "path": str(path.relative_to(output)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_meshes": sources,
        "dimensions_m": actual,
        "triangles": sum(len(triangles) for triangles, _ in surfaces),
        "point_count": len(cloud),
    }


def main():
    """@entry Freeze the vehicle slice; sample full meshes without part crops."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--points", type=int, default=8192)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError("Use a fresh prepared directory")
    data_path = args.prepared / "dataset.json"
    data = json.loads(data_path.read_text())
    data["cases"] = [c for c in data["cases"] if c["scene"] == 1]
    output.mkdir(parents=True)
    (output / "points").mkdir()
    (output / "views").mkdir()
    for c in data["cases"]:
        source = args.prepared / c["views"]
        if hashlib.sha256(source.read_bytes()).hexdigest() != c["views_sha256"]:
            raise ValueError("Source complete view changed")
        shutil.copyfile(source, output / c["views"])
    specs, hashes = source_specs()
    raw = ROOT / "outputs/aerodynamic-expansion-20261109/raw/drivaerrim"
    shared, shared_sources, expected = rim_body(raw)
    result = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = [
            pool.submit(
                prepare_case,
                c,
                specs.get(c["case_id"]),
                output,
                args.points,
                shared,
                shared_sources,
                expected,
            )
            for c in data["cases"]
        ]
        for future in as_completed(jobs):
            cid, row = future.result()
            result[cid] = row
            if len(result) % 25 == 0:
                print(
                    json.dumps({"prepared": len(result), "total": len(jobs)}),
                    flush=True,
                )
    data["source_joint_dataset"] = str(data_path.resolve())
    data["source_joint_dataset_sha256"] = hashlib.sha256(
        data_path.read_bytes()
    ).hexdigest()
    data["source_counts"] = {
        k: v
        for k, v in data["source_counts"].items()
        if k in {c["source"] for c in data["cases"]}
    }
    data["unknown_counts"] = {
        k: data["unknown_counts"][k] for k in data["source_counts"]
    }
    (output / "dataset.json").write_text(json.dumps(data, indent=2) + "\n")
    (output / "point-manifest.json").write_text(
        json.dumps(
            {
                "complete": True,
                "points": args.points,
                "seed": 42,
                "source_manifest_hashes": hashes,
                "cases": result,
                "scope": "Complete original surfaces; area-stratified samples, canonical source axes/scale; no geometry repair, local crops, normals or wall-distance features",
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"complete": True, "geometries": len(result)}), flush=True)


if __name__ == "__main__":
    main()
