"""Render genuine full-car Rim five-view caches from unchanged raw triangles."""

import argparse
import copy
import csv
import hashlib
import json
import resource
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyvista as pv

from .prepare_drivaerrim import merge_depth
from .render import VIEW_AXES, _rasterize, save_five_views

SHARED = (
    "Car_Body",
    "WheelHouse_FL",
    "WheelHouse_RL",
    "WheelSupport_FL",
    "WheelSupport_RL",
)


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def render_component(mesh, origin, length, size):
    """Keep the old full_frame normalization, fixed window and rasterizer margin."""
    points = ((mesh.points - origin) / length).astype(np.float32)
    if np.max(abs(points[:, :2])) > 0.65 or points[:, 2].min() < -1e-5:
        raise ValueError("Part falls outside the declared full-car frame")
    faces = np.asarray(mesh.faces).reshape(-1, 4)[:, 1:]
    return {
        name: _rasterize(points, faces, view=name, size=size, window_l=1.3)
        for name in VIEW_AXES
    }


def shared_views(raw, entries):
    """Render each shared triangle once per resolution; never append a full car."""
    meshes, hashes = [], {}
    for name in SHARED:
        relative = f"Geometry/{name}/{name}.stl"
        path = raw / relative
        hashes[relative] = sha256(path)
        if hashes[relative] != entries[relative]["sha256"]:
            raise ValueError(f"Shared source hash mismatch: {relative}")
        meshes.append(pv.read(path))
    bounds = np.asarray([mesh.bounds for mesh in meshes]).reshape(-1, 3, 2)
    mirrored = bounds[1:].copy()
    mirrored[:, 1, :] = -mirrored[:, 1, ::-1]
    bounds = np.concatenate((bounds, mirrored))
    low, high = bounds[:, :, 0].min(0), bounds[:, :, 1].max(0)
    tyre_path = raw / "Geometry/Wheel_FL/Wheel_FL_0001.stl"
    if sha256(tyre_path) != entries["Geometry/Wheel_FL/Wheel_FL_0001.stl"]["sha256"]:
        raise ValueError("Frame tyre source hash mismatch")
    tyre = pv.read(tyre_path)
    low[2] = tyre.bounds[4]
    del tyre
    dimensions = high - low
    origin = np.asarray([(low[0] + high[0]) / 2, (low[1] + high[1]) / 2, low[2]])
    buffers = {256: None, 512: None}
    for name, mesh in zip(SHARED, meshes):
        for mirror in range(1 if name == "Car_Body" else 2):
            if mirror:
                mesh.points[:, 1] *= -1
            for size, previous in buffers.items():
                current = render_component(mesh, origin, dimensions[0], size)
                buffers[size] = (
                    current if previous is None else merge_depth(previous, current)
                )
    return buffers, origin, dimensions, hashes


def render_case(
    case, raw, source_root, output, entries, body, origin, dimensions, replay
):
    started = time.monotonic()
    gid = case["case_id"].rsplit("-", 1)[-1]
    old_path = source_root / case["views"]
    if sha256(old_path) != case["views_sha256"]:
        raise ValueError(f"Existing256 cache hash mismatch: {gid}")
    sizes = (256, 512) if replay else (512,)
    views, hashes, digest = {s: None for s in sizes}, {}, hashlib.sha256()
    for position in ("FL", "RL"):
        relative = f"Geometry/Wheel_{position}/Wheel_{position}_{gid}.stl"
        path = raw / relative
        actual = sha256(path)
        if actual != entries[relative]["sha256"]:
            raise ValueError(f"Wheel source hash mismatch: {relative}")
        hashes[relative] = actual
        digest.update(actual.encode())
        mesh = pv.read(path)
        if not np.isclose(mesh.bounds[4], origin[2], atol=1e-5):
            raise ValueError("Unexpected tyre contact height")
        for mirror in (False, True):
            if mirror:
                mesh.points[:, 1] *= -1
            for size in sizes:
                current = render_component(mesh, origin, dimensions[0], size)
                views[size] = (
                    current
                    if views[size] is None
                    else merge_depth(views[size], current)
                )
    if digest.hexdigest() != case["geometry_hash"]:
        raise ValueError(f"Prepared geometry signature mismatch: {gid}")
    for size in sizes:
        merge_depth(views[size], body[size])
    difference = None
    if replay:
        with np.load(old_path, allow_pickle=False) as old:
            difference = {
                name: float(abs(np.stack((v.depth, v.mask)) - old[name]).max())
                for name, v in views[256].items()
            }
        if max(difference.values()) > 1e-7:
            raise ValueError(f"Raw256 replay differs: {difference}")
    result = copy.deepcopy(case)
    result["views_256"] = str(old_path.resolve())
    result["views_256_sha256"] = case["views_sha256"]
    result["views"] = f"views/{case['geometry_id']}.npz"
    save_five_views(output / result["views"], views[512])
    result["views_sha256"] = sha256(output / result["views"])
    result.pop("numerical_views_sha256", None)
    result["source_views_sha256"] = result["views_256_sha256"]
    return result, {
        "source_hashes": hashes,
        "seconds": time.monotonic() - started,
        "replay256_max_abs": difference,
    }


def source_cases(prepared, raw):
    dataset = prepared / "dataset.json" if prepared.is_dir() else prepared
    data = json.loads(dataset.read_text())
    cases = sorted(
        (c for c in data["cases"] if c["source"] == "drivaerrim"),
        key=lambda c: c["case_id"],
    )
    if len(cases) != 904 or len({c["case_id"] for c in cases}) != 904:
        raise ValueError("Expected904 unique prepared Rim cases")
    labels = {
        f"{int(r['Rim']):04d}": [float(r["Cd"]), float(r["Cl"])]
        for r in csv.DictReader(
            (raw / "NumData/Force_Coefficients_Combined.csv").open()
        )
    }
    splits = {}
    for split, filename in (
        ("train", "case_01_train_720/train_ids.txt"),
        ("val", "validation_ids.txt"),
        ("test", "test_ids.txt"),
    ):
        for line in (raw / "Dataset_Splits" / filename).read_text().splitlines():
            gid = line.strip().rsplit("_", 1)[-1]
            if gid in splits:
                raise ValueError("Source ID appears in multiple splits")
            splits[gid] = split
    for case in cases:
        gid = case["case_id"].rsplit("-", 1)[-1]
        if case["targets"] != labels[gid] or case["split"] != splits[gid]:
            raise ValueError("Source labels or official splits differ")
    if [sum(c["split"] == s for c in cases) for s in ("train", "val", "test")] != [
        720,
        92,
        92,
    ]:
        raise ValueError("Unexpected Rim split counts")
    return dataset, data, cases


def prepare(prepared, raw, output, workers=2, limit=None):
    """@entry Fresh-only full512 cache, or training-only benchmark with --limit."""
    prepared, raw, output = Path(prepared), Path(raw), Path(output)
    if output.exists():
        raise FileExistsError(
            "Use a fresh output directory; partial outputs are not resumed"
        )
    if workers < 1 or (limit is not None and not 1 <= limit <= 720):
        raise ValueError("Require workers>=1 and training-only limit in1..720")
    dataset, data, cases = source_cases(prepared, raw)
    if limit is not None:
        cases = [c for c in cases if c["split"] == "train"][:limit]
    anchor = next(c["case_id"] for c in cases if c["split"] == "train")
    inventory_path = raw / "inventory-geometry.json"
    inventory = json.loads(inventory_path.read_text())
    if not inventory["complete"]:
        raise ValueError("Incomplete raw source inventory")
    entries = {r["path"]: r for r in inventory["files"]}
    output.mkdir(parents=True)
    started = time.monotonic()
    body, origin, dimensions, hashes = shared_views(raw, entries)
    body_seconds = time.monotonic() - started
    print(json.dumps({"shared_render_seconds": body_seconds}), flush=True)
    rendered, audit = {}, {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = [
            pool.submit(
                render_case,
                c,
                raw,
                dataset.parent,
                output,
                entries,
                body,
                origin,
                dimensions,
                c["case_id"] == anchor,
            )
            for c in cases
        ]
        for future in as_completed(jobs):
            case, evidence = future.result()
            rendered[case["case_id"]], audit[case["case_id"]] = case, evidence
            print(
                json.dumps(
                    {
                        "prepared": len(rendered),
                        "total": len(cases),
                        "case_id": case["case_id"],
                        "seconds": evidence["seconds"],
                    }
                ),
                flush=True,
            )
    result = copy.deepcopy(data)
    result["cases"] = [rendered[c["case_id"]] for c in cases]
    result["source_counts"] = {"drivaerrim": len(cases)}
    result["render_config"] = {
        "size": 512,
        "window_l": 1.3,
        "margin": 4,
        "scale": "declared_body_length",
        "source": "raw triangles; no resize/crop/decimation",
    }
    result["cache_conversion"] = (
        "None for512: direct raw-triangle raster; original256 caches referenced separately."
    )
    result["resolution_preparation"] = {
        "train_ready": limit is None,
        "training_only_smoke_limit": limit,
        "source_prepared": str(dataset.resolve()),
        "source_dataset_sha256": sha256(dataset),
        "inventory_sha256": sha256(inventory_path),
        "source_revision": inventory["revision"],
        "origin_m": origin.tolist(),
        "dimensions_m": dimensions.tolist(),
        "shared_source_hashes": hashes,
        "case_audit": audit,
        "shared_render_seconds": body_seconds,
        "elapsed_seconds": time.monotonic() - started,
        "workers": workers,
        "peak_rss_platform_units": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "peak_rss_note": "macOS bytes; Linux KiB; includes rasterizer and VTK allocations",
        "code_sha256": {
            str(p.resolve()): sha256(p)
            for p in (
                Path(__file__),
                Path(__file__).with_name("render.py"),
                Path(__file__).with_name("prepare_drivaerrim.py"),
            )
        },
    }
    (output / "dataset.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "finished": len(rendered),
                "elapsed_seconds": result["resolution_preparation"]["elapsed_seconds"],
                "train_ready": limit is None,
            }
        ),
        flush=True,
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare"])
    parser.add_argument("prepared", type=Path)
    parser.add_argument("raw", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    prepare(args.prepared, args.raw, args.output, args.workers, args.limit)


if __name__ == "__main__":
    main()
