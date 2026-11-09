"""Collect pinned AhmedML and WindsorML geometry/CFD-label datasets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from http.client import IncompleteRead
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener, urlopen

from vehicle_cd_similarity_lib.mesh_io import read_mesh

from .render import canonical_points

PROFILE_PATH = (
    Path(__file__).resolve().parents[2] / "assets" / "public-cfd-profiles.json"
)
DATASETS = json.loads(PROFILE_PATH.read_text())

SPLIT_COUNTS = {"train": 16, "val": 4, "test": 4}
SPLIT_MANIFEST_NAME = "splits-manifest.json"
SPLIT_SOURCE_PATH = "splits/manifest.json"
FULL_MANIFEST_NAME = "full-source.json"
MAX_RETRIES = 3


class SourceAssetMissing(FileNotFoundError):
    """A pinned run asset is not published at the requested source revision."""


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _hash_file(path: Path, algorithm: str) -> str:
    if algorithm == "sha256":
        digest = hashlib.sha256()
    elif algorithm == "git_blob_sha1":
        digest = hashlib.sha1()
        digest.update(f"blob {path.stat().st_size}\0".encode())
    else:
        raise ValueError(f"Unsupported file digest: {algorithm}")
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256(path: Path) -> str:
    return _hash_file(path, "sha256")


def _actual_metadata(path: Path, expected: dict) -> dict:
    actual = {"bytes": path.stat().st_size}
    for algorithm in ("sha256", "git_blob_sha1"):
        if algorithm in expected:
            actual[algorithm] = _hash_file(path, algorithm)
    return actual


def _manifest_url(dataset: dict) -> str:
    return (
        f"https://huggingface.co/datasets/{dataset['repo']}/resolve/"
        f"{dataset['revision']}/splits/manifest.json"
    )


def _source_file_url(dataset: dict, relative_path: str) -> str:
    return (
        f"https://huggingface.co/datasets/{dataset['repo']}/resolve/"
        f"{dataset['revision']}/{relative_path}"
    )


def _pinned_file_metadata(url: str, revision: str) -> dict:
    headers = None
    for attempt in range(MAX_RETRIES):
        opener = build_opener(_NoRedirect())
        request = Request(url, method="HEAD")
        try:
            response = opener.open(request, timeout=30)
            headers = response.headers
            break
        except HTTPError as response:
            if response.code in (301, 302, 303, 307, 308):
                headers = response.headers
                break
            if response.code in (404, 410):
                raise SourceAssetMissing(url) from response
            if (
                response.code not in (408, 429, 500, 502, 503, 504)
                or attempt + 1 == MAX_RETRIES
            ):
                raise
        except (URLError, TimeoutError, ConnectionError):
            if attempt + 1 == MAX_RETRIES:
                raise
        time.sleep(0.5 * (2**attempt))
    if headers is None:
        raise RuntimeError(f"Could not read pinned source metadata: {url}")
    if headers.get("x-repo-commit") != revision:
        raise ValueError(f"Resolved revision differs from pinned commit: {url}")
    size = headers.get("x-linked-size")
    etag = headers.get("x-linked-etag", "").strip('"')
    if not size:
        raise ValueError(f"Pinned source did not expose file size: {url}")
    if len(etag) == 64:
        return {"bytes": int(size), "sha256": etag}
    if len(etag) == 40:
        return {"bytes": int(size), "git_blob_sha1": etag}
    raise ValueError(f"Pinned source did not expose a recognized object hash: {url}")


def _download(url: str, destination: Path, expected: dict) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        actual = _actual_metadata(destination, expected)
        if actual != expected:
            raise ValueError(
                f"Existing asset differs from pinned source: {destination}"
            )
        return actual
    for attempt in range(MAX_RETRIES):
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=destination.parent,
                prefix=destination.name,
                suffix=".partial",
                delete=False,
            ) as output:
                temporary = Path(output.name)
                sha256 = hashlib.sha256() if "sha256" in expected else None
                git_blob_sha1 = (
                    hashlib.sha1(f"blob {expected['bytes']}\0".encode())
                    if "git_blob_sha1" in expected
                    else None
                )
                size = 0
                try:
                    response = urlopen(url, timeout=120)
                except HTTPError as error:
                    if error.code in (404, 410):
                        raise SourceAssetMissing(url) from error
                    if (
                        error.code not in (408, 429, 500, 502, 503, 504)
                        or attempt + 1 == MAX_RETRIES
                    ):
                        raise
                    time.sleep(0.5 * (2**attempt))
                    continue
                with response:
                    while block := response.read(1024 * 1024):
                        output.write(block)
                        if sha256:
                            sha256.update(block)
                        if git_blob_sha1:
                            git_blob_sha1.update(block)
                        size += len(block)
            actual = {"bytes": size}
            if sha256:
                actual["sha256"] = sha256.hexdigest()
            if git_blob_sha1:
                actual["git_blob_sha1"] = git_blob_sha1.hexdigest()
            if actual["bytes"] != expected["bytes"]:
                if attempt + 1 == MAX_RETRIES:
                    raise ValueError(f"Downloaded asset has wrong pinned size: {url}")
                time.sleep(0.5 * (2**attempt))
                continue
            if actual != expected:
                raise ValueError(
                    f"Downloaded asset failed pinned size/hash check: {url}"
                )
            temporary.replace(destination)
            return actual
        except (URLError, TimeoutError, ConnectionError, IncompleteRead):
            if attempt + 1 == MAX_RETRIES:
                raise
            time.sleep(0.5 * (2**attempt))
        finally:
            if temporary and temporary.exists():
                temporary.unlink()
    raise RuntimeError(f"Could not download pinned source asset: {url}")


def _inventory_files(catalogue: Path, name: str) -> dict:
    inventory = json.loads((catalogue / "inventory.json").read_text())
    dataset = DATASETS[name]
    if inventory["revision"] != dataset["revision"]:
        raise ValueError(f"Catalogue revision mismatch for {name}")
    verified = {}
    for item in inventory["files"]:
        path = catalogue / item["path"]
        if not path.is_file() or path.stat().st_size != item["bytes"]:
            raise ValueError(f"Catalogue file missing or wrong size: {path}")
        if _sha256(path) != item["sha256"]:
            raise ValueError(f"Catalogue file hash mismatch: {path}")
        verified[item["path"]] = item
    return verified


def _ensure_catalogue(catalogue: Path, name: str) -> dict:
    dataset = DATASETS[name]
    catalogue.mkdir(parents=True, exist_ok=True)
    inventory_path = catalogue / "inventory.json"
    required = (
        "README.md",
        "LICENSE.txt",
        dataset["label_file"],
        "geo_parameters_all.csv",
    )
    if inventory_path.is_file():
        files = _inventory_files(catalogue, name)
        if any(path not in files for path in required):
            raise ValueError(
                f"Catalogue inventory is missing required source files: {catalogue}"
            )
    else:
        entries = []
        source_paths = [*required, SPLIT_SOURCE_PATH]
        for source_path in source_paths:
            local_name = (
                SPLIT_MANIFEST_NAME if source_path == SPLIT_SOURCE_PATH else source_path
            )
            url = (
                _manifest_url(dataset)
                if source_path == SPLIT_SOURCE_PATH
                else _source_file_url(dataset, source_path)
            )
            expected = _pinned_file_metadata(url, dataset["revision"])
            local_path = catalogue / local_name
            _download(url, local_path, expected)
            entries.append(
                {
                    "path": local_name,
                    "bytes": local_path.stat().st_size,
                    "sha256": _sha256(local_path),
                    "source_hash": expected.get(
                        "sha256", expected.get("git_blob_sha1")
                    ),
                    "table_rows": None,
                }
            )
        inventory_path.write_text(
            json.dumps(
                {
                    "repository": dataset["repo"],
                    "revision": dataset["revision"],
                    "source_url": f"https://huggingface.co/datasets/{dataset['repo']}",
                    "files": entries,
                    "geometry_downloaded": False,
                    "use": "Pinned metadata, geometry parameters, and force labels only; no volume or field data.",
                },
                indent=2,
            )
            + "\n"
        )
        files = _inventory_files(catalogue, name)

    split_path = catalogue / SPLIT_MANIFEST_NAME
    if not split_path.is_file():
        url = _manifest_url(dataset)
        _download(url, split_path, _pinned_file_metadata(url, dataset["revision"]))
    return files


def _split_selection(
    catalogue: Path, name: str, full: bool = False
) -> tuple[dict, dict]:
    dataset = DATASETS[name]
    path = catalogue / SPLIT_MANIFEST_NAME
    if not path.is_file():
        raise ValueError(f"Pinned split manifest is missing: {path}")
    url = _manifest_url(dataset)
    remote = _pinned_file_metadata(url, dataset["revision"])
    if _actual_metadata(path, remote) != remote:
        raise ValueError(f"Local split manifest differs from pinned source: {path}")
    local = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    manifest = json.loads(path.read_text())
    selected = {}
    official_counts = {}
    for split, count in SPLIT_COUNTS.items():
        key = f"full_{split}"
        if key not in manifest or not isinstance(manifest[key], list):
            raise ValueError(f"Pinned official split is missing: {key}")
        official_counts[split] = len(manifest[key])
        if not full and len(manifest[key]) < count:
            raise ValueError(f"Official {key} has fewer than {count} run IDs")
        selected[split] = manifest[key][:] if full else manifest[key][:count]
    metadata = {
        "path": SPLIT_MANIFEST_NAME,
        "url": url,
        "revision": dataset["revision"],
        **local,
        **(
            {"git_blob_sha1": remote["git_blob_sha1"]}
            if "git_blob_sha1" in remote
            else {}
        ),
        "selection": (
            "all full_train, full_val, and full_test IDs in the pinned official manifest"
            if full
            else "first 16 full_train, first 4 full_val, and first 4 full_test IDs in the pinned official manifest"
        ),
        "official_partition_counts": official_counts,
    }
    return selected, metadata


def _table(path: Path) -> dict[int, dict]:
    with path.open(newline="") as source:
        rows = csv.DictReader(source, skipinitialspace=True)
        if not rows.fieldnames or "run" not in rows.fieldnames:
            raise ValueError(f"Table has no run key: {path}")
        return {int(row["run"]): row for row in rows}


def _mesh_url(dataset: dict, run: int) -> str:
    return (
        f"https://huggingface.co/datasets/{dataset['repo']}/resolve/"
        f"{dataset['revision']}/run_{run}/{dataset['mesh_name']}_{run}.stl"
    )


def _case_runs(selection: dict) -> list[tuple[str, int]]:
    return [
        (split, int(run.removeprefix("run_")))
        for split in SPLIT_COUNTS
        for run in selection[split]
    ]


def _split_counts(selection: dict) -> dict[str, int]:
    return {split: len(selection[split]) for split in SPLIT_COUNTS}


def _pilot_geometry_metadata(output: Path, name: str) -> dict[int, dict]:
    path = output / "source.json"
    if not path.is_file():
        return {}
    source = json.loads(path.read_text())
    spec = DATASETS[name]
    provenance = source.get("provenance", {})
    if (
        provenance.get("repository") != spec["repo"]
        or provenance.get("revision") != spec["revision"]
    ):
        raise ValueError(
            f"Existing pilot source does not match pinned revision: {path}"
        )
    return {
        int(asset["run"]): {"bytes": asset["bytes"], "sha256": asset["sha256"]}
        for asset in provenance.get("geometry_assets", [])
    }


def _available_full_selection(
    name: str,
    catalogue: Path,
    output: Path,
    selection: dict,
    workers: int,
) -> tuple[dict, dict[int, dict], list[dict]]:
    spec = DATASETS[name]
    labels = _table(catalogue / spec["label_file"])
    geometry_rows = _table(catalogue / "geo_parameters_all.csv")
    selected = {split: [] for split in SPLIT_COUNTS}
    excluded = []
    candidates = []
    for split, run in _case_runs(selection):
        reasons = []
        if run not in labels:
            reasons.append("missing_variable_reference_labels")
        if run not in geometry_rows:
            reasons.append("missing_geometry_parameters")
        if reasons:
            excluded.append({"split": split, "run": run, "reasons": reasons})
        else:
            selected[split].append(f"run_{run}")
            candidates.append((split, run))

    cached_metadata = _pilot_geometry_metadata(output, name)
    expected: dict[int, dict] = {}
    missing_meshes: dict[int, dict] = {}
    futures = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for _split, run in candidates:
            existing = output / "geometries" / f"{name}-run-{run}.stl"
            if existing.is_file() and run in cached_metadata:
                actual = _actual_metadata(existing, cached_metadata[run])
                if actual != cached_metadata[run]:
                    raise ValueError(
                        f"Cached pilot STL differs from its checksum: {existing}"
                    )
                expected[run] = cached_metadata[run]
            else:
                futures[
                    pool.submit(
                        _pinned_file_metadata, _mesh_url(spec, run), spec["revision"]
                    )
                ] = run
        for future in as_completed(futures):
            run = futures[future]
            try:
                expected[run] = future.result()
            except SourceAssetMissing:
                missing_meshes[run] = {
                    "reason": "mesh_not_published",
                    "source_url": _mesh_url(spec, run),
                }
    for split, run in candidates:
        if run in missing_meshes:
            selected[split].remove(f"run_{run}")
            excluded.append(
                {
                    "split": split,
                    "run": run,
                    "reasons": [missing_meshes[run]["reason"]],
                    "source_url": missing_meshes[run]["source_url"],
                }
            )
    return (
        selected,
        expected,
        sorted(excluded, key=lambda item: (item["split"], item["run"])),
    )


def _collect_meshes(
    name: str,
    catalogue: Path,
    output: Path,
    selection: dict,
    workers: int,
    asset_metadata: dict[int, dict] | None = None,
    progress_name: str | None = None,
    missing_assets: list[dict] | None = None,
) -> dict[int, dict]:
    dataset = DATASETS[name]
    target_dir = output / "geometries"
    tasks = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for _split, run in _case_runs(selection):
            url = _mesh_url(dataset, run)
            expected = (
                asset_metadata[run]
                if asset_metadata is not None
                else _pinned_file_metadata(url, dataset["revision"])
            )
            destination = target_dir / f"{name}-run-{run}.stl"
            cached = catalogue / f"run_{run}" / f"{dataset['mesh_name']}_{run}.stl"
            if cached.is_file() and not destination.exists():
                actual = _actual_metadata(cached, expected)
                if actual != expected:
                    raise ValueError(
                        f"Catalogue STL differs from pinned asset: {cached}"
                    )
                destination.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.link(cached, destination)
                except OSError:
                    shutil.copyfile(cached, destination)
            if destination.exists():
                actual = _actual_metadata(destination, expected)
                if actual != expected:
                    raise ValueError(
                        f"Existing geometry differs from pinned asset: {destination}"
                    )
                tasks[run] = actual
            else:
                tasks[run] = pool.submit(_download, url, destination, expected)
        futures = {
            value: run for run, value in tasks.items() if hasattr(value, "result")
        }
        completed = 0
        for future in as_completed(futures):
            run = futures[future]
            try:
                tasks[run] = future.result()
            except SourceAssetMissing:
                if missing_assets is None:
                    raise
                missing_assets.append(
                    {"run": run, "source_url": _mesh_url(dataset, run)}
                )
                tasks.pop(run)
                continue
            completed += 1
            if progress_name and completed % 25 == 0:
                print(
                    f"{progress_name}: downloaded and verified {completed} new STLs",
                    flush=True,
                )
    if progress_name:
        print(
            f"{progress_name}: all {len(tasks)} selected STLs are verified", flush=True
        )
    return tasks


def _source_manifest(
    name: str,
    catalogue: Path,
    output: Path,
    selection: dict,
    split_metadata: dict,
    files: dict,
    mesh_metadata: dict,
    excluded_runs: list[dict] | None = None,
    full: bool = False,
) -> dict:
    spec = DATASETS[name]
    labels = _table(catalogue / spec["label_file"])
    geometry_rows = _table(catalogue / "geo_parameters_all.csv")
    selected_runs = _case_runs(selection)
    if any(
        run not in labels or run not in geometry_rows for _split, run in selected_runs
    ):
        raise ValueError(f"A selected run is missing geometry or CFD labels: {name}")

    cases = []
    dimensions_by_run = {}
    for split, run in selected_runs:
        geometry_path = output / "geometries" / f"{name}-run-{run}.stl"
        geometry_values = geometry_rows[run]
        ground_clearance_m = spec.get("ground_clearance_m", 0.0)
        mesh = read_mesh(geometry_path)
        _, dimensions = canonical_points(
            mesh.points,
            unit_scale_to_m=1.0,
            front_axis=spec["front_axis"],
            up_axis=spec["up_axis"],
            ground_clearance_m=ground_clearance_m,
        )
        dimensions_by_run[run] = dimensions
        row = labels[run]
        if name == "ahmedml":
            reynolds = (
                spec["speed_m_s"]
                * dimensions["length_m"]
                / spec["kinematic_viscosity_m2_s"]
            )
        else:
            kinematic_viscosity = (
                spec["speed_m_s"]
                * spec["reference_body_length_m"]
                / spec["reference_reynolds"]
            )
            reynolds = spec["speed_m_s"] * dimensions["length_m"] / kinematic_viscosity
        case = {
            "case_id": f"{name}-run-{run}",
            "geometry_id": f"{name}-run-{run}",
            "family_id": spec["family_id"],
            "split": split,
            "geometry": {
                "path": f"geometries/{name}-run-{run}.stl",
                "unit_scale_to_m": 1.0,
                "front_axis": spec["front_axis"],
                "up_axis": spec["up_axis"],
                "ground_clearance_m": ground_clearance_m,
            },
            "reynolds": reynolds,
            "yaw_deg": spec["yaw_deg"],
            "cd": float(row["cd"]),
            "cl": float(row["cl"]),
            "design_value": geometry_values,
            "cd_source": spec["label_file"],
            "cd_confidence": "published CFD coefficient; not experimental truth",
            "assumptions": {
                "conditions_assumed": False,
                "source_conditions_verified": False,
                "source_conditions_note": "Published solver conditions are retained; averaging-window details are incomplete or conflict across source descriptions.",
            },
        }
        if name == "windsorml":
            case["pair_id"] = "windsorml-fixed-cfd-condition"
        cases.append(case)

    provenance = json.loads(json.dumps(spec["provenance"]))
    provenance.update(
        {
            "repository": spec["repo"],
            "revision": spec["revision"],
            "source_url": f"https://huggingface.co/datasets/{spec['repo']}",
            "license_file": "LICENSE.txt",
            "source_files": [
                {
                    "path": path,
                    "bytes": item["bytes"],
                    "sha256": item["sha256"],
                }
                for path, item in sorted(files.items())
                if path
                in (
                    spec["label_file"],
                    "geo_parameters_all.csv",
                    "README.md",
                    "LICENSE.txt",
                )
            ],
            "geometry_assets": [
                {
                    "run": run,
                    "path": f"geometries/{name}-run-{run}.stl",
                    "source_url": _mesh_url(spec, run),
                    **mesh_metadata[run],
                    "length_m": dimensions_by_run[run]["length_m"],
                    "body_clearance_m": float(geometry_rows[run]["clearance"]) / 1000
                    if name == "windsorml"
                    else spec["ground_clearance_m"],
                }
                for _split, run in selected_runs
            ],
        }
    )
    return {
        "schema_version": 1,
        "data_kind": "cfd",
        "target_names": ["cd", "cl"],
        "conditions": spec["conditions"],
        "provenance": provenance,
        "split_policy": "geometry_disjoint",
        "split_provenance": {
            "method": (
                "all available cases from official source full partitions"
                if full
                else "official source full split, bounded deterministic prefix"
            ),
            "official_manifest": split_metadata,
            "official_partition_counts": split_metadata.get(
                "official_partition_counts", _split_counts(selection)
            ),
            "counts": _split_counts(selection),
            "family_id": spec["family_id"],
            "excluded_runs": excluded_runs or [],
        },
        "cases": cases,
    }


def collect(
    catalogue_root: Path, output_root: Path, workers: int = 4, full: bool = False
) -> dict:
    if not 1 <= workers <= 6:
        raise ValueError("workers must be between 1 and 6")
    if not full:
        existing = [
            name for name in DATASETS if (output_root / name / "source.json").exists()
        ]
        if existing:
            raise FileExistsError(
                f"Refusing to overwrite completed datasets under {output_root}: {existing}"
            )
    output_root.mkdir(parents=True, exist_ok=True)
    reports = {}
    result = {}
    for name in DATASETS:
        catalogue = catalogue_root / name
        files = _ensure_catalogue(catalogue, name)
        selection, split_metadata = _split_selection(catalogue, name, full=full)
        dataset_output = output_root / name
        manifest_name = FULL_MANIFEST_NAME if full else "source.json"
        manifest_path = dataset_output / manifest_name
        if full and manifest_path.is_file():
            source = json.loads(manifest_path.read_text())
            if (
                source.get("provenance", {}).get("revision")
                != DATASETS[name]["revision"]
            ):
                raise ValueError(
                    f"Existing full manifest differs from pinned source: {manifest_path}"
                )
            for asset in source["provenance"]["geometry_assets"]:
                path = dataset_output / asset["path"]
                expected = {
                    key: asset[key]
                    for key in ("bytes", "sha256", "git_blob_sha1")
                    if key in asset
                }
                if _actual_metadata(path, expected) != expected:
                    raise ValueError(
                        f"Existing full source asset failed checksum: {path}"
                    )
            result[name] = {
                "source": str(manifest_path),
                "case_count": len(source["cases"]),
            }
            print(f"{name}: existing full manifest and assets verified", flush=True)
            continue

        dataset_output.mkdir(exist_ok=True)
        excluded = []
        asset_metadata = None
        missing_assets = None
        if full:
            selection, asset_metadata, excluded = _available_full_selection(
                name, catalogue, dataset_output, selection, workers
            )
            print(
                f"{name}: official candidates {split_metadata['official_partition_counts']}; available paired cases {_split_counts(selection)}; excluded {len(excluded)}",
                flush=True,
            )
            missing_assets = []
        mesh_metadata = _collect_meshes(
            name,
            catalogue,
            dataset_output,
            selection,
            workers,
            asset_metadata=asset_metadata,
            progress_name=name if full else None,
            missing_assets=missing_assets,
        )
        for missing in missing_assets or []:
            run = missing["run"]
            split = next(
                split for split, runs in selection.items() if f"run_{run}" in runs
            )
            selection[split].remove(f"run_{run}")
            excluded.append(
                {
                    "split": split,
                    "run": run,
                    "reasons": ["mesh_not_published"],
                    "source_url": missing["source_url"],
                }
            )
        excluded.sort(key=lambda item: (item["split"], item["run"]))
        source = _source_manifest(
            name,
            catalogue,
            dataset_output,
            selection,
            split_metadata,
            files,
            mesh_metadata,
            excluded_runs=excluded,
            full=full,
        )
        if full:
            manifest_path.write_text(
                json.dumps(source, indent=2, allow_nan=False) + "\n"
            )
            result[name] = {
                "source": str(manifest_path),
                "case_count": len(source["cases"]),
            }
            print(
                f"{name}: full manifest ready with {_split_counts(selection)} cases at {manifest_path}",
                flush=True,
            )
        else:
            reports[name] = {"output": dataset_output, "source": source}

    if full:
        return result
    for name, item in reports.items():
        path = item["output"] / "source.json"
        path.write_text(json.dumps(item["source"], indent=2, allow_nan=False) + "\n")
        result[name] = {"source": str(path), "case_count": len(item["source"]["cases"])}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalogue-root",
        type=Path,
        default=Path("outputs/predict-vehicle-cd/source-catalogue"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/predict-vehicle-cd/public-data"),
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--full",
        action="store_true",
        help="collect all available cases from official full splits",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            collect(args.catalogue_root, args.output_root, args.workers, args.full),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
