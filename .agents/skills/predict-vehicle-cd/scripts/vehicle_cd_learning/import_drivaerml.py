"""Fetch a bounded DrivAerML pilot or every official geometry/coefficient case."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import http.client
import json
import os
import random
import re
import shutil
import tempfile
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

DATASET = "neashton/drivaerml"
REVISION = "5d448b209bf654503c64ce7261c34fa125f46392"
BASE_URL = f"https://huggingface.co/datasets/{DATASET}"
SPEED_M_S = 38.889
KINEMATIC_VISCOSITY_M2_S = 1.507e-5
MAX_PILOT_CASES = 120
MAX_RETRIES = 3
FAMILY_ID = "ford-ocda-drivaer-notchback-morphs"
_DIRECTORY_METADATA: dict[str, dict[str, dict]] = {}
_METADATA_LOCK = threading.Lock()
_PROGRESS_LOCK = threading.Lock()
_STL_VERTEX = re.compile(
    rb"^[ \t]*vertex[ \t]+(\S+)[ \t]+(\S+)[ \t]+(\S+)[ \t\r]*$",
    re.MULTILINE,
)


def _url(path: str) -> str:
    encoded = urllib.parse.quote(path, safe="/")
    return f"{BASE_URL}/resolve/{REVISION}/{encoded}?download=true"


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_metadata(directory: str) -> dict[str, dict]:
    """Read pinned file hashes for one source directory."""
    with _METADATA_LOCK:
        cached = _DIRECTORY_METADATA.get(directory)
    if cached is not None:
        return cached
    suffix = f"/{urllib.parse.quote(directory, safe='/')}" if directory else ""
    url = f"https://huggingface.co/api/datasets/{DATASET}/tree/{REVISION}{suffix}?recursive=false"
    request = urllib.request.Request(
        url, headers={"User-Agent": "cad-surface-mesher-public-data/1.0"}
    )
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                entries = json.load(response)
            break
        except (OSError, http.client.HTTPException, ValueError) as error:
            if attempt == MAX_RETRIES:
                raise
            print(
                f"retry {attempt}/{MAX_RETRIES - 1} listing {directory or 'source'}: {error}",
                flush=True,
            )
            time.sleep(attempt)
    result = {entry["path"]: entry for entry in entries if entry.get("type") == "file"}
    with _METADATA_LOCK:
        _DIRECTORY_METADATA[directory] = result
    return result


def _source_digest(path: Path, algorithm: str) -> str:
    digest = hashlib.sha256() if algorithm == "sha256" else hashlib.sha1()
    if algorithm == "git-sha1":
        digest.update(f"blob {path.stat().st_size}\0".encode("ascii"))
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_hash(entry: dict) -> tuple[str, str]:
    lfs = entry.get("lfs")
    if lfs:
        return "sha256", lfs["oid"]
    return "git-sha1", entry["oid"]


def _write_sidecar(path: Path, metadata: dict) -> None:
    sidecar = Path(f"{path}.source.json")
    temporary = Path(f"{sidecar}.part")
    temporary.write_text(json.dumps(metadata, sort_keys=True) + "\n")
    temporary.replace(sidecar)


def _download_with_hf_xet(path: str, destination: Path, entry: dict) -> dict:
    """Try Hugging Face Hub's optional Xet transport after urllib is exhausted."""
    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import HfHubHTTPError
    except ImportError as error:
        raise OSError("Hugging Face Hub Xet support is unavailable") from error

    algorithm, source_hash = _expected_hash(entry)
    expected_size = int(entry["size"])
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        with tempfile.TemporaryDirectory(
            prefix="drivaerml-hf-cache-", dir=destination.parent
        ) as cache_dir:
            downloaded = Path(
                hf_hub_download(
                    repo_id=DATASET,
                    filename=path,
                    repo_type="dataset",
                    revision=REVISION,
                    cache_dir=cache_dir,
                    force_download=True,
                    etag_timeout=60,
                )
            )
            if downloaded.stat().st_size != expected_size:
                raise OSError(
                    f"{path}: Xet received {downloaded.stat().st_size} bytes, "
                    f"expected {expected_size}"
                )
            if _source_digest(downloaded, algorithm) != source_hash:
                raise OSError(f"{path}: Xet result does not match pinned source hash")
            try:
                os.link(downloaded, temporary)
            except OSError:
                shutil.copyfile(downloaded, temporary)
            temporary.replace(destination)
    except HfHubHTTPError as error:
        temporary.unlink(missing_ok=True)
        raise OSError(
            f"{path}: Hugging Face Hub Xet request failed: {error}"
        ) from error
    except (OSError, ValueError, RuntimeError):
        temporary.unlink(missing_ok=True)
        raise

    cache = {
        "revision": REVISION,
        "source_path": path,
        "source_size_bytes": expected_size,
        "source_hash_algorithm": algorithm,
        "source_hash": source_hash,
    }
    _write_sidecar(destination, cache)
    print(f"verified {path} through Hugging Face Xet", flush=True)
    return {
        **cache,
        "size_bytes": expected_size,
        "sha256": _file_hash(destination),
        "cache_hit": False,
        "transport": "hf_xet",
    }


def _download(
    path: str,
    destination: Path,
    entry: dict,
) -> dict:
    """Verify a pinned cache or atomically fetch one public source file."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    algorithm, source_hash = _expected_hash(entry)
    expected_size = int(entry["size"])
    cache = {
        "revision": REVISION,
        "source_path": path,
        "source_size_bytes": expected_size,
        "source_hash_algorithm": algorithm,
        "source_hash": source_hash,
    }
    if destination.is_file() and destination.stat().st_size == expected_size:
        digest = _source_digest(destination, algorithm)
        if digest == source_hash:
            _write_sidecar(destination, cache)
            return {
                **cache,
                "size_bytes": expected_size,
                "sha256": _file_hash(destination),
                "cache_hit": True,
            }

    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(
        _url(path), headers={"User-Agent": "cad-surface-mesher-public-data/1.0"}
    )
    url_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        sha256 = hashlib.sha256()
        source_digest = hashlib.sha256() if algorithm == "sha256" else hashlib.sha1()
        if algorithm == "git-sha1":
            source_digest.update(f"blob {expected_size}\0".encode("ascii"))
        size = 0
        status = None
        content_length = None
        content_range = None
        content_encoding = None
        try:
            with (
                urllib.request.urlopen(request, timeout=120) as response,
                temporary.open("wb") as out,
            ):
                status = getattr(response, "status", None)
                content_length = response.headers.get("Content-Length")
                content_range = response.headers.get("Content-Range")
                content_encoding = response.headers.get("Content-Encoding")
                if (
                    content_length
                    and content_length.isdigit()
                    and int(content_length) != expected_size
                ):
                    raise OSError(
                        f"{path}: HTTP {status} Content-Length={content_length} "
                        f"expected={expected_size} Content-Range={content_range!r} "
                        f"Content-Encoding={content_encoding!r}"
                    )
                while chunk := response.read(1024 * 1024):
                    out.write(chunk)
                    sha256.update(chunk)
                    source_digest.update(chunk)
                    size += len(chunk)
            actual_source_hash = source_digest.hexdigest()
            if size != expected_size or actual_source_hash != source_hash:
                raise OSError(
                    f"{path}: HTTP {status} received={size} expected={expected_size} "
                    f"source_hash={actual_source_hash[:12]} "
                    f"expected_hash={source_hash[:12]} "
                    f"Content-Length={content_length!r} "
                    f"Content-Range={content_range!r} "
                    f"Content-Encoding={content_encoding!r}"
                )
            temporary.replace(destination)
            break
        except (OSError, http.client.HTTPException, ValueError) as error:
            temporary.unlink(missing_ok=True)
            if attempt == MAX_RETRIES:
                url_error = error
                break
            print(
                f"retry {attempt}/{MAX_RETRIES - 1} downloading {path}: {error}",
                flush=True,
            )
            time.sleep(attempt)
    if url_error is not None:
        try:
            return _download_with_hf_xet(path, destination, entry)
        except (
            OSError,
            http.client.HTTPException,
            ValueError,
            RuntimeError,
            ImportError,
        ) as xet_error:
            raise OSError(
                f"{path}: urllib failed after {MAX_RETRIES} attempts ({url_error}); "
                f"Hugging Face Xet fallback failed ({xet_error})"
            ) from xet_error
    _write_sidecar(destination, cache)
    return {
        **cache,
        "size_bytes": size,
        "sha256": sha256.hexdigest(),
        "cache_hit": False,
    }


def _sample_ids(
    split_manifest: dict, counts: dict[str, int], seed: int
) -> dict[str, list[int]]:
    rng = random.Random(seed)
    selected = {}
    for split, count in counts.items():
        if count <= 0:
            raise ValueError(f"Requested {split} case count must be positive")
        source = split_manifest[f"full_{split}"]
        ids = sorted(
            (
                int(match.group(1))
                for name in source
                if (match := re.fullmatch(r"run_(\d+)", name))
            ),
        )
        if len(ids) != len(source) or count > len(ids):
            raise ValueError(f"Invalid full_{split} source split or requested count")
        if split == "train" and count and 1 in ids:
            chosen = [1, *rng.sample([value for value in ids if value != 1], count - 1)]
        else:
            chosen = rng.sample(ids, count)
        selected[split] = sorted(chosen)
    return selected


def _all_ids(split_manifest: dict) -> dict[str, list[int]]:
    """Select every case from the three official full split partitions."""
    selected = {}
    for split in ("train", "val", "test"):
        source = split_manifest[f"full_{split}"]
        ids = sorted(
            int(match.group(1))
            for name in source
            if (match := re.fullmatch(r"run_(\d+)", name))
        )
        if len(ids) != len(source) or len(ids) != len(set(ids)):
            raise ValueError(f"Invalid or duplicate IDs in full_{split} source split")
        selected[split] = ids
    if {split: len(ids) for split, ids in selected.items()} != {
        "train": 400,
        "val": 34,
        "test": 50,
    }:
        raise ValueError("Pinned official DrivAerML split sizes differ from 400/34/50")
    return selected


def _read_csv(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    if len(rows) != 1:
        raise ValueError(f"Expected one row in {path}, found {len(rows)}")
    return {key.strip().lower(): value.strip() for key, value in rows[0].items() if key}


def _stl_bounds(path: Path) -> tuple[list[float], list[float]]:
    """Read exact ASCII STL coordinates in bounded NumPy float64 blocks."""
    low = np.full(3, np.inf, dtype=np.float64)
    high = np.full(3, -np.inf, dtype=np.float64)
    count = 0

    def include_vertices(data: bytes) -> None:
        nonlocal count, low, high
        rows = _STL_VERTEX.findall(data)
        if len(rows) != data.count(b"vertex "):
            raise ValueError(f"Malformed STL vertex in {path}")
        values = np.fromstring(
            b" ".join(token for row in rows for token in row),
            sep=" ",
            dtype=np.float64,
        )
        if values.size != 3 * len(rows):
            raise ValueError(f"Malformed STL vertex in {path}")
        points = values.reshape(-1, 3)
        if not np.isfinite(points).all():
            raise ValueError(f"Nonfinite STL vertex in {path}")
        if len(points):
            low = np.minimum(low, points.min(axis=0))
            high = np.maximum(high, points.max(axis=0))
            count += len(points)

    with path.open("rb") as file:
        tail = b""
        while block := file.read(8 * 1024 * 1024):
            data = tail + block
            end = data.rfind(b"\n") + 1
            if end:
                include_vertices(data[:end])
                tail = data[end:]
            else:
                tail = data
        if tail:
            include_vertices(tail)
    if not count:
        raise ValueError(f"No ASCII STL vertices found in {path}")
    return low.tolist(), high.tolist()


def _case_record(output: Path, run_id: int, split: str) -> dict:
    run_dir = output / f"run_{run_id}"
    geometry_path = run_dir / f"drivaer_{run_id}.stl"
    force_path = run_dir / f"force_mom_{run_id}.csv"
    reference_path = run_dir / f"geo_ref_{run_id}.csv"
    label = _read_csv(force_path)
    reference = _read_csv(reference_path)
    lower, upper = _stl_bounds(geometry_path)
    body_length_m = upper[0] - lower[0]
    if not 3.0 <= body_length_m <= 7.0:
        raise ValueError(
            f"Unexpected DrivAer body length for run_{run_id}: {body_length_m}"
        )
    if "cd" not in label or "cl" not in label:
        raise ValueError(f"Missing Cd/Cl labels in {force_path}")

    return {
        "case_id": f"drivaerml-run-{run_id:03d}",
        "geometry_id": f"drivaerml-run-{run_id:03d}",
        "family_id": FAMILY_ID,
        "split": split,
        "geometry": {
            "path": f"run_{run_id}/drivaer_{run_id}.stl",
            "unit_scale_to_m": 1.0,
            "front_axis": "-x",
            "up_axis": "+z",
            "ground_clearance_m": 0.0,
        },
        "reynolds": SPEED_M_S * body_length_m / KINEMATIC_VISCOSITY_M2_S,
        "yaw_deg": 0.0,
        "cd": float(label["cd"]),
        "cl": float(label["cl"]),
        "cd_source": f"DrivAerML run_{run_id}/force_mom_{run_id}.csv at revision {REVISION}; variable per-geometry references",
        "assumptions": {
            "reynolds_reference_length": "raw mesh body length along x",
            "reynolds_reference_length_m": body_length_m,
            "source_force_reference_area_m2": float(reference["aref"]),
            "source_force_reference_length_m": float(reference["lref"]),
            "source_force_reference_area_kind": "per-geometry frontal area",
            "source_ground_convention": "wheel-contact minimum of the source STL is ground; ground_clearance_m is zero",
            "source_raw_min_z_m": lower[2],
        },
    }


def import_drivaerml(
    output: Path,
    counts: dict[str, int],
    seed: int,
    workers: int,
    all_cases: bool = False,
) -> Path:
    total = sum(counts.values())
    if not all_cases and (
        total < 3
        or total > MAX_PILOT_CASES
        or any(count <= 0 for count in counts.values())
    ):
        raise ValueError(f"Pilot size must be 3..{MAX_PILOT_CASES} cases")
    if not 1 <= workers <= 8:
        raise ValueError("workers must be in [1, 8]")
    output = output.expanduser().resolve()
    source_dir = output / "source"
    small_sources = (
        "README.md",
        "LICENSE.txt",
        "splits/manifest.json",
        "force_mom_all.csv",
    )
    source_files = {}
    for remote in small_sources:
        local = source_dir / remote
        directory = remote.rpartition("/")[0]
        metadata = _directory_metadata(directory)
        if remote not in metadata:
            raise FileNotFoundError(f"Pinned DrivAerML file is missing: {remote}")
        source_files[remote] = _download(remote, local, metadata[remote])

    official_splits = json.loads((source_dir / "splits/manifest.json").read_text())
    selected = (
        _all_ids(official_splits)
        if all_cases
        else _sample_ids(official_splits, counts, seed)
    )
    jobs = [(run_id, split) for split, ids in selected.items() for run_id in ids]

    print(
        "selected official cases: "
        + ", ".join(f"{split}={len(ids)}" for split, ids in selected.items())
        + f" ({len(jobs)} total)",
        flush=True,
    )

    case_metadata = {}
    metadata_failures = []

    def list_case(run_id: int) -> tuple[int, dict[str, dict]]:
        metadata = _directory_metadata(f"run_{run_id}")
        case_files = {}
        for filename in (
            f"drivaer_{run_id}.stl",
            f"force_mom_{run_id}.csv",
            f"geo_ref_{run_id}.csv",
        ):
            remote = f"run_{run_id}/{filename}"
            if remote not in metadata:
                raise FileNotFoundError(f"Pinned DrivAerML file is missing: {remote}")
            case_files[remote] = metadata[remote]
        return run_id, case_files

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(list_case, run_id): run_id for run_id, _ in jobs}
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            run_id = futures[future]
            try:
                _, case_metadata[run_id] = future.result()
            except (OSError, http.client.HTTPException, ValueError) as error:
                metadata_failures.append({"run_id": run_id, "error": str(error)})
            if index % 25 == 0 or index == len(futures):
                print(
                    f"source metadata listed for {index}/{len(futures)} cases",
                    flush=True,
                )

    expected_case_bytes = sum(
        int(entry["size"])
        for files in case_metadata.values()
        for entry in files.values()
    )
    expected_source_bytes = sum(
        int(item["size_bytes"]) for item in source_files.values()
    )
    expected_total_bytes = expected_case_bytes + expected_source_bytes
    expected_files = len(source_files) + 3 * len(jobs)
    completed_files = len(source_files)
    cached_files = sum(bool(item.get("cache_hit")) for item in source_files.values())
    print(
        f"expected {expected_total_bytes:,} bytes across {expected_files} files "
        f"({expected_case_bytes:,} case-file bytes; {len(case_metadata)}/{len(jobs)} "
        "case metadata sets listed)",
        flush=True,
    )
    failures = [
        {
            "run_id": item["run_id"],
            "stage": "source_metadata",
            "error": item["error"],
        }
        for item in metadata_failures
    ]

    def fetch_case(run_id: int) -> dict:
        nonlocal completed_files, cached_files
        root = output / f"run_{run_id}"
        files = {}
        metadata = case_metadata.get(run_id)
        if metadata is None:
            raise RuntimeError(f"No pinned source metadata available for run_{run_id}")
        for filename in (
            f"drivaer_{run_id}.stl",
            f"force_mom_{run_id}.csv",
            f"geo_ref_{run_id}.csv",
        ):
            remote = f"run_{run_id}/{filename}"
            downloaded_file = _download(remote, root / filename, metadata[remote])
            files[remote] = downloaded_file
            with _PROGRESS_LOCK:
                completed_files += 1
                cached_files += bool(downloaded_file["cache_hit"])
                size_mb = downloaded_file["size_bytes"] / (1024 * 1024)
                print(
                    f"[file {completed_files}/{expected_files}] {remote}: "
                    f"{size_mb:.1f} MiB verified "
                    f"({'cache' if downloaded_file['cache_hit'] else 'download'})",
                    flush=True,
                )
        return files

    downloaded = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(fetch_case, run_id): run_id
            for run_id, _ in jobs
            if run_id in case_metadata
        }
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            run_id = futures[future]
            try:
                downloaded[run_id] = future.result()
            except (OSError, http.client.HTTPException, ValueError) as error:
                failures.append(
                    {"run_id": run_id, "stage": "download", "error": str(error)}
                )
            if index % 25 == 0 or index == len(futures):
                print(
                    f"case files verified for {index}/{len(futures)} scheduled cases",
                    flush=True,
                )

    record_jobs = [
        (run_id, split)
        for split, ids in selected.items()
        for run_id in ids
        if run_id in downloaded
    ]
    records = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(workers, 4)) as pool:
        futures = {
            pool.submit(_case_record, output, run_id, split): run_id
            for run_id, split in record_jobs
        }
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            run_id = futures[future]
            try:
                records[run_id] = future.result()
            except (OSError, csv.Error, KeyError, ValueError) as error:
                failures.append(
                    {"run_id": run_id, "stage": "case_record", "error": str(error)}
                )
            if index % 25 == 0 or index == len(futures):
                print(
                    f"geometry records parsed for {index}/{len(futures)} cases",
                    flush=True,
                )

    cases = [
        records[run_id]
        for split, ids in selected.items()
        for run_id in ids
        if run_id in records
    ]

    if failures:
        failure_path = output / "full-import-failures.json"
        failure_path.write_text(json.dumps(failures, indent=2) + "\n")
        raise RuntimeError(
            f"DrivAerML import failed for {len(failures)} cases; details: {failure_path}"
        )

    training_manifest = {
        "schema_version": 1,
        "data_kind": "cfd",
        "split_policy": "geometry_disjoint",
        "split_provenance": {
            "source": "DrivAerML source/splits/manifest.json full partitions",
            "selection": (
                "All case IDs from each official source partition"
                if all_cases
                else "Deterministic bounded sample by seed from each official source partition"
            ),
            "geometry_disjoint": True,
            "family_disjoint": False,
            "shared_family_id": FAMILY_ID,
            "scope": "Within-family interpolation among morphed Ford OCDA DrivAer notchback designs; no unseen vehicle-family generalization claim",
        },
        "target_names": ["cd", "cl"],
        "conditions": {
            "speed": {"value_m_s": SPEED_M_S, "inflow_yaw_deg": 0.0},
            "air": {
                "temperature_k": 293.15,
                "kinematic_viscosity_m2_s": KINEMATIC_VISCOSITY_M2_S,
                "density_kg_m3": None,
                "density_status": "not reported in the source paper; not inferred",
            },
            "ground": "static floor with upstream slip segment and downstream no-slip segment",
            "wheels": "static, non-rotating",
            "turbulence": "hybrid RANS-LES, Spalart-Allmaras near-wall model with LES sigma-model in separated regions",
            "solver": "OpenFOAM v2212 with UpstreamCFD workflow modifications",
            "averaging": "time-average after automatic initial-transient removal; target drag CI +/- 1.5 counts, with 40-60 convective-unit cap",
            "reference_area": "per-geometry frontal Aref from geo_ref_i.csv; Cd and Cl from force_mom_i.csv",
            "force_axes": "source x is streamwise; Cd = Fx/(0.5*rho*U^2*Aref); source z is vertical; Cl = Fz/(0.5*rho*U^2*Aref); zero yaw",
            "reynolds_definition": "38.889 m/s times each raw STL body length along x divided by 1.507e-5 m^2/s; source paper separately reports baseline Re=7.19e6 using 2.786 m wheelbase",
        },
        "provenance": {
            "dataset": "DrivAerML: High-Fidelity Computational Fluid Dynamics Dataset for Road-Car External Aerodynamics",
            "dataset_id": DATASET,
            "revision": REVISION,
            "license": "CC BY-SA 4.0",
            "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
            "paper_url": "https://arxiv.org/abs/2408.11969",
            "source_url": f"{BASE_URL}",
            "geometry_family": "500 parametric morphs of one Ford OCDA DrivAer notchback family; family_id intentionally shared across all cases",
            "geometry_convention": {
                "source_axes": "x streamwise, y lateral, z vertical",
                "front_axis": "-x",
                "up_axis": "+z",
                "unit_scale_to_m": 1.0,
                "ground_z_convention": "raw tire-contact minimum z maps to the road; set per-case ground_clearance_m=0",
                "orientation_check": "run_1 source STL side projection confirms the nose is at raw -x and the tires touch the raw minimum-z line",
                "sample_preview": "run_1/sample-side-preview.png (local validation artifact)",
            },
            "split_manifest": "source/splits/manifest.json, official full_train/full_val/full_test partitions",
            **(
                {"official_full_selection": {"selected_run_ids": selected}}
                if all_cases
                else {
                    "pilot_sampling": {
                        "seed": seed,
                        "counts": counts,
                        "selected_run_ids": selected,
                    }
                }
            ),
            "source_reynolds": "Paper reports baseline Reynolds number on 2.786 m wheelbase; this manifest recomputes input Re on each mesh body length to match the learner's reference-length convention.",
            "source_files": source_files,
            "case_file_hashes": {
                f"run_{run_id}": files for run_id, files in sorted(downloaded.items())
            },
        },
        "cases": cases,
    }
    manifest_path = output / (
        "full-training_manifest.json" if all_cases else "training_manifest.json"
    )
    temporary_manifest = Path(f"{manifest_path}.part")
    temporary_manifest.write_text(
        json.dumps(training_manifest, indent=2, allow_nan=False) + "\n"
    )
    temporary_manifest.replace(manifest_path)
    print(
        f"complete: {len(cases)} cases, {cached_files}/{expected_files} files reused "
        f"from verified cache; manifest={manifest_path}",
        flush=True,
    )
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/predict-vehicle-cd/public-data/drivaerml"),
    )
    parser.add_argument("--train-count", type=int, default=40)
    parser.add_argument("--val-count", type=int, default=10)
    parser.add_argument("--test-count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--all",
        action="store_true",
        help="import all 484 cases in the official 400/34/50 geometry splits",
    )
    args = parser.parse_args()
    path = import_drivaerml(
        args.output_dir,
        {"train": args.train_count, "val": args.val_count, "test": args.test_count},
        args.seed,
        args.workers,
        all_cases=args.all,
    )
    print(path)


if __name__ == "__main__":
    main()
