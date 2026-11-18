"""Download pinned public geometry, coefficients and conditions, excluding fields."""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from http.client import IncompleteRead
from pathlib import Path
from urllib.error import URLError

from huggingface_hub import hf_hub_download
from requests import RequestException
from vehicle_cd_learning.collect_bluff_bodies import _download


def select_files(source, name):
    files = source["siblings"]
    if source.get("gated"):
        raise ValueError("Gated source: list for application; do not download")
    if name == "drivaerrim":
        return [
            f
            for f in files
            if f["rfilename"].startswith(("Geometry/", "Dataset_Splits/"))
            or f["rfilename"]
            in ("NumData/Force_Coefficients_Combined.csv", "README.md")
        ]
    if name == "superwing":
        allowed = {
            "origingeom.npy",
            "index.npy",
            "configs.dat",
            "running_info.jsonl",
            "training_samples_index.txt",
            "README.md",
        }
        return [f for f in files if f["rfilename"] in allowed]
    # The ten STL copies for each HiLift geometry have identical source SHA-256.
    # Fetch one copy and retain all source paths in the pinned repository listing.
    selected, hashes = [], set()
    for f in files:
        p = f["rfilename"]
        if p in (
            "README.md",
            "force_mom_all.csv",
            "geo_values_all.csv",
            "force_mom_surface_overrides_v1.json",
        ) or p.startswith("splits/"):
            selected.append(f)
        elif p.endswith(".stl") and f["lfs"]["sha256"] not in hashes:
            hashes.add(f["lfs"]["sha256"])
            selected.append(f)
        elif "/ref_values_" in p and "_AoA_4/" in p:
            selected.append(f)
    return selected


def fetch(source, entry, root):
    relative = entry["rfilename"]
    expected = {"bytes": entry["size"]}
    if "lfs" in entry:
        expected["sha256"] = entry["lfs"]["sha256"]
        if not (root / relative).exists():
            hf_hub_download(
                source["id"],
                relative,
                repo_type="dataset",
                revision=source["sha"],
                local_dir=root,
            )
    else:
        expected["git_blob_sha1"] = entry["blobId"]
    url = f"https://huggingface.co/datasets/{source['id']}/resolve/{source['sha']}/{relative}"
    actual = _download(url, root / relative, expected)
    return {"path": relative, **actual}


def main():
    """@entry Select a source-specific allowlist and verify every downloaded file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", choices=("drivaerrim", "hilift", "superwing"))
    parser.add_argument("--catalogue", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    source = json.loads(args.catalogue.read_text())
    selected = select_files(source, args.source)
    if args.metadata_only:
        selected = [
            f
            for f in selected
            if not f["rfilename"].endswith((".stl", "origingeom.npy"))
        ]
    root = args.output_dir
    root.mkdir(parents=True, exist_ok=True)
    plan = {
        "repo": source["id"],
        "revision": source["sha"],
        "metadata_only": args.metadata_only,
        "files": selected,
        "bytes": sum(f["size"] for f in selected),
    }
    suffix = "metadata" if args.metadata_only else "geometry"
    (root / f"download-plan-{suffix}.json").write_text(
        json.dumps(plan, indent=2) + "\n"
    )
    print(
        json.dumps(
            {"repo": source["id"], "files": len(selected), "bytes": plan["bytes"]}
        ),
        flush=True,
    )
    verified, failed = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(fetch, source, f, root): f for f in selected}
        with (root / f"download-{suffix}.jsonl").open("a") as log:
            for future in as_completed(pending):
                entry = pending[future]
                try:
                    result = future.result()
                    verified.append(result)
                except (
                    OSError,
                    ValueError,
                    RuntimeError,
                    URLError,
                    IncompleteRead,
                    RequestException,
                ) as error:
                    result = {"path": entry["rfilename"], "error": str(error)}
                    failed.append(result)
                log.write(json.dumps(result) + "\n")
                log.flush()
                if len(verified) % 20 == 0 or "error" in result:
                    print(
                        json.dumps(
                            {
                                "verified": len(verified),
                                "failed": len(failed),
                                "total": len(selected),
                                "last": result["path"],
                            }
                        ),
                        flush=True,
                    )
    report = {
        "repo": source["id"],
        "revision": source["sha"],
        "files": verified,
        "failures": failed,
        "complete": not failed,
    }
    (root / f"inventory-{suffix}.json").write_text(json.dumps(report, indent=2) + "\n")
    if failed:
        raise RuntimeError(f"{len(failed)} files failed integrity/download checks")


if __name__ == "__main__":
    main()
