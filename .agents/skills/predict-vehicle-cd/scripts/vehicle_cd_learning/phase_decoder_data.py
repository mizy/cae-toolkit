"""Prepare the retained decoder experiment's true-rendered Rim phase inputs."""

import argparse
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from .delta_experiment import sha256, write_json
from .phase_views import MAIN, phase_id, render_case, shared_buffers
from .sampling_experiment import load as load_sampling

OFF_GRID = ((0.173, -0.317), (-0.362, 0.119))


def prepare(design_path, sampling_root, output):
    """@entry Reuse verified phase images, render missing geometry/phase pairs, checkpoint per mesh."""
    design = json.loads(design_path.read_text())
    sp, cases = load_sampling(sampling_root)
    parent_root = Path(sp["parent_experiment"])
    parent = json.loads((parent_root / "protocol.json").read_text())
    old_root = Path(sp["phase_root"])
    old_file = old_root / "dataset.json"
    assert (
        sha256(old_file)
        == json.loads((sampling_root / "data-lock.json").read_text())["dataset_sha256"]
    )
    old = json.loads(old_file.read_text())["cases"]
    nominal_root = Path(parent["prepared512"])
    nominal_file = nominal_root / "dataset.json"
    assert (
        sha256(nominal_file)
        == json.loads((parent_root / "data512-lock.json").read_text())["dataset_sha256"]
    )
    nominal = json.loads(nominal_file.read_text())
    by_id = {c["case_id"]: c for c in nominal["cases"]}
    ids = {
        split: sorted(c["case_id"] for c in cases if c["split"] == split)
        for split in ("train", "val", "test")
    }
    train = list(map(tuple, design["train_phase_pixels"]))
    val = train + list(map(tuple, design["additional_validation_phase_pixels"]))
    test = sorted(
        np.random.default_rng(42).choice(ids["test"], 12, replace=False).tolist()
    )
    groups = {cid: train for cid in design["train_phase_case_ids"]}
    groups.update({cid: val for cid in design["validation_phase_case_ids"]})
    groups.update({cid: list(OFF_GRID) for cid in test})
    groups.update({cid: val + list(OFF_GRID) for cid in sp["anchors"]})
    if not (
        set(design["train_phase_case_ids"]) <= set(ids["train"]) - set(sp["anchors"])
        and set(design["validation_phase_case_ids"]) <= set(ids["val"])
        and len(groups) == 64 + 24 + 12 + 3
    ):
        raise ValueError(
            "Expanded train, validation, test or anchor membership mismatch"
        )
    inventory = Path(parent["inventory"])
    assert sha256(inventory) == parent["input_sha256"][str(inventory)]
    old_by_key = {(c["source_case_id"], c["phase_id"]): c for c in old}
    output.mkdir(parents=True, exist_ok=True)
    plan = {
        "design": str(design_path.resolve()),
        "design_sha256": sha256(design_path),
        "sampling_root": str(sampling_root.resolve()),
        "nominal_root": str(nominal_root),
        "source_sha256": {
            str(f.resolve()): sha256(f)
            for f in (Path(__file__), old_file, inventory, nominal_file)
        },
        "groups": {i: [list(ph) for ph in phases] for i, phases in groups.items()},
        "off_grid_test_ids": test,
        "off_grid_phases": [list(x) for x in OFF_GRID],
    }
    if (output / "plan.json").exists() and json.loads(
        (output / "plan.json").read_text()
    ) != plan:
        raise ValueError("Partial output belongs to a different phase plan")
    write_json(output / "plan.json", plan)
    (output / "records").mkdir(exist_ok=True)
    frame = nominal["resolution_preparation"]
    origin = np.array(frame["origin_m"])
    length = frame["dimensions_m"][0]
    entries = {f["path"]: f for f in json.loads(inventory.read_text())["files"]}
    rows = {}
    pending = {}
    counts = Counter()
    replay = {}
    started = time.monotonic()
    for cid, phases in groups.items():
        record_file = output / "records" / f"{cid}.json"
        rows[cid] = {}
        if record_file.exists():
            saved = json.loads(record_file.read_text())
            rows[cid] = saved["views"]
            replay[cid] = saved["replay"]
            if set(rows[cid]) != {phase_id(ph) for ph in phases}:
                raise ValueError("Saved phase list differs")
            for v in rows[cid].values():
                if sha256(v["path"]) != v["sha256"]:
                    raise ValueError("Saved raw view content changed")
            counts["resumed"] += len(phases)
            continue
        need = []
        for ph in phases:
            name = phase_id(ph)
            source = old_by_key.get((cid, name))
            if source is None:
                need.append(ph)
                continue
            path = old_root / source["views"]
            if sha256(path) != source["views_sha256"] or any(
                source[k] != by_id[cid][k]
                for k in ("split", "targets", "geometry_hash", "parameters")
            ):
                raise ValueError("Reused phase image or conditions changed")
            rows[cid][name] = {
                "path": str(path.resolve()),
                "sha256": source["views_sha256"],
            }
            counts["reused"] += 1
        if need:
            # The existing anchor replay verifies its original +0.25 example during rendering.
            if cid == sp["anchors"][0] and MAIN not in need:
                need.append(MAIN)
            pending[cid] = [
                {
                    "source_case_id": cid,
                    "phase_id": phase_id(ph),
                    "phase_pixels": list(ph),
                }
                for ph in need
            ]
    if pending:
        phases = sorted(
            {
                tuple(item["phase_pixels"])
                for items in pending.values()
                for item in items
            }
            | {(0.0, 0.0)}
        )
        body, hashes = shared_buffers(inventory.parent, entries, origin, length, phases)
        if hashes != frame["shared_source_hashes"]:
            raise ValueError("Shared car components changed")
        print(
            "Shared components ready",
            len(phases),
            round(time.monotonic() - started, 1),
            flush=True,
        )
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {
                pool.submit(
                    render_case,
                    by_id[cid],
                    items,
                    inventory.parent,
                    entries,
                    body,
                    origin,
                    length,
                    nominal_root,
                    output,
                    cid
                    in (
                        design["train_phase_case_ids"][0],
                        design["validation_phase_case_ids"][0],
                        test[0],
                    ),
                    parent_root / "sampling-control/0001-512.npz",
                ): cid
                for cid, items in pending.items()
            }
            for job in as_completed(futures):
                cid = futures[job]
                rendered, evidence = job.result()
                replay[cid] = evidence["phase0_replay_max_abs"]
                for c in rendered:
                    name = c["phase_id"]
                    path = output / c["views"]
                    if name not in rows[cid]:
                        counts["rendered"] += 1
                    rows[cid][name] = {
                        "path": str(path.resolve()),
                        "sha256": c["views_sha256"],
                    }
                write_json(
                    output / "records" / f"{cid}.json",
                    {"views": rows[cid], "replay": replay[cid]},
                )
                print(
                    cid,
                    len(rows[cid]),
                    round(time.monotonic() - started, 1),
                    flush=True,
                )
    if any(
        set(rows[i]) != {phase_id(x) for x in phases} for i, phases in groups.items()
    ):
        raise ValueError("Incomplete phase coverage")
    write_json(
        output / "manifest.json",
        {
            "views": rows,
            "counts": dict(counts),
            "phase0_replays": {i: v for i, v in replay.items() if v is not None},
            "plan_sha256": sha256(output / "plan.json"),
            "elapsed_s": time.monotonic() - started,
        },
    )


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for n in ("design", "sampling", "output"):
        cli.add_argument(n, type=Path)
    a = cli.parse_args()
    prepare(a.design, a.sampling, a.output)
