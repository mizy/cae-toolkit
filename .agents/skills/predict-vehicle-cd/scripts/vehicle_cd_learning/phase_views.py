"""Genuine complete-car camera-phase caches from unchanged DrivAerRim triangles."""

import argparse
import copy
import hashlib
import json
import resource
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyvista as pv

from .delta_experiment import ANCHORS, load, sha256, write_json
from .delta_sampling import shifted_views
from .prepare_drivaerrim import merge_depth
from .render import VIEW_AXES, _rasterize, save_five_views
from .rim_resolution import SHARED

SEED = 20261117
TRAIN_PHASES = ((0.25, 0.0), (-0.25, 0.0), (0.0, 0.25), (0.0, -0.25))
DIAGNOSTIC_PHASES = ((0.125, 0.0), (0.5, 0.0))
PHASES = TRAIN_PHASES + DIAGNOSTIC_PHASES
MAIN = TRAIN_PHASES[0]
TEST_QUERY = "drivaerrim__rim-0002"


def phase_id(phase):
    x, y = (
        f"{v:+g}".replace("+", "p").replace("-", "m").replace(".", "d") for v in phase
    )
    return f"x{x}y{y}"


def camera_views(mesh, origin, length, size, phase):
    """Positive phase moves sampling right/down; each view's depth stays fixed."""
    x, y = phase
    if y == 0:
        return shifted_views(mesh, origin, length, size, x)
    points = ((mesh.points - origin) / length).astype(np.float32)
    faces = np.asarray(mesh.faces).reshape(-1, 4)[:, 1:]
    views = {}
    for name, (horizontal, vertical, _, _) in VIEW_AXES.items():
        shifted = points.copy()
        shifted[:, horizontal] += np.float32(x * 1.3 / (size - 8))
        shifted[:, vertical] -= np.float32(y * 1.3 / (size - 8))
        views[name] = _rasterize(shifted, faces, view=name, size=size, window_l=1.3)
    return views


def selection(cases):
    """Select IDs before loading geometry, without reading target values."""
    splits = {
        s: sorted(c["case_id"] for c in cases if c["split"] == s)
        for s in ("train", "val", "test")
    }
    if [len(splits[s]) for s in splits] != [720, 92, 92]:
        raise ValueError("Expected unchanged official 720/92/92 splits")
    if not set(ANCHORS) <= set(splits["train"]) or TEST_QUERY not in splits["test"]:
        raise ValueError("Predeclared anchors/query have unexpected splits")

    def sample(values, count):
        return sorted(
            np.random.default_rng(SEED).choice(values, count, replace=False).tolist()
        )

    selected = {
        "train": sample([i for i in splits["train"] if i not in ANCHORS], 32),
        "val": sample(splits["val"], 12),
        "test": sorted(
            [TEST_QUERY] + sample([i for i in splits["test"] if i != TEST_QUERY], 23)
        ),
        "all_test": splits["test"],
        "anchors": list(ANCHORS),
    }
    items = []
    for split in ("train", "val"):
        for identity in selected[split]:
            for phase in TRAIN_PHASES:
                items.append(
                    {
                        "source_case_id": identity,
                        "split": split,
                        "phase_id": phase_id(phase),
                        "phase_pixels": list(phase),
                        "role": split,
                    }
                )
    for identity in selected["all_test"]:
        phases = (MAIN,) + (
            (DIAGNOSTIC_PHASES + ((0.0, 0.25),)) if identity in selected["test"] else ()
        )
        for phase in phases:
            items.append(
                {
                    "source_case_id": identity,
                    "split": "test",
                    "phase_id": phase_id(phase),
                    "phase_pixels": list(phase),
                    "role": "test_diagnostic",
                }
            )
    for identity in ANCHORS:
        for phase in PHASES:
            items.append(
                {
                    "source_case_id": identity,
                    "split": "train",
                    "phase_id": phase_id(phase),
                    "phase_pixels": list(phase),
                    "role": "anchor",
                }
            )
    return selected, items


def checked_mesh(raw, relative, entries, hashes):
    path = raw / relative
    actual = sha256(path)
    if actual != entries[relative]["sha256"]:
        raise ValueError(f"Raw source hash mismatch: {relative}")
    hashes[relative] = actual
    mesh = pv.read(path)
    if not mesh.is_all_triangles:
        raise ValueError(f"Raw source is not triangle-only: {relative}")
    return mesh


def shared_buffers(raw, entries, origin, length, phases):
    """One immutable shared body z-buffer per phase; never assemble a full mesh."""
    buffers = {p: None for p in phases}
    hashes = {}
    for name in SHARED:
        relative = f"Geometry/{name}/{name}.stl"
        mesh = checked_mesh(raw, relative, entries, hashes)
        for mirror in range(1 if name == "Car_Body" else 2):
            if mirror:
                mesh.points[:, 1] *= -1
            for phase in phases:
                current = camera_views(mesh, origin, length, 512, phase)
                buffers[phase] = (
                    current
                    if buffers[phase] is None
                    else merge_depth(buffers[phase], current)
                )
        del mesh
    for views in buffers.values():
        for view in views.values():
            view.depth.flags.writeable = False
            view.mask.flags.writeable = False
    return buffers, hashes


def cache_difference(views, path):
    with np.load(path, allow_pickle=False) as previous:
        difference = {
            name: float(np.max(np.abs(np.stack((v.depth, v.mask)) - previous[name])))
            for name, v in views.items()
        }
    if max(difference.values()) > 1e-7:
        raise ValueError(f"Raw triangle replay differs: {path}: {difference}")
    return difference


def render_case(
    case,
    items,
    raw,
    entries,
    body,
    origin,
    length,
    nominal_root,
    output,
    replay,
    control,
):
    started = time.monotonic()
    identity = case["case_id"]
    gid = identity.rsplit("-", 1)[-1]
    nominal = nominal_root / case["views"]
    if sha256(nominal) != case["views_sha256"]:
        raise ValueError(f"Nominal cache changed: {identity}")
    phases = list(dict.fromkeys(tuple(i["phase_pixels"]) for i in items))
    if replay:
        phases.append((0.0, 0.0))
    buffers = {p: None for p in phases}
    hashes, digest = {}, hashlib.sha256()
    for position in ("FL", "RL"):
        relative = f"Geometry/Wheel_{position}/Wheel_{position}_{gid}.stl"
        mesh = checked_mesh(raw, relative, entries, hashes)
        digest.update(hashes[relative].encode())
        if not np.isclose(mesh.bounds[4], origin[2], atol=1e-5):
            raise ValueError(f"Unexpected tyre contact height: {identity}")
        for mirror in (False, True):
            if mirror:
                mesh.points[:, 1] *= -1
            for phase in phases:
                current = camera_views(mesh, origin, length, 512, phase)
                buffers[phase] = (
                    current
                    if buffers[phase] is None
                    else merge_depth(buffers[phase], current)
                )
        del mesh
    if digest.hexdigest() != case["geometry_hash"]:
        raise ValueError(f"Source wheel signature mismatch: {identity}")
    for phase, views in buffers.items():
        merge_depth(views, body[phase])
    evidence = {
        "source_hashes": hashes,
        "geometry_signature": digest.hexdigest(),
        "phase0_replay_max_abs": None,
        "old_sampling_replay_max_abs": None,
    }
    if replay:
        evidence["phase0_replay_max_abs"] = cache_difference(
            buffers[(0.0, 0.0)], nominal
        )
    if identity == ANCHORS[0]:
        evidence["old_sampling_replay_max_abs"] = cache_difference(
            buffers[MAIN], control
        )
    records = []
    for item in items:
        record = copy.deepcopy(case)
        record.update(item)
        record["case_id"] = identity + "__phase-" + item["phase_id"]
        record["nominal_views"] = str(nominal.resolve())
        record["nominal_views_sha256"] = case["views_sha256"]
        record["views"] = f"views/{gid}-{item['phase_id']}.npz"
        save_five_views(output / record["views"], buffers[tuple(item["phase_pixels"])])
        record["views_sha256"] = sha256(output / record["views"])
        records.append(record)
    evidence["seconds"] = time.monotonic() - started
    evidence["rendered_phase_count"] = len(phases)
    return records, evidence


def prepare(experiment, output, workers=2, smoke=None):
    """@entry Freeze selection first, verify raw geometry, render and audit phases."""
    experiment, output = Path(experiment), Path(output)
    if output.exists():
        raise FileExistsError("Use a fresh output directory")
    if workers not in (1, 2) or (smoke is not None and not 1 <= smoke <= 32):
        raise ValueError("Use one/two workers and smoke training count in 1..32")
    started = time.monotonic()
    protocol, cases = load(experiment)
    nominal_root = Path(protocol["prepared512"])
    source_path = nominal_root / "dataset.json"
    source = json.loads(source_path.read_text())
    frame = source["resolution_preparation"]
    lock = json.loads((experiment / "data512-lock.json").read_text())
    if not frame["train_ready"] or sha256(source_path) != lock["dataset_sha256"]:
        raise ValueError("Nominal512 manifest is incomplete or changed")
    nominal_cases = {c["case_id"]: c for c in source["cases"]}
    if set(nominal_cases) != set(protocol["case_ids"]):
        raise ValueError("Nominal512 membership changed")
    for case in cases:
        if any(
            case[k] != nominal_cases[case["case_id"]][k]
            for k in ("split", "targets", "parameters", "geometry_hash")
        ):
            raise ValueError("Nominal512 labels, splits or geometry changed")
    inventory_path = Path(protocol["inventory"])
    inventory = json.loads(inventory_path.read_text())
    if not inventory["complete"] or sha256(inventory_path) != frame["inventory_sha256"]:
        raise ValueError("Raw inventory incomplete or changed")
    entries = {f["path"]: f for f in inventory["files"]}
    selected, full_items = selection(cases)
    replay_id = selected["train"][0]
    items = full_items
    if smoke is not None:
        allowed = set(selected["train"][:smoke])
        items = [
            i
            for i in full_items
            if i["source_case_id"] in allowed
            or (i["source_case_id"] == ANCHORS[0] and tuple(i["phase_pixels"]) == MAIN)
        ]
    control = experiment / "sampling-control/0001-512.npz"
    control_plan = experiment / "sampling-control/plan.json"
    previous = json.loads(control_plan.read_text())
    if (
        previous["reference"] != ANCHORS[0]
        or previous["horizontal_camera_phase_pixels"] != MAIN[0]
    ):
        raise ValueError("Unexpected historical sampling-control phase")
    source_files = [
        experiment / "protocol.json",
        source_path,
        inventory_path,
        experiment / "data512-lock.json",
        control,
        control_plan,
    ]
    code_files = [
        Path(__file__),
        Path(__file__).with_name("delta_sampling.py"),
        Path(__file__).with_name("render.py"),
        Path(__file__).with_name("prepare_drivaerrim.py"),
    ]
    plan = {
        "seed": SEED,
        "selection": selected,
        "items": items,
        "full_item_count": len(full_items),
        "smoke_training_count": smoke,
        "phase0_replay_case_id": replay_id,
        "train_phases": [list(p) for p in TRAIN_PHASES],
        "diagnostic_phases": [list(p) for p in DIAGNOSTIC_PHASES],
        "source_sha256": {str(p.resolve()): sha256(p) for p in source_files},
        "code_sha256": {str(p.resolve()): sha256(p) for p in code_files},
        "source_frame": {k: frame[k] for k in ("origin_m", "dimensions_m")},
        "phase_semantics": "pixels right/down in every full 512 view; normalized h += x*1.3/504, v -= y*1.3/504; depth unchanged; labels unchanged",
        "selection_scope": "Seeded IDs only; no target-based selection; .125/.5px test phases are held-out magnitude diagnostics",
        "render_config": {
            "size": 512,
            "window_l": 1.3,
            "margin": 4,
            "geometry": "raw full triangles; no crop, warp, resize or decimation",
        },
    }
    output.mkdir(parents=True)
    (output / "views").mkdir()
    write_json(output / "plan.json", plan)
    write_json(
        output / "manifest.json",
        {
            "stage_complete": False,
            "train_ready": False,
            "plan_sha256": sha256(output / "plan.json"),
        },
    )
    phases = sorted({tuple(i["phase_pixels"]) for i in items} | {(0.0, 0.0)})
    origin, length = np.asarray(frame["origin_m"]), frame["dimensions_m"][0]
    shared_started = time.monotonic()
    body, shared_hashes = shared_buffers(
        inventory_path.parent, entries, origin, length, phases
    )
    if shared_hashes != frame["shared_source_hashes"]:
        raise ValueError("Shared geometry differs from nominal512 preparation")
    shared_seconds = time.monotonic() - shared_started
    print(
        json.dumps({"shared_seconds": shared_seconds, "phases": len(phases)}),
        flush=True,
    )
    grouped = {}
    for item in items:
        grouped.setdefault(item["source_case_id"], []).append(item)
    records, audit = {}, {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = {
            pool.submit(
                render_case,
                nominal_cases[i],
                group,
                inventory_path.parent,
                entries,
                body,
                origin,
                length,
                nominal_root,
                output,
                i == replay_id,
                control,
            ): i
            for i, group in grouped.items()
        }
        for job in as_completed(jobs):
            identity = jobs[job]
            current, evidence = job.result()
            records[identity], audit[identity] = current, evidence
            print(
                json.dumps(
                    {
                        "prepared_geometries": len(records),
                        "total_geometries": len(grouped),
                        "case_id": identity,
                        "seconds": evidence["seconds"],
                    }
                ),
                flush=True,
            )
    for path, expected in {**plan["source_sha256"], **plan["code_sha256"]}.items():
        if sha256(path) != expected:
            raise ValueError(f"Source changed during preparation: {path}")
    elapsed = time.monotonic() - started
    wheel_seconds = sum(a["seconds"] for a in audit.values())
    phase_count = sum(a["rendered_phase_count"] for a in audit.values())
    manifest = {
        "stage_complete": True,
        "train_ready": smoke is None,
        "smoke_training_count": smoke,
        "plan_sha256": sha256(output / "plan.json"),
        "source_sha256": plan["source_sha256"],
        "code_sha256": plan["code_sha256"],
        "source_frame": plan["source_frame"],
        "shared_source_hashes": shared_hashes,
        "case_audit": audit,
        "item_count": len(items),
        "geometry_count": len(grouped),
        "shared_seconds": shared_seconds,
        "elapsed_seconds": elapsed,
        "workers": workers,
        "estimated_full_seconds": shared_seconds * 7 / len(phases)
        + wheel_seconds / phase_count * (len(full_items) + 1) / workers,
        "peak_rss_platform_units": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "peak_rss_note": "macOS bytes; Linux KiB; shared immutable body buffers, at most two wheel workers",
    }
    dataset = {
        "cases": [record for identity in grouped for record in records[identity]],
        "parameter_names": source["parameter_names"],
        "phase_preparation": manifest,
        "render_config": plan["render_config"],
    }
    write_json(output / "dataset.json", dataset)
    write_json(
        output / "manifest.json",
        {**manifest, "dataset_sha256": sha256(output / "dataset.json")},
    )
    print(
        json.dumps(
            {
                k: manifest[k]
                for k in (
                    "item_count",
                    "elapsed_seconds",
                    "estimated_full_seconds",
                    "train_ready",
                )
            }
        ),
        flush=True,
    )
    return dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare"])
    parser.add_argument("experiment", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--smoke", type=int)
    args = parser.parse_args()
    prepare(args.experiment, args.output, args.workers, args.smoke)


if __name__ == "__main__":
    main()
