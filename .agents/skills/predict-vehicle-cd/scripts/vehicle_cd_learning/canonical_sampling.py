"""Canonical geometry-to-view boundary for paired inference with declared sampling provenance."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyvista as pv
import torch

from .delta_experiment import sha256, write_json
from .model import VIEWS
from .prepare_drivaerrim import merge_depth
from .render import save_five_views
from .rim_resolution import SHARED, render_component
from .sampling_experiment import SamplingCases
from .sampling_inference import load_selected


def canonical_views(
    parts, frame, cache_root, source_phase=(0.0, 0.0), component_cache=None, force=False
):
    """@entry Rebuild a canonical full view from geometry, or reuse its verified exact cache.

    source_phase records the upstream representation only. Source triangles, not
    a shifted depth image or a coefficient label, determine the canonical pixels.
    """
    phase = np.asarray(source_phase, float)
    if phase.shape != (2,) or not np.isfinite(phase).all() or not parts:
        raise ValueError(
            "Source geometry and finite two-axis sampling provenance required"
        )
    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    geometry = [
        {"sha256": sha256(part["path"]), "mirror_y": bool(part.get("mirror_y", False))}
        for part in parts
    ]
    camera = {
        "resolution": int(frame.get("resolution", 512)),
        "window_l": 1.3,
        "margin": 4,
        "origin_m": list(frame["origin_m"]),
        "length_m": float(frame["length_m"]),
        "phase_pixels": [0.0, 0.0],
        "axes": "metres,nose-X,up+Z; input parts share coordinates",
    }
    if (
        camera["length_m"] <= 0
        or not np.isfinite(camera["length_m"])
        or camera["resolution"] < 16
        or np.asarray(camera["origin_m"]).shape != (3,)
        or not np.isfinite(camera["origin_m"]).all()
    ):
        raise ValueError("Invalid shared canonical frame")
    renderer = {
        str(f.resolve()): sha256(f)
        for f in (
            Path(__file__),
            Path(__file__).with_name("render.py"),
            Path(__file__).with_name("rim_resolution.py"),
            Path(__file__).with_name("prepare_drivaerrim.py"),
        )
    }
    identity = {"geometry": geometry, "camera": camera, "renderer_sha256": renderer}
    key = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    view_path, meta_path = cache_root / f"{key}.npz", cache_root / f"{key}.json"
    reused = False
    if view_path.exists() and meta_path.exists() and not force:
        meta = json.loads(meta_path.read_text())
        if meta["identity"] != identity or sha256(view_path) != meta["views_sha256"]:
            raise ValueError("Canonical cache content/provenance mismatch")
        reused = True
    else:
        parts_cache = {} if component_cache is None else component_cache
        result = None
        for part, signature in zip(parts, geometry):
            part_key = json.dumps(
                {"part": signature, "camera": camera, "renderer": renderer},
                sort_keys=True,
            )
            if part_key not in parts_cache:
                mesh = pv.read(part["path"])
                if (
                    not mesh.is_all_triangles
                    or not mesh.n_points
                    or not np.isfinite(mesh.points).all()
                ):
                    raise ValueError("Finite triangle source required")
                if signature["mirror_y"]:
                    mesh.points[:, 1] *= -1
                parts_cache[part_key] = render_component(
                    mesh,
                    np.asarray(camera["origin_m"]),
                    camera["length_m"],
                    camera["resolution"],
                )
            if result is None:
                result = {
                    n: type(v)(
                        mask=v.mask.copy(), depth=v.depth.copy(), normal=v.normal.copy()
                    )
                    for n, v in parts_cache[part_key].items()
                }
            else:
                merge_depth(result, parts_cache[part_key])
        if any(
            sha256(part["path"]) != record["sha256"]
            for part, record in zip(parts, geometry)
        ):
            raise ValueError("Geometry changed during canonical rendering")
        save_five_views(view_path, result)
        meta = {
            "identity": identity,
            "views_sha256": sha256(view_path),
            "geometry_labels_used": False,
        }
        write_json(meta_path, meta)
    with np.load(view_path, allow_pickle=False) as source:
        arrays = {name: source[name].copy() for name in VIEWS}
    return arrays, {
        "key": key,
        "canonical_views": str(view_path.resolve()),
        "source_phase_pixels": phase.tolist(),
        "canonical_phase_pixels": [0.0, 0.0],
        "geometry": geometry,
        "reused_canonical_cache": reused,
        "method": "Geometry re-render or content-and-frame-verified canonical cache; no image inverse warp",
    }


def rim_parts(raw, identity):
    gid = identity.rsplit("-", 1)[-1]
    parts = []
    for name in SHARED:
        path = str((raw / f"Geometry/{name}/{name}.stl").resolve())
        parts.append({"path": path, "mirror_y": False})
        if name != "Car_Body":
            parts.append({"path": path, "mirror_y": True})
    for pos in ["FL", "RL"]:
        path = str((raw / f"Geometry/Wheel_{pos}/Wheel_{pos}_{gid}.stl").resolve())
        parts.extend(
            [{"path": path, "mirror_y": False}, {"path": path, "mirror_y": True}]
        )
    return parts


def predict(
    experiment,
    reference,
    query,
    reference_phase=(0.0, 0.0),
    query_phase=(0.0, 0.0),
    force=False,
):
    """@entry Same geometry and shared frame always produce the same model inputs."""
    root = Path(experiment)
    torch.set_num_threads(2)
    model, p, cases, selection = load_selected(root)
    parent = json.loads((Path(p["parent_experiment"]) / "protocol.json").read_text())
    source = json.loads((Path(parent["prepared512"]) / "dataset.json").read_text())[
        "resolution_preparation"
    ]
    frame = {
        "origin_m": source["origin_m"],
        "length_m": source["dimensions_m"][0],
        "resolution": 512,
    }
    raw = Path(parent["inventory"]).parent
    data = SamplingCases(p, cases, with_phases=False)
    a, b = data.index[reference], data.index[query]
    if cases[a]["parameters"] != cases[b]["parameters"]:
        raise ValueError("Paired physical/numeric conditions differ")
    shared = {}
    av, am = canonical_views(
        rim_parts(raw, reference),
        frame,
        root / "canonical-inputs",
        reference_phase,
        shared,
        force,
    )
    bv, bm = canonical_views(
        rim_parts(raw, query),
        frame,
        root / "canonical-inputs",
        query_phase,
        shared,
        force,
    )
    with torch.inference_mode():
        pred = (
            model(
                {n: torch.from_numpy(v[None]) for n, v in av.items()},
                {n: torch.from_numpy(v[None]) for n, v in bv.items()},
                data.parameters[a : a + 1],
            ).numpy()[0]
            * p["delta_std"]
        )
    return {
        "reference": reference,
        "query": query,
        "delta_cd": float(pred[0]),
        "delta_cl": float(pred[1]),
        "canonicalization": {"reference": am, "query": bm},
        "model_raw_phase_invariance_validated": selection["validation_passed"],
        "scope": "Sampling-contract repair from original geometry. This does not establish invariance of the neural model to arbitrary shifted raster inputs.",
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("experiment", type=Path)
    p.add_argument("reference")
    p.add_argument("query")
    p.add_argument("--reference-phase", nargs=2, type=float, default=(0.0, 0.0))
    p.add_argument("--query-phase", nargs=2, type=float, default=(0.0, 0.0))
    p.add_argument("--force-render", action="store_true")
    a = p.parse_args()
    print(
        json.dumps(
            predict(
                a.experiment,
                a.reference,
                a.query,
                a.reference_phase,
                a.query_phase,
                a.force_render,
            ),
            indent=2,
        )
    )
