"""Opt-in geometry-to-coefficients replay for the retained Rim, COCOANet and HiLift contracts."""

import argparse
import json
import time
from pathlib import Path

import torch

from .delta_experiment import sha256
from .dimensionless import normalize_parameters
from .joint_sampling import load, load_checkpoint
from .model import VIEWS
from .rim_resolution import SHARED
from .subpixel_geometry import aircraft_triangles, rim_triangles
from .subpixel_sampling import registered_views


def predict(experiment, query, reference=None, phases=None):
    """@entry Read the source meshes and their audited physical frame; preserve independent sensor phases."""
    started = time.perf_counter()
    p, cases = load(Path(experiment))
    by_id = {c["case_id"]: c for c in cases}
    retained = json.loads((Path(experiment) / "retention.json").read_text())
    if sha256(retained["checkpoint"]) != retained["sha256"]:
        raise ValueError("Retained checkpoint changed")
    model, absolute = load_checkpoint(retained["checkpoint"])
    model.eval()
    absolute.eval()
    torch.set_num_threads(2)
    zero = {n: (0.0, 0.0) for n in VIEWS}
    phases = phases or {}
    endpoints = {"query": query}
    if reference is not None:
        if (
            any(by_id[c]["source"] != "drivaerrim" for c in (reference, query))
            or by_id[reference]["parameters"] != by_id[query]["parameters"]
        ):
            raise ValueError(
                "Direct paired inference requires the matched Rim source contract"
            )
        endpoints["reference"] = reference
    if set(phases) - set(endpoints):
        raise ValueError("Unknown phase-map endpoint")
    if any(
        by_id[c]["source"] not in ("drivaerrim", "cocoanet", "hilift")
        for c in endpoints.values()
    ):
        raise ValueError(
            "This replay accepts the three validated native-render source contracts"
        )
    shared = []
    raw = None
    entries = None
    frame = None
    if by_id[query]["source"] == "drivaerrim":
        parent = json.loads(
            (Path(p["parent_experiment"]) / "protocol.json").read_text()
        )
        inventory = Path(parent["inventory"])
        raw = inventory.parent
        if sha256(inventory) != parent["input_sha256"][str(inventory)]:
            raise ValueError("Source inventory changed")
        entries = {f["path"]: f for f in json.loads(inventory.read_text())["files"]}
        frame = json.loads((Path(parent["prepared512"]) / "dataset.json").read_text())[
            "resolution_preparation"
        ]
        shared = rim_triangles(
            raw,
            entries,
            [(f"Geometry/{n}/{n}.stl", n != "Car_Body") for n in SHARED],
            frame,
        )
    coeff = {}
    pair_features = {}
    stats = p["normalization"]
    with torch.inference_mode():
        for role, cid in endpoints.items():
            c = by_id[cid]
            offsets = phases.get(role, zero)
            if c["source"] == "drivaerrim":
                gid = cid.rsplit("-", 1)[-1]
                wheels = [
                    (f"Geometry/Wheel_{pos}/Wheel_{pos}_{gid}.stl", True)
                    for pos in ("FL", "RL")
                ]
                parts = shared + rim_triangles(raw, entries, wheels, frame)
            else:
                parts = aircraft_triangles(c)
            for size in [256, 512] if reference else [256]:
                arrays = registered_views(parts, size, offsets)
                inputs = {n: torch.from_numpy(v[None]) for n, v in arrays.items()}
                encoded = model.encode(inputs)
                if size == 512:
                    pair_features[role] = encoded
                else:
                    x = torch.from_numpy(
                        normalize_parameters(c["parameters"], stats)[None]
                    )
                    value = (
                        absolute(encoded, x).numpy()[0] * stats["target_std"]
                        + stats["target_mean"]
                    )
                    coeff[role] = {
                        "case_id": cid,
                        "Cd": float(value[0]),
                        "Cl": float(value[1]),
                    }
        result = {
            "coefficients": coeff,
            "checkpoint_sha256": retained["sha256"],
            "source_phase_pixels": {r: phases.get(r, zero) for r in endpoints},
        }
        if reference is not None:
            x = torch.from_numpy(
                normalize_parameters(by_id[reference]["parameters"], p)[None]
            )
            delta = (
                model.predict_pair(
                    pair_features["reference"], pair_features["query"], x
                ).numpy()[0]
                * p["delta_std"]
            )
            result["direct_paired_delta"] = {
                "Cd": float(delta[0]),
                "Cl": float(delta[1]),
            }
    result["seconds_including_geometry_and_model_load"] = time.perf_counter() - started
    return result


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("experiment", type=Path)
    cli.add_argument("query")
    cli.add_argument("--reference")
    cli.add_argument(
        "--phases",
        type=Path,
        help="JSON mapping query/reference to the five (dx,dy) native-pixel offsets",
    )
    a = cli.parse_args()
    print(
        json.dumps(
            predict(
                a.experiment,
                a.query,
                a.reference,
                json.loads(a.phases.read_text()) if a.phases else None,
            ),
            indent=2,
        )
    )
