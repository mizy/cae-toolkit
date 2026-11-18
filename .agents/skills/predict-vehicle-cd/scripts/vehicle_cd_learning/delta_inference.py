"""Replay a complete-geometry Rim pair through the selected paired-delta models."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .delta_experiment import batch, dataset_for, load, sha256, write_json
from .model import VIEWS
from .paired_delta import PairedDelta
from .prepare_drivaerrim import merge_depth, wheel_views
from .render import save_five_views
from .rim_resolution import render_case, shared_views


def predict(experiment, arm, reference, query):
    """@entry Predict query−reference from verified full-view inputs, without using their labels."""
    p, cases = load(experiment)
    ck = torch.load(
        Path(experiment) / arm / "model.pt", map_location="cpu", weights_only=True
    )
    if ck["protocol_sha256"] != sha256(Path(experiment) / "protocol.json"):
        raise ValueError("Checkpoint and input protocol differ")
    model = PairedDelta(**ck["model_config"]).eval()
    model.load_state_dict(ck["model_state"], strict=True)
    dataset = dataset_for(p, cases, p["arms"][arm]["resolution"])
    a, b = dataset.index[reference], dataset.index[query]
    views, parameters, _ = batch(dataset, [a, b], torch.device("cpu"))
    with torch.inference_mode():
        delta = model(
            {n: v[:1] for n, v in views.items()},
            {n: v[1:] for n, v in views.items()},
            parameters[:1],
        ).numpy()[0] * np.asarray(p["delta_std"])
    return {
        "reference": reference,
        "query": query,
        "delta_cd": float(delta[0]),
        "delta_cl": float(delta[1]),
        "resolution": p["arms"][arm]["resolution"],
        "pool_size": p["arms"][arm]["pool_size"],
        "scope": "Fixed-body, same-condition Rim paired-model experiment; no independent-vehicle or variable-Re guarantee.",
    }


def raw_replay(experiment, reference, query):
    """Rerasterize the predefined pair from unchanged full source geometry at both resolutions."""
    root = Path(experiment)
    p, cases = load(root)
    if not (root / "comparison.json").exists():
        raise ValueError(
            "Freeze/evaluate all four arms before this illustrative test replay"
        )
    output = root / "raw-replay"
    if output.exists():
        raise FileExistsError("Raw replay already exists")
    output.mkdir()
    raw = Path(p["inventory"]).parent
    inventory = json.loads(Path(p["inventory"]).read_text())
    entries = {r["path"]: r for r in inventory["files"]}
    started = time.perf_counter()
    body, origin, dimensions, source_hashes = shared_views(raw, entries)
    selected = {c["case_id"]: c for c in cases if c["case_id"] in (reference, query)}
    if len(selected) != 2:
        raise ValueError("Expected two distinct prepared geometries")
    views = {256: {}, 512: {}}
    audit = {}
    data512 = json.loads((Path(p["prepared512"]) / "dataset.json").read_text())
    records512 = {c["case_id"]: c for c in data512["cases"]}
    for identity in (reference, query):
        case = selected[identity]
        gid = identity.rsplit("-", 1)[-1]
        _, rendered, signature = wheel_views(gid, raw, origin, dimensions, entries)
        if signature != case["geometry_hash"]:
            raise ValueError("Raw wheel signature differs")
        merge_depth(rendered, body[256])
        file = output / f"{gid}-256.npz"
        save_five_views(file, rendered)
        current, evidence = render_case(
            case,
            raw,
            Path(p["prepared"]),
            output,
            entries,
            body,
            origin,
            dimensions,
            False,
        )
        sources = {256: file, 512: output / current["views"]}
        originals = {
            256: Path(p["prepared"]) / case["views"],
            512: Path(p["prepared512"]) / records512[identity]["views"],
        }
        audit[identity] = {}
        for size in (256, 512):
            with (
                np.load(sources[size], allow_pickle=False) as a,
                np.load(originals[size], allow_pickle=False) as b,
            ):
                difference = {n: float(abs(a[n] - b[n]).max()) for n in VIEWS}
                if max(difference.values()) > 1e-7:
                    raise ValueError("Raw input replay differs")
                views[size][identity] = {n: torch.from_numpy(a[n][None]) for n in VIEWS}
            audit[identity][str(size)] = difference
        source_hashes.update(evidence["source_hashes"])
    dataset = dataset_for(p, cases, 256)
    parameters = dataset.parameters[dataset.index[reference]][None]
    comparison = json.loads((root / "comparison.json").read_text())
    result = {}
    torch.set_num_threads(2)
    for arm, config in p["arms"].items():
        ck = torch.load(root / arm / "model.pt", map_location="cpu", weights_only=True)
        model = PairedDelta(**ck["model_config"]).eval()
        model.load_state_dict(ck["model_state"])
        with torch.inference_mode():
            delta = model(
                views[config["resolution"]][reference],
                views[config["resolution"]][query],
                parameters,
            ).numpy()[0] * np.asarray(p["delta_std"])
        expected = np.array(comparison["models"][arm]["predictions"][reference][query])
        gap = abs(delta - expected)
        if gap.max() > 2e-5:
            raise ValueError("Raw forward differs from cached feature evaluation")
        result[arm] = {"delta": delta.tolist(), "gap_vs_evaluation": gap.tolist()}
    write_json(
        output / "verification.json",
        {
            "source_hashes": source_hashes,
            "inputs": audit,
            "models": result,
            "reference": reference,
            "query": query,
            "true_delta": (
                np.array(selected[query]["targets"]) - selected[reference]["targets"]
            ).tolist(),
            "raw_reconstruction_and_prediction_seconds": time.perf_counter() - started,
            "scope": "Known predefined reference/query replay; complete body+mirrored wheels, no local crops or simplified meshes.",
        },
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    a = sub.add_parser("predict")
    a.add_argument("experiment", type=Path)
    a.add_argument("arm")
    a.add_argument("reference")
    a.add_argument("query")
    a = sub.add_parser("raw-replay")
    a.add_argument("experiment", type=Path)
    a.add_argument("reference")
    a.add_argument("query")
    a = p.parse_args()
    if a.action == "predict":
        print(json.dumps(predict(a.experiment, a.arm, a.reference, a.query), indent=2))
    else:
        raw_replay(a.experiment, a.reference, a.query)
