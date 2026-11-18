"""Post-training camera-sampling diagnostic on the predeclared complete-car pair."""

import argparse
import json
from pathlib import Path

import numpy as np
import pyvista as pv
import torch

from .delta_experiment import dataset_for, load, sha256, write_json
from .paired_delta import PairedDelta
from .prepare_drivaerrim import merge_depth
from .render import VIEW_AXES, _rasterize, save_five_views
from .rim_resolution import SHARED


def shifted_views(mesh, origin, length, size, pixels):
    points = ((mesh.points - origin) / length).astype(np.float32)
    faces = np.asarray(mesh.faces).reshape(-1, 4)[:, 1:]
    result = {}
    for view, (horizontal, _, _, _) in VIEW_AXES.items():
        shifted = points.copy()
        shifted[:, horizontal] += np.float32(pixels * 1.3 / (size - 8))
        # Move the camera-plane sampling origin, leaving the depth coordinate intact.
        result[view] = _rasterize(shifted, faces, view=view, size=size, window_l=1.3)
    return result


def run(experiment):
    root = Path(experiment)
    p, cases = load(root)
    if not (root / "comparison.json").exists():
        raise ValueError("Complete frozen evaluation first")
    out = root / "sampling-control"
    if out.exists():
        raise FileExistsError("Sampling diagnostic already exists")
    out.mkdir()
    plan = json.loads((root / "evaluation-plan.json").read_text())["raw_demo"]
    identities = [plan["reference"], plan["query"]]
    data = json.loads((Path(p["prepared512"]) / "dataset.json").read_text())
    frame = data["resolution_preparation"]
    origin = np.asarray(frame["origin_m"])
    length = frame["dimensions_m"][0]
    raw = Path(p["inventory"]).parent
    inventory = json.loads(Path(p["inventory"]).read_text())
    entries = {f["path"]: f for f in inventory["files"]}
    pixels = 0.25
    write_json(
        out / "plan.json",
        {
            "horizontal_camera_phase_pixels": pixels,
            "reference": identities[0],
            "query": identities[1],
            "scope": "One post-training diagnostic, no selection use. Same physical geometry; camera sampling origin changes horizontally in each complete view.",
        },
    )
    body = {256: None, 512: None}
    for name in SHARED:
        relative = f"Geometry/{name}/{name}.stl"
        file = raw / relative
        if sha256(file) != entries[relative]["sha256"]:
            raise ValueError("Raw shared geometry changed")
        mesh = pv.read(file)
        for mirror in range(1 if name == "Car_Body" else 2):
            if mirror:
                mesh.points[:, 1] *= -1
            for size, previous in body.items():
                views = shifted_views(mesh, origin, length, size, pixels)
                body[size] = views if previous is None else merge_depth(previous, views)
    shifted = {256: {}, 512: {}}
    for identity in identities:
        gid = identity.rsplit("-", 1)[-1]
        buffers = {256: None, 512: None}
        for pos in ["FL", "RL"]:
            relative = f"Geometry/Wheel_{pos}/Wheel_{pos}_{gid}.stl"
            file = raw / relative
            if sha256(file) != entries[relative]["sha256"]:
                raise ValueError("Raw wheel geometry changed")
            mesh = pv.read(file)
            for mirror in (False, True):
                if mirror:
                    mesh.points[:, 1] *= -1
                for size, previous in buffers.items():
                    views = shifted_views(mesh, origin, length, size, pixels)
                    buffers[size] = (
                        views if previous is None else merge_depth(previous, views)
                    )
        for size, views in buffers.items():
            merge_depth(views, body[size])
            save_five_views(out / f"{gid}-{size}.npz", views)
            shifted[size][identity] = {
                n: torch.from_numpy(
                    np.stack((v.depth, v.mask)).astype(np.float32)[None]
                )
                for n, v in views.items()
            }
    torch.set_num_threads(2)
    comparison = json.loads((root / "comparison.json").read_text())
    result = {}
    for arm, config in p["arms"].items():
        ck = torch.load(root / arm / "model.pt", map_location="cpu", weights_only=True)
        model = PairedDelta(**ck["model_config"]).eval()
        model.load_state_dict(ck["model_state"])
        ds = dataset_for(p, cases, config["resolution"])
        original, params, _ = ds[ds.index[identities[0]]]
        first, second = [shifted[config["resolution"]][i] for i in identities]
        with torch.inference_mode():
            pair = model(first, second, params[None]).numpy()[0] * np.asarray(
                p["delta_std"]
            )
            self_shift = model(
                {n: v[None] for n, v in original.items()}, first, params[None]
            ).numpy()[0] * np.asarray(p["delta_std"])
        reference = np.array(
            comparison["models"][arm]["predictions"][identities[0]][identities[1]]
        )
        result[arm] = {
            "same_mesh_different_phase_delta": self_shift.tolist(),
            "same_pair_common_phase_prediction": pair.tolist(),
            "same_pair_common_phase_minus_nominal": (pair - reference).tolist(),
        }
    write_json(
        out / "results.json",
        {
            "phase_pixels": pixels,
            "arms": result,
            "scope": "One fixed pair; phase sensitivity is representation error, not a new aerodynamic effect or a validated general noise bound. No training/model choice was altered.",
        },
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    run(parser.parse_args().experiment)
