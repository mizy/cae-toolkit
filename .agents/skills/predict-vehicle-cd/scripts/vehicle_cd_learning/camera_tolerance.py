"""Measure absolute coefficient changes when only the five observation cameras change."""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from .absolute_view_audit import predict_grid, view_metrics
from .delta_experiment import sha256, write_json
from .dimensionless import normalize_parameters
from .joint_sampling import load, load_checkpoint
from .model import VIEWS
from .phase_decoder_data import OFF_GRID
from .render import VIEW_AXES
from .rim_resolution import SHARED
from .subpixel_geometry import rim_triangles
from .subpixel_sampling import capture, project_vertices, resolve

CAMERAS = [("fixed", 0.0, dx, dy) for dx, dy in ((0, 0), (0.25, 0), *OFF_GRID)] + [
    (axis, signed * degrees, 0, 0)
    for degrees in (0.5, 1.0)
    for axis in ("azimuth", "elevation")
    for signed in (-1, 1)
]
FP16 = len(CAMERAS)


def camera_points(points, pivot, view, axis, degrees):
    """Express a fixed object in a camera turned about the shared vehicle centre."""
    if axis == "fixed":
        return points
    horizontal, vertical, depth, sign = VIEW_AXES[view]
    side = horizontal if axis == "azimuth" else vertical
    theta = np.deg2rad(degrees)
    cosine, sine = np.cos(theta), np.sin(theta)
    lateral = points[:, side].astype(float) - pivot[side]
    toward_camera = sign * (points[:, depth].astype(float) - pivot[depth])
    rotated = points.copy()
    rotated[:, side] = pivot[side] + cosine * lateral - sine * toward_camera
    rotated[:, depth] = pivot[depth] + sign * (sine * lateral + cosine * toward_camera)
    return rotated


def render_parts(parts, pivot, camera, base=None):
    axis, degrees, dx, dy = camera
    result = {}
    for view in VIEWS:
        depth = None if base is None else base[view].copy()
        for points, faces in parts:
            projected = project_vertices(
                camera_points(points, pivot, view, axis, degrees), view, 256
            )
            depth = capture(projected, faces, 256, 1, (dx, dy), (0.0, 0.0), depth)
        result[view] = depth
    return result


def vehicle_images(parts, pivot, shared, nominal_file):
    """Combine actual first hits of the shared body and wheel parts at each camera."""
    images = []
    audit = {}
    for index, (camera, body) in enumerate(zip(CAMERAS, shared)):
        depth = render_parts(parts, pivot, camera, body)
        images.append({view: resolve(depth[view], 256, 1) for view in VIEWS})
        audit[index] = {}
        for view, array in images[-1].items():
            values, mask = array
            visible = mask > 0
            edge = visible[:, 0].any() or visible[:, -1].any() or visible[[0, -1]].any()
            audit[index][view] = {
                "mask_changed": int(np.count_nonzero(mask != images[0][view][1])),
                "viewport_edge_occupied": bool(edge),
                "depth_outside_unit_interval": int(
                    np.count_nonzero(visible & ((values < 0) | (values > 1)))
                ),
                "minimum_visible_depth": float(values[visible].min()),
                "maximum_visible_depth": float(values[visible].max()),
            }
    with np.load(nominal_file, allow_pickle=False) as original:
        for view in VIEWS:
            np.testing.assert_array_equal(images[0][view], original[view])
    rounded = {v: a.copy() for v, a in images[0].items()}
    for array in rounded.values():
        array[0] = array[0].astype(np.float16).astype(np.float32)
    images.append(rounded)
    return {v: torch.from_numpy(np.stack([a[v] for a in images])) for v in VIEWS}, audit


def embeddings(model, readout, images, parameters, stats, device):
    tokens = []
    for start in range(0, FP16 + 1, 4):
        pooled = model.encode(
            {v: images[v][start : start + 4].to(device) for v in VIEWS}
        )
        projected = [
            (readout.bottom_project if v == "bottom" else readout.image_project)(
                pooled[v]
            )
            for v in VIEWS
        ]
        tokens.append(torch.stack(projected, 1))
    tokens = torch.cat(tokens)
    check = np.array(
        [[0] * 5, [1, 0, 0, 2, 3], [0, 4, 0, 0, 0], [8, 9, 0, 10, 11], [FP16] * 5]
    )
    mixed = {v: images[v][check[:, i]].to(device) for i, v in enumerate(VIEWS)}
    direct = (
        readout(model.encode(mixed), parameters.expand(len(check), -1)).cpu().numpy()
    )
    direct = direct * stats["target_std"] + stats["target_mean"]
    factored = predict_grid(readout, tokens[None], parameters, check, stats)[0]
    error = float(np.abs(direct - factored).max())
    if error > 2e-6:
        raise ValueError(f"Full forward disagrees with per-view encoding by {error}")
    # The reference depth plane of the bottom camera touches the ground at zero angle.
    # Check whether a few signed first-hit depths crossing zero explain single-view drift.
    angular = [i for i, (axis, *_) in enumerate(CAMERAS) if axis != "fixed"]
    bottom = images["bottom"][angular].to(device).clone()
    bottom[:, 0].clamp_(0, 1)
    bounded = tokens.clone()
    bounded[angular, VIEWS.index("bottom")] = readout.bottom_project(
        model.bottom_encoder.pooled_features(bottom)
    )
    single = np.zeros((len(angular), len(VIEWS)), dtype=int)
    single[:, VIEWS.index("bottom")] = angular
    before = predict_grid(readout, tokens[None], parameters, single, stats)
    after = predict_grid(readout, bounded[None], parameters, single, stats)
    return tokens.cpu(), error, np.abs(before - after).max((0, 1))


def camera_metrics(readout, tokens, params, targets, stats):
    groups = {
        "pixel_shift": [0, 1, 2, 3],
        "angle_0.5": [0, 4, 5, 6, 7],
        "angle_1": [0, 8, 9, 10, 11],
    }
    scores, arrays = {}, {}
    for name, indices in groups.items():
        grid = np.asarray(list(itertools.product(indices, repeat=len(VIEWS))))
        values = predict_grid(readout, tokens, params, grid, stats)
        metrics = view_metrics(values, targets, grid)
        single = np.count_nonzero(grid, axis=1) == 1
        drift = np.abs(values[:, single] - values[:, :1])
        metrics["mean_over_vehicles_and_single_view_offsets"] = drift.mean(
            (0, 1)
        ).tolist()
        metrics["mean_vehicle_worst_single_view_offset"] = drift.max(1).mean(0).tolist()
        metrics["single_view_count"] = int(single.sum())
        metrics["all_patterns_count"] = len(grid)
        scores[name] = metrics
        arrays[name] = values
        arrays[name + "_grid"] = grid
    rounded = predict_grid(readout, tokens, params, np.asarray([[FP16] * 5]), stats)
    scores["float16_depth"] = {
        "mean_absolute_drift": np.abs(rounded[:, 0] - arrays["pixel_shift"][:, 0])
        .mean(0)
        .tolist()
    }
    arrays["float16_depth"] = rounded[:, 0]
    return scores, arrays


def observations(rows, prepared, raw, entries, frame, pivot):
    body = rim_triangles(
        raw,
        entries,
        [(f"Geometry/{n}/{n}.stl", n != "Car_Body") for n in SHARED],
        frame,
    )
    shared = [render_parts(body, pivot, camera) for camera in CAMERAS]
    del body
    for case in rows:
        gid = case["case_id"].rsplit("-", 1)[-1]
        specs = [
            (f"Geometry/Wheel_{pos}/Wheel_{pos}_{gid}.stl", True)
            for pos in ("FL", "RL")
        ]
        path = prepared / case["views"]
        if sha256(path) != case["views_sha256"]:
            raise ValueError(f"Nominal views changed for {gid}")
        yield vehicle_images(
            rim_triangles(raw, entries, specs, frame), pivot, shared, path
        )


def run(experiment, evaluation, output):
    """@entry Replay matched fifth-epoch weights on the same native 256-pixel images."""
    p, cases = load(experiment)
    ids = json.loads(evaluation.read_text())["test_ids"]
    all_cases = {c["case_id"]: c for c in cases}
    rows = [all_cases[cid] for cid in ids]
    if any(
        c["source"] != "drivaerrim"
        or c["split"] != "test"
        or c["parameters"] != rows[0]["parameters"]
        for c in rows
    ):
        raise ValueError("Expected matched historical DrivAerRim test records")
    parent = json.loads((Path(p["parent_experiment"]) / "protocol.json").read_text())
    inventory = Path(parent["inventory"])
    if sha256(inventory) != parent["input_sha256"][str(inventory)]:
        raise ValueError("Rim inventory changed")
    entries = {f["path"]: f for f in json.loads(inventory.read_text())["files"]}
    frame = json.loads((Path(parent["prepared512"]) / "dataset.json").read_text())[
        "resolution_preparation"
    ]
    pivot = np.array(
        [0.0, 0.0, 0.5 * frame["dimensions_m"][2] / frame["dimensions_m"][0]]
    )
    locked = json.loads((experiment / "test-plan.json").read_text())["models"]
    models = {name: locked[name] for name in ("control-matched", "online-diagnostic")}
    for ck in models.values():
        if ck["epoch"] != 5 or sha256(ck["checkpoint"]) != ck["checkpoint_sha256"]:
            raise ValueError("Expected the preserved matched epoch-5 checkpoints")
    output.mkdir(parents=True, exist_ok=False)
    plan = {
        "models": models,
        "ids": ids,
        "cameras_axis_degrees_dx_dy": CAMERAS,
        "fp16_state": FP16,
        "pivot_in_vehicle_lengths": pivot.tolist(),
        "target_order": ["Cd", "Cl"],
        "input_size": 256,
        "views": VIEWS,
        "evaluation_sha256": sha256(evaluation),
        "inventory_sha256": sha256(inventory),
        "script_sha256": sha256(Path(__file__)),
        "orientation": "Positive azimuth or elevation turns the camera toward the original screen right or up about the body centre. Mesh, flight conditions, physical yaw, camera window and signed depth reference are fixed.",
    }
    write_json(output / "plan.json", plan)
    torch.set_num_threads(2)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    nets = {
        k: tuple(m.eval() for m in load_checkpoint(v["checkpoint"], device))
        for k, v in models.items()
    }
    tokens = {k: [] for k in models}
    parity = {k: [] for k in models}
    depth_clipping = {k: [] for k in models}
    camera_audit = {}
    stats = p["normalization"]
    params = torch.from_numpy(
        np.stack([normalize_parameters(c["parameters"], stats) for c in rows])
    ).to(device)
    with torch.inference_mode():
        rendered = observations(
            rows, Path(p["joint_prepared"]), inventory.parent, entries, frame, pivot
        )
        for i, (images, audit) in enumerate(rendered):
            cid = ids[i]
            camera_audit[cid] = audit
            for name, (model, readout) in nets.items():
                vectors, error, clipping = embeddings(
                    model, readout, images, params[i : i + 1], stats, device
                )
                tokens[name].append(vectors)
                parity[name].append(error)
                depth_clipping[name].append(clipping)
            print(cid, flush=True)
        write_json(output / "image-audit.json", camera_audit)
        result = {}
        for name, (_, readout) in nets.items():
            scores, values = camera_metrics(
                readout,
                torch.stack(tokens[name]).to(device),
                params,
                np.array([c["targets"] for c in rows]),
                stats,
            )
            scores["direct_forward_max_difference"] = max(parity[name])
            scores["bottom_depth_clipping_max_single_view_effect"] = np.max(
                depth_clipping[name], axis=0
            ).tolist()
            path = output / (name + ".npz")
            np.savez_compressed(path, case_ids=ids, **values)
            scores["predictions_sha256"] = sha256(path)
            result[name] = scores
        write_json(output / "comparison.json", result)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for arg in ("experiment", "evaluation", "output"):
        cli.add_argument(arg, type=Path)
    args = cli.parse_args()
    run(args.experiment, args.evaluation, args.output)
