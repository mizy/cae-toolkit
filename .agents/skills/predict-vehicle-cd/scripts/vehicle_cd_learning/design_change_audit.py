"""Contrast geometry deformations and matched-condition component replacements with raster changes."""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pyvista as pv
import torch
from aircraft_learning.prepare_scenes import resample_views

from . import sampling_experiment
from .absolute_view_audit import predict_grid, project
from .absolute_view_renders import PHASES
from .cl_sampling_repair import encode_requests
from .delta_experiment import sha256, write_json
from .delta_metrics import score_delta
from .detail_sensitivity import edit_mirrors
from .dimensionless import normalize_parameters
from .joint_sampling import FullCases, batch_cases, load, load_checkpoint
from .model import VIEWS
from .phase_views import camera_views
from .render import canonical_points


def pair_outputs(model, bank, requests, numeric, scale, device):
    predictions = []
    for start in range(0, len(requests), 64):
        left, right = zip(*requests[start : start + 64])
        a = {
            k: torch.stack([bank[t][k] for t in left]).to(device) for k in bank[left[0]]
        }
        b = {k: torch.stack([bank[t][k] for t in right]).to(device) for k in a}
        predictions.append(
            model.predict_pair(a, b, numeric.expand(len(left), -1)).cpu().numpy()
        )
    return np.concatenate(predictions).astype(float) * scale


def resolution_check(model, head, full, rim, sp, p, device):
    val = sorted(c["case_id"] for c in rim.cases if c["split"] == "val")
    ids = sp["anchors"] + val
    geo_pairs = list(itertools.combinations(val, 2))
    labels = {c["case_id"]: np.array(c["targets"]) for c in rim.cases}
    numeric = rim.parameters[rim.index[val[0]]].to(device)
    if any(
        not torch.equal(rim.parameters[rim.index[i]].to(device), numeric) for i in ids
    ):
        raise ValueError("Mixed conditions in Rim resolution comparison")
    truth = np.array([labels[b] - labels[a] for a, b in geo_pairs])
    queries = np.array([labels[q] for q in val])
    result = {}
    for size in (256, 512):
        bank, absolute = {}, {}
        for start in range(0, len(ids), 8):
            group = ids[start : start + 8]
            fi = [full.index[i] for i in group]
            if size == 256:
                views, _, _ = batch_cases(full, fi, device)
            else:
                views, _, _ = sampling_experiment.batch_views(
                    rim, [(rim.index[i], None) for i in group], device
                )
            encoded = model.encode(views)
            y = (
                head(encoded, full.parameters[fi].to(device))
                .cpu()
                .numpy()
                .astype(float)
            )
            y = y * p["normalization"]["target_std"] + p["normalization"]["target_mean"]
            absolute.update(zip(group, y))
            for i, cid in enumerate(group):
                bank[cid] = {k: v[i].cpu() for k, v in encoded.items()}
        direct = pair_outputs(model, bank, geo_pairs, numeric, sp["delta_std"], device)
        from_absolute = np.array([absolute[b] - absolute[a] for a, b in geo_pairs])
        base = np.array([labels[a] for a in sp["anchors"]])
        anchored = pair_outputs(
            model,
            bank,
            list(itertools.product(sp["anchors"], val)),
            numeric,
            sp["delta_std"],
            device,
        ).reshape(3, -1, 2)
        result[str(size)] = {
            "absolute_nominal_mae": np.abs(
                np.array([absolute[q] for q in val]) - queries
            )
            .mean(0)
            .tolist(),
            "absolute_difference": score_delta(truth, from_absolute),
            "direct_difference": score_delta(truth, direct),
            "known_reference_plus_direct_delta_mae": np.abs(
                (base[:, None] + anchored).mean(0) - queries
            )
            .mean(0)
            .tolist(),
            "same_reference_constant_mae": np.abs(base.mean(0) - queries)
            .mean(0)
            .tolist(),
            "absolute_difference_vs_direct_mae": np.abs(from_absolute - direct)
            .mean(0)
            .tolist(),
        }
    return result


def stress_patterns():
    """Return explicit endpoint phase patterns for the direct nonlinear pair readout."""
    zero = (0,) * 5
    patterns = {(zero, zero)}
    for j in range(len(VIEWS)):
        for u, v in itertools.product(range(5), repeat=2):
            a, b = list(zero), list(zero)
            a[j], b[j] = u, v
            patterns.add((tuple(a), tuple(b)))
    for u, v in itertools.product(range(5), repeat=2):
        patterns.add(((u,) * 5, (v,) * 5))
    rng = np.random.default_rng(42)
    for a, b in rng.integers(0, 5, size=(128, 2, 5)):
        patterns.add((tuple(a.tolist()), tuple(b.tolist())))
    return sorted(patterns)


def stress_swaps(model, rim, sp, ids, device, output):
    phase_lookup = {
        tuple(v): key
        for key, v in zip(
            sp["phase_ids"], [(0.25, 0), (-0.25, 0), (0, 0.25), (0, -0.25)]
        )
    }
    phase_ids = [None, *[phase_lookup[xy] for xy in PHASES[1:]]]
    cache = encode_requests(
        model, rim, [(rim.index[c], s) for c in ids for s in phase_ids], device
    )
    pairs = list(itertools.combinations(ids, 2))
    truth = np.array(
        [
            np.asarray(rim.cases[rim.index[b]]["targets"])
            - rim.cases[rim.index[a]]["targets"]
            for a, b in pairs
        ]
    )
    numeric = rim.parameters[rim.index[ids[0]]].to(device).expand(len(pairs), -1)
    sides = [[rim.index[q] for q in s] for s in zip(*pairs)]
    all_states = []
    for side in sides:
        all_states.append(
            [
                {
                    k: torch.stack([cache[(i, phase)][k] for i in side]).to(device)
                    for k in cache[(side[0], phase)]
                }
                for phase in phase_ids
            ]
        )
    patterns = stress_patterns()
    values = []
    for pattern in patterns:
        ends = []
        for side, states in zip(all_states, pattern):
            ends.append(
                {
                    k: side[state][k]
                    for n, state in zip(VIEWS, states)
                    for k in (n, n + "_mask")
                }
            )
        values.append(
            model.predict_pair(*ends, numeric).cpu().numpy().astype(float)
            * sp["delta_std"]
        )
    values = np.stack(values, axis=1)
    if any(any(x) for x in patterns[0]):
        raise ValueError("First pair pattern must be nominal")
    np.savez_compressed(
        output,
        pairs=np.asarray(pairs),
        truth=truth,
        patterns=np.asarray(patterns),
        predictions=values,
    )
    scores = score_delta(truth, values[:, 0])
    lo, hi = values.min(1), values.max(1)
    for j, n in enumerate(("Cd", "Cl")):
        target = truth[:, j]
        nz = np.abs(target) > 1e-12
        good = ((target > 0) & (lo[:, j] > 5e-6)) | ((target < 0) & (hi[:, j] < -5e-6))
        scores[n].update(
            nonzero=int(nz.sum()),
            correct_every_tested_pattern=int((good & nz).sum()),
            can_reverse=int(((lo[:, j] < -5e-6) & (hi[:, j] > 5e-6) & nz).sum()),
            mean_worst_sampling_drift=float(
                np.abs(values[:, :, j] - values[:, 0:1, j]).max(1).mean()
            ),
            mean_worst_delta_mae=float(
                np.maximum(np.abs(lo[:, j] - target), np.abs(hi[:, j] - target)).mean()
            ),
        )
    return {
        "case_count": len(ids),
        "pair_count": len(pairs),
        "pattern_count": len(patterns),
        "scores": scores,
        "arrays_sha256": sha256(output),
    }


def mirror_sweep(model, head, data, p, source, device):
    report = json.loads((source / "geometry-report.json").read_text())
    mesh_path = Path(report["source_mesh"])
    if sha256(mesh_path) != report["selection"]["source_sha256"]:
        raise ValueError("Original mirror geometry changed")
    mesh = pv.read(mesh_path).triangulate()
    original = np.asarray(mesh.points, dtype=float).copy()
    origin, dims = canonical_points(original, unit_scale_to_m=1)
    case = data.cases[data.index[report["case"]["case_id"]]]
    parameters = case["parameters"]
    stats = p["normalization"]
    length = dims["length_m"]
    grid = np.array(list(itertools.product(range(5), repeat=5)), np.int16)
    responses = []
    views_only = []
    max_change = []
    probe_positions = (0.0, 0.075, 0.15, 0.225, 0.30)
    for fraction in probe_positions:
        physical = (
            edit_mirrors(original, 1 - fraction, report["selection"])
            .astype(mesh.points.dtype)
            .astype(float)
        )
        mesh.points = origin + (physical - original).astype(np.float32) / length
        width, height = np.ptp(physical, axis=0)[1:]
        max_change.append(float(np.linalg.norm(physical - original, axis=1).max()))
        updated = [width / length, height / length, *parameters[2:]]
        tokens = []
        for dx, dy in PHASES:
            # Native legacy render at 512, then its original 256 sampling transform.
            scale = (512 - 8) / (256 - 8)
            images = camera_views(mesh, np.zeros(3), 1.0, 512, (dx * scale, dy * scale))
            arrays = resample_views(
                {
                    k: np.stack((v.depth, v.mask)).astype(np.float32)
                    for k, v in images.items()
                }
            )
            if (dx, dy) == PHASES[0] and fraction in (0.0, 0.15, 0.30):
                archive = next(
                    v
                    for v in report["variants"]
                    if abs(v["outboard_extension_scale"] - (1 - fraction)) < 1e-8
                )
                with np.load(source / archive["views"]["0"]) as f:
                    for n in VIEWS:
                        np.testing.assert_array_equal(arrays[n], f[n])
                if fraction == 0:
                    nominal, _, _ = data.get(data.index[case["case_id"]])
                    for n in VIEWS:
                        np.testing.assert_array_equal(arrays[n], nominal[n].numpy())
            tokens.append(
                project(
                    model,
                    head,
                    {
                        n: torch.from_numpy(v[None]).to(device)
                        for n, v in arrays.items()
                    },
                )
            )
        bank = torch.stack(tokens, dim=1)
        for numbers, results in [(updated, responses), (parameters, views_only)]:
            param = torch.from_numpy(normalize_parameters(numbers, stats)[None]).to(
                device
            )
            results.append(predict_grid(head, bank, param, grid, stats)[0])
    values = np.stack(responses)
    delta = values - values[:1]
    img = np.stack(views_only)
    return {
        "case_id": case["case_id"],
        "source_mesh_sha256": sha256(mesh_path),
        "base_CFD": case["targets"],
        "fraction_shortened": probe_positions,
        "maximum_displacement_m": max_change,
        "nominal_coefficients": values[:, 0].tolist(),
        "nominal_change": delta[:, 0].tolist(),
        "image_only_nominal_change": (img[:, 0] - img[0, 0]).tolist(),
        "same_phase_delta_min": delta.min(1).tolist(),
        "same_phase_delta_max": delta.max(1).tolist(),
        "independent_phase_delta_min": (values.min(1) - values[0].max(0)).tolist(),
        "independent_phase_delta_max": (values.max(1) - values[0].min(0)).tolist(),
        "modified_CFD_available": False,
        "source_report_sha256": sha256(source / "geometry-report.json"),
    }


def run(experiment, raw, source, output):
    p, cases = load(experiment)
    sp, rim_cases = sampling_experiment.load(p["sampling_parent"])
    retained = json.loads((experiment / "retention.json").read_text())
    if sha256(retained["checkpoint"]) != retained["sha256"]:
        raise ValueError("Checkpoint changed")
    ids = sorted(
        cid
        for gid, record in json.loads((raw / "manifest.json").read_text())[
            "geometries"
        ].items()
        if record["source"] == "drivaerrim"
        for cid in record["case_ids"]
    )
    if len(ids) != 12:
        raise ValueError("Expected the frozen Rim validation subset")
    output.mkdir(parents=True, exist_ok=False)
    write_json(
        output / "plan.json",
        {
            "epoch5_sha256": retained["sha256"],
            "case_ids": ids,
            "pair_patterns": stress_patterns(),
            "three_known_anchors": sp["anchors"],
            "anchor_aggregation": "unweighted mean of CFD(anchor) plus direct predicted delta",
            "mirror_fraction_shortened": [0, 0.075, 0.15, 0.225, 0.3],
            "source_sha256": sha256(Path(__file__)),
            "scope": "Frozen coefficient pathways, same-conditions Rim wheel replacements. Mirror edits use the full-pool absolute head, measured CFD exists only for the original car.",
        },
    )
    torch.set_num_threads(2)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model, head = load_checkpoint(retained["checkpoint"], device)
    model.eval()
    head.eval()
    full, rim = FullCases(p, cases), sampling_experiment.SamplingCases(sp, rim_cases)
    with torch.inference_mode():
        results = {
            "resolution": resolution_check(model, head, full, rim, sp, p, device),
            "swap": stress_swaps(
                model, rim, sp, ids, device, output / "replacement-predictions.npz"
            ),
            "mirror": mirror_sweep(model, head, full, p, source, device),
        }
    write_json(output / "results.json", results)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for key in ("experiment", "raw", "source", "output"):
        cli.add_argument(key, type=Path)
    a = cli.parse_args()
    run(a.experiment, a.raw, a.source, a.output)
