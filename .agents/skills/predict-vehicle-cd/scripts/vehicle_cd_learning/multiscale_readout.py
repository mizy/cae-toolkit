"""A fixed mixture of geometry scales for shared absolute Cd/Cl kernel readout."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from aircraft_learning.prepare_scenes import resample_views
from aircraft_learning.scene_experiment import sampling_weights

from . import dimensionless as dim
from .detail_kernel import distances
from .end_to_end_evaluation import radiant_scores
from .head_experiment import physical
from .kernel_experiment import bootstrap
from .kernel_experiment import load as load_parent
from .kernel_readout import KernelReadout
from .readout_experiment import absolute_scores
from .render import render_geometry

CONTRACT = "vehicle_multiscale_readout_v1"
SCALES = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0)


def mixture(distance):
    return sum(np.exp(-distance / (2 * s**2)) for s in SCALES) / len(SCALES)


class MultiScaleReadout(KernelReadout):
    def predict_features(self, features, base):
        if len(self.kernel_reference) == 0:
            return base
        x = (features.double() - self.feature_mean) / self.feature_scale
        ref = self.kernel_reference
        distance = (
            (x * x).mean(1)[:, None]
            + (ref * ref).mean(1)[None]
            - 2 * (x @ ref.T) / x.shape[1]
        ).clamp_min(0)
        kernel = sum(torch.exp(-distance / (2 * s**2)) for s in SCALES) / len(SCALES)
        return (
            kernel @ self.kernel_dual
            + self.kernel_bias
            + self.base_weight * base.double()
        ).to(base.dtype)


def load_model(path):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    if (
        ck.get("input_contract") != CONTRACT
        or tuple(ck["kernel_scales"]) != SCALES
        or ck["parameter_names"] != dim.NAMES
    ):
        raise ValueError("Use the fixed multiscale checkpoint loader")
    state = ck["model_state"]
    model = MultiScaleReadout(
        "pooled",
        len(state["feature_mean"]),
        len(state["kernel_reference"]),
        **ck["model_config"],
    )
    model.load_state_dict(state, strict=True)
    if (
        not all(torch.isfinite(v).all() for v in state.values())
        or (model.feature_scale <= 0).any()
    ):
        raise ValueError("Invalid multiscale checkpoint")
    return model.eval().requires_grad_(False), ck


def fit(parent, external, output):
    """@entry Test a fixed scale mixture; select ridge strength/mode on validation only."""
    if output.exists():
        raise FileExistsError("Use a fresh multiscale experiment directory")
    p, data, features = load_parent(parent)
    output.mkdir(parents=True)
    files = [Path(__file__), Path(__file__).with_name("kernel_readout.py"), external]
    protocol = {
        "parent": str(parent.resolve()),
        "checkpoint": p["checkpoint"],
        "scales": SCALES,
        "alphas": [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
        "modes": ["direct", "residual"],
        "selection": p["selection"],
        "hypothesis": "One globally selected length scale underweights fine Rim variations; average six positive Gaussian kernels without a source/category input.",
        "sha256": {
            **p["sha256"],
            **{
                str(f.resolve()): hashlib.sha256(f.read_bytes()).hexdigest()
                for f in files
            },
        },
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    torch.set_num_threads(2)
    source = torch.load(p["checkpoint"], map_location="cpu", weights_only=True)
    train = np.array([c["split"] == "train" for c in data["cases"]])
    val = np.array([c["split"] == "val" for c in data["cases"]])
    x = features["pooled"].astype(float)
    mean, scale = x[train].mean(0), np.maximum(x[train].std(0), 0.05)
    g = x.shape[1] - 36
    scale[:g] *= np.sqrt(g / x.shape[1])
    scale[g:] *= np.sqrt(36 / x.shape[1])
    x = (x - mean) / scale
    weights = sampling_weights(
        [c for c in data["cases"] if c["split"] == "train"]
    ).numpy()
    targets, base = (
        features["targets"][train].astype(float),
        features["base"].astype(float),
    )
    gram, kv = (
        mixture(distances(x[train], x[train])),
        mixture(distances(x[val], x[train])),
    )
    systems = []
    for j in range(2):
        observed = np.isfinite(targets[:, j])
        root = np.sqrt(weights[observed] / weights[observed].mean())
        responses = np.stack(
            (targets[observed, j], targets[observed, j] - base[train, j][observed]),
            axis=1,
        )
        bias = np.average(responses, weights=weights[observed], axis=0)
        eigen, vectors = np.linalg.eigh(
            gram[np.ix_(observed, observed)] * root[:, None] * root[None]
        )
        projected = vectors.T @ ((responses - bias) * root[:, None])
        systems.append((observed, root, eigen, vectors, projected, bias))
    limits = p["baseline_validation"]["macro"]
    best, chosen, candidates = sum(limits.values()), None, []
    model = MultiScaleReadout("pooled", x.shape[1], **source["model_config"])
    missing, extra = model.load_state_dict(source["model_state"], strict=False)
    if extra or set(missing) != {
        "feature_mean",
        "feature_scale",
        "kernel_reference",
        "kernel_dual",
        "kernel_bias",
        "kernel_bandwidth",
        "base_weight",
    }:
        raise ValueError("Unexpected source weights")
    model.feature_mean.copy_(torch.from_numpy(mean))
    model.feature_scale.copy_(torch.from_numpy(scale))
    for alpha in protocol["alphas"]:
        dual, bias = np.zeros((int(train.sum()), 2, 2)), np.zeros((2, 2))
        for j, (observed, root, eigen, vectors, projected, center) in enumerate(
            systems
        ):
            dual[observed, j] = root[:, None] * (
                vectors @ (projected / (np.maximum(eigen, 0)[:, None] + alpha))
            )
            bias[j] = center
        for mode, name in enumerate(protocol["modes"]):
            pred = kv @ dual[:, :, mode] + bias[:, mode] + mode * base[val]
            metrics = absolute_scores(
                [c for c in data["cases"] if c["split"] == "val"],
                physical(pred, p["normalization"]),
            )
            eligible = all(metrics["macro"][k] <= limits[k] for k in limits)
            candidate = {
                "alpha": alpha,
                "mode": name,
                "validation": metrics,
                "eligible": eligible,
            }
            candidates.append(candidate)
            if eligible and sum(metrics["macro"].values()) < best:
                chosen, best = candidate, sum(metrics["macro"].values())
                model.kernel_reference = torch.tensor(x[train], dtype=torch.float64)
                model.kernel_dual = torch.tensor(dual[:, :, mode], dtype=torch.float64)
                model.kernel_bias.copy_(torch.from_numpy(bias[:, mode]))
                model.base_weight.fill_(mode)
    ck = dict(source)
    for key in (
        "epoch",
        "pair_arm",
        "migration",
        "validation_selection_score",
        "validation_absolute_mae",
    ):
        ck.pop(key, None)
    ck.update(
        input_contract=CONTRACT,
        kernel_scales=SCALES,
        readout_stage="pooled",
        model_state=model.state_dict(),
        kernel_selection=chosen,
        training_mode="fixed_multiscale_kernel",
        source_checkpoint=p["checkpoint"],
        source_checkpoint_sha256=p["sha256"][p["checkpoint"]],
    )
    torch.save(ck, output / "model.pt")
    selection = {
        "selected": chosen,
        "candidates": candidates,
        "model_sha256": hashlib.sha256((output / "model.pt").read_bytes()).hexdigest(),
        "test_used": False,
        "external_used": False,
    }
    (output / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    print(
        json.dumps({"selected": chosen, "candidate_count": len(candidates)}, indent=2)
    )


def evaluate(parent, external, output):
    """Evaluate the validation-selected mixture after its model hash is frozen."""
    p, data, features = load_parent(parent)
    protocol = json.loads((output / "protocol.json").read_text())
    for name, sha in protocol["sha256"].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != sha:
            raise ValueError("Frozen multiscale input changed")
    selection = json.loads((output / "selection.json").read_text())
    if (
        hashlib.sha256((output / "model.pt").read_bytes()).hexdigest()
        != selection["model_sha256"]
    ):
        raise ValueError("Selected multiscale model changed")
    torch.set_num_threads(2)
    model, _ = load_model(output / "model.pt")
    normalized = []
    with torch.inference_mode():
        for start in range(0, len(data["cases"]), 128):
            part = slice(start, start + 128)
            normalized.append(
                model.predict_features(
                    torch.from_numpy(features["pooled"][part]),
                    torch.from_numpy(features["base"][part]),
                ).numpy()
            )
    pred = physical(np.concatenate(normalized), p["normalization"])
    baseline = physical(features["base"], p["normalization"])
    scores = {}
    for split in ("train", "val", "test"):
        mask = np.array([c["split"] == split for c in data["cases"]])
        scores[split] = absolute_scores(
            [c for c in data["cases"] if c["split"] == split], pred[mask]
        )
    test = np.array([c["split"] == "test" for c in data["cases"]])
    result = {
        "scores": scores,
        "paired_bootstrap": bootstrap(
            [c for c in data["cases"] if c["split"] == "test"],
            baseline[test],
            pred[test],
        ),
        "radiant": radiant_scores(
            data["cases"], pred, json.loads(external.read_text())
        ),
        "rows": [
            {
                "case_id": c["case_id"],
                "source": c["source"],
                "split": c["split"],
                "truth": c["targets"],
                "prediction": v.tolist(),
            }
            for c, v in zip(data["cases"], pred)
        ],
    }
    (output / "comparison.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "test": scores["test"]["macro"],
                "radiant": {
                    k: v for k, v in result["radiant"].items() if k != "by_anchor"
                },
            },
            indent=2,
        )
    )


def predict_case(checkpoint, case_path, output):
    torch.set_num_threads(2)
    model, ck = load_model(checkpoint)
    case = json.loads(case_path.read_text())
    g = case["geometry"]
    views, dimensions = render_geometry(
        (case_path.parent / g["path"]).resolve(), g, ck["source_render_config"]
    )
    arrays = resample_views(
        {k: np.stack((v.depth, v.mask)).astype(np.float32) for k, v in views.items()}
    )
    raw = dim.parameters(dimensions, case["conditions"])
    values = dim.predict_views(model, ck, arrays, raw)
    names = (
        ["cd"] if case.get("condition_profile") == "builtin_assumed" else ["cd", "cl"]
    )
    warnings = []
    for name, value in zip(dim.PARAMETERS, raw):
        bounds = ck["parameter_ranges"].get(name)
        if value is not None and (
            bounds is None or value < bounds[0] - 1e-6 or value > bounds[1] + 1e-6
        ):
            warnings.append(f"{name} outside observed support")
    if case["conditions"].get("re_body") is None:
        warnings.append("Re unknown; no Reynolds-response validation")
    result = {
        "coefficients": {
            k: float(v) for k, v in zip(("cd", "cl"), values) if k in names
        },
        "input_contract": CONTRACT,
        "warnings": warnings,
        "scope": "Vehicle known-family multiscale experiment",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    for name in ("fit", "evaluate"):
        q = sub.add_parser(name)
        q.add_argument("parent", type=Path)
        q.add_argument("external", type=Path)
        q.add_argument("output", type=Path)
    q = sub.add_parser("predict")
    q.add_argument("checkpoint", type=Path)
    q.add_argument("case", type=Path)
    q.add_argument("output", type=Path)
    a = p.parse_args()
    if a.action == "predict":
        print(json.dumps(predict_case(a.checkpoint, a.case, a.output), indent=2))
    else:
        (fit if a.action == "fit" else evaluate)(a.parent, a.external, a.output)
