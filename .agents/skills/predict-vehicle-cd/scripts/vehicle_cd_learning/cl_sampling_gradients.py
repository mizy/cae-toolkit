"""Train-only local gradient evidence for real-render Cl sampling sensitivity."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .delta_experiment import sha256, write_json
from .image_perturbation import translate_views
from .joint_sampling import FullCases, batch_cases, load, load_checkpoint
from .model import VIEWS
from .sampling_experiment import SamplingCases, batch_views
from .sampling_experiment import load as load_sampling


def target_objectives(prediction, targets, weights, loss_weights):
    """Split the existing weighted full-pool objective into its Cd/Cl terms."""
    nominal, augmented = prediction.chunk(2)
    observed = torch.isfinite(targets)
    safe = torch.where(observed, targets, 0)
    square = (
        loss_weights["absolute_nominal"] * (nominal - safe).square()
        + loss_weights["absolute_augmented"] * (augmented - safe).square()
        + loss_weights["absolute_consistency"] * (augmented - nominal.detach()).square()
    ).masked_fill(~observed, 0)
    return (square * weights[:, None] / observed.sum(1).clamp_min(1)[:, None]).mean(0)


def cnn_gradient(loss, model, retain_graph=False):
    parameters = [
        p
        for n, p in model.named_parameters()
        if n.startswith(("image_encoder.", "bottom_encoder."))
    ]
    values = torch.autograd.grad(
        loss, parameters, retain_graph=retain_graph, allow_unused=True
    )
    return torch.cat(
        [
            torch.zeros_like(p).flatten() if g is None else g.detach().flatten()
            for p, g in zip(parameters, values)
        ]
    ).double()


def describe_gradient(gradient, references):
    norm = float(gradient.norm())
    row = {"norm": norm, "cosine": {}}
    for name, reference in references.items():
        denominator = norm * float(reference.norm())
        row["cosine"][name] = (
            float(gradient @ reference) / denominator if denominator > 1e-20 else None
        )
    return row


def real_noise_gradient(model, data, ids, phases, scale):
    """Use only existing TRAIN real-render phases, never a validation residual."""
    gradients, predictions = [], []
    for start in range(0, len(ids), 4):
        current = ids[start : start + 4]
        requests = [(data.index[i], None) for i in current]
        requests += [(data.index[i], phase) for phase in phases for i in current]
        views, params, _ = batch_views(data, requests, torch.device("cpu"))
        encoded, n = model.encode(views), len(current)
        left = torch.arange(n).repeat(len(phases))
        right = torch.arange(n, n * (1 + len(phases)))
        output = model.predict_pair(
            {k: v[left] for k, v in encoded.items()},
            {k: v[right] for k, v in encoded.items()},
            params[left],
        )
        gradients.append(cnn_gradient(output[:, 1].square().mean(), model))
        predictions.append(output.detach().numpy() * scale)
    return torch.stack(gradients).mean(0), np.concatenate(predictions)


def rim_label_gradients(model, data, left, right, scale):
    requests = [(int(i), None) for i in np.concatenate((left, right))]
    views, params, targets = batch_views(data, requests, torch.device("cpu"))
    encoded, n = model.encode(views), len(left)
    output = model.predict_pair(
        {k: v[:n] for k, v in encoded.items()},
        {k: v[n:] for k, v in encoded.items()},
        params[:n],
    )
    target = (targets[n:] - targets[:n]) / torch.tensor(scale, dtype=torch.float32)
    return {
        "Rim_nominal_Cd": cnn_gradient(
            F.mse_loss(output[:, 0], target[:, 0]) * 0.5, model, True
        ),
        "Rim_nominal_Cl": cnn_gradient(
            F.mse_loss(output[:, 1], target[:, 1]) * 0.5, model
        ),
    }


def run(experiment, output):
    """@entry Diagnose local train gradients without taking optimizer steps."""
    root, output = Path(experiment), Path(output)
    if output.exists():
        raise FileExistsError("Use a fresh gradient-diagnosis directory")
    p, cases = load(root)
    retained = json.loads((root / "retention.json").read_text())
    if sha256(retained["checkpoint"]) != retained["sha256"]:
        raise ValueError("Retained source checkpoint changed")
    sp, rim_cases = load_sampling(p["sampling_parent"])
    phase_plan = json.loads((Path(sp["phase_root"]) / "plan.json").read_text())
    rng = np.random.default_rng(42)
    groups = {}
    for source in p["source_counts"]:
        indices = [
            i
            for i, c in enumerate(cases)
            if c["source"] == source and c["split"] == "train"
        ]
        groups[source] = rng.choice(indices, 6, replace=False).reshape(2, 3).tolist()
    noise_ids = phase_plan["selection"]["train"][:8]
    phases = [sp["phase_ids"][0], sp["phase_ids"][2]]
    with np.load(root / "draws.npz", allow_pickle=False) as source:
        left, right = source["left"][0, :4], source["right"][0, :4]
    files = [
        Path(__file__),
        Path(retained["checkpoint"]),
        root / "protocol.json",
        root / "draws.npz",
    ]
    plan = {
        "checkpoint": retained["checkpoint"],
        "source_case_ids": {
            s: [[cases[i]["case_id"] for i in batch] for batch in batches]
            for s, batches in groups.items()
        },
        "noise_train_ids": noise_ids,
        "noise_real_phase_ids": phases,
        "rim_label_pairs": [
            [rim_cases[int(a)]["case_id"], rim_cases[int(b)]["case_id"]]
            for a, b in zip(left, right)
        ],
        "offsets_pixels": [0.25, 0.0],
        "source_sha256": {str(f.resolve()): sha256(f) for f in files},
        "scope": "Local first-order CNN gradients on fixed TRAIN mini-batches. Negative dot with noise gradient predicts noise loss increase under a sufficiently small plain gradient step; Adam/history and population effects are not established.",
    }
    if any(
        cases[i]["split"] != "train"
        for batches in groups.values()
        for batch in batches
        for i in batch
    ):
        raise ValueError("Probe included a held-out case")
    output.mkdir(parents=True)
    write_json(output / "plan.json", plan)
    torch.set_num_threads(2)
    model, readout = load_checkpoint(retained["checkpoint"], "cpu")
    model.eval(), readout.eval()
    data, rim = FullCases(p, cases), SamplingCases(sp, rim_cases)
    noise, values = real_noise_gradient(model, rim, noise_ids, phases, p["delta_std"])
    references = {"raw_phase_Cl_noise": noise}
    references.update(rim_label_gradients(model, rim, left, right, p["delta_std"]))
    result = {
        "reference_gradients": {
            k: describe_gradient(g, references) for k, g in references.items()
        },
        "sources": {},
    }
    stored = {k: g.numpy() for k, g in references.items()}
    for source, batches in groups.items():
        rows, accum = [], []
        for number, batch in enumerate(batches):
            views, params, targets = batch_cases(data, batch, torch.device("cpu"))
            offsets = torch.full((len(batch), 2), 0.25)
            translated = translate_views(views, offsets)
            pooled = model.encode(
                {n: torch.cat((views[n], translated[n])) for n in VIEWS}
            )
            prediction = readout(pooled, params.repeat(2, 1))
            objectives = target_objectives(
                prediction, targets, data.weights[batch], p["loss_weights"]
            )
            cd = cnn_gradient(objectives[0], model, True)
            cl = cnn_gradient(objectives[1], model)
            accum.append((cd, cl))
            rows.append(
                {
                    "Cd": describe_gradient(cd, references),
                    "Cl": describe_gradient(cl, references),
                    "total": describe_gradient(cd + cl, references),
                }
            )
            stored[f"{source}_{number}_Cd"] = cd.numpy()
            stored[f"{source}_{number}_Cl"] = cl.numpy()
        cd, cl = (torch.stack([g[j] for g in accum]).mean(0) for j in range(2))
        result["sources"][source] = {
            "mean_Cd": describe_gradient(cd, references),
            "mean_Cl": describe_gradient(cl, references),
            "mean_total": describe_gradient(cd + cl, references),
            "batches": rows,
        }
        print(json.dumps({"source": source, **result["sources"][source]}), flush=True)
    np.savez_compressed(
        output / "gradients.npz", **stored, noise_physical_predictions=values
    )
    result.update(
        scope=plan["scope"],
        sampling_train_mae=np.abs(values).mean(0).tolist(),
        gradients_sha256=sha256(output / "gradients.npz"),
        checkpoint_unchanged=sha256(retained["checkpoint"]) == retained["sha256"],
    )
    write_json(output / "results.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.experiment, args.output)
