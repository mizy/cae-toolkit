"""Train both frozen full-pool sampling arms, retaining every epoch."""

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .delta_experiment import sha256, tensor_sha256, write_json
from .image_perturbation import translate_views
from .joint_sampling import AbsoluteReadout, FullCases, batch_cases, load
from .model import VIEWS
from .paired_delta import PairedDelta
from .sampling_experiment import SamplingCases, batch_views
from .sampling_experiment import load as load_sampling


def masked_case_mse(prediction, target, weights, observed):
    """Average observed coefficients per case, then apply source case weights."""
    safe_target = torch.where(observed, target, 0)
    square = (prediction - safe_target).square().masked_fill(~observed, 0)
    return (weights * square.sum(1) / observed.sum(1).clamp_min(1)).mean()


def absolute_objective(prediction, target, weights):
    nominal, augmented = prediction.chunk(2)
    observed = torch.isfinite(target)
    return {
        "absolute_nominal": masked_case_mse(nominal, target, weights, observed),
        "absolute_augmented": masked_case_mse(augmented, target, weights, observed),
        "absolute_consistency": masked_case_mse(
            augmented, nominal.detach(), weights, observed
        ),
    }


def rim_objective(model, encoded, parameters, targets, scale, teacher):
    """Supervise nominal, three shifted pairs, two self pairs and frozen teacher."""
    n = len(targets) // 2

    def predict(a, b, condition):
        return model.predict_pair(
            {k: v[a] for k, v in encoded.items()},
            {k: v[b] for k, v in encoded.items()},
            condition,
        )

    a, b, aa, bb = (
        torch.arange(i * n, (i + 1) * n, device=parameters.device) for i in range(4)
    )
    nominal = predict(a, b, parameters[:n])
    augmented = predict(
        torch.cat((aa, a, aa)), torch.cat((bb, bb, b)), parameters[:n].repeat(3, 1)
    )
    self_pair = predict(torch.cat((a, b)), torch.cat((aa, bb)), parameters)
    target = (targets[n:] - targets[:n]) / scale
    return {
        "rim_nominal": F.mse_loss(nominal, target),
        "rim_augmented": F.mse_loss(augmented, target.repeat(3, 1)),
        "rim_self": self_pair.square().mean(),
        "rim_consistency": F.mse_loss(augmented, nominal.detach().repeat(3, 1)),
        "rim_teacher": F.mse_loss(nominal, teacher),
    }


def save_checkpoint(path, model, readout, p, arm, epoch, source, protocol_sha256=None):
    """Own compatible delta checkpoints and their auxiliary-head provenance."""
    state = dict(source)
    state.pop("validation", None)
    for key, module in (("model_state", model), ("absolute_state", readout)):
        state[key] = {
            n: v.detach().cpu().clone() for n, v in module.state_dict().items()
        }
        if any(not torch.isfinite(v).all() for v in state[key].values()):
            raise FloatingPointError("Nonfinite checkpoint tensor")
    state.update(
        model_config={"pool_size": 8, "parameter_count": 36},
        absolute_tensor_sha256=tensor_sha256(readout),
        absolute_normalization=p["normalization"],
        delta_std=p["delta_std"],
        parameter_mean=p["parameter_mean"],
        parameter_std=p["parameter_std"],
        source_sha256=p["source_sha256"],
        protocol_sha256=protocol_sha256,
        coverage_floor=0.0,
        filter_input=False,
        arm=arm,
        epoch=epoch,
    )
    if path.exists():
        raise FileExistsError(f"Checkpoint already exists: {path}")
    torch.save(state, path)


def warm_start(p, cases):
    """@entry Fit the auxiliary head for twenty fixed train-only cached epochs."""
    root = Path(p["experiment"])
    if any((root / name).exists() for name in ("initial.pt", "warmup.json")):
        raise FileExistsError("Warm start already exists")
    torch.set_num_threads(2)
    torch.manual_seed(p["seed"])
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    source = torch.load(p["checkpoint"], map_location="cpu", weights_only=True)
    model = PairedDelta(8, 36).to(device)
    model.load_state_dict(source["model_state"], strict=True)
    model.eval().requires_grad_(False)
    if tensor_sha256(model) != p["initial_tensor_sha256"]:
        raise ValueError("Warm-start B differs from the frozen original")
    if p["warm_epochs"] != 20:
        raise ValueError("Warm start requires exactly twenty fixed epochs")
    data = FullCases(p, cases)
    indices = [data.index[i] for i in p["train_ids"]]
    if len(set(indices)) != len(indices) or set(indices) != {
        i for i, c in enumerate(cases) if c["split"] == "train"
    }:
        raise ValueError("Warm start must include all and only training records")
    keys = {i: (cases[i]["geometry_hash"], cases[i]["views_sha256"]) for i in indices}
    representatives = list({key: i for i, key in keys.items()}.values())
    cache, started = {}, time.monotonic()
    with torch.no_grad():
        for start in range(0, len(representatives), p["batch_size"]):
            current = representatives[start : start + p["batch_size"]]
            views, _, _ = batch_cases(data, current, device)
            pooled = model.encode(views)
            for k, i in enumerate(current):
                cache[keys[i]] = {n: pooled[n][k].detach().cpu().clone() for n in VIEWS}
            data.cache.clear()
    encoding_s = time.monotonic() - started
    readout = AbsoluteReadout().to(device).train()
    optimizer = torch.optim.Adam(readout.parameters(), lr=1e-3)
    rng, trace, history = np.random.default_rng(p["seed"]), hashlib.sha256(), []
    for epoch in range(1, 21):
        epoch_started, losses = time.monotonic(), []
        order = rng.permutation(indices).astype(np.int32)
        for start in range(0, len(order), p["batch_size"]):
            current = order[start : start + p["batch_size"]]
            pooled = {
                n: torch.stack([cache[keys[int(i)]][n] for i in current]).to(device)
                for n in VIEWS
            }
            target = data.standardized_targets[current].to(device)
            prediction = readout(pooled, data.parameters[current].to(device))
            loss = masked_case_mse(
                prediction,
                target,
                data.weights[current].to(device),
                torch.isfinite(target),
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite warm-start loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                readout.parameters(), 10, error_if_nonfinite=True
            )
            optimizer.step()
            losses.append((len(current), float(loss.detach())))
            trace.update(current.tobytes())
        record = {
            "epoch": epoch,
            "loss": sum(n * v for n, v in losses) / len(indices),
            "elapsed_s": time.monotonic() - epoch_started,
        }
        history.append(record)
        print(json.dumps({"warmup": record}), flush=True)
    if tensor_sha256(model) != p["initial_tensor_sha256"]:
        raise AssertionError("Warm start changed frozen B weights")
    save_checkpoint(root / "initial.pt", model, readout, p, "original_B", 0, source)
    write_json(
        root / "warmup.json",
        {
            "train_ids": p["train_ids"],
            "encoded_case_ids": [cases[i]["case_id"] for i in representatives],
            "encoded_unique_geometry_count": len(cache),
            "train_record_count": len(indices),
            "cached_float32_bytes": sum(
                v.numel() * v.element_size() for c in cache.values() for v in c.values()
            ),
            "trace_sha256": trace.hexdigest(),
            "history": history,
            "encoding_s": encoding_s,
            "elapsed_s": time.monotonic() - started,
            "initial_tensor_sha256": tensor_sha256(model),
            "initial_absolute_sha256": tensor_sha256(readout),
            "device": str(device),
            "selection": "Fixed twenty epochs; no held-out image encoding or epoch selection",
        },
    )


def full_backward(model, readout, data, indices, offsets, device, loss_weights, online):
    """Release the complete absolute-supervision graph before the Rim branch."""
    views, parameters, targets = batch_cases(data, indices, device)
    if online:
        augmented = translate_views(views, offsets)
        encoded = model.encode({n: torch.cat((views[n], augmented[n])) for n in VIEWS})
    else:
        encoded = {n: torch.cat((v, v)) for n, v in model.encode(views).items()}
    prediction = readout(encoded, parameters.repeat(2, 1))
    losses = absolute_objective(prediction, targets, data.weights[indices].to(device))
    total = sum(loss_weights[k] * v for k, v in losses.items())
    if not torch.isfinite(total):
        raise FloatingPointError("Nonfinite full-pool objective")
    total.backward()
    return {k: float(v.detach()) for k, v in losses.items()}


def rim_backward(model, data, left, right, offsets, teacher, device, p, online):
    requests = [(int(i), None) for i in np.concatenate((left, right))]
    views, parameters, targets = batch_views(data, requests, device)
    shift = torch.from_numpy(np.concatenate((offsets[:, 0], offsets[:, 1]))).to(device)
    if online:
        augmented = translate_views(views, shift)
        encoded = model.encode({n: torch.cat((views[n], augmented[n])) for n in VIEWS})
    else:
        encoded = {n: torch.cat((v, v)) for n, v in model.encode(views).items()}
    losses = rim_objective(
        model,
        encoded,
        parameters,
        targets,
        torch.tensor(p["delta_std"], dtype=torch.float32, device=device),
        torch.from_numpy(teacher).to(device),
    )
    total = sum(p["loss_weights"][k] * v for k, v in losses.items())
    if not torch.isfinite(total):
        raise FloatingPointError("Nonfinite Rim objective")
    total.backward()
    return {k: float(v.detach()) for k, v in losses.items()}


def training_draws(root, p, cases, rim_cases):
    """Verify exact inherited pair order and train-only full/endpoint membership."""
    with np.load(root / "draws.npz", allow_pickle=False) as stored:
        draws = {
            k: stored[k].copy()
            for k in ("full_order", "full_offsets", "left", "right", "rim_offsets")
        }
    with np.load(Path(p["sampling_parent"]) / "draws.npz", allow_pickle=False) as old:
        if any(not np.array_equal(draws[k], old[k]) for k in ("left", "right")):
            raise ValueError("Inherited Rim pair schedule changed")
    full_train = [i for i, c in enumerate(cases) if c["case_id"] in set(p["train_ids"])]
    rim_train = [i for i, c in enumerate(rim_cases) if c["split"] == "train"]
    for key, expected in (
        ("full_order", full_train),
        ("left", rim_train),
        ("right", rim_train),
    ):
        if draws[key].shape != (p["epochs"], len(expected)) or any(
            not np.array_equal(np.sort(row), expected) for row in draws[key]
        ):
            raise ValueError(f"Invalid train-only schedule: {key}")
    if any(cases[i]["split"] != "train" for i in full_train):
        raise ValueError("Full-pool schedule includes held-out records")
    for key, shape in (
        ("full_offsets", (*draws["full_order"].shape, 2)),
        ("rim_offsets", (*draws["left"].shape, 2, 2)),
    ):
        if (
            draws[key].shape != shape
            or not np.isfinite(draws[key]).all()
            or np.abs(draws[key]).max() > 0.5
        ):
            raise ValueError("Invalid frozen online offsets")
    with np.load(
        Path(p["sampling_parent"]) / "teacher.npz", allow_pickle=False
    ) as stored:
        teacher = stored["prediction"].copy()
        if set(stored["encoded_train_indices"].tolist()) != set(rim_train):
            raise ValueError(
                "Teacher encoded indices differ from Rim training membership"
            )
    if teacher.shape != (*draws["left"].shape, 2) or not np.isfinite(teacher).all():
        raise ValueError("Invalid frozen teacher predictions")
    if (
        hashlib.sha256(draws["full_order"].astype(np.int32).tobytes()).hexdigest()
        != p["full_trace_sha256"]
    ):
        raise ValueError("Frozen full-pool order differs")
    pair_sha = hashlib.sha256(
        np.stack((draws["left"], draws["right"]), -1).astype(np.int32).tobytes()
    ).hexdigest()
    if pair_sha != p["pair_trace_sha256"]:
        raise ValueError("Frozen pair trace differs")
    return draws, teacher


def train(root, arm):
    """@entry Run one fresh arm, preserving epoch00..06 without model selection."""
    root = Path(root)
    p, cases = load(root)
    if arm not in p["arms"]:
        raise ValueError("Expected control or online arm")
    output = root / arm
    if output.exists():
        raise FileExistsError("Arm output already exists")
    sp, rim_cases = load_sampling(p["sampling_parent"])
    draws, teacher = training_draws(root, p, cases, rim_cases)
    torch.set_num_threads(2)
    torch.manual_seed(p["seed"])
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    initial = torch.load(root / "initial.pt", map_location="cpu", weights_only=True)
    model, readout = PairedDelta(8, 36).to(device), AbsoluteReadout().to(device)
    model.load_state_dict(initial["model_state"], strict=True)
    readout.load_state_dict(initial["absolute_state"], strict=True)
    if (
        tensor_sha256(model) != p["initial_tensor_sha256"]
        or tensor_sha256(readout) != p["initial_absolute_sha256"]
    ):
        raise ValueError("Arm initialization differs from the frozen common state")
    model.train(), readout.train()
    cnn, pair_head = [], []
    for name, parameter in model.named_parameters():
        (
            cnn if name.startswith(("image_encoder.", "bottom_encoder.")) else pair_head
        ).append(parameter)
    optimizer = torch.optim.Adam(
        [
            {"params": cnn, "lr": p["cnn_lr"]},
            {"params": pair_head, "lr": p["head_lr"]},
            {"params": readout.parameters(), "lr": p["head_lr"]},
        ]
    )
    data, rim = FullCases(p, cases), SamplingCases(sp, rim_cases, with_phases=False)
    protocol_sha = sha256(root / "protocol.json")
    output.mkdir()
    save_checkpoint(
        output / "epoch-00.pt", model, readout, p, arm, 0, initial, protocol_sha
    )
    full_trace, pair_trace, offset_trace = (
        hashlib.sha256(),
        hashlib.sha256(),
        hashlib.sha256(),
    )
    full_counts, rim_counts = Counter(), Counter()
    report = {
        "arm": arm,
        "history": [],
        "optimizer_steps": 0,
        "device": str(device),
        "initial_tensor_sha256": tensor_sha256(model),
        "initial_absolute_sha256": tensor_sha256(readout),
        "protocol_sha256": protocol_sha,
        "logical_image_exposures": {"full": 0, "rim": 0},
        "actual_cnn_image_encodings": {"full": 0, "rim": 0},
        "selection": "No selection during training; retain every epoch for separate validation",
    }
    started = time.monotonic()
    for epoch in range(p["epochs"]):
        epoch_started, totals = time.monotonic(), Counter()
        order = draws["full_order"][epoch]
        steps = (len(order) + p["batch_size"] - 1) // p["batch_size"]
        boundaries = np.linspace(0, len(draws["left"][epoch]), steps + 1).astype(int)
        for step, start in enumerate(range(0, len(order), p["batch_size"])):
            indices = order[start : start + p["batch_size"]]
            offsets = draws["full_offsets"][epoch, start : start + len(indices)]
            a, b = boundaries[step : step + 2]
            left, right = draws["left"][epoch, a:b], draws["right"][epoch, a:b]
            optimizer.zero_grad(set_to_none=True)
            parts = full_backward(
                model,
                readout,
                data,
                indices,
                torch.from_numpy(offsets).to(device),
                device,
                p["loss_weights"],
                arm == "online",
            )
            if len(left):
                parts.update(
                    rim_backward(
                        model,
                        rim,
                        left,
                        right,
                        draws["rim_offsets"][epoch, a:b],
                        teacher[epoch, a:b],
                        device,
                        p,
                        arm == "online",
                    )
                )
            norm = torch.nn.utils.clip_grad_norm_(
                [*model.parameters(), *readout.parameters()],
                p["clip_norm"],
                error_if_nonfinite=True,
            )
            if norm == 0:
                raise FloatingPointError("All joint gradients are zero")
            optimizer.step()
            totals.update(parts)
            report["optimizer_steps"] += 1
            full_trace.update(indices.astype(np.int32).tobytes())
            pair_trace.update(np.stack((left, right), -1).astype(np.int32).tobytes())
            offset_trace.update(
                (offsets if arm == "online" else np.zeros_like(offsets)).tobytes()
            )
            rim_offsets = draws["rim_offsets"][epoch, a:b]
            offset_trace.update(
                (
                    rim_offsets if arm == "online" else np.zeros_like(rim_offsets)
                ).tobytes()
            )
            full_counts.update(cases[int(i)]["case_id"] for i in indices)
            rim_counts.update(
                rim_cases[int(i)]["case_id"] for i in np.concatenate((left, right))
            )
            report["logical_image_exposures"]["full"] += 2 * len(indices)
            report["logical_image_exposures"]["rim"] += 4 * len(left)
            report["actual_cnn_image_encodings"]["full"] += (
                2 if arm == "online" else 1
            ) * len(indices)
            report["actual_cnn_image_encodings"]["rim"] += (
                4 if arm == "online" else 2
            ) * len(left)
        record = {
            "epoch": epoch + 1,
            "loss_components": {k: v / steps for k, v in totals.items()},
            "elapsed_s": time.monotonic() - epoch_started,
        }
        report["history"].append(record)
        report.update(
            executed_full_trace_sha256=full_trace.hexdigest(),
            executed_pair_sha256=pair_trace.hexdigest(),
            actual_offset_trace_sha256=offset_trace.hexdigest(),
            full_case_exposures=dict(full_counts),
            perturbed_full_case_exposures={
                i: n if arm == "online" else 0 for i, n in full_counts.items()
            },
            rim_endpoint_exposures=dict(rim_counts),
            rim_augmented_endpoint_exposures={
                i: n if arm == "online" else 0 for i, n in rim_counts.items()
            },
            elapsed_s=time.monotonic() - started,
        )
        save_checkpoint(
            output / f"epoch-{epoch + 1:02d}.pt",
            model,
            readout,
            p,
            arm,
            epoch + 1,
            initial,
            protocol_sha,
        )
        write_json(output / "training.json", report)
        print(json.dumps({"arm": arm, **record}), flush=True)
    if (
        report["optimizer_steps"] != p["training_steps"]
        or full_trace.hexdigest() != p["full_trace_sha256"]
        or pair_trace.hexdigest() != p["pair_trace_sha256"]
    ):
        raise AssertionError(
            "Executed training schedule differs from the frozen protocol"
        )
    if any(n != p["epochs"] for n in full_counts.values()) or any(
        n != 2 * p["epochs"] for n in rim_counts.values()
    ):
        raise AssertionError("Unexpected full-pool or Rim exposure count")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    command = sub.add_parser("train")
    command.add_argument("root", type=Path)
    command.add_argument("--arm", choices=("control", "online"), required=True)
    args = parser.parse_args()
    train(args.root, args.arm)
