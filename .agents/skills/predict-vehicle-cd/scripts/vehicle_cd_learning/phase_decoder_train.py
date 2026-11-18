"""Matched three-epoch decoder continuations on genuine full-view sampling changes."""

import argparse
import copy
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .delta_experiment import sha256, write_json
from .joint_sampling import load, load_checkpoint
from .model import VIEWS
from .phase_views import phase_id
from .sampling_experiment import load as load_sampling

ARMS = ("nominal", "real_phase")


def dataset(data_root, experiment):
    p, _ = load(experiment)
    sp, cases = load_sampling(p["sampling_parent"])
    prepared = json.loads((data_root / "plan.json").read_text())
    raw = json.loads((data_root / "manifest.json").read_text())
    if raw["plan_sha256"] != sha256(data_root / "plan.json"):
        raise ValueError("Phase preparation changed")
    old_root = Path(sp["phase_root"])
    old = json.loads((old_root / "dataset.json").read_text())
    paths = {
        (r["source_case_id"], r["phase_id"]): (old_root / r["views"], r["views_sha256"])
        for r in old["cases"]
    }
    for cid, views in raw["views"].items():
        paths.update(
            {
                (cid, name): (Path(row["path"]), row["sha256"])
                for name, row in views.items()
            }
        )
    nominal_root = Path(prepared["nominal_root"])
    for c in json.loads((nominal_root / "dataset.json").read_text())["cases"]:
        paths[(c["case_id"], None)] = (nominal_root / c["views"], c["views_sha256"])
    return p, sp, cases, paths


def feature_bank(model, paths, ids, phases, device):
    samples = [(cid, ph) for cid in ids for ph in phases]
    values = {}
    with torch.inference_mode():
        for start in range(0, len(samples), 8):
            rows = []
            for key in samples[start : start + 8]:
                path, expected = paths[key]
                if sha256(path) != expected:
                    raise ValueError(f"Raw phase changed: {key}")
                with np.load(path, allow_pickle=False) as f:
                    rows.append({n: torch.from_numpy(f[n]) for n in VIEWS})
            inputs = {n: torch.stack([r[n] for r in rows]).to(device) for n in VIEWS}
            for k, value in model.encode(inputs).items():
                values.setdefault(k, []).append(value.cpu())
    # Clone outside inference_mode before using tensors as frozen inputs to autograd.
    return {
        k: torch.cat(rows).reshape(len(ids), len(phases), *rows[0].shape[1:]).clone()
        for k, rows in values.items()
    }


def mixed(bank, indices, phases):
    return {
        k: bank[k][indices, phases[:, j]]
        for j, n in enumerate(VIEWS)
        for k in (n, n + "_mask")
    }


def draws(plan, sp, cases):
    ids = plan["train_nominal_case_ids"]
    index = {c: i for i, c in enumerate(ids)}
    with np.load(Path(sp["experiment"]) / "draws.npz", allow_pickle=False) as previous:
        a, b = (
            np.array(
                [[index[cases[i]["case_id"]] for i in row] for row in previous[k][:3]],
                np.int32,
            )
            for k in ("left", "right")
        )
    rng = np.random.default_rng(plan["seed"])
    count = len(plan["train_phase_case_ids"])
    states = len(plan["train_phase_pixels"])
    u = rng.integers(count, size=a.shape, dtype=np.int32)
    v = (u + rng.integers(1, count, size=a.shape, dtype=np.int32)) % count
    phase = rng.integers(1, states + 1, size=(*a.shape, 2, 5), dtype=np.int32)
    # Half single-view (all five represented), one quarter all views share the offset,
    # one quarter independent. Half the physical pairs use a common endpoint pattern.
    mode = np.arange(a.size).reshape(a.shape) % 4
    for side in range(2):
        single = rng.integers(5, size=a.shape)
        select = np.arange(5) == single[:, :, None]
        phase[:, :, side] = np.where(
            (mode < 2)[:, :, None] & ~select, 0, phase[:, :, side]
        )
        phase[:, :, side] = np.where(
            (mode == 2)[:, :, None], phase[:, :, side, :1], phase[:, :, side]
        )
    common = rng.random(a.shape) < 0.5
    phase[:, :, 1] = np.where(common[:, :, None], phase[:, :, 0], phase[:, :, 1])
    return {"a": a, "b": b, "u": u, "v": v, "phase": phase}


def losses(model, teacher, parameters, bank, local, truth, local_truth, draw, online):
    a, b, u, v = (draw[k] for k in ("a", "b", "u", "v"))
    phases = draw["phase"]
    zeros = torch.zeros((len(a), 5), dtype=torch.long, device=parameters.device)
    aa, bb = mixed(bank, a, zeros), mixed(bank, b, zeros)
    lu, lv = mixed(local, u, zeros), mixed(local, v, zeros)
    au = mixed(local, u, phases[:, 0]) if online else lu
    av = mixed(local, v, phases[:, 1]) if online else lv
    params = parameters.expand(len(a), -1)
    with torch.no_grad():
        nominal_teacher = teacher.predict_pair(aa, bb, params)
    nominal = model.predict_pair(aa, bb, params)
    local_nominal = model.predict_pair(lu, lv, params)

    def joined(*items):
        return {k: torch.cat([r[k] for r in items]) for k in items[0]}

    altered = model.predict_pair(
        joined(au, lu, au), joined(av, av, lv), params.repeat(3, 1)
    )
    self_pair = model.predict_pair(joined(lu, lv), joined(au, av), params.repeat(2, 1))
    return {
        "true_delta_CdCl": 0.5
        * (
            F.mse_loss(nominal, truth[b] - truth[a])
            + F.mse_loss(altered, (local_truth[v] - local_truth[u]).repeat(3, 1))
        ),
        "epoch5_nominal_fidelity_CdCl": F.mse_loss(nominal, nominal_teacher),
        "real_phase_self_Cl": self_pair[:, 1].square().mean(),
        "real_phase_pair_consistency_CdCl": F.mse_loss(
            altered, local_nominal.detach().repeat(3, 1)
        ),
    }


def constrain(model, teacher, fraction):
    """Constrain each trainable module's displacement relative to the retained decoder."""
    with torch.no_grad():
        for m, t in [
            (model.adapter, teacher.adapter),
            (model.head[0], teacher.head[0]),
        ]:
            pairs = list(zip(m.parameters(), t.parameters()))
            delta = sum((w - ref).square().sum() for w, ref in pairs).sqrt()
            limit = fraction * sum(ref.square().sum() for _, ref in pairs).sqrt()
            factor = (limit / delta.clamp_min(1e-20)).clamp_max(1)
            for w, ref in pairs:
                w.copy_(ref + (w - ref) * factor)


def train_arm(
    output, plan, ck, teacher, bank, local, truth, local_truth, schedule, params, arm
):
    folder = output / arm
    folder.mkdir(exist_ok=False)
    model = copy.deepcopy(teacher)
    model.requires_grad_(False)
    model.adapter.requires_grad_(True)
    model.head[0].requires_grad_(True)
    model.train()
    optimizer = torch.optim.Adam(
        [w for w in model.parameters() if w.requires_grad], lr=plan["learning_rate"]
    )
    size = plan["batch_size_pairs"]
    device = params.device
    history = []
    start = time.monotonic()
    counts = Counter()
    phase_ids = set()
    executed = hashlib.sha256()
    for e in range(plan["epochs_each"]):
        sums = Counter()
        for i in range(0, len(truth), size):
            host = {k: v[e, i : i + size] for k, v in schedule.items()}
            draw = {
                k: torch.from_numpy(v).to(device=device, dtype=torch.long)
                for k, v in host.items()
            }
            optimizer.zero_grad(set_to_none=True)
            terms = losses(
                model,
                teacher,
                params,
                bank,
                local,
                truth,
                local_truth,
                draw,
                arm == "real_phase",
            )
            loss = sum(plan["loss_weights"][k] * val for k, val in terms.items())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [w for w in model.parameters() if w.requires_grad],
                10,
                error_if_nonfinite=True,
            )
            optimizer.step()
            constrain(model, teacher, plan["maximum_relative_module_displacement"])
            for k, val in terms.items():
                sums[k] += len(host["a"]) * float(val.detach())
            counts.update(
                plan["train_nominal_case_ids"][i]
                for i in np.concatenate([host["a"], host["b"]])
            )
            executed.update(
                np.stack((host["a"], host["b"], host["u"], host["v"]), axis=-1)
                .astype(np.int32)
                .tobytes()
            )
            if arm == "real_phase":
                for j, key in enumerate(("u", "v")):
                    phase_ids.update(
                        plan["train_phase_case_ids"][i]
                        for i in host[key][np.any(host["phase"][:, j] > 0, axis=1)]
                    )
        record = {
            "epoch": e + 1,
            "loss": {k: v / len(truth) for k, v in sums.items()},
            "elapsed_s": time.monotonic() - start,
        }
        history.append(record)
        state = dict(ck)
        state.update(
            model_state={
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            },
            arm=arm,
            epoch=e + 1,
            parent_checkpoint_sha256=plan["initial_checkpoint_sha256"],
            protocol_sha256=sha256(output / "protocol.json"),
        )
        for name, val in state["model_state"].items():
            if not name.startswith(("adapter.", "head.0.")) and not torch.equal(
                val, ck["model_state"][name]
            ):
                raise ValueError("A frozen parameter changed")
        torch.save(state, folder / f"epoch-{e + 1:02}.pt")
        print(arm, record, flush=True)
    write_json(
        folder / "training.json",
        {
            "arm": arm,
            "history": history,
            "updates": len(history) * ((len(truth) + size - 1) // size),
            "true_pair_endpoint_exposures": dict(counts),
            "actual_draw_sha256": executed.hexdigest(),
            "actual_phase_train_ids": sorted(phase_ids),
            "absolute_output_frozen": True,
            "changed_tensors": [
                n
                for n, v in state["model_state"].items()
                if not torch.equal(v, ck["model_state"][n])
            ],
        },
    )


def train(experiment, phase_root, output):
    _, sp, cases, paths = dataset(phase_root, experiment)
    prep = json.loads((phase_root / "plan.json").read_text())
    design_path = Path(prep["design"])
    if sha256(design_path) != prep["design_sha256"]:
        raise ValueError("Frozen design changed")
    plan = json.loads(design_path.read_text())
    source = Path(plan["initial_checkpoint"])
    if sha256(source) != plan["initial_checkpoint_sha256"]:
        raise ValueError("Retained decoder changed")
    ids, subset = plan["train_nominal_case_ids"], plan["train_phase_case_ids"]
    train_ids = {c["case_id"] for c in cases if c["split"] == "train"}
    if (
        set(ids) != train_ids
        or len(ids) != 720
        or not set(subset) <= train_ids - set(sp["anchors"])
    ):
        raise ValueError("Unexpected training membership")
    scale = np.asarray(sp["delta_std"])
    labels = {c["case_id"]: c["targets"] for c in cases}
    schedule = draws(plan, sp, cases)
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "draws.npz", **schedule)
    plan.update(
        experiment=str(experiment.resolve()),
        phases=str(phase_root.resolve()),
        arms=list(ARMS),
        status="frozen three-epoch experimental protocol",
        draw_sha256=sha256(output / "draws.npz"),
        geometry_manifest_sha256=sha256(phase_root / "manifest.json"),
        code_sha256={
            str(f.resolve()): sha256(f)
            for f in (
                Path(__file__),
                Path(__file__).with_name("phase_decoder_scores.py"),
            )
        },
        objective="Average original CFD pair and mixed augmented local CFD pair MSE; weights, displacement limits and nominal fidelity inherited. All five single views and independent multi-view phases included.",
        acceptance="Worst per-sample drift mean <=0.05 and p95 <=0.10 of original train delta scale per target; each phase self/drift <=105% of E5; nominal and shifted geometry preservation relative to E5. Validation-only selection; show best rejected attempted epoch if none pass.",
    )
    write_json(output / "protocol.json", plan)
    torch.set_num_threads(2)
    torch.manual_seed(42)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    teacher, _ = load_checkpoint(source, device)
    teacher.eval().requires_grad_(False)
    bank = feature_bank(teacher, paths, ids, [None], device)
    local = feature_bank(
        teacher,
        paths,
        subset,
        [None] + [phase_id(q) for q in plan["train_phase_pixels"]],
        device,
    )
    bank = {k: v.to(device) for k, v in bank.items()}
    local = {k: v.to(device) for k, v in local.items()}
    ck = torch.load(source, map_location="cpu", weights_only=True)
    raw = np.asarray(cases[0]["parameters"], float)
    known = np.isfinite(raw)
    if any(c["parameters"] != cases[0]["parameters"] for c in cases):
        raise ValueError("Expected identical physical conditions")
    params = torch.tensor(
        np.r_[
            np.where(
                known,
                (raw - np.asarray(sp["parameter_mean"][: len(raw)]))
                / sp["parameter_std"][: len(raw)],
                0,
            ),
            known,
        ],
        dtype=torch.float32,
        device=device,
    )
    truth = torch.tensor(
        np.array([labels[i] for i in ids]) / scale, dtype=torch.float32, device=device
    )
    local_truth = torch.tensor(
        np.array([labels[i] for i in subset]) / scale,
        dtype=torch.float32,
        device=device,
    )
    for arm in ARMS:
        train_arm(
            output,
            plan,
            ck,
            teacher,
            bank,
            local,
            truth,
            local_truth,
            schedule,
            params,
            arm,
        )


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for n in ("experiment", "phases", "output"):
        cli.add_argument(n, type=Path)
    a = cli.parse_args()
    train(a.experiment, a.phases, a.output)
