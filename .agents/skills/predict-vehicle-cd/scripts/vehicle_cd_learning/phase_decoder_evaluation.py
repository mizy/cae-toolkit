"""Select on real independent validation phases, then open the historical and off-grid test diagnostics."""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from .delta_experiment import sha256, write_json
from .joint_sampling import load_checkpoint
from .phase_decoder_scores import (
    groups_and_pairs,
    pair_grid,
    patterns,
    stability,
    summarize,
)
from .phase_decoder_train import ARMS, dataset, feature_bank
from .phase_views import phase_id
from .sampling_experiment import SamplingCases
from .sampling_validation import score_checkpoint


def check_trials(root, plan):
    traces = set()
    for arm in ARMS:
        run = json.loads((root / arm / "training.json").read_text())
        visits = run["true_pair_endpoint_exposures"]
        if (
            run["updates"] != 69
            or set(visits) != set(plan["train_nominal_case_ids"])
            or set(visits.values()) != {6}
        ):
            raise ValueError("Real geometry pair budget mismatch")
        allowed = set(plan["train_phase_case_ids"]) if arm == "real_phase" else set()
        if set(run["actual_phase_train_ids"]) != allowed:
            raise ValueError("Phase training membership mismatch")
        traces.add(run["actual_draw_sha256"])
    if len(traces) != 1:
        raise ValueError("Training schedules differ")


def sampled_guard(values, reference, nself):
    """Compare the distribution in every prespecified pattern against E5 in that pattern."""
    counts = {}
    for name, old, new in [
        ("self", reference[:, :nself], values[:, :nself]),
        (
            "drift",
            reference[:, nself:] - reference[:1, nself:],
            values[:, nself:] - values[:1, nself:],
        ),
    ]:
        for metric in ("mae", "p95"):
            reduce = (
                (lambda x: np.abs(x).mean(1))
                if metric == "mae"
                else (lambda x: np.quantile(np.abs(x), 0.95, axis=1))
            )
            base, trial = reduce(old), reduce(new)
            counts[name + "_" + metric] = ((trial > base * 1.05 + 1e-8).sum(0)).tolist()
    return {
        "pass": all(sum(c) == 0 for c in counts.values()),
        "failed_patterns_by_target": counts,
    }


def compare_scope(
    entries,
    ids,
    queries,
    anchors,
    bank,
    grid,
    labels,
    params,
    scale,
    device,
    output,
    title,
    store_arrays,
):
    grouped = groups_and_pairs(queries, anchors)
    pairs = list(itertools.chain.from_iterable(grouped.values()))
    results = {}
    reference = None
    for name, path in entries.items():
        model, _ = load_checkpoint(path, device)
        model.eval()
        with torch.inference_mode():
            values = pair_grid(model, bank, ids, pairs, grid, params, scale)
        if reference is None:
            reference = values
        metric = summarize(
            values, grid, grouped, labels, reference, strict_patterns=True
        )
        metric["stability"] = stability(metric, scale)
        metric["sampling_guard"] = sampled_guard(values, reference, len(queries))
        # Include the pooled three-anchor response, alongside the separate anchor gates.
        a_start = len(queries) + len(grouped["pairs"])
        target = np.array(
            [np.asarray(labels[b]) - labels[a] for a, b in pairs[a_start:]]
        )
        from .delta_metrics import score_delta
        from .phase_metrics import preservation_gate

        gate_count = 0
        for j in range(len(grid)):
            trial = score_delta(target, values[j, a_start:])
            base = score_delta(target, reference[j, a_start:])
            if not preservation_gate(base, trial, min_amplitude_fraction=0.95)["pass"]:
                gate_count += 1
            if j == 0:
                metric["pooled_anchors"] = trial
        metric["pooled_anchor_failed_patterns"] = gate_count
        results[name] = metric
        if store_arrays:
            np.savez_compressed(
                output / f"{title}-{name}.npz",
                prediction=values,
                patterns=grid,
                pairs=np.asarray(pairs),
            )
        print(
            title,
            name,
            metric["stability"]["worst_threshold_multiple"],
            metric["response_failures"],
            flush=True,
        )
    return results


def run(root):
    plan = json.loads((root / "protocol.json").read_text())
    check_trials(root, plan)
    paths = plan["code_sha256"]
    for f, expected in paths.items():
        if sha256(f) != expected:
            raise ValueError("Training source changed")
    run_root = Path(plan["experiment"])
    data_root = Path(plan["phases"])
    _, sp, cases, paths = dataset(data_root, run_root)
    if sha256(data_root / "manifest.json") != plan["geometry_manifest_sha256"]:
        raise ValueError("Data changed")
    source = Path(plan["initial_checkpoint"])
    val = list(plan["validation_phase_case_ids"])
    anchors = sp["anchors"]
    validation_states = [None] + [
        phase_id(x)
        for x in plan["train_phase_pixels"] + plan["additional_validation_phase_pixels"]
    ]
    prep = json.loads((data_root / "plan.json").read_text())
    off = prep["off_grid_test_ids"]
    test_states = [None, phase_id((0.25, 0))] + [
        phase_id(x) for x in prep["off_grid_phases"]
    ]
    eval_plan = {
        "validation_ids": val,
        "validation_states": validation_states,
        "validation_mixed_draw_seed": 42,
        "test_ids": off,
        "test_states": test_states,
        "no_test_selection": True,
        "engineering_scales": sp["delta_std"],
        "source_sha256": {
            str(f.resolve()): sha256(f)
            for f in (
                Path(__file__),
                root / "protocol.json",
                data_root / "plan.json",
                source,
            )
        },
    }
    out = root / "evaluation"
    out.mkdir(exist_ok=False)
    write_json(out / "plan.json", eval_plan)
    torch.set_num_threads(2)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    initial, _ = load_checkpoint(source, device)
    initial.eval().requires_grad_(False)
    labels = {c["case_id"]: c["targets"] for c in cases}
    original_data = SamplingCases(sp, cases)
    params = original_data.parameters[0].to(device)
    scale = sp["delta_std"]
    selected = {"epoch5": source}
    for arm in ARMS:
        for epoch in range(1, 4):
            selected[f"{arm}-{epoch}"] = root / arm / f"epoch-{epoch:02}.pt"
    nominal = plan["validation_nominal_case_ids"]
    ids = anchors + nominal
    bank = feature_bank(initial, paths, ids, [None], device)
    nominal_scores = compare_scope(
        selected,
        ids,
        nominal,
        anchors,
        bank,
        patterns(1, 0),
        labels,
        params,
        scale,
        device,
        out,
        "nominal-val",
        False,
    )
    ids = anchors + val
    bank = feature_bank(initial, paths, ids, validation_states, device)
    val_scores = compare_scope(
        selected,
        ids,
        val,
        anchors,
        bank,
        patterns(len(validation_states)),
        labels,
        params,
        scale,
        device,
        out,
        "phase-val",
        True,
    )
    passing = []
    for name, rec in val_scores.items():
        rec["full_nominal_preserved"] = (
            not nominal_scores[name]["response_failures"]
            and not nominal_scores[name]["pooled_anchor_failed_patterns"]
        )
        rec["passes_joint"] = bool(
            rec["full_nominal_preserved"]
            and not rec["response_failures"]
            and not rec["pooled_anchor_failed_patterns"]
            and rec["sampling_guard"]["pass"]
            and rec["stability"]["passes"]
        )
        if rec["passes_joint"] and name != "epoch5":
            passing.append(name)
    rank = lambda n: (val_scores[n]["stability"]["worst_threshold_multiple"], n)
    winner = min(passing, key=rank) if passing else "epoch5"
    diag = min([n for n in selected if n.startswith("real_phase-")], key=rank)
    matched = diag.replace("real_phase", "nominal")
    write_json(
        out / "validation.json", {"nominal": nominal_scores, "phase": val_scores}
    )
    policy = {
        "chosen": winner,
        "eligible": passing,
        "diagnostic": diag,
        "matched_control": matched,
        "baseline_sha256": sha256(source),
        "validation_sha256": sha256(out / "validation.json"),
        "selection_scope": "Validation only; rejected diagnostic does not replace retained epoch5.",
    }
    write_json(out / "selection.json", policy)
    representatives = {
        n: selected[n] for n in dict.fromkeys(["epoch5", diag, matched, winner])
    }
    write_json(
        out / "test-plan.json",
        {n: {"path": str(f), "sha256": sha256(f)} for n, f in representatives.items()},
    )
    ids = anchors + off
    bank = feature_bank(initial, paths, ids, test_states, device)
    tests = compare_scope(
        representatives,
        ids,
        off,
        anchors,
        bank,
        patterns(len(test_states), exhaustive_singles=True),
        labels,
        params,
        scale,
        device,
        out,
        "offgrid-test",
        True,
    )
    history = json.loads((Path(sp["phase_root"]) / "plan.json").read_text())[
        "selection"
    ]
    standard = {}
    for name, f in representatives.items():
        model, _ = load_checkpoint(f, device)
        model.eval()
        stats = score_checkpoint(
            model,
            original_data,
            sp,
            cases,
            history["all_test"],
            [phase_id((0.25, 0))],
            split="test",
        )
        arrays = stats.pop("phase_scoring")
        np.savez_compressed(
            out / f"main-test-{name}.npz",
            **{k: np.asarray(v) for k, v in arrays.items()},
        )
        standard[name] = {
            "nominal_anchors": stats["nominal_anchors"],
            "patterns": stats["patterns"],
        }
    write_json(
        out / "comparison.json",
        {"selection": policy, "off_grid": tests, "historical_main": standard},
    )


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("root", type=Path)
    run(cli.parse_args().root)
