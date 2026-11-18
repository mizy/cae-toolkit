"""Validation-only stage restoration with one fixed E5 encoder feature cache.

No weights are saved. Component restoration probes neural co-adaptation and
nonlinear interactions; it cannot attribute aerodynamic force contributions.
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from . import sampling_experiment as sampling
from .cl_sampling_diagnosis import (
    CHECKPOINTS,
    compact,
    phase_distribution,
    score_from_features,
    view_scores,
)
from .cl_sampling_repair import EXPECTED_CHECKPOINT, encode_requests
from .paired_delta import PairedDelta
from .sampling_validation import candidate_gates

GROUPS = ("adapter", "parameter_encoder", "head_hidden", "Cl_row")
DEFAULT_OUT = Path("outputs/aerodynamic-cl-readout-diagnosis-20261117")
OLD_DIAGNOSIS = Path("outputs/aerodynamic-cl-sampling-diagnosis-20261117")


def restored_state(original, baseline, groups, full_readout=False):
    """Copy exact tensors, preserving E5 encoder and normally its Cd final row."""
    state = {name: value.clone() for name, value in original.items()}
    prefixes = {
        "adapter": "adapter.",
        "parameter_encoder": "parameter_encoder.",
        "head_hidden": "head.0.",
    }
    for group, prefix in prefixes.items():
        if group in groups or full_readout:
            for name in state:
                if name.startswith(prefix):
                    state[name] = baseline[name].clone()
    if "Cl_row" in groups or full_readout:
        state["head.2.weight"][1] = baseline["head.2.weight"][1]
    if full_readout:
        state["head.2.weight"][0] = baseline["head.2.weight"][0]
    return state


def owner_metadata(groups, full_readout=False):
    return {
        "image_encoder": "E5",
        "bottom_encoder": "E5",
        **{name: "B" if name in groups or full_readout else "E5" for name in GROUPS},
        "Cd_row": "B" if full_readout else "E5",
        "absolute_head": "excluded",
        "restored_groups": list(groups),
        "diagnostic_only": True,
    }


def score_summary(metrics, views, phases):
    phase = metrics["phase_scoring"]
    return {
        **compact(metrics),
        "self_distribution": phase_distribution(phase["self_phase_pred"], phases),
        "single_view": {
            name: {k: value for k, value in row.items() if k != "predictions"}
            for name, row in views.items()
        },
    }


def contrasts(summaries):
    """Exact factorial contrasts, conditional on all other stage owners."""
    output = {}
    for group in GROUPS:
        rows = []
        other = [name for name in GROUPS if name != group]
        for mask in itertools.product((False, True), repeat=3):
            selected = [name for name, enabled in zip(other, mask) if enabled]
            a, b = variant_name(selected), variant_name(selected + [group])
            values = {}
            for target in ("Cd", "Cl"):
                before = summaries[a]["self_distribution"]
                after = summaries[b]["self_distribution"]
                values[target] = {
                    "all_phase_mae_change": after["pooled"][target]["mae"]
                    - before["pooled"][target]["mae"],
                    "positive_x_mae_change": after["by_phase"]["xp0d25yp0"][target][
                        "mae"
                    ]
                    - before["by_phase"]["xp0d25yp0"][target]["mae"],
                    "positive_x_signed_mean_change": after["by_phase"]["xp0d25yp0"][
                        target
                    ]["signed_mean"]
                    - before["by_phase"]["xp0d25yp0"][target]["signed_mean"],
                }
            rows.append({"before": a, "after": b, "metrics": values})
        output[group] = rows
    return output


def variant_name(groups):
    return (
        "restore_" + "+".join(name for name in GROUPS if name in groups)
        if groups
        else "original_E5"
    )


def prepare(output, prior_root=OLD_DIAGNOSIS):
    """Freeze validation-only ownership and old artifacts before encoding."""
    if output.exists():
        raise FileExistsError("Readout diagnosis output must be fresh")
    old_plan = json.loads((prior_root / "plan.json").read_text())
    for path, expected in old_plan["source_sha256"].items():
        if sampling.sha256(path) != expected:
            raise ValueError("Prior diagnosis source changed: " + path)
    paths = list(Path(__file__).parent.glob("*.py")) + list(prior_root.glob("*.json"))
    paths += [Path(CHECKPOINTS[name]) for name in ("B", "E5")]
    hashes = {str(path.resolve()): sampling.sha256(path) for path in paths}
    if sampling.sha256(CHECKPOINTS["E5"]) != EXPECTED_CHECKPOINT:
        raise ValueError("Retained E5 checkpoint identity changed")
    variants = {
        variant_name([g for g, bit in zip(GROUPS, mask) if bit]): owner_metadata(
            [g for g, bit in zip(GROUPS, mask) if bit]
        )
        for mask in itertools.product((False, True), repeat=4)
    }
    variants["full_B_readout"] = owner_metadata(GROUPS, full_readout=True)
    plan = {
        "validation_case_ids": old_plan["validation_case_ids"],
        "phase_ids": old_plan["phase_ids"],
        "anchors": old_plan["anchors"],
        "preservation": old_plan["preservation"],
        "stability": old_plan["stability"],
        "variants": variants,
        "source_sha256": hashes,
        "primary_diagnostic": "positive-x quarter-pixel same-geometry Cl, signed bias and bottom-only response",
        "scope": "Fixed E5 encoder, validation only: 92 nominal query geometries, 3 train anchors and 12x4 raw phases. No training, held-out scoring, model selection, or checkpoint publication. Cd final row stays E5 except explicit full_B_readout control. Component co-adaptation limits interpretation; single-view outputs are nonlinear and cannot be added as forces.",
    }
    output.mkdir(parents=True)
    sampling.write_json(output / "plan.json", plan)
    return plan


def run(output=DEFAULT_OUT, prior_root=OLD_DIAGNOSIS):
    """@entry Score all predefined stage restorations from one 143-image cache."""
    output = Path(output)
    prior_root = Path(prior_root)
    plan = prepare(output, prior_root)
    sp, cases = sampling.load("outputs/vehicle-sampling-consistency-20261117")
    data = sampling.SamplingCases(sp, cases)
    data.nominal.limit = 8
    identities, phases = plan["validation_case_ids"], plan["phase_ids"]
    queries = sorted(
        c["case_id"]
        for c in cases
        if c["source"] == "drivaerrim" and c["split"] == "val"
    )
    requests = [(data.index[i], None) for i in dict.fromkeys(sp["anchors"] + queries)]
    requests += [(data.index[i], phase) for phase in phases for i in identities]
    torch.set_num_threads(2)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    checkpoints = {
        name: torch.load(CHECKPOINTS[name], map_location="cpu", weights_only=True)
        for name in ("B", "E5")
    }
    original, baseline = (checkpoints[name]["model_state"] for name in ("E5", "B"))
    model = PairedDelta(8, 36).to(device).eval()
    model.load_state_dict(original, strict=True)
    cache = encode_requests(model, data, requests, device)
    data.nominal.cache.clear()
    data.cache.clear()
    previous = json.loads((prior_root / "predictions.json").read_text())["crosses"]
    results, summaries, gates = {}, {}, {}
    with torch.inference_mode():
        for name, owners in plan["variants"].items():
            state = restored_state(
                original, baseline, owners["restored_groups"], name == "full_B_readout"
            )
            model.load_state_dict(state, strict=True)
            metrics = score_from_features(
                model, cache, data, sp, cases, identities, phases
            )
            views = view_scores(model, cache, data, sp, identities, phases)
            if name in ("original_E5", "full_B_readout"):
                old = previous["E5__E5" if name == "original_E5" else "E5__B"]
                np.testing.assert_allclose(
                    metrics["phase_scoring"]["self_phase_pred"],
                    old["phase_scoring"]["self_phase_pred"],
                    rtol=3e-5,
                    atol=2e-7,
                )
            results[name] = {"owners": owners, "metrics": metrics, "single_view": views}
            summaries[name] = score_summary(metrics, views, phases)
            gates[name] = candidate_gates(
                previous["B__B"], metrics, plan, sp["delta_std"]
            )
            x = summaries[name]["self_distribution"]["by_phase"]["xp0d25yp0"]
            print(
                json.dumps(
                    {
                        "variant": name,
                        "positive_x_Cl": x["Cl"]["mae"],
                        "positive_x_Cd": x["Cd"]["mae"],
                    }
                ),
                flush=True,
            )
    sampling.write_json(output / "predictions.json", results)
    summary = {
        "variants": summaries,
        "gates_vs_original_B": gates,
        "factorial_contrasts": contrasts(summaries),
        "encoded_images": len(cache),
        "scope": plan["scope"],
        "device": str(device),
        "prior_controls_reproduced": True,
    }
    for path, expected in plan["source_sha256"].items():
        if sampling.sha256(path) != expected:
            raise ValueError("Frozen source changed during stage diagnosis: " + path)
    summary["source_unchanged"] = True
    sampling.write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--prior-root", type=Path, default=OLD_DIAGNOSIS)
    args = parser.parse_args()
    run(args.output, args.prior_root)


if __name__ == "__main__":
    main()
