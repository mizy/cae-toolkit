"""Predict paired coefficients with the validation-selected sampling-consistency model."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .delta_experiment import sha256
from .sampling_experiment import SamplingCases, batch_views, load
from .stable_delta import initialize_from_checkpoint


def load_selected(experiment):
    root = Path(experiment)
    p, cases = load(root)
    selection = json.loads((root / "selection.json").read_text())
    checkpoint = Path(selection["selected_checkpoint"])
    if sha256(checkpoint) != selection["selected_model_sha256"]:
        raise ValueError("Selected sampling checkpoint changed")
    ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = initialize_from_checkpoint(
        ck, ck.get("coverage_floor", 0), ck.get("filter_input", False)
    )
    return model.eval().requires_grad_(False), p, cases, selection


def predict(experiment, reference, query, reference_phase=None, query_phase=None):
    """@entry Complete-view pair inference; phase arguments select diagnostic inputs only."""
    torch.set_num_threads(2)
    model, p, cases, selection = load_selected(experiment)
    data = SamplingCases(
        p, cases, with_phases=reference_phase is not None or query_phase is not None
    )
    requests = [
        (data.index[reference], reference_phase),
        (data.index[query], query_phase),
    ]
    views, parameters, _ = batch_views(data, requests, torch.device("cpu"))
    with torch.inference_mode():
        values = model(
            {k: v[:1] for k, v in views.items()},
            {k: v[1:] for k, v in views.items()},
            parameters[:1],
        ).numpy()[0] * np.asarray(p["delta_std"])
    return {
        "reference": reference,
        "query": query,
        "reference_phase": reference_phase,
        "query_phase": query_phase,
        "delta_cd": float(values[0]),
        "delta_cl": float(values[1]),
        "validation_sampling_gate_passed": selection["validation_passed"],
        "scope": "Rim complete-view paired model; camera phases are numerical diagnostics, not changed physical conditions.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("reference")
    parser.add_argument("query")
    parser.add_argument("--reference-phase")
    parser.add_argument("--query-phase")
    a = parser.parse_args()
    print(
        json.dumps(
            predict(
                a.experiment, a.reference, a.query, a.reference_phase, a.query_phase
            ),
            indent=2,
        )
    )
