"""Held-out sampling consistency, retained aerodynamic response and regression replay."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .delta_experiment import sha256, write_json
from .delta_metrics import compare_radiant
from .phase_views import phase_id
from .sampling_experiment import SamplingCases, batch_views, load
from .sampling_validation import candidate_gates, choose_candidate, score_checkpoint
from .stable_delta import initialize_from_checkpoint


def regression_pair(model, data, p, reference, query, phase, device):
    requests = [
        (data.index[reference], None),
        (data.index[query], None),
        (data.index[reference], phase),
        (data.index[query], phase),
    ]
    with torch.inference_mode():
        views, parameters, _ = batch_views(data, requests, device)
        encoded = model.encode(views)

        def predict(a, b):
            return (
                model.predict_pair(
                    {n: v[a : a + 1] for n, v in encoded.items()},
                    {n: v[b : b + 1] for n, v in encoded.items()},
                    parameters[a : a + 1],
                )
                .cpu()
                .numpy()[0]
                * p["delta_std"]
            ).tolist()

        nominal = predict(0, 1)
        result = {
            "nominal": nominal,
            "self_reference": predict(0, 2),
            "self_query": predict(1, 3),
            "common": predict(2, 3),
            "query_shifted": predict(0, 3),
            "reference_shifted": predict(2, 1),
        }
    result["drift"] = {
        name: (np.array(result[name]) - nominal).tolist()
        for name in ["common", "query_shifted", "reference_shifted"]
    }
    return result


def evaluate(experiment):
    """@entry Evaluate only after the validation-only candidate is sealed."""
    root = Path(experiment)
    p, cases = load(root)
    if (root / "comparison.json").exists():
        raise FileExistsError("Sampling comparison already recorded")
    selection = json.loads((root / "selection.json").read_text())
    validation = json.loads((root / "validation-metrics.json").read_text())
    if (
        sha256(root / "validation-metrics.json")
        != selection["validation_metrics_sha256"]
    ):
        raise ValueError("Validation metrics changed")
    if sha256(selection["selected_checkpoint"]) != selection["selected_model_sha256"]:
        raise ValueError("Selected model changed")
    plan = json.loads((root / "validation-plan.json").read_text())
    phase_plan = json.loads((Path(p["phase_root"]) / "plan.json").read_text())
    test_ids = phase_plan["selection"]["all_test"]
    subset = phase_plan["selection"]["test"]
    main_phase = [phase_id((0.25, 0))]
    extras = [phase_id((0.125, 0)), phase_id((0.5, 0)), phase_id((0, 0.25))]
    representatives = {"original_B": validation["baseline"]}
    for arm in p["arms"]:
        chosen = choose_candidate(
            [c for c in validation["candidates"] if c["arm"] == arm],
            validation["baseline"],
        )
        representatives[arm] = next(
            c
            for c in [validation["baseline"], *validation["candidates"]]
            if c["checkpoint"] == chosen["selected_checkpoint"]
        )
        # Also expose the best stability attempt, even when preservation rejected it.
        representatives[arm + "-diagnostic"] = min(
            [c for c in validation["candidates"] if c["arm"] == arm and c["epoch"] > 0],
            key=lambda c: c["gates"]["score"],
        )
    representatives["selected"] = next(
        c
        for c in [validation["baseline"], *validation["candidates"]]
        if c["checkpoint"] == selection["selected_checkpoint"]
    )
    write_json(
        root / "test-plan.json",
        {
            "models": {
                n: {
                    "checkpoint": c["checkpoint"],
                    "sha256": c["checkpoint_sha256"],
                    "chosen_arm": c["arm"],
                    "epoch": c["epoch"],
                }
                for n, c in representatives.items()
            },
            "main_phase_ids": main_phase,
            "main_geometry_count": len(test_ids),
            "extra_phase_ids": extras,
            "extra_geometry_count": len(subset),
            "test_selection_used": False,
            "scope": "Representatives and overall winner fixed from validation. Diagnostic rows expose rejected attempts selected by validation stability only; they are not adopted models. No test re-ranking.",
        },
    )
    torch.set_num_threads(2)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    data = SamplingCases(p, cases)
    cache = {}
    result = {}
    external = json.loads(
        Path(
            json.loads((Path(p["parent_experiment"]) / "protocol.json").read_text())[
                "external"
            ]
        ).read_text()
    )
    for name, entry in representatives.items():
        identity = (
            entry["checkpoint_sha256"],
            entry["coverage_floor"],
            entry["filter_input"],
        )
        if identity not in cache:
            ck = torch.load(entry["checkpoint"], map_location="cpu", weights_only=True)
            model = (
                initialize_from_checkpoint(
                    ck, entry["coverage_floor"], entry["filter_input"]
                )
                .eval()
                .to(device)
            )
            main = score_checkpoint(
                model, data, p, cases, test_ids, main_phase, split="test"
            )
            extra = score_checkpoint(
                model, data, p, cases, subset, extras, split="test"
            )
            regression = regression_pair(
                model,
                data,
                p,
                p["anchors"][0],
                "drivaerrim__rim-0002",
                main_phase[0],
                device,
            )
            cache[identity] = {
                "checkpoint": entry["checkpoint"],
                "chosen_arm": entry["arm"],
                "epoch": entry["epoch"],
                "main": main,
                "extra": extra,
                "regression": regression,
                "radiant_nominal": compare_radiant(
                    cases, p["anchors"], main["nominal_anchor_predictions"], external
                ),
            }
            del model
            if device.type == "mps":
                torch.mps.empty_cache()
        result[name] = cache[identity]
        print(
            json.dumps(
                {
                    "evaluated": name,
                    "checkpoint_arm": entry["arm"],
                    "epoch": entry["epoch"],
                    "nominal_mae": result[name]["main"]["nominal_anchors"][
                        "mean_anchors"
                    ],
                    "main_sampling": result[name]["main"]["patterns"]["common"][
                        "targets"
                    ],
                }
            ),
            flush=True,
        )
    baseline = result["original_B"]
    for name, record in result.items():
        record["test_main_gates"] = candidate_gates(
            baseline["main"], record["main"], plan, p["delta_std"]
        )
        record["test_extra_gates"] = candidate_gates(
            baseline["extra"], record["extra"], plan, p["delta_std"]
        )
    summary = {
        "selection": selection,
        "models": result,
        "scope": "Held-out phase geometries remain the historical Rim test; correlated phases/pairs, not independent new CFD. Original regression and wider offsets are evaluated after validation choice.",
    }
    write_json(root / "comparison.json", summary)
    chosen = result["selected"]
    print(
        json.dumps(
            {
                "selected": selection["chosen_arm"],
                "epoch": selection["epoch"],
                "validation_passed": selection["validation_passed"],
                "test_main_passed": chosen["test_main_gates"]["validation_passed"],
                "test_extra_passed": chosen["test_extra_gates"]["validation_passed"],
                "regression": chosen["regression"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("experiment", type=Path)
    evaluate(p.parse_args().experiment)
