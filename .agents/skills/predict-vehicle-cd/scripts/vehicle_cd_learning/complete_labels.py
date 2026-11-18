"""Complete nominal Re and record unverified Cl estimates without changing truth."""

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from .dimensionless import PARAMETERS, load_model, predict_views, read_prepared
from .model import VIEWS


def fill_re(case, length_m):
    """Fill only missing body-length Re; air and operating assumptions stay explicit."""
    index = PARAMETERS.index("log_re_body")
    if case["parameters"][index] is not None:
        return
    source = case["source"]
    if source not in ("drivaerrim", "builtin_assumed"):
        raise ValueError(f"No approved Re assumption for {source}")
    speed = case["conditions"]["velocity_m_s"]
    density, viscosity = 1.204, 1.813e-5 if source == "drivaerrim" else 1.81e-5
    if not np.isfinite([length_m, speed]).all() or min(length_m, speed) <= 0:
        raise ValueError("Body length and nominal speed must be finite and positive")
    re = density * speed * length_m / viscosity
    case["parameters"][index] = float(np.log(re))
    case["conditions"]["re_body"] = re
    case["source_evidence"]["re_completion"] = {
        "status": "estimated",
        "source_conditions_verified": False,
        "formula": "Re_body = rho * U * L_body / mu",
        "reference_length": "full body length in metres from audited source geometry",
        "audit_inputs": {
            "length_m": length_m,
            "velocity_m_s": speed,
            "density_kg_m3": density,
            "dynamic_viscosity_pa_s": viscosity,
        },
        "basis": (
            "Source speed; assumed air properties. Search-cached old dataset card lists "
            "mu=1.813e-5, but the pinned current card omits viscosity; not verified CFD Re."
            if source == "drivaerrim"
            else "User-approved 120 km/h nominal profile and existing assumed air properties; not original source conditions."
        ),
    }


def estimate_cl(model, checkpoint, views, before, after, teacher_hash):
    """Keep pseudo-labels outside supervised targets; response spread is not uncertainty."""
    original = predict_views(model, checkpoint, views, before)
    completed = predict_views(model, checkpoint, views, after)
    return {
        "value": float(completed[1]),
        "status": "model_pseudo_label",
        "source_conditions_verified": False,
        "eligible_for_supervised_loss": False,
        "teacher_sha256": teacher_hash,
        "before_re_completion": float(original[1]),
        "re_completion_shift": float(completed[1] - original[1]),
        "uncertainty": None,
        "limit": "No measured Cl on these cars; same-model completion adds no independent lift supervision. Re response and cross-family accuracy are unvalidated.",
    }


def write_table(path, cases):
    """Export observed values and estimates with explicit training eligibility."""
    fields = [
        "case_id",
        "source",
        "split",
        "re_body",
        "re_status",
        "cd",
        "cl",
        "cl_status",
        "cl_supervised",
        "cl_before_re_completion",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for case in cases:
            estimate = case.get("target_estimates", {}).get("cl", {})
            observed = case["targets"][1] is not None
            writer.writerow(
                {
                    "case_id": case["case_id"],
                    "source": case["source"],
                    "split": case["split"],
                    "re_body": case["conditions"]["re_body"],
                    "re_status": case["source_evidence"]
                    .get("re_completion", {})
                    .get("status", "source_derived"),
                    "cd": case["targets"][0],
                    "cl": case["targets"][1] if observed else estimate["value"],
                    "cl_status": "source_label" if observed else estimate["status"],
                    "cl_supervised": observed,
                    "cl_before_re_completion": estimate.get("before_re_completion"),
                }
            )


def complete(source, builtin_audit, rim_prepared, checkpoint, output):
    """@entry Write a fresh dataset and a separate, auditable completed-value table."""
    prepared = output / "prepared"
    if prepared.exists():
        raise FileExistsError("Use a fresh prepared directory")
    data = read_prepared(source)
    paths = [
        source / "dataset.json",
        builtin_audit,
        rim_prepared / "dataset.json",
        checkpoint,
    ]
    frozen = {
        str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
    }
    builtin = {
        c["id"]: c["dimensions"]["length_m"]
        for c in json.loads(builtin_audit.read_text())
    }
    rim = json.loads((rim_prepared / "dataset.json").read_text())["geometry"]
    model, teacher = load_model(checkpoint)
    teacher_hash = frozen[str(checkpoint.resolve())]
    torch.set_num_threads(2)
    prepared.mkdir(parents=True)
    for case in data["cases"]:
        before = list(case["parameters"])
        if before[PARAMETERS.index("log_re_body")] is None:
            key = case["geometry_id"].split("__", 1)[1]
            length = (
                builtin[key]
                if case["source"] == "builtin_assumed"
                else rim[key]["dimensions"][0]
            )
            fill_re(case, length)
        if case["targets"][1] is None:
            if case["source"] != "builtin_assumed" or case["split"] != "train":
                raise ValueError(
                    "Cl pseudo-label completion is limited to train-only reference cars"
                )
            with np.load(source / case["views"], allow_pickle=False) as raw:
                views = {k: raw[k] for k in VIEWS}
            case["target_estimates"] = {
                "cl": estimate_cl(
                    model, teacher, views, before, case["parameters"], teacher_hash
                )
            }
        target = prepared / case["views"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / case["views"], target)
    save_completion(data, source, checkpoint, frozen, output)
    read_prepared(prepared)
    if any(
        hashlib.sha256(Path(p).read_bytes()).hexdigest() != sha
        for p, sha in frozen.items()
    ):
        raise ValueError("Source artifact changed during completion")
    return data["label_completion"]


def save_completion(data, source, checkpoint, frozen, output):
    index = PARAMETERS.index("log_re_body")
    counts = Counter(
        c["source"] for c in data["cases"] if "re_completion" in c["source_evidence"]
    )
    data["source_prepared"] = str((source / "dataset.json").resolve())
    data["source_dataset_sha256"] = frozen[data["source_prepared"]]
    data["unknown_counts"] = {
        s: {
            k: sum(
                c["parameters"][i] is None for c in data["cases"] if c["source"] == s
            )
            for i, k in enumerate(PARAMETERS)
        }
        for s in data["source_counts"]
    }
    data["label_completion"] = {
        "date": "2026-11-16",
        "source_artifacts": frozen,
        "teacher_checkpoint": str(checkpoint.resolve()),
        "estimated_re_by_source": dict(counts),
        "remaining_missing_re": sum(
            c["parameters"][index] is None for c in data["cases"]
        ),
        "estimated_cl": sum("target_estimates" in c for c in data["cases"]),
        "observed_cl": sum(c["targets"][1] is not None for c in data["cases"]),
        "training_policy": "Nominal Re enters the input vector. target_estimates are audit-only and excluded from normalization, loss and metrics; observed targets are unchanged.",
        "known_mask_definition": "A supplied value, including an explicit estimate; does not certify source verification.",
        "teacher_policy": "Frozen dimensionless A candidate, original and completed Re inputs, same full five views. No retraining or accuracy claim.",
    }
    (output / "prepared/dataset.json").write_text(
        json.dumps(data, indent=2, allow_nan=False) + "\n"
    )
    (output / "completion.json").write_text(
        json.dumps(data["label_completion"], indent=2) + "\n"
    )
    write_table(output / "completed-values.csv", data["cases"])
    write_table(
        output / "builtin-estimates.csv",
        [c for c in data["cases"] if c["source"] == "builtin_assumed"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--builtin-audit", type=Path, required=True)
    parser.add_argument("--rim-prepared", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            complete(
                args.source,
                args.builtin_audit,
                args.rim_prepared,
                args.checkpoint,
                args.output_dir,
            ),
            indent=2,
        )
    )
