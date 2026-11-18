"""Prespecified fourfold Rim training-budget check using the frozen trainer."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from .conditioning_diagnostics import evaluate_checkpoint, paired_cd_interval
from .conditioning_experiment import cases_for, load, train_run


def prepare(experiment):
    p, _ = load(experiment)
    out = experiment / "budget-dose"
    if out.exists():
        raise FileExistsError("Budget-dose protocol exists")
    out.mkdir()
    run = next(r for r in p["runs"] if r["name"] == "rim-local")
    p.update(
        epochs=160,
        runs=[run],
        normalization={"rim-local": p["normalization"]["rim-local"]},
        selection_std={"rim-local": p["selection_std"]["rim-local"]},
        scope="Budget dose only: same initial model, Rim target scale, inputs and sampler seed;160 vs40 epochs. First40 draw/optimization prefix should reproduce rim-local. More optimizer steps/exposures are the intended treatment; not a matched-budget architecture comparison.",
    )
    parent = experiment / "protocol.json"
    p["input_sha256"][str(parent.resolve())] = hashlib.sha256(
        parent.read_bytes()
    ).hexdigest()
    p["code_sha256"][str(Path(__file__).resolve())] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    (out / "protocol.json").write_text(json.dumps(p, indent=2) + "\n")


def compare(experiment, device_name="mps"):
    dose = experiment / "budget-dose"
    p, data = load(dose)
    run = p["runs"][0]
    short = json.loads((experiment / "rim-local/training_report.json").read_text())
    long = json.loads((dose / "rim-local/training_report.json").read_text())
    if len(long["history"]) != 160:
        raise ValueError("Budget dose incomplete")
    cases = [c for c in cases_for(data, run) if c["split"] == "train"]
    expected_steps = 160 * ((len(cases) + p["batch_size"] - 1) // p["batch_size"])
    if long["optimizer_steps"] != expected_steps:
        raise ValueError("Dose optimizer budget differs")
    result = {
        "first40_val_curve_max_difference": max(
            abs(a["val_mse"] - b["val_mse"])
            for a, b in zip(short["history"], long["history"][:40])
        ),
        "short_optimizer_steps": short["optimizer_steps"],
        "long_optimizer_steps": long["optimizer_steps"],
        "short_rim_exposures": sum(short["supervised_counts"].values()),
        "long_rim_exposures": sum(long["supervised_counts"].values()),
    }
    device = torch.device(device_name)
    oldp, olddata = load(experiment)
    for name, folder, protocol, source in (
        ("40", experiment, oldp, olddata),
        ("160", dose, p, data),
    ):
        result[name] = {
            role: evaluate_checkpoint(
                folder, protocol, source, run, filename, True, device
            )
            for role, filename in (("selected", "model.pt"), ("last", "last.pt"))
        }
    result["paired_interval"] = paired_cd_interval(
        result["40"]["selected"]["rows"],
        result["160"]["selected"]["rows"],
        "drivaerrim",
    )
    result["scope"] = (
        "Same optimization prefix and validation selector; increased training budget. Single seed, historical test; fit/overfit evidence, not universal sample complexity."
    )
    (experiment / "budget-dose-comparison.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "dose_complete": True,
                "prefix_difference": result["first40_val_curve_max_difference"],
                "40": result["40"]["selected"]["metrics"]["test/vehicle"],
                "160": result["160"]["selected"]["metrics"]["test/vehicle"],
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "train", "compare"))
    parser.add_argument("experiment", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.experiment)
    elif args.action == "train":
        train_run(args.experiment / "budget-dose", "rim-local", "mps")
    else:
        compare(args.experiment)


if __name__ == "__main__":
    main()
