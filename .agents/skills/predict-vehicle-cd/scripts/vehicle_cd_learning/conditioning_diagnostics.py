"""Verify and compare the condition/allocation development experiment."""

import argparse
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from aircraft_learning.compare_scenes import grouped_scores
from aircraft_learning.scene_experiment import sampling_weights

from .conditioning_experiment import cases_for, load
from .conditioning_report import plot, write_report
from .detail_kernel import evaluate_probe
from .geometry_diagnostics import pairwise_ranking, predict_rows, spread_scores
from .point_baseline import make_model


def verify_runs(experiment, p, data):
    initialization, jitter, traces = set(), set(), {}
    for run in p["runs"]:
        root = experiment / run["name"]
        report = json.loads((root / "training_report.json").read_text())
        history = report["history"]
        if [h["epoch"] for h in history] != list(range(1, p["epochs"] + 1)):
            raise ValueError(f"Incomplete run: {run['name']}")
        cases = cases_for(data, run)
        train = [c for c in cases if c["split"] == "train"]
        rng = torch.Generator().manual_seed(p["seed"] + 3)
        counts, rim_trace, steps = np.zeros(len(train), int), hashlib.sha256(), 0
        for _ in range(p["epochs"]):
            ids = torch.multinomial(
                sampling_weights(train), len(train), replacement=True, generator=rng
            ).numpy()
            counts += np.bincount(ids, minlength=len(train))
            for start in range(0, len(ids), p["batch_size"]):
                subset = ids[start : start + p["batch_size"]]
                rim = np.asarray(
                    [i for i in subset if train[i]["source"] == "drivaerrim"], np.int32
                )
                rim_trace.update(np.asarray([steps, len(rim)], np.int32).tobytes())
                rim_trace.update(rim.tobytes())
                steps += 1
        expected = dict(zip((c["case_id"] for c in train), counts.tolist()))
        supervised = {
            c["case_id"]: int(counts[i])
            if not run["focus"] or c["source"] == "drivaerrim"
            else 0
            for i, c in enumerate(train)
        }
        if (
            report["sampling_counts"] != expected
            or report["supervised_counts"] != supervised
        ):
            raise ValueError("Sample or supervised exposures differ from frozen replay")
        if (
            report["rim_trace_sha256"] != rim_trace.hexdigest()
            or report["optimizer_steps"] != steps
        ):
            raise ValueError("Rim batch trace / optimizer timing differs")
        if (
            report["sampling_rng_sha256"]
            != hashlib.sha256(rng.get_state().numpy().tobytes()).hexdigest()
        ):
            raise ValueError("Sampling RNG differs")
        if run["fold"] in traces and traces[run["fold"]] != rim_trace.hexdigest():
            raise ValueError("Matched runs have different Rim exposures or timing")
        traces[run["fold"]] = rim_trace.hexdigest()
        if run["jitter"]:
            jitter.add(report["jitter_rng_sha256"])
        for file in ("model.pt", "last.pt") + (
            ("rim-model.pt",) if not run["focus"] and run["fold"] == "primary" else ()
        ):
            ck = torch.load(root / file, map_location="cpu", weights_only=True)
            expected_epoch = (
                p["epochs"]
                if file == "last.pt"
                else int(
                    np.argmin(
                        [
                            h["rim_val_mse" if file == "rim-model.pt" else "val_mse"]
                            for h in history
                        ]
                    )
                )
                + 1
            )
            if (
                ck["epoch"] != expected_epoch
                or ck["run"] != run
                or ck["normalization"] != p["normalization"][run["name"]]
            ):
                raise ValueError("Checkpoint selection or input contract differs")
            if set(ck["supervised_train_ids"]) != {
                k for k, v in supervised.items() if v > 0
            }:
                raise ValueError("Actual supervised members differ")
            initialization.add(ck["initial_model_sha256"])
    if len(initialization) != 1 or len(jitter) != 1:
        raise ValueError("Initial weights or full-batch jitter RNG not matched")


def evaluate_checkpoint(experiment, p, data, run, filename, rim_only, device):
    ck = torch.load(
        experiment / run["name"] / filename, map_location="cpu", weights_only=True
    )
    cases = cases_for(data, run)
    if run["focus"] or rim_only:
        cases = [c for c in cases if c["source"] == "drivaerrim"]
    model = make_model("views", p["seed"])
    model.load_state_dict(ck["model_state"], strict=True)
    model.to(device).eval()
    rows = predict_rows(
        model, Path(p["prepared"]), cases, ck["normalization"], {}, "views", device
    )
    result = {
        "epoch": ck["epoch"],
        "checkpoint_sha256": hashlib.sha256(
            (experiment / run["name"] / filename).read_bytes()
        ).hexdigest(),
        "metrics": grouped_scores(rows),
        "spread": spread_scores(rows),
        "ranking": pairwise_ranking(rows),
        "rows": rows,
    }
    del model
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    return result


def paired_cd_interval(a, b, source=None):
    left = {
        r["case_id"]: r
        for r in a
        if r["split"] == "test" and (source is None or r["source"] == source)
    }
    right = {
        r["case_id"]: r
        for r in b
        if r["split"] == "test" and (source is None or r["source"] == source)
    }
    if left.keys() != right.keys():
        raise ValueError("Paired test members differ")
    rng = np.random.default_rng(42)
    means, draws = [], []
    for group in sorted({r["source"] for r in left.values()}):
        ids = sorted(k for k, r in left.items() if r["source"] == group)
        delta = np.array(
            [
                abs(left[k]["prediction"][0] - left[k]["truth"][0])
                - abs(right[k]["prediction"][0] - right[k]["truth"][0])
                for k in ids
            ]
        )
        means.append(delta.mean())
        draws.append(delta[rng.integers(0, len(delta), (2000, len(delta)))].mean(1))
    return {
        "cd_mae_improvement": float(np.mean(means)),
        "conditional_interval95": np.quantile(
            np.mean(draws, axis=0), [0.025, 0.975]
        ).tolist(),
        "scope": "Positive is better; fixed-model, source-stratified geometry resampling; excludes seed and new-family uncertainty",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    p, data = load(args.experiment)
    verify_runs(args.experiment, p, data)
    torch.set_num_threads(2)
    device, results = torch.device(args.device), {}
    for run in p["runs"]:
        r = {
            "run": run,
            "selected": evaluate_checkpoint(
                args.experiment, p, data, run, "model.pt", False, device
            ),
        }
        if run["fold"] == "primary":
            r["last_rim"] = evaluate_checkpoint(
                args.experiment, p, data, run, "last.pt", True, device
            )
            r["rim_selected"] = (
                r["selected"]
                if run["focus"]
                else evaluate_checkpoint(
                    args.experiment, p, data, run, "rim-model.pt", True, device
                )
            )
        results[run["name"]] = r
        print(
            json.dumps(
                {
                    "evaluated": run["name"],
                    "scores": r["selected"]["metrics"]["test/vehicle"],
                }
            ),
            flush=True,
        )
    old = json.loads((Path(p["previous"]) / "comparison.json").read_text())
    intervals = {}
    for a, b in (
        ("full", "dimensionless"),
        ("full-jitter", "dimensionless-jitter"),
        ("full", "full-jitter"),
        ("dimensionless", "dimensionless-jitter"),
    ):
        intervals[f"{a}->{b}"] = paired_cd_interval(
            results[a]["selected"]["rows"], results[b]["selected"]["rows"]
        )
    for a, b in (
        ("full", "rim-global"),
        ("rim-global", "rim-local"),
        ("rim-local", "rim-local-jitter"),
    ):
        for role in ("last_rim", "rim_selected"):
            intervals[f"{a}->{b}/{role}"] = paired_cd_interval(
                results[a][role]["rows"], results[b][role]["rows"], "drivaerrim"
            )
    intervals["holdout-mask-velocity"] = paired_cd_interval(
        old["runs"]["views-ahmed-holdout"]["rows"],
        results["holdout-no-velocity"]["selected"]["rows"],
    )
    previous_history = json.loads(
        (Path(p["previous"]) / "views/training_report.json").read_text()
    )["history"]
    now = json.loads((args.experiment / "full/training_report.json").read_text())[
        "history"
    ]
    reproduction = {
        "max_val_curve_difference": float(
            max(abs(a["val_mse"] - b["val_mse"]) for a, b in zip(previous_history, now))
        ),
        "same_best_epoch": results["full"]["selected"]["epoch"]
        == old["runs"]["views"]["best_epoch"],
    }
    result = {
        "protocol": p,
        "runs": results,
        "paired_intervals": intervals,
        "reproduction": reproduction,
        "prior_holdout": {
            name: old["runs"][name]
            for name in ("views-ahmed-holdout", "views-ahmed-no-scale")
        },
        "kernel": evaluate_probe(args.experiment, results["full"]["selected"]["rows"]),
    }
    (args.experiment / "comparison.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    plot(args.experiment, result)
    write_report(args.experiment, result)
    for path, expected in p["preserved_artifacts"].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Previous artifact changed: {path}")
    (args.experiment / "final-verification.json").write_text(
        json.dumps(
            {
                "runs_complete": len(results),
                "all_preserved_artifacts_match": True,
                "same_initial_weights_verified": True,
                "sample_and_supervised_exposures_verified": True,
                "same_rim_batch_trace_and_optimizer_steps_verified": True,
                "matched_jitter_rng_verified": True,
                "comparison_sha256": hashlib.sha256(
                    (args.experiment / "comparison.json").read_bytes()
                ).hexdigest(),
                "default_weights_changed": False,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
