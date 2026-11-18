"""Train aircraft coefficient models with source-geometry-disjoint evaluation."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from vehicle_cd_learning.evaluate import metrics
from vehicle_cd_learning.model import VIEWS, CoefficientModel
from vehicle_cd_learning.train import train_model, training_normalization


class AircraftCases(Dataset):
    def __init__(self, root, cases, stats, mode):
        self.root, self.cases, self.stats, self.mode = root, cases, stats, mode
        self.cache = {}

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, index):
        case = self.cases[index]
        if self.mode == "views":
            gid = case["geometry_id"]
            if gid not in self.cache:
                with np.load(
                    self.root / "views" / f"{gid}.npz", allow_pickle=False
                ) as f:
                    self.cache[gid] = {
                        k: torch.tensor(f[k], dtype=torch.float32) for k in VIEWS
                    }
            views = self.cache[gid]
            values = case["parameters"]
        elif self.mode == "design":
            views = {}
            values = case["design_parameters"] + case.get(
                "condition_parameters", case["parameters"][3:]
            )
        else:
            views = {}
            values = case.get("condition_parameters", case["parameters"][3:])
        x = (np.array(values) - self.stats["parameter_mean"]) / self.stats[
            "parameter_std"
        ]
        y = (np.array(case["targets"]) - self.stats["target_mean"]) / self.stats[
            "target_std"
        ]
        return (
            views,
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )


def evaluate(checkpoint_path, data, root, mode, splits=("val", "test")):
    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = CoefficientModel(**ck["model_config"])
    model.load_state_dict(ck["model_state"])
    model.eval()
    result = {}
    train_geometry = {x["geometry_id"] for x in data["cases"] if x["split"] == "train"}
    for split in splits:
        cases = [x for x in data["cases"] if x["split"] == split]
        if any(x["geometry_id"] in train_geometry for x in cases):
            raise ValueError("Geometry leakage")
        ds = AircraftCases(root, cases, ck, mode)
        pred = []
        with torch.inference_mode():
            for views, x, _ in ds:
                y = model({k: v[None] for k, v in views.items()}, x[None])[0].numpy()
                pred.append(y * np.array(ck["target_std"]) + ck["target_mean"])
        p = np.array(pred)
        truth = np.array([x["targets"] for x in cases])
        scores = metrics(truth, p, ["cd", "cl"])
        nonzero = abs(p[:, 0]) > 1e-6
        scores["cl_over_cd_mae"] = (
            float(
                abs(
                    p[nonzero, 1] / p[nonzero, 0]
                    - truth[nonzero, 1] / truth[nonzero, 0]
                ).mean()
            )
            if nonzero.any()
            else None
        )
        scores["ratio_valid_count"] = int(nonzero.sum())
        scores["nonpositive_cd"] = int((p[:, 0] <= 0).sum())
        scores["cl_sign_errors"] = int((np.sign(p[:, 1]) != np.sign(truth[:, 1])).sum())
        scores["cl_sign_errors_abs_truth_ge_0_05"] = int(
            (
                (np.sign(p[:, 1]) != np.sign(truth[:, 1])) & (abs(truth[:, 1]) >= 0.05)
            ).sum()
        )
        scores["cl_non_near_zero_count"] = int((abs(truth[:, 1]) >= 0.05).sum())
        scores["cd_mape"] = float(np.mean(abs(p[:, 0] - truth[:, 0]) / truth[:, 0]))
        mean = np.tile(ck["target_mean"], (len(cases), 1))
        result[split] = {
            "cases": len(cases),
            "geometries": len({x["geometry_id"] for x in cases}),
            "metrics": scores,
            "training_mean_baseline": metrics(truth, mean, ["cd", "cl"]),
            "predictions": [
                {
                    "case_id": c["case_id"],
                    "geometry_id": c["geometry_id"],
                    "truth": c["targets"],
                    "prediction": v.tolist(),
                    "conditions": c["conditions"],
                }
                for c, v in zip(cases, p)
            ],
        }
    return {
        "mode": mode,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "best_epoch": ck["epoch"],
        "scope": data.get(
            "scope",
            "Unseen geometry within one CCA family, source CFD coefficients; no arbitrary aircraft-family or flight-condition extrapolation claim",
        ),
        "splits": result,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("views", "design", "conditions"), default="views"
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--encoder-checkpoint", type=Path)
    args = parser.parse_args()
    data = json.loads((args.prepared / "dataset.json").read_text())
    owners = {}
    identities = set()
    for case in data["cases"]:
        if case["case_id"] in identities:
            raise ValueError("Duplicate case ID")
        identities.add(case["case_id"])
        geometry = data["geometry"][case["geometry_id"]]
        for key in (case["geometry_id"], geometry["sha256"]):
            if key in owners and owners[key] != case["split"]:
                raise ValueError("Geometry crosses train/validation/test")
            owners[key] = case["split"]
    for gid, geometry in data["geometry"].items():
        if (
            hashlib.sha256(
                (args.prepared / "views" / f"{gid}.npz").read_bytes()
            ).hexdigest()
            != geometry["views_sha256"]
        ):
            raise ValueError("Prepared views changed")
    train = [x for x in data["cases"] if x["split"] == "train"]
    val = [x for x in data["cases"] if x["split"] == "val"]
    names = (
        data["parameter_names"]
        if args.mode == "views"
        else data["design_names"]
        + data.get("condition_names", data["parameter_names"][3:])
        if args.mode == "design"
        else data.get("condition_names", data["parameter_names"][3:])
    )
    x = [
        c["parameters"]
        if args.mode == "views"
        else c["design_parameters"] + c.get("condition_parameters", c["parameters"][3:])
        if args.mode == "design"
        else c.get("condition_parameters", c["parameters"][3:])
        for c in train
    ]
    y = [c["targets"] for c in train]
    stats = training_normalization(x, y, ["cd", "cl"])
    metadata = {
        "render_config": data["render_config"],
        "conditions": data.get(
            "training_conditions",
            {
                "beta_deg": 0,
                "velocity_max_m_s": 80,
                "source": "COCOANet declared velocity, altitude, alpha; Re and Mach not inferred without verified source atmosphere",
            },
        ),
        "parameter_ranges": {
            k: [
                float(v)
                for v in [np.min(np.asarray(x)[:, i]), np.max(np.asarray(x)[:, i])]
            ]
            for i, k in enumerate(names)
        },
        "data_kind": "cfd",
        "provenance": {
            "dataset": data["dataset"],
            "revision": data["revision"],
            "license": data["license"],
            "mode": args.mode,
            "geometry_convention": data["geometry_convention"],
        },
        "train_ids": [c["case_id"] for c in train],
        "train_geometry_hashes": sorted(
            {data["geometry"][c["geometry_id"]]["sha256"] for c in train}
        ),
        "heldout_ids": [c["case_id"] for c in data["cases"] if c["split"] != "train"],
        "train_family_ids": sorted({c["family_id"] for c in train}),
        "split_policy": "geometry_disjoint",
        "dataset_sha256": hashlib.sha256(
            (args.prepared / "dataset.json").read_bytes()
        ).hexdigest(),
    }
    checkpoint = train_model(
        AircraftCases(args.prepared, train, stats, args.mode),
        AircraftCases(args.prepared, val, stats, args.mode),
        parameter_names=names,
        target_names=["cd", "cl"],
        normalization=stats,
        metadata=metadata,
        output_dir=args.output_dir,
        variant="five" if args.mode == "views" else "parameters",
        epochs=args.epochs,
        batch_size=16,
        learning_rate=0.001,
        seed=args.seed,
        device_name=args.device,
        augment=args.mode == "views",
        model_size="small",
        encoder="pyramid_detail_attention" if args.mode == "views" else "plain",
        fusion="attention" if args.mode == "views" else "concat",
        preload=True,
        encoder_checkpoint=args.encoder_checkpoint,
    )
    result = evaluate(checkpoint, data, args.prepared, args.mode)
    (args.output_dir / "evaluation.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps({k: v["metrics"] for k, v in result["splits"].items()}, indent=2),
        flush=True,
    )


if __name__ == "__main__":
    main()
