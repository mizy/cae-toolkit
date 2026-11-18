"""Freeze and train matched A/B/C whole-vehicle scene-conditioning experiments."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from vehicle_cd_learning.model import VIEWS
from vehicle_cd_learning.train import train_model

MODES = {"A": "none", "B": "concat", "C": "film"}
CODE = Path(__file__).resolve()


def normalization(cases):
    values = np.asarray([c["parameters"] for c in cases], dtype=float)
    observed = np.isfinite(values)
    counts = observed.sum(0)
    mean = np.divide(
        np.where(observed, values, 0).sum(0),
        counts,
        out=np.zeros(values.shape[1]),
        where=counts > 0,
    )
    std = np.sqrt(
        np.divide(
            np.where(observed, (values - mean) ** 2, 0).sum(0),
            counts,
            out=np.ones(values.shape[1]),
            where=counts > 0,
        )
    )
    targets = np.asarray([c["targets"] for c in cases], dtype=float)
    if np.any(np.isfinite(targets).sum(0) == 0):
        raise ValueError("Training data must observe both Cd and Cl")
    return {
        "parameter_mean": [*mean.tolist(), *([0.0] * len(mean))],
        "parameter_std": [*np.maximum(std, 1e-8).tolist(), *([1.0] * len(mean))],
        "target_mean": np.nanmean(targets, 0).tolist(),
        "target_std": np.nanstd(targets, 0).tolist(),
        "parameter_observed_counts": counts.tolist(),
    }


class SceneCases(Dataset):
    """Read raw condition truth; missing-value masks are derived at this boundary."""

    def __init__(self, root, cases, stats):
        self.root, self.cases, self.stats = root, cases, stats
        self.cache = {}

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, index):
        case = self.cases[index]
        path = self.root / case["views"]
        if path not in self.cache:
            with np.load(path, allow_pickle=False) as data:
                self.cache[path] = {
                    k: torch.tensor(data[k], dtype=torch.float32) for k in VIEWS
                }
        raw = np.asarray(case["parameters"], dtype=float)
        known = np.isfinite(raw)
        mean = np.asarray(self.stats["parameter_mean"][: len(raw)])
        std = np.asarray(self.stats["parameter_std"][: len(raw)])
        x = np.concatenate(
            (np.where(known, (raw - mean) / std, 0), known.astype(float))
        )
        y = (
            np.asarray(case["targets"], dtype=float) - self.stats["target_mean"]
        ) / self.stats["target_std"]
        return (
            self.cache[path],
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
            torch.tensor(case["scene"], dtype=torch.long),
        )


def sampling_weights(cases):
    counts = Counter((c["scene"], c["source"]) for c in cases)
    sources = {
        scene: sorted({source for s, source in counts if s == scene})
        for scene in (1, 2)
    }
    weights = []
    for c in cases:
        scene, source = c["scene"], c["source"]
        aux = "builtin_assumed" in sources[scene]
        source_share = (
            (0.05 if source == "builtin_assumed" else 0.95 / (len(sources[scene]) - 1))
            if aux
            else 1 / len(sources[scene])
        )
        weights.append(0.5 * source_share / counts[(scene, source)])
    return torch.tensor(weights, dtype=torch.float64)


def verify_data(root, data):
    identities, owners, paths = set(), {}, {}
    for c in data["cases"]:
        if c["case_id"] in identities or c["scene"] not in (1, 2):
            raise ValueError("Duplicate case or invalid scene")
        identities.add(c["case_id"])
        for key in (c["geometry_id"], c["geometry_hash"]):
            if key in owners and owners[key] != c["split"]:
                raise ValueError("Geometry crosses split")
            owners[key] = c["split"]
        path = (root / c["views"]).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("Prepared view is outside the dataset")
        paths[path] = c["views_sha256"]
    for path, expected in paths.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"View checksum changed: {path}")
    return len(paths)


def freeze(prepared, output, epochs, batch_size):
    if (output / "protocol.json").exists():
        raise FileExistsError("Experiment already frozen")
    data = json.loads((prepared / "dataset.json").read_text())
    geometries = verify_data(prepared, data)
    train = [c for c in data["cases"] if c["split"] == "train"]
    names = data["parameter_names"] + [
        f"{name}_known" for name in data["parameter_names"]
    ]
    files = [
        CODE,
        CODE.with_name("prepare_scenes.py"),
        CODE.with_name("compare_scenes.py"),
        CODE.parents[3] / "predict-vehicle-cd/scripts/vehicle_cd_learning/model.py",
        CODE.parents[3] / "predict-vehicle-cd/scripts/vehicle_cd_learning/train.py",
    ]
    protocol = {
        "prepared": str(prepared.resolve()),
        "dataset_sha256": hashlib.sha256(
            (prepared / "dataset.json").read_bytes()
        ).hexdigest(),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": 0.001,
        "seeds": [42, 43, 44],
        "modes": MODES,
        "scene_dropout": 0.1,
        "parameter_names": names,
        "normalization": normalization(train),
        "source_counts": dict(
            Counter(f"{c['split']}/{c['source']}" for c in data["cases"])
        ),
        "view_sets": geometries,
        "initialization": "matched random, no pretrained geometry leakage; scene columns zero; FiLM identity",
        "sampling": "50% per scene; equal CFD sources within scene; builtin assumed aux gets 5% of vehicle budget; replacement samples=len(train) each epoch",
        "selection": "validation mean of per-scene standardized observed-target MSE",
        "test_policy": "No test evaluation until all nine prespecified runs complete; existing historical splits reused, not new blind/family tests",
        "code_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files
        },
        "torch_version": torch.__version__,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "protocol.json").write_text(
        json.dumps(protocol, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "train": len(train),
                "cases": len(data["cases"]),
                "geometries": geometries,
                "numeric_inputs": len(names),
                "runs": 9,
                "epochs": epochs,
            }
        ),
        flush=True,
    )


def train_run(experiment, mode, seed, device):
    protocol = json.loads((experiment / "protocol.json").read_text())
    if seed not in protocol["seeds"] or mode not in protocol["modes"]:
        raise ValueError("Run is outside frozen protocol")
    for filename, expected in protocol["code_sha256"].items():
        if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen training/evaluation code changed: {filename}")
    root = Path(protocol["prepared"])
    if (
        hashlib.sha256((root / "dataset.json").read_bytes()).hexdigest()
        != protocol["dataset_sha256"]
    ):
        raise ValueError("Frozen dataset changed")
    data = json.loads((root / "dataset.json").read_text())
    verify_data(root, data)
    train = [c for c in data["cases"] if c["split"] == "train"]
    val = [c for c in data["cases"] if c["split"] == "val"]
    names, stats = protocol["parameter_names"], protocol["normalization"]
    weights = sampling_weights(train)
    sampler = WeightedRandomSampler(
        weights,
        len(train),
        replacement=True,
        generator=torch.Generator().manual_seed(seed + 3),
    )
    metadata = {
        "scene_experiment": True,
        "render_config": {"ordinary_size": 256, "bottom_size": 256, "window_l": 1.3},
        "conditions": {
            "contract": data["parameter_names"],
            "missing": "unknown physical values normalized to zero with explicit known mask",
        },
        "parameter_ranges": {},
        "data_kind": "cfd_and_assumed_reference",
        "provenance": {"experiment": str(experiment.resolve()), "mode": mode},
        "train_ids": [c["case_id"] for c in train],
        "train_geometry_hashes": sorted({c["geometry_hash"] for c in train}),
        "heldout_ids": [c["case_id"] for c in data["cases"] if c["split"] != "train"],
        "train_family_ids": sorted({c["family_id"] for c in train}),
        "split_policy": "existing_geometry_splits_no_pretrained_initialization",
        "dataset_sha256": protocol["dataset_sha256"],
        "train_source_counts": dict(Counter(c["source"] for c in train)),
    }
    folder = experiment / f"{mode}-seed-{seed}"
    train_model(
        SceneCases(root, train, stats),
        SceneCases(root, val, stats),
        parameter_names=names,
        normalization=stats,
        metadata=metadata,
        output_dir=folder,
        epochs=protocol["epochs"],
        batch_size=protocol["batch_size"],
        learning_rate=protocol["learning_rate"],
        seed=seed,
        device_name=device,
        augment=True,
        model_size="small",
        encoder="pyramid_detail_attention",
        fusion="attention",
        preload=True,
        scene_mode=protocol["modes"][mode],
        scene_dropout=0 if mode == "A" else protocol["scene_dropout"],
        train_sampler=sampler,
    )
    replay = torch.Generator().manual_seed(seed + 3)
    counts = torch.zeros(len(train), dtype=torch.long)
    for _ in range(protocol["epochs"]):
        drawn = torch.multinomial(
            weights, len(train), replacement=True, generator=replay
        )
        counts += torch.bincount(drawn, minlength=len(train))
    if not torch.equal(
        replay.get_state(), sampler.generator.get_state()
    ) or not torch.all(counts > 0):
        raise ValueError(
            "Actual sampling RNG does not match protocol or omitted a training case"
        )
    (folder / "sampling-proof.json").write_text(
        json.dumps(
            {
                "all_members_sampled": True,
                "minimum_exposures": int(counts.min()),
                "counts": dict(zip(metadata["train_ids"], counts.tolist())),
                "final_rng_sha256": hashlib.sha256(
                    replay.get_state().numpy().tobytes()
                ).hexdigest(),
            },
            indent=2,
        )
        + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("prepared", type=Path)
    f.add_argument("--output-dir", type=Path, required=True)
    f.add_argument("--epochs", type=int, default=40)
    f.add_argument("--batch-size", type=int, default=32)
    t = sub.add_parser("train")
    t.add_argument("experiment", type=Path)
    t.add_argument("--mode", choices=MODES, required=True)
    t.add_argument("--seed", type=int, required=True)
    t.add_argument("--device", default="mps")
    args = parser.parse_args()
    if args.action == "freeze":
        freeze(args.prepared, args.output_dir, args.epochs, args.batch_size)
    else:
        train_run(args.experiment, args.mode, args.seed, args.device)


if __name__ == "__main__":
    main()
