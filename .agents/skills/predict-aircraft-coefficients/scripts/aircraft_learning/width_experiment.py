"""Train one wider A model against an existing, verified single-seed baseline."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import WeightedRandomSampler
from vehicle_cd_learning.model import CoefficientModel
from vehicle_cd_learning.train import seed_everything, train_model

from .scene_experiment import SceneCases, sampling_weights, verify_data


def initialization_hash(model, width_common=False):
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        if width_common and (
            ".project." in key
            or key in ("head.0.weight", "view_attention.score.0.weight")
        ):
            continue
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def train(source, output, width, seed, device):
    """@entry Reuse baseline data and sampling only after proving its initialization."""
    if output.exists():
        raise FileExistsError("Use a fresh experiment directory")
    source_protocol = json.loads((source / "protocol.json").read_text())
    if seed not in source_protocol["seeds"]:
        raise ValueError("Seed is absent from the source experiment")
    for name, expected in source_protocol["code_sha256"].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Source experiment code changed: {name}")
    root = Path(source_protocol["prepared"])
    data = json.loads((root / "dataset.json").read_text())
    if (
        hashlib.sha256((root / "dataset.json").read_bytes()).hexdigest()
        != source_protocol["dataset_sha256"]
    ):
        raise ValueError("Source dataset changed")
    verify_data(root, data)
    baseline_folder = source / f"A-seed-{seed}"
    baseline = torch.load(
        baseline_folder / "model.pt", map_location="cpu", weights_only=True
    )
    report = json.loads((baseline_folder / "training_report.json").read_text())
    if len(report["history"]) != source_protocol["epochs"]:
        raise ValueError("Baseline did not finish its budget")
    seed_everything(seed)
    reference = CoefficientModel(**baseline["model_config"])
    if initialization_hash(reference) != baseline["scene_common_initial_sha256"]:
        raise ValueError("Fresh 32-width initialization differs from actual baseline")
    source_hashes = dict(source_protocol["code_sha256"])
    source_hashes[str(Path(__file__).resolve())] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    protocol = {
        "source_experiment": str(source.resolve()),
        "seed": seed,
        "widths": [32, width],
        "epochs": source_protocol["epochs"],
        "batch_size": source_protocol["batch_size"],
        "learning_rate": source_protocol["learning_rate"],
        "dataset_sha256": source_protocol["dataset_sha256"],
        "prepared": str(root),
        "parameter_names": baseline["parameter_names"],
        "baseline_checkpoint": str((baseline_folder / "model.pt").resolve()),
        "baseline_checkpoint_sha256": hashlib.sha256(
            (baseline_folder / "model.pt").read_bytes()
        ).hexdigest(),
        "baseline_sampling_proof_sha256": hashlib.sha256(
            (baseline_folder / "sampling-proof.json").read_bytes()
        ).hexdigest(),
        "reference_full_initial_sha256": initialization_hash(reference),
        "reference_width_common_initial_sha256": initialization_hash(reference, True),
        "code_sha256": source_hashes,
        "scope": "One prespecified seed42 width comparison; reused historical geometry holdouts; no new blind/family test; model selection validation only",
        "change": "Only per-view projection width and dependent attention/head input matrices; same CNN/pooling, physical inputs, shared Cd/Cl head hidden64/16, sampling, jitter and optimizer",
    }
    del reference
    output.mkdir(parents=True)
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    train_cases = [c for c in data["cases"] if c["split"] == "train"]
    val_cases = [c for c in data["cases"] if c["split"] == "val"]
    if set(baseline["train_ids"]) != {c["case_id"] for c in train_cases}:
        raise ValueError("Training members differ from baseline")
    metadata = {
        k: baseline[k]
        for k in (
            "render_config",
            "conditions",
            "parameter_ranges",
            "data_kind",
            "train_ids",
            "train_geometry_hashes",
            "heldout_ids",
            "train_family_ids",
            "split_policy",
            "dataset_sha256",
            "train_source_counts",
        )
    }
    metadata.update(
        scene_experiment=True,
        provenance={
            "experiment": str(output.resolve()),
            "comparison": "A/no scene width32 versus width128",
            "seed": seed,
        },
    )
    weights = sampling_weights(train_cases)
    sampler = WeightedRandomSampler(
        weights,
        len(train_cases),
        replacement=True,
        generator=torch.Generator().manual_seed(seed + 3),
    )
    folder = output / f"width-{width}-seed-{seed}"
    stats = source_protocol["normalization"]
    train_model(
        SceneCases(root, train_cases, stats),
        SceneCases(root, val_cases, stats),
        parameter_names=baseline["parameter_names"],
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
        view_feature_size=width,
        scene_mode="none",
        scene_dropout=0,
        train_sampler=sampler,
    )
    replay = torch.Generator().manual_seed(seed + 3)
    counts = torch.zeros(len(train_cases), dtype=torch.long)
    for _ in range(protocol["epochs"]):
        indices = torch.multinomial(
            weights, len(train_cases), replacement=True, generator=replay
        )
        counts += torch.bincount(indices, minlength=len(train_cases))
    actual_counts = dict(zip(baseline["train_ids"], counts.tolist()))
    old_proof = json.loads((baseline_folder / "sampling-proof.json").read_text())
    if (
        not torch.equal(replay.get_state(), sampler.generator.get_state())
        or actual_counts != old_proof["counts"]
    ):
        raise ValueError("Actual sampling differs from baseline")
    checkpoint = torch.load(folder / "model.pt", map_location="cpu", weights_only=True)
    if (
        checkpoint["common_initial_sha256"]
        != protocol["reference_width_common_initial_sha256"]
    ):
        raise ValueError("Shared pre-projection initialization differs")
    proof = {
        "all_members_sampled": bool(torch.all(counts > 0)),
        "minimum_exposures": int(counts.min()),
        "counts": actual_counts,
        "final_rng_sha256": hashlib.sha256(
            replay.get_state().numpy().tobytes()
        ).hexdigest(),
        "matches_existing_32_baseline": True,
    }
    (folder / "sampling-proof.json").write_text(json.dumps(proof, indent=2) + "\n")
    print(
        json.dumps(
            {
                "training_complete": True,
                "width": width,
                "seed": seed,
                "shared_initialization_verified": True,
                "sampling_verified": True,
            }
        ),
        flush=True,
    )


def compare(experiment, device):
    from .compare_scenes import evaluate_run

    protocol = json.loads((experiment / "protocol.json").read_text())
    if (experiment / "comparison.json").exists():
        raise FileExistsError("Comparison already exists")
    for name, expected in protocol["code_sha256"].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen code changed: {name}")
    root = Path(protocol["prepared"])
    if (
        hashlib.sha256((root / "dataset.json").read_bytes()).hexdigest()
        != protocol["dataset_sha256"]
    ):
        raise ValueError("Frozen dataset changed")
    data = json.loads((root / "dataset.json").read_text())
    cases = [
        c
        for c in data["cases"]
        if c["split"] in ("val", "test") or c["source"] == "builtin_assumed"
    ]
    result = {"protocol": protocol, "runs": {}}
    torch.set_num_threads(2)
    for width in protocol["widths"]:
        path = (
            Path(protocol["baseline_checkpoint"])
            if width == 32
            else experiment / f"width-{width}-seed-{protocol['seed']}/model.pt"
        )
        report = json.loads(path.with_name("training_report.json").read_text())
        if len(report["history"]) != protocol["epochs"]:
            raise ValueError("Incomplete training budget")
        ck = torch.load(path, map_location="cpu", weights_only=True)
        if (
            width == 32
            and hashlib.sha256(path.read_bytes()).hexdigest()
            != protocol["baseline_checkpoint_sha256"]
        ):
            raise ValueError("Baseline checkpoint changed")
        scores = evaluate_run(ck, cases, root, device)
        scores.update(
            checkpoint=str(path.resolve()),
            best_epoch=ck["epoch"],
            parameter_count=report["parameter_count"],
            checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        result["runs"][str(width)] = scores
        print(json.dumps({"evaluated_width": width}), flush=True)
    (experiment / "comparison.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                w: {
                    k: v
                    for k, v in r["scores"].items()
                    if k.startswith("test_source_macro/")
                }
                for w, r in result["runs"].items()
            },
            indent=2,
        )
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    t = sub.add_parser("train")
    t.add_argument("source", type=Path)
    t.add_argument("--output-dir", type=Path, required=True)
    t.add_argument("--width", type=int, default=128)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--device", default="mps")
    c = sub.add_parser("compare")
    c.add_argument("experiment", type=Path)
    c.add_argument("--device", default="mps")
    args = p.parse_args()
    if args.action == "train":
        train(args.source, args.output_dir, args.width, args.seed, args.device)
    else:
        compare(args.experiment, args.device)


if __name__ == "__main__":
    main()
