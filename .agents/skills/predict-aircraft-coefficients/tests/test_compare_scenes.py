import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

SKILL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL / "scripts"))
sys.path.insert(0, str(SKILL.parent / "predict-vehicle-cd" / "scripts"))
from aircraft_learning.compare_scenes import (
    aggregate,
    check_runs,
    evaluate_run,
    grouped_scores,
    paired_bootstrap,
    score,
    write_report,
)
from vehicle_cd_learning.model import VIEWS, CoefficientModel


def row(identity, source="a", scene=1, truth=(0.2, 0.3), pred=(0.3, 0.4)):
    return {
        "case_id": identity,
        "geometry_id": identity,
        "source": source,
        "scene": scene,
        "split": "test",
        "truth": list(truth),
        "prediction": list(pred),
        "unknown_scene_prediction": list(pred),
    }


def test_metrics_mask_missing_lift_and_disclose_invalid_drag_ratios():
    rows = [
        row("a", truth=(0.2, None), pred=(0.3, 99)),
        row("b", truth=(0.4, -0.2), pred=(-0.1, 0.2)),
        row("c", truth=(0.4, 0.2), pred=(0.2, 0.1)),
    ]
    scores = score(rows)
    assert scores["cd"]["observed_count"] == 3
    assert scores["cl"]["observed_count"] == 2
    assert scores["cl"]["mae"] == pytest.approx(0.25)
    assert scores["ratio_valid_count"] == 1
    assert scores["ratio_invalid_count"] == 1
    assert scores["nonpositive_cd"] == 1
    assert scores["signed_cl_over_cd_mae"] == 0
    assert scores["cl_sign_errors_abs_truth_ge_0_05"] == 1


def test_source_macro_keeps_large_source_from_dominating():
    rows = [row(str(i), source="large", pred=(1.2, 0.3)) for i in range(3)]
    rows.append(row("small", source="small", pred=(3.2, 0.3)))
    grouped = grouped_scores(rows)
    assert grouped["test/vehicle"]["cd_mae"] == pytest.approx(1.5)
    assert grouped["test_source_macro/vehicle"]["cd_mae"] == pytest.approx(2)
    runs = [{"mode": "A", "scores": {"test/vehicle": {"cd_mae": x}}} for x in (1, 2, 3)]
    result = aggregate(runs)["A"]["test/vehicle"]["cd_mae"]
    assert result["mean"] == 2 and result["std"] == 1 and result["seeds"] == 3


def test_paired_geometry_bootstrap_preserves_constant_improvement():
    predictions = {}
    for mode, error in [("A", (0.03, 0.04)), ("B", (0.02, 0.02)), ("C", (0.01, 0.01))]:
        predictions[mode] = {}
        for seed in (42, 43):
            rows = []
            for scene in (1, 2):
                for index in range(4):
                    item = row(
                        f"{scene}-{index}",
                        source=f"s{scene}",
                        scene=scene,
                        pred=(0.2 + error[0], 0.3 + error[1]),
                    )
                    item["geometry_id"] = f"g{scene}-{index // 2}"
                    rows.append(item)
            predictions[mode][seed] = rows
    result = paired_bootstrap(predictions, [42, 43], resamples=100)
    for scene in ("vehicle", "aircraft"):
        pair = result["pairs"]["A_vs_B"][scene]
        assert pair["cd"]["improvement"] == pytest.approx(0.01)
        np.testing.assert_allclose(pair["cl"]["ci95"], [0.02, 0.02], atol=1e-14)


def frozen_run_fixture(root):
    stats = {
        "parameter_mean": [0.0],
        "parameter_std": [1.0],
        "target_mean": [0.0, 0.0],
        "target_std": [1.0, 1.0],
    }
    protocol = {
        "seeds": [42],
        "modes": {"A": "none", "B": "concat", "C": "film"},
        "epochs": 2,
        "dataset_sha256": "dataset",
        "parameter_names": ["p"],
        "normalization": stats,
    }
    data = {
        "cases": [
            {"case_id": f"case-{i}", "geometry_hash": f"hash-{i}", "split": "train"}
            for i in range(2)
        ]
    }
    for mode, scene_mode in protocol["modes"].items():
        folder = root / f"{mode}-seed-42"
        folder.mkdir()
        report = {
            "history": [{"epoch": 1, "val_mse": 0.3}, {"epoch": 2, "val_mse": 0.2}],
            "best_epoch": 2,
            "parameter_count": 100,
        }
        proof = {
            "counts": {"case-0": 2, "case-1": 2},
            "all_members_sampled": True,
            "final_rng_sha256": "rng",
        }
        checkpoint = {
            "seed": 42,
            "epoch": 2,
            "train_ids": ["case-0", "case-1"],
            "train_geometry_hashes": ["hash-0", "hash-1"],
            "dataset_sha256": "dataset",
            "parameter_names": ["p"],
            "model_config": {"scene_mode": scene_mode},
            **stats,
            "scene_common_initial_sha256": "common",
        }
        (folder / "training_report.json").write_text(json.dumps(report))
        (folder / "sampling-proof.json").write_text(json.dumps(proof))
        torch.save(checkpoint, folder / "model.pt")
    return protocol, data


@pytest.mark.parametrize("mutation", ["incomplete", "sampling", "initialization"])
def test_validation_gate_rejects_any_unpaired_run_before_prediction(tmp_path, mutation):
    protocol, data = frozen_run_fixture(tmp_path)
    assert len(check_runs(tmp_path, protocol, data)) == 3
    folder = tmp_path / "C-seed-42"
    if mutation == "incomplete":
        path = folder / "training_report.json"
        report = json.loads(path.read_text())
        report["history"].pop()
        path.write_text(json.dumps(report))
    elif mutation == "sampling":
        path = folder / "sampling-proof.json"
        proof = json.loads(path.read_text())
        proof["counts"] = {"case-0": 1, "case-1": 3}
        path.write_text(json.dumps(proof))
    else:
        path = folder / "model.pt"
        checkpoint = torch.load(path, weights_only=True)
        checkpoint["scene_common_initial_sha256"] = "different"
        torch.save(checkpoint, path)
    with pytest.raises(ValueError):
        check_runs(tmp_path, protocol, data)


def test_batch_evaluation_handles_unknown_scene_and_missing_labels(tmp_path):
    torch.set_num_threads(2)
    model = CoefficientModel(
        4,
        "five",
        2,
        size="small",
        encoder="pyramid_detail_attention",
        fusion="attention",
        scene_mode="film",
    )
    stats = {
        "parameter_mean": [0.0, 1.0, 0.0, 0.0],
        "parameter_std": [1.0] * 4,
        "target_mean": [0.2, 0.3],
        "target_std": [0.1, 0.2],
    }
    ck = {
        **stats,
        "model_config": {
            "parameter_count": 4,
            "variant": "five",
            "target_count": 2,
            "size": "small",
            "encoder": "pyramid_detail_attention",
            "fusion": "attention",
            "scene_mode": "film",
        },
        "model_state": model.state_dict(),
    }
    arrays = {v: np.ones((2, 32, 32), dtype=np.float32) for v in VIEWS}
    np.savez(tmp_path / "views.npz", **arrays)
    cases = [
        {
            "case_id": str(i),
            "geometry_id": str(i),
            "source": f"s{i}",
            "scene": i,
            "split": "test",
            "views": "views.npz",
            "parameters": [None, 1.0],
            "targets": [0.2, None if i == 1 else 0.3],
        }
        for i in (1, 2)
    ]
    result = evaluate_run(ck, cases, tmp_path, "cpu")
    assert len(result["predictions"]) == 2
    assert result["scores"]["test/vehicle"]["cl_mae"] is None
    assert "test/aircraft" in result["unknown_scene_scores"]
    assert result["latency"]["samples"] == 20
    json.dumps(result, allow_nan=False)


def test_report_writes_comparable_source_macro_plot_and_table(tmp_path):
    aggregates = {
        m: {
            f"test_source_macro/{s}": {
                k: {"mean": 0.1, "std": 0.01}
                for k in ("cd_mae", "cl_mae", "signed_cl_over_cd_mae")
            }
            for s in ("vehicle", "aircraft")
        }
        for m in ("A", "B", "C")
    }
    write_report(tmp_path, {"aggregate": aggregates})
    assert (tmp_path / "comparison.png").stat().st_size > 1000
    assert "not a new blind test" in (tmp_path / "README.md").read_text()
