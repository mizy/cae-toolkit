"""Compare full-view multiview matching with the legacy side-view predictor."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
from vehicle_cd_learning.data import load_dataset
from vehicle_cd_similarity_lib.artifacts import load_artifacts
from vehicle_cd_similarity_lib.multiview import predict_descriptors, score_descriptors
from vehicle_cd_similarity_lib.scoring import compare_artifacts, score_library, softmax

FOLD_COUNT = 3
SPLIT_SEED = 42
TOP_K = 3
TEMPERATURE = 0.08
EPSILON = 1e-6
TIMING_REPEATS = 15


def family_folds(
    cases: list[dict[str, Any]], folds: int = FOLD_COUNT, seed: int = SPLIT_SEED
) -> list[tuple[str, ...]]:
    """Partition sorted families with the learning pipeline's fixed RNG protocol."""
    families = sorted({str(case["family_id"]) for case in cases})
    if not 2 <= folds <= len(families):
        raise ValueError("folds must be between 2 and the number of families")
    np.random.default_rng(seed).shuffle(families)
    return [
        tuple(str(value) for value in fold) for fold in np.array_split(families, folds)
    ]


def training_reference_indices(
    records: list[dict[str, Any]],
    cases: list[dict[str, Any]],
    heldout_cases: list[dict[str, Any]],
) -> list[int]:
    """Select fold references and reject ID, family, or geometry leakage."""
    cases_by_id = {str(case["case_id"]): case for case in cases}
    records_by_id = {str(record["id"]): record for record in records}
    if len(cases_by_id) != len(cases) or len(records_by_id) != len(records):
        raise ValueError("Case and reference IDs must be unique")
    if set(cases_by_id) != set(records_by_id):
        raise ValueError("Prepared cases and multiview reference IDs differ")

    for car_id, record in records_by_id.items():
        case = cases_by_id[car_id]
        if str(record["family_id"]) != str(case["family_id"]) or str(
            record["geometry_hash"]
        ) != str(case["geometry_hash"]):
            raise ValueError(f"Reference metadata differs from prepared case {car_id}")

    heldout_ids = {str(case["case_id"]) for case in heldout_cases}
    heldout_families = {str(case["family_id"]) for case in heldout_cases}
    heldout_hashes = {str(case["geometry_hash"]) for case in heldout_cases}
    if not heldout_ids or not heldout_families:
        raise ValueError("A fold must hold out at least one case and family")
    indices = [
        index
        for index, record in enumerate(records)
        if str(record["family_id"]) not in heldout_families
    ]
    train_ids = {str(records[index]["id"]) for index in indices}
    train_hashes = {str(records[index]["geometry_hash"]) for index in indices}
    if not indices:
        raise ValueError("A fold has no training references")
    if train_ids & heldout_ids:
        raise ValueError("Out-of-fold split leaks reference IDs")
    if train_hashes & heldout_hashes:
        raise ValueError("Out-of-fold split leaks geometry hashes")
    return indices


def _subset_bank(
    bank: dict[str, np.ndarray], indices: list[int]
) -> dict[str, np.ndarray]:
    return {name: np.asarray(values)[indices] for name, values in bank.items()}


def _descriptor_at(bank: dict[str, np.ndarray], index: int) -> dict[str, np.ndarray]:
    return {name: np.asarray(values)[index] for name, values in bank.items()}


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metrics(truth: list[float], predictions: list[float]) -> dict[str, float]:
    error = np.abs(
        np.asarray(predictions, dtype=np.float64) - np.asarray(truth, dtype=np.float64)
    )
    return {
        "mae": float(error.mean()),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "p95_absolute_error": float(np.quantile(error, 0.95)),
    }


def _legacy_prediction(
    query: Any, library: dict[str, Any], library_dir: Path
) -> tuple[float, list[dict[str, Any]]]:
    results = score_library(query, library, library_dir, query_category=None)
    top = results[: min(TOP_K, len(results))]
    distances = [max(0.0, 1.0 - float(row["score"])) for row in top]
    logits = [TEMPERATURE / (distance + EPSILON) for distance in distances]
    weights = softmax(logits)
    matches = [
        {"id": row["id"], "score": float(row["score"]), "weight": float(weight)}
        for row, weight in zip(top, weights)
    ]
    estimate = sum(float(row["cd"]) * weight for row, weight in zip(top, weights))
    return float(estimate), matches


def _timings(
    query_id: str,
    query_descriptor: dict[str, np.ndarray],
    bank: dict[str, np.ndarray],
    query_artifact: Any,
    all_reference_artifacts: list[Any],
    legacy_library: dict[str, Any],
    legacy_dir: Path,
) -> dict[str, Any]:
    """Time warmed scoring with 26 references; image and descriptor prep are excluded."""
    operations = {
        "multiview_cached_bank": lambda: score_descriptors(query_descriptor, bank),
        "legacy_preloaded_references": lambda: [
            compare_artifacts(query_artifact, reference)
            for reference in all_reference_artifacts
        ],
        "legacy_score_library_png_io": lambda: score_library(
            query_artifact, legacy_library, legacy_dir, query_category=None
        ),
    }
    for operation in operations.values():
        operation()

    samples: dict[str, list[float]] = {name: [] for name in operations}
    for _ in range(TIMING_REPEATS):
        for name, operation in operations.items():
            start = time.perf_counter()
            operation()
            samples[name].append(time.perf_counter() - start)
    return {
        "query_id": query_id,
        "reference_count": 26,
        "repetitions": TIMING_REPEATS,
        "scope": (
            "Warm scoring only. Query views/descriptors, multiview library loading, "
            "and preloaded legacy artifacts are outside the timer. The normal legacy "
            "score_library path includes PNG reads. No rendering or descriptor preparation."
        ),
        "methods": {
            name: {
                "median_seconds": float(statistics.median(values)),
                "mean_seconds": float(statistics.mean(values)),
                "min_seconds": float(min(values)),
                "max_seconds": float(max(values)),
            }
            for name, values in samples.items()
        },
    }


def validate(
    library_json: Path, prepared_dir: Path, legacy_library_path: Path, output_dir: Path
) -> dict[str, Any]:
    """Run fixed family-disjoint out-of-fold prediction and legacy comparison."""
    from vehicle_cd_multiview import load_library

    library_json = library_json.resolve()
    prepared_dir = prepared_dir.resolve()
    legacy_library_path = legacy_library_path.resolve()
    output_dir = output_dir.resolve()

    metadata, bank = load_library(library_json)
    prepared_path = prepared_dir / "dataset.json"
    dataset = load_dataset(prepared_path)
    if dataset["target_names"] != ["cd"]:
        raise ValueError("Multiview similarity validation requires a Cd-only dataset")
    prepared_sha256 = _hash_file(prepared_path)
    if metadata.get("source_dataset_sha256") != prepared_sha256:
        raise ValueError("Multiview library source hash differs from prepared dataset")
    for key in ("render_config", "conditions", "data_kind"):
        if metadata.get(key) != dataset.get(key):
            raise ValueError(f"Multiview metadata {key} differs from prepared dataset")

    cases = dataset["cases"]
    records = metadata["records"]
    if len(cases) != 26 or len(records) != 26:
        raise ValueError("Validation protocol requires the complete 26-car dataset")
    if any(np.asarray(values).shape[0] != len(records) for values in bank.values()):
        raise ValueError("Descriptor bank row count differs from reference metadata")

    legacy_library = json.loads(legacy_library_path.read_text())
    legacy_records = legacy_library.get("cars", [])
    legacy_by_id = {str(car["id"]): car for car in legacy_records}
    record_by_id = {str(record["id"]): record for record in records}
    case_by_id = {str(case["case_id"]): case for case in cases}
    if len(legacy_by_id) != len(legacy_records):
        raise ValueError("Legacy library IDs must be unique")
    expected_ids = set(case_by_id)
    if set(record_by_id) != expected_ids or set(legacy_by_id) != expected_ids:
        raise ValueError("Prepared, multiview, and legacy library IDs must match")
    for car_id, case in case_by_id.items():
        record = record_by_id[car_id]
        legacy = legacy_by_id[car_id]
        truth = float(case["targets"][0])
        for source_name, source in (("multiview", record), ("legacy", legacy)):
            if not math.isclose(float(source["cd"]), truth, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"{source_name} Cd label differs for {car_id}")
        if str(record.get("cd_source", "")) != str(case.get("cd_source", "")):
            raise ValueError(f"Multiview Cd provenance differs for {car_id}")
        if str(legacy.get("cd_source", "")) != str(case.get("cd_source", "")):
            raise ValueError(f"Legacy Cd provenance differs for {car_id}")

    bank_row = {str(record["id"]): index for index, record in enumerate(records)}
    legacy_artifacts = {
        car_id: load_artifacts(legacy_library_path.parent, legacy["artifacts"])
        for car_id, legacy in legacy_by_id.items()
    }
    fold_groups = family_folds(cases)
    all_predictions: dict[str, dict[str, Any]] = {}
    fold_reports = []
    for fold_index, families in enumerate(fold_groups):
        heldout_family_set = set(families)
        heldout_cases = [
            case for case in cases if str(case["family_id"]) in heldout_family_set
        ]
        train_indices = training_reference_indices(records, cases, heldout_cases)
        train_records = [records[index] for index in train_indices]
        train_bank = _subset_bank(bank, train_indices)
        train_legacy = {
            **legacy_library,
            "cars": [
                legacy_by_id[str(records[index]["id"])] for index in train_indices
            ],
        }
        train_ids = {str(record["id"]) for record in train_records}
        if train_ids & {str(case["case_id"]) for case in heldout_cases}:
            raise ValueError("Held-out ID entered a fold reference bank")
        if {str(record["geometry_hash"]) for record in train_records} & {
            str(case["geometry_hash"]) for case in heldout_cases
        }:
            raise ValueError("Held-out geometry entered a fold reference bank")

        for case in heldout_cases:
            car_id = str(case["case_id"])
            query = _descriptor_at(bank, bank_row[car_id])
            new = predict_descriptors(
                query,
                train_bank,
                train_records,
                top_k=TOP_K,
                temperature=TEMPERATURE,
                epsilon=EPSILON,
            )
            old_prediction, old_matches = _legacy_prediction(
                legacy_artifacts[car_id], train_legacy, legacy_library_path.parent
            )
            all_predictions[car_id] = {
                "case_id": car_id,
                "fold": fold_index,
                "family_id": str(case["family_id"]),
                "geometry_hash": str(case["geometry_hash"]),
                "truth": float(case["targets"][0]),
                "multiview_prediction": float(new["cd"]),
                "legacy_side_prediction": old_prediction,
                "multiview_top_k": [
                    {"id": row["id"], "score": row["score"], "weight": row["weight"]}
                    for row in new["top_k"]
                ],
                "legacy_side_top_k": old_matches,
                "training_reference_count": len(train_records),
            }
        fold_reports.append(
            {
                "fold": fold_index,
                "heldout_families": list(families),
                "heldout_ids": [str(case["case_id"]) for case in heldout_cases],
                "training_reference_count": len(train_records),
                "training_ids": [str(record["id"]) for record in train_records],
                "id_overlap": [],
                "geometry_hash_overlap": [],
            }
        )

    prediction_rows = [all_predictions[str(case["case_id"])] for case in cases]
    if len(prediction_rows) != 26 or set(all_predictions) != expected_ids:
        raise ValueError("Out-of-fold validation must produce exactly 26 predictions")
    truth = [row["truth"] for row in prediction_rows]
    report = {
        "scope": (
            "Fixed three-fold family-disjoint comparison of full-view multiscale "
            "matching against the legacy side-view predictor, using all 26 cases. "
            "Labels are mixed-source references and are not verified CFD accuracy."
        ),
        "protocol": {
            "folds": FOLD_COUNT,
            "seed": SPLIT_SEED,
            "top_k": TOP_K,
            "temperature": TEMPERATURE,
            "epsilon": EPSILON,
            "legacy_category_bonus": False,
            "query_features": "Full-bank descriptors selected by case ID; no query Cd label is passed to prediction.",
            "part_crops": False,
            "prediction_count": len(prediction_rows),
        },
        "sources": {
            "prepared_dataset_sha256": prepared_sha256,
            "multiview_library_sha256": _hash_file(library_json),
            "multiview_metadata_source_dataset_sha256": metadata[
                "source_dataset_sha256"
            ],
            "legacy_reference_library_sha256": _hash_file(legacy_library_path),
            "prepared_source_hash_matches": True,
            "ids_and_cd_provenance_match": True,
        },
        "metrics": {
            "multiview_full_view": _metrics(
                truth, [row["multiview_prediction"] for row in prediction_rows]
            ),
            "legacy_side_view": _metrics(
                truth, [row["legacy_side_prediction"] for row in prediction_rows]
            ),
        },
        "folds_detail": fold_reports,
        "cases": prediction_rows,
    }

    first_case_id = str(cases[0]["case_id"])
    report["timing"] = _timings(
        first_case_id,
        _descriptor_at(bank, bank_row[first_case_id]),
        bank,
        legacy_artifacts[first_case_id],
        [legacy_artifacts[str(car["id"])] for car in legacy_records],
        legacy_library,
        legacy_library_path.parent,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "library_json", type=Path, help="Prepared multiview descriptor library"
    )
    parser.add_argument(
        "prepared_dir", type=Path, help="Prepared 26-car dataset directory"
    )
    parser.add_argument(
        "legacy_library", type=Path, help="Legacy side-view reference_cars.json"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = validate(
        args.library_json, args.prepared_dir, args.legacy_library, args.output_dir
    )
    print(
        json.dumps({"metrics": report["metrics"], "timing": report["timing"]}, indent=2)
    )


if __name__ == "__main__":
    main()
