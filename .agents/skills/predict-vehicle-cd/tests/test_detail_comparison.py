import copy
import json
import sys
from pathlib import Path

import pytest

SKILL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL / "scripts"))
sys.path.insert(0, str(SKILL.parent / "predict-aircraft-coefficients/scripts"))

from vehicle_cd_learning.detail_comparison import collect


def test_external_comparison_rejects_changed_geometry_labels_and_aggregate_errors(
    tmp_path,
):
    diagnosis, paired = tmp_path / "diagnosis", tmp_path / "paired"
    diagnosis.mkdir()
    paired.mkdir()
    (diagnosis / "comparison.json").write_text("{}")
    rows = [
        {
            "case_id": "query1",
            "source": "drivaerrim",
            "truth": [0.25, 0.1],
            "prediction": [0.2501, 0.1],
        },
        {
            "case_id": "query2",
            "source": "drivaerrim",
            "truth": [0.26, 0.1],
            "prediction": [0.2599, 0.1],
        },
    ]
    model = {"rows": rows, "anchor_scores": {"base": {"delta_mae": [0.0001, 0]}}}
    (paired / "comparison.json").write_text(
        json.dumps(
            {
                "protocol": {"anchor_id": "base"},
                "models": {name: model for name in ["A0", "A1", "A2"]},
            }
        )
    )
    external = {
        "primary_anchor": "base",
        "primary_anchor_cd": 0.24,
        "rows_by_anchor": {
            "base": [
                {
                    "case_id": r["case_id"],
                    "truth": r["truth"][0],
                    "delta_prediction": r["prediction"][0] - 0.24,
                    "delta_truth": r["truth"][0] - 0.24,
                }
                for r in rows
            ]
        },
        "metrics_by_anchor": {"base": {"delta_cd_mae": 0.0001}},
        "direct_test_pair_metrics": {
            "both_directions_delta_cd_mae": 0.0002,
            "ranking_accuracy": 1.0,
        },
    }
    path = tmp_path / "external.json"
    path.write_text(json.dumps(external))
    assert (
        collect(diagnosis, paired, path)["models"]["Radiant_SDF"]["pair_mae"] == 0.0002
    )
    for key, changed, reason in [
        ("case_id", "unmatched", "geometries"),
        ("truth", 0.99, "labels"),
    ]:
        invalid = copy.deepcopy(external)
        invalid["rows_by_anchor"]["base"][0][key] = changed
        path.write_text(json.dumps(invalid))
        with pytest.raises(ValueError, match=reason):
            collect(diagnosis, paired, path)
    external["metrics_by_anchor"]["base"]["delta_cd_mae"] = 0.1
    path.write_text(json.dumps(external))
    with pytest.raises(ValueError, match="aggregate"):
        collect(diagnosis, paired, path)
