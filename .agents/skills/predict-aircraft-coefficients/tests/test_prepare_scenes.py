import sys
from pathlib import Path

import numpy as np

SKILL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL / "scripts"))
from aircraft_learning.prepare_scenes import (
    PARAMETERS,
    VIEWS,
    parameters,
    resample_views,
    vehicle_records,
)


def test_missing_physics_stays_missing_and_declared_altitude_is_retained():
    values = parameters(2, 1, 0.5, {"velocity_m_s": 40, "altitude_m": 1000})
    d = dict(zip(PARAMETERS, values))
    assert len(values) == 20
    assert d["log_re_body"] is None and d["mach"] is None
    assert d["ground_present"] is None and d["alpha_sin"] is None
    assert d["altitude_over_10000m"] == 0.1
    np.testing.assert_allclose(values[:4], [0.5, 0.25, np.log(2), np.log(40)])
    geometry_only = parameters(2, 1, 0.5, {})
    assert all(value is None for value in geometry_only[3:])


def test_declared_zero_conditions_are_distinct_from_missing_values():
    d = dict(
        zip(
            PARAMETERS,
            parameters(
                2,
                1,
                0.5,
                {
                    "alpha_deg": 0,
                    "beta_deg": 0,
                    "ground_present": 0,
                    "wheel_rotating": 1,
                    "reference_area_m2": 2,
                },
            ),
        )
    )
    assert d["alpha_sin"] == 0 and d["alpha_cos"] == 1
    assert d["ground_present"] == 0
    assert d["wheel_rotating"] == 1
    assert d["wheel_rotation_speed_over_velocity"] is None
    assert d["log_reference_area_over_length2"] == np.log(0.5)


def test_legacy_camera_resampling_preserves_pixel_center_coordinates():
    old, new, margin = 512, 256, 4
    ids = np.arange(old * old).reshape(old, old)
    image = np.stack([ids / (old * old - 1), np.ones_like(ids)]).astype(np.float32)
    result = resample_views({v: image for v in VIEWS})["top"]
    for row, col in [(16, 33), (96, 127), (200, 240)]:
        sample = round(float(result[0, row, col]) * (old * old - 1))
        old_row, old_col = divmod(sample, old)
        source_x = (old_col + 0.5 - margin) / (old - 2 * margin)
        source_y = (old - 1 - old_row - 0.5 - margin) / (old - 2 * margin)
        target_x = (col + 0.5 - margin) / (new - 2 * margin)
        target_y = (new - 1 - row - 0.5 - margin) / (new - 2 * margin)
        assert abs(source_x - target_x) <= 0.5 / (old - 2 * margin)
        assert abs(source_y - target_y) <= 0.5 / (old - 2 * margin)
    assert np.isin(result[1], [0, 1]).all()
    assert not result[1, 0].any() and not result[1, :, 0].any()


def test_windsor_side_force_rotation_and_clearance_contract():
    original = {
        "case_id": "windsorml-run-0",
        "geometry_id": "windsorml-run-0",
        "condition_profile": "windsorml",
        "parameters": [0.4, 0.3],
        "reynolds": 2.9e6,
        "yaw_deg": 2.5,
        "targets": [0.2, 0.5],
    }
    sources = {
        "builtin_orientation": [],
        "ahmed_source": {"cases": []},
        "windsor_source": {
            "cases": [
                {
                    "case_id": original["case_id"],
                    "design_value": {"frontal_area": "0.1", "clearance": "50"},
                }
            ]
        },
        "windsor_forces": {"0": {"cd": "0.2", "cs": "0.1", "cl": "0.5"}},
    }
    record = next(
        vehicle_records(
            {
                "cases": [original],
                "condition_profiles": {"windsorml": {"conditions": {"yaw_deg": 2.5}}},
            },
            sources,
        )
    )
    _, _, length, _, _, physical, targets, _ = record
    np.testing.assert_allclose(
        targets, [0.2 * np.cos(np.deg2rad(2.5)) - 0.1 * np.sin(np.deg2rad(2.5)), 0.5]
    )
    assert physical["clearance_m"] == 0.05
    assert physical["reference_area_m2"] == 0.1
    assert physical["beta_deg"] == 2.5 and length == 1.044


def test_builtin_nominal_conditions_are_inputs_with_assumed_provenance():
    original = {
        "case_id": "builtin-car",
        "geometry_id": "builtin-car",
        "condition_profile": "builtin_assumed",
        "parameters": [0.4, 0.3],
        "reynolds": 10_000_000,
        "yaw_deg": 0.0,
        "targets": [0.25, None],
        "assumptions": {"source_conditions_verified": False},
        "cd_source": "public reference Cd",
    }
    sources = {
        "builtin_orientation": [{"id": "builtin-car", "dimensions": {"length_m": 4}}],
        "ahmed_source": {"cases": []},
        "windsor_source": {"cases": []},
    }
    data = {
        "cases": [original],
        "condition_profiles": {
            "builtin_assumed": {
                "conditions": {"speed": {"value_kmh": 120.0, "assumed": True}}
            }
        },
    }
    _, _, length, width, height, physical, targets, evidence = next(
        vehicle_records(data, sources)
    )
    assert physical == {
        "velocity_m_s": 120 / 3.6,
        "alpha_deg": 0.0,
        "beta_deg": 0.0,
        "ground_present": 1.0,
        "ground_speed_over_velocity": 1.0,
        "wheels_present": 1.0,
        "wheel_rotating": 1.0,
        "wheel_rotation_speed_over_velocity": 1.0,
    }
    values = dict(zip(PARAMETERS, parameters(length, width, height, physical)))
    assert values["beta_sin"] == 0 and values["beta_cos"] == 1
    assert values["log_velocity_m_s"] == np.log(120 / 3.6)
    assert values["log_re_body"] is None and values["mach"] is None
    assert values["log_reference_area_over_length2"] is None
    assert targets == [0.25, None]
    provenance = evidence["assumptions_not_verified_physics"]
    assert provenance["source_conditions_verified"] is False
    assert set(provenance["assumed_fields"]) == set(physical)
    assert data["cases"][0]["assumptions"] == {"source_conditions_verified": False}
