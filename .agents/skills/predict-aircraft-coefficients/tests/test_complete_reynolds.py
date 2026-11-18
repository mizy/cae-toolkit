import copy
import sys
from pathlib import Path

import numpy as np
import pytest

SKILL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL / "scripts"))
sys.path.insert(0, str(SKILL.parent / "predict-vehicle-cd/scripts"))

from aircraft_learning.complete_reynolds import fill_aircraft_re, standard_air


def test_standard_air_matches_sea_level_and_ten_km_reference():
    temperature, pressure, density, viscosity = standard_air(0)
    assert temperature == 288.15 and pressure == 101325
    assert density == pytest.approx(1.225, rel=1e-6)
    assert viscosity == pytest.approx(1.7893e-5, rel=1e-4)
    temperature, pressure, density, _ = standard_air(10000)
    assert temperature == pytest.approx(223.15)
    assert pressure == pytest.approx(26436.24, rel=1e-6)
    assert density == pytest.approx(0.412706, rel=2e-6)
    for altitude in (-1, 11001, np.inf, np.nan):
        with pytest.raises(ValueError):
            standard_air(altitude)


def test_aircraft_re_uses_body_extent_and_preserves_labels_and_source_conditions():
    case = {
        "source": "cocoanet",
        "parameters": [None] * 20,
        "conditions": {"velocity_m_s": 50, "altitude_m": 1000},
        "targets": [0.03, 0.5],
        "source_evidence": {"source_conditions": {"altitude_m": 1000}},
    }
    original = copy.deepcopy(case)
    geometry = {"dimensions": {"length_m": 5, "width_m": 15}, "reference_chord_m": 1.2}
    fill_aircraft_re(case, geometry, 4)
    _, _, rho, mu = standard_air(1000)
    assert case["conditions"]["re_body"] == rho * 50 * 5 / mu
    assert case["parameters"][4] == np.log(case["conditions"]["re_body"])
    assert case["targets"] == original["targets"]
    assert (
        case["source_evidence"]["source_conditions"]
        == original["source_evidence"]["source_conditions"]
    )
    assert case["source_evidence"]["re_completion"]["status"] == "estimated"
    filled = copy.deepcopy(case)
    fill_aircraft_re(case, {"dimensions": {"length_m": 10}}, 4)
    assert filled == case
