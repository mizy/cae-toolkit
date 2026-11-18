import sys
from pathlib import Path

import numpy as np
import pytest
import pyvista as pv

SKILL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL / "scripts"))
sys.path.insert(0, str(SKILL.parent / "predict-vehicle-cd" / "scripts"))
from aircraft_learning.prepare import parameters, render
from aircraft_learning.train import AircraftCases
from vehicle_cd_learning.model import VIEWS


def test_full_span_render_preserves_geometry_and_all_views():
    # mm geometry with span five times body length; the car's framing rejects it.
    mesh = pv.Box(bounds=(-1000, 1000, -5000, 5000, -500, 500)).triangulate()
    views, d = render(mesh, 32)
    assert d["length_m"] == pytest.approx(2)
    assert d["width_m"] == pytest.approx(10)
    assert tuple(views) == VIEWS
    for view in views.values():
        assert view.mask.any() and np.isfinite(view.depth).all()
        occupied = np.argwhere(view.mask)
        assert occupied.min() >= 3 and occupied.max() <= 28


def test_aircraft_alpha_is_explicit_and_sideslip_is_restricted():
    d = {"length_m": 2, "width_m": 10, "height_m": 1}
    row = {
        "alpha_deg": 30,
        "beta_deg": 0,
        "velocity_magnitude_ms": 40,
        "altitude_m": 1000,
    }
    p = parameters(d, row)
    np.testing.assert_allclose(
        p, [5, 0.5, np.log(2), np.log(40), 0.1, 0.5, np.sqrt(3) / 2]
    )
    with pytest.raises(ValueError, match="sideslip"):
        parameters(d, {**row, "beta_deg": 2})


def test_render_unit_conversion_is_consistent():
    mm = pv.Box(bounds=(-1000, 1000, -3000, 3000, -500, 500)).triangulate()
    meters = mm.copy()
    meters.points = meters.points * 0.001
    first, a = render(mm, 64)
    second, b = render(meters, 64, unit_scale_to_m=1)
    assert a["length_m"] == pytest.approx(b["length_m"])
    for name in VIEWS:
        np.testing.assert_array_equal(first[name].mask, second[name].mask)
        np.testing.assert_allclose(first[name].depth, second[name].depth, atol=1e-6)


def test_loader_reuses_geometry_across_conditions(tmp_path):
    (tmp_path / "views").mkdir()
    np.savez(
        tmp_path / "views/1.npz", **{n: np.ones((2, 32, 32), np.float32) for n in VIEWS}
    )
    cases = [
        {"geometry_id": "1", "parameters": [i] * 7, "targets": [0.05, 0.2]}
        for i in (1, 2)
    ]
    stats = {
        "parameter_mean": [0] * 7,
        "parameter_std": [1] * 7,
        "target_mean": [0, 0],
        "target_std": [1, 1],
    }
    ds = AircraftCases(tmp_path, cases, stats, "views")
    a, x, _ = ds[0]
    b, y, _ = ds[1]
    assert a["front"].data_ptr() == b["front"].data_ptr()
    assert len(ds.cache) == 1
    assert not x.equal(y)


@pytest.mark.parametrize(
    "mode,expected", [("conditions", [14, 0.2, 0, 1]), ("design", [3, 14, 0.2, 0, 1])]
)
def test_explicit_condition_schema_preserves_reynolds_for_baselines(
    tmp_path, mode, expected
):
    case = {
        "parameters": [1, 0.2, 14, 0.2, 0, 1],
        "condition_parameters": [14, 0.2, 0, 1],
        "design_parameters": [3],
        "targets": [0.2, 1.5],
    }
    stats = {
        "parameter_mean": [0] * len(expected),
        "parameter_std": [1] * len(expected),
        "target_mean": [0, 0],
        "target_std": [1, 1],
    }
    _, value, _ = AircraftCases(tmp_path, [case], stats, mode)[0]
    np.testing.assert_array_equal(value.numpy(), np.asarray(expected, np.float32))


def test_component_depth_composition_matches_complete_geometry():
    from vehicle_cd_learning.prepare_drivaerrim import full_frame, merge_depth

    a = pv.Box(bounds=(-0.5, 0.5, -0.2, 0.2, 0.1, 0.4)).triangulate()
    b = pv.Box(bounds=(-0.25, 0.25, -0.3, 0.3, 0, 0.2)).triangulate()
    origin = np.zeros(3)
    composed = merge_depth(full_frame(a, origin, 1), full_frame(b, origin, 1))
    complete = full_frame(a.append_polydata(b), origin, 1)
    for view in VIEWS:
        np.testing.assert_array_equal(composed[view].mask, complete[view].mask)
        np.testing.assert_allclose(
            composed[view].depth, complete[view].depth, atol=1e-6
        )
