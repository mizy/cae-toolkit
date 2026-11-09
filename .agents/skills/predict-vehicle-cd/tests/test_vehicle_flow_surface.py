import numpy as np
import pytest
import pyvista as pv
from vehicle_flow_learning.surface import sample_surface


def test_box_area_sampling_and_scale():
    geometry = {"unit_scale_to_m": 1.0, "front_axis": "-x", "up_axis": "+z"}
    box = pv.Box(bounds=(-1, 1, -0.5, 0.5, 0, 1)).triangulate()
    samples, info = sample_surface(box, geometry, 100000, 7)
    assert info["closed_by_edge_incidence"]
    assert info["inconsistent_winding_edges"] == 0
    assert samples["weights_m2"].sum() == pytest.approx(10)
    assert (
        np.linalg.norm(
            (samples["normals"] * samples["weights_m2"][:, None]).sum(axis=0)
        )
        < 0.1
    )
    scaled, scaled_info = sample_surface(
        box, {**geometry, "unit_scale_to_m": 2}, 1000, 7
    )
    assert scaled_info["normalization"]["length_m"] == pytest.approx(4)
    assert scaled["weights_m2"].sum() == pytest.approx(40)


def test_open_surface_is_reported():
    box = pv.Box().triangulate().extract_cells(range(10)).extract_surface()
    _, info = sample_surface(
        box, {"unit_scale_to_m": 1, "front_axis": "-x", "up_axis": "+z"}, 100, 1
    )
    assert not info["closed_by_edge_incidence"]
    assert info["boundary_edges"] > 0


def test_reversed_box_is_auto_oriented():
    geometry = {"unit_scale_to_m": 1, "front_axis": "-x", "up_axis": "+z"}
    box = pv.Box().triangulate().flip_faces()
    _, info, mesh = sample_surface(box, geometry, 1000, 5, return_mesh=True)
    assert info["signed_volume_m3"] > 0
    assert info["constant_pressure_force_residual_fraction"] < 1e-12
    assert mesh.n_cells == box.n_cells


def test_systematic_surface_sampling_is_area_balanced_and_reproducible():
    geometry = {"unit_scale_to_m": 1, "front_axis": "-x", "up_axis": "+z"}
    box = pv.Box(bounds=(-1, 1, -1, 1, 0, 2)).triangulate()
    first, info = sample_surface(box, geometry, 1200, 9, sampling="systematic")
    second, _ = sample_surface(box, geometry, 1200, 9, sampling="systematic")
    _, counts = np.unique(first["normals"], axis=0, return_counts=True)
    np.testing.assert_array_equal(counts, np.full(6, 200))
    np.testing.assert_array_equal(first["points_m"], second["points_m"])
    assert first["weights_m2"].sum() == pytest.approx(24)
    assert info["sampling"] == "systematic"
    with pytest.raises(ValueError, match="sampling must be"):
        sample_surface(box, geometry, 1200, 9, sampling="unknown")
