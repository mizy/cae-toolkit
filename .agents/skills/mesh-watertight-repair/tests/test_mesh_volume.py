from pathlib import Path
import sys

import numpy as np
import pytest
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from mesh_metrics import component_face_rows, mesh_report, signed_volume
from print_geometry import orient_material


def test_component_face_rows_group_in_label_order_and_preserve_face_order():
    labels = np.asarray([2, 1, 2, 0, 1, 0, 2], dtype=np.int64)
    groups = component_face_rows(labels)
    assert [rows.tolist() for rows in groups] == [[3, 5], [1, 4], [0, 2, 6]]
    assert component_face_rows(np.empty(0, dtype=np.int64)) == []
    with pytest.raises(ValueError):
        component_face_rows(np.asarray([0, -1]))


@pytest.mark.parametrize("scale", [1e-3, 1., 1e3])
def test_closed_volume_survives_unit_changes_translation_and_reversed_winding(scale):
    box = trimesh.creation.box(extents=[2., 3., 4.])
    points = box.vertices * scale + np.array([1e6, -2e6, 3e6]) * scale
    assert signed_volume(points, box.faces) == pytest.approx(24. * scale**3, rel=1e-8)
    assert signed_volume(points, box.faces[:, ::-1]) == pytest.approx(-24. * scale**3, rel=1e-8)


@pytest.mark.parametrize("scale", [1e-3, 1., 1e3])
def test_small_nested_material_shell_volume_and_orientation_are_unit_invariant(scale):
    offset = np.asarray([2e6, -1e6, 3e6]) * scale
    outer = trimesh.creation.box(extents=[2. * scale] * 3)
    inner = trimesh.creation.box(extents=[1e-6 * scale] * 3)
    points = np.vstack((outer.vertices + offset, inner.vertices + offset))
    faces = np.vstack((outer.faces, inner.faces[:, ::-1] + len(outer.vertices)))

    report = mesh_report(points, faces)
    assert report["topology"]["closed_manifold"]
    assert report["volume"]["reliable"]
    expected_volumes = [8. * scale**3, -1e-18 * scale**3]
    for actual, expected, error_bound in zip(
        report["volume"]["component_signed"], expected_volumes,
        report["volume"]["component_signed_error_bound"],
    ):
        assert abs(actual - expected) <= error_bound

    oriented, shells = orient_material(trimesh.Trimesh(points, faces, process=False))
    assert [(shell["depth"], shell["closed"]) for shell in shells] == [(0, True), (1, True)]
    assert signed_volume(oriented.vertices, oriented.faces[:len(outer.faces)]) > 0
    assert signed_volume(oriented.vertices, oriented.faces[len(outer.faces):]) < 0


def test_flat_closed_shell_volume_remains_unreliable():
    points = np.asarray([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [1., 1., 0.]])
    faces = np.asarray([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]])
    report = mesh_report(points, faces)
    assert report["topology"]["closed_manifold"]
    assert not report["volume"]["reliable"]
