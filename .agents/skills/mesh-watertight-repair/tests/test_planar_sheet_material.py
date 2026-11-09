import sys
from pathlib import Path

import numpy as np
import pyvista as pv
import pytest
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from material_cleanup import geometry_gate_arrays
from planar_sheet_material import build_planar_sheet_material
from source_material_repair import repair_source_file, repair_source_mesh


def overlapping_sheets():
    points = np.asarray([
        [0, 0, 0], [2, 0, 0], [2, 1, 0], [0, 1, 0],
        [1, .5, 1e-7], [3, .5, 1e-7], [3, 1.5, 1e-7], [1, 1.5, 1e-7],
    ], dtype=float)
    faces = np.asarray([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]])
    return points, faces


def test_planar_union_wall_has_expected_volume_and_geometry():
    points, faces = overlapping_sheets()
    vertices, result, rows, report = build_planar_sheet_material(points, faces, wall_depth=.1)
    assert geometry_gate_arrays(vertices, result)["passed"]
    assert np.all(rows == -1)  # New triangles do not claim exact source-facet lineage.
    assert report["source_components"] == 2
    assert report["projection_max_displacement"] < 1e-6
    mesh = trimesh.Trimesh(vertices=vertices, faces=result, process=False)
    assert mesh.volume == pytest.approx(.35, rel=1e-5)


def test_planar_backend_respects_fidelity_and_serialized_geometry_contract():
    points, faces = overlapping_sheets()
    source = pv.PolyData(points, np.column_stack((np.full(len(faces), 3), faces)).ravel())
    candidate, report = repair_source_mesh(source, max_error=.1, boundary_mode="wall", grid_size=32,
                                           union_backend="planar")
    assert report["accepted"], report.get("failure_reason")
    assert report["candidate_surface_distance"]["numerical_continuous_bound"]["passed"]
    assert report["candidate_surface_distance"]["continuous_one_sided_bound"] <= .1
    assert geometry_gate_arrays(np.asarray(candidate.points), candidate.faces.reshape(-1, 4)[:, 1:])["passed"]


def test_planar_backend_rejects_curved_or_closed_input():
    points, faces = overlapping_sheets()
    points[7, 2] = .1
    with pytest.raises(ValueError, match="plane deviation|face normals"):
        build_planar_sheet_material(points, faces, wall_depth=.1)
    box = trimesh.creation.box()
    with pytest.raises(ValueError, match="open sheet"):
        build_planar_sheet_material(box.vertices, box.faces, wall_depth=.1)


def test_part_that_intersects_supplied_assembly_cannot_be_accepted(tmp_path):
    points, faces = overlapping_sheets()
    source = pv.PolyData(points, np.column_stack((np.full(len(faces), 3), faces)).ravel())
    context = pv.Cube(center=(1, .75, -.025), x_length=.3, y_length=.3, z_length=.3).triangulate()
    source_path, context_path = tmp_path / "source.vtp", tmp_path / "context.vtp"
    source.save(source_path)
    context.save(context_path)
    report = repair_source_file(source_path, tmp_path / "out", max_error=.1, boundary_mode="wall",
                                grid_size=32, union_backend="planar", assembly_context=context_path)
    assert report["acceptance_checks"]["geometry"]
    assert not report["accepted"]
    assert not report["assembly_context_validation"]["passed"]
    assert report["assembly_context_validation"]["after"]["cross_pairs"] > 0
    assert report["outputs"]["accepted_mesh_vtp"] is None


def test_disjoint_assembly_context_does_not_block_local_acceptance(tmp_path):
    points, faces = overlapping_sheets()
    source = pv.PolyData(points, np.column_stack((np.full(len(faces), 3), faces)).ravel())
    context = pv.Cube(center=(20, 20, 20)).triangulate()
    source_path, context_path = tmp_path / "source.vtp", tmp_path / "context.vtp"
    source.save(source_path)
    context.save(context_path)
    report = repair_source_file(source_path, tmp_path / "out", max_error=.1, boundary_mode="wall",
                                grid_size=32, union_backend="planar", assembly_context=context_path)
    assert report["accepted"]
    assert report["assembly_context_validation"]["passed"]
    assert report["assembly_context_validation"]["context_faces_in_bounds"] == 0
