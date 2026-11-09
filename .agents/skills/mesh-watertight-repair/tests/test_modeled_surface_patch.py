import sys
from pathlib import Path

import numpy as np
import pyvista as pv
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from modeled_patch import attach_modeled_patch
from modeled_surface_patch import build_patch, build_sphere_patch
from mesh_metrics import edge_topology, mesh_report
from repair_inventory import build_inventory


def curved_sheet():
    x, y = np.meshgrid(np.linspace(-1, 1, 9), np.linspace(-1, 1, 9))
    points = np.column_stack((x.ravel(), y.ravel(), (0.2*x*x + 0.1*y*y).ravel()))
    faces = []
    for i in range(8):
        for j in range(8):
            if 2 <= i < 6 and 2 <= j < 6:
                continue
            a = 9*i + j
            faces.extend(((a, a+1, a+9), (a+1, a+10, a+9)))
    faces = np.asarray(faces)
    inventory = build_inventory(points, faces, np.arange(len(faces)), "test", max_items=1000)
    region = min(inventory["boundary_regions"]["items"], key=lambda r: r["diameter"])
    return points, faces, region


def test_quadratic_patch_uses_exact_source_boundary_and_recovers_curvature():
    points, faces, region = curved_sheet()
    original = points.copy(), faces.copy()
    patch_points, patch_faces, report = build_patch(points, faces, region, rings=3, max_fit_error=1e-8, max_offset=1, subdivisions=3)
    loop = np.asarray(region["ordered_vertex_ids"])
    np.testing.assert_array_equal(patch_points[:len(loop)], points[loop])
    expected = 0.2*patch_points[:, 0]**2 + 0.1*patch_points[:, 1]**2
    np.testing.assert_allclose(patch_points[:, 2], expected, atol=1e-10)
    _, edges = edge_topology(patch_faces)
    boundary = {edge for edge, incident in edges.items() if len(incident) == 1}
    assert boundary == {tuple(sorted((i, (i+1) % len(loop)))) for i in range(len(loop))}
    assert mesh_report(patch_points, patch_faces)["quality"]["aspect_ratio"]["max"] < 5
    assert report["mesh_method"] == "gmsh"
    assert report["status"] == "candidate_only"
    np.testing.assert_array_equal(points, original[0])
    np.testing.assert_array_equal(faces, original[1])


def test_fit_rejects_source_outside_declared_error_budget():
    points, faces, region = curved_sheet()
    points[0, 2] += 0.5
    with pytest.raises(ValueError, match="exceeds budget"):
        build_patch(points, faces, region, rings=12, max_fit_error=1e-5, max_offset=1, subdivisions=1)


def test_patch_offset_budget_rejects_excessive_extrapolation():
    points, faces, region = curved_sheet()
    with pytest.raises(ValueError, match="patch offset"):
        build_patch(points, faces, region, rings=3, max_fit_error=1e-8, max_offset=1e-6, subdivisions=2)


def test_gmsh_rejects_unbounded_resolution_before_meshing():
    points, faces, region = curved_sheet()
    with pytest.raises(ValueError, match="triangle budget"):
        build_patch(points, faces, region, rings=3, max_fit_error=1e-8, max_offset=1,
                    subdivisions=2, mesh_size=1e-6)


def test_centroid_triangulation_remains_an_explicit_option():
    points, faces, region = curved_sheet()
    _, patch_faces, report = build_patch(points, faces, region, rings=3, max_fit_error=1e-8,
                                        max_offset=1, subdivisions=1, mesh_method="centroid")
    assert report["mesh_method"] == "centroid"
    assert len(patch_faces) == 3 * (len(region["ordered_vertex_ids"]) - 2)


def test_fitted_spherical_piece_attaches_and_preserves_the_original_faces():
    original = pv.Sphere(radius=1, theta_resolution=32, phi_resolution=25).triangulate()
    points = np.asarray(original.points, dtype=float)
    faces = original.faces.reshape(-1, 4)[:, 1:]
    faces = faces[points[faces].mean(axis=1)[:, 2] < 0.5]
    inventory = build_inventory(points, faces, np.arange(len(faces)), "sphere", max_items=1000)
    region = inventory["boundary_regions"]["items"][0]
    vertices, patch_faces, report = build_sphere_patch(
        points, faces, region, rings=4, max_fit_error=1e-5, patch_rings=8
    )
    source = pv.PolyData(points, np.column_stack((np.full(len(faces), 3), faces)).ravel())
    patch = pv.PolyData(vertices, np.column_stack((np.full(len(patch_faces), 3), patch_faces)).ravel())
    assembled, result = attach_modeled_patch(source, patch, seam_tolerance=1e-8, max_seam_angle_deg=20)
    assert report["fit_max_residual"] < 1e-5
    assert result["geometry_accepted"] and result["source_faces_unchanged"]
    np.testing.assert_allclose(np.linalg.norm(vertices, axis=1), 1, atol=1e-6)
    np.testing.assert_array_equal(assembled.faces.reshape(-1, 4)[:len(faces), 1:], faces)
