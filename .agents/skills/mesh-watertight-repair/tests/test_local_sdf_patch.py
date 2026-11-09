from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import trimesh

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from local_defect_regions import extract_defect_regions
from local_sdf_patch import (
    build_local_sdf_patch,
)
from local_voxel_field import closest_distances
from mesh_metrics import edge_topology
from source_patch_collar import connect_source_collar
from source_projected_validation import exact_self_intersection_report


def perturbed_sphere_case():
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    points = np.asarray(sphere.vertices).copy()
    faces = np.asarray(sphere.faces).copy()
    moved = (points[:, 2] > 0.83) & (points[:, 0] ** 2 + points[:, 1] ** 2 < 0.12)
    points[moved, 2] += 0.025
    centers = points[faces].mean(axis=1)
    rows = np.flatnonzero((centers[:, 2] > 0.72) & (centers[:, 0] ** 2 + centers[:, 1] ** 2 < 0.3))
    selected = points[faces[rows].ravel()]
    bounds = np.stack((selected.min(axis=0), selected.max(axis=0)))
    return points, faces, rows, bounds


def smooth_box_ring_case():
    box = trimesh.creation.box(extents=(2.0, 2.0, 1.0))
    points, faces = np.asarray(box.vertices), np.asarray(box.faces)
    for _ in range(3):
        points, faces = trimesh.remesh.subdivide(points, faces)
    centers = points[faces].mean(axis=1)
    rows = np.flatnonzero((centers[:, 2] > 0.49) & (np.abs(centers[:, 0]) < 0.72) & (np.abs(centers[:, 1]) < 0.72))
    return points, faces, rows


class LocalSdfPatchTest(unittest.TestCase):
    def test_failed_stitch_is_reported_without_publishing_a_patch(self):
        points, faces, rows, bounds = perturbed_sphere_case()
        original = points.copy()
        patch_points, patch_faces, report = build_local_sdf_patch(
            points,
            faces,
            defect_face_rows=rows,
            roi_bounds=bounds,
            pitch=0.04,
            max_voxels=100_000,
            max_patch_area=2.0,
            max_error=0.07,
            max_bridge_length=0.001,
        )
        self.assertIsNone(patch_points)
        self.assertIsNone(patch_faces)
        self.assertEqual(report["zipper"]["status"], "rejected")
        self.assertEqual(report["zipper"]["max_normal_angle"], 20.0)
        self.assertIn("boundary-normal", report["zipper"]["reason"])
        self.assertEqual(report["zipper_faces"], [])
        np.testing.assert_array_equal(points, original)

    def test_closest_distance_queries_fail_closed_on_nonfinite_locator_output(self):
        class InvalidLocator:
            def find_closest_cell(self, queries, *, return_closest_point):
                return np.zeros(len(queries), dtype=np.int64), np.full_like(queries, np.nan)

        with self.assertRaisesRegex(ValueError, "invalid cells or points"):
            closest_distances(InvalidLocator(), np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64))

    def test_perturbed_sphere_patch_passes_without_smoothing(self):
        points, faces, rows, bounds = perturbed_sphere_case()
        original_points, original_faces = points.copy(), faces.copy()
        patch_points, patch_faces, report = build_local_sdf_patch(
            points,
            faces,
            defect_face_rows=rows,
            roi_bounds=bounds,
            pitch=0.04,
            max_voxels=100_000,
            max_patch_area=2.0,
            max_error=0.07,
            max_bridge_length=0.5,
        )

        self.assertIsNotNone(patch_points)
        self.assertIsNotNone(patch_faces)
        self.assertEqual(report["status"], "ready")
        self.assertEqual(len(report["source_boundary_loops"]), 1)
        self.assertEqual(len(report["patch_boundary_loops"]), 1)
        self.assertLessEqual(report["grid"]["voxels"], 100_000)
        self.assertLessEqual(report["source_distance"]["max_error"], 0.07)
        self.assertGreater(report["source_roi"]["removed_boundary_ring_faces"], 0)
        self.assertEqual(report["zipper"]["status"], "ready")
        self.assertEqual(report["zipper"]["max_normal_angle"], 20.0)
        for loop_key, normal_key in (
            ("collar_inner_loop", "collar_inner_edge_normals"),
            ("patch_boundary_loops", "patch_boundary_edge_normals"),
        ):
            loop = report[loop_key]
            if loop_key == "patch_boundary_loops":
                loop = loop[0]
            normals = np.asarray(report[normal_key], dtype=np.float64)
            self.assertEqual(normals.shape, (len(loop) - 1, 3))
            self.assertTrue(np.isfinite(normals).all())
            np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-8)
        np.testing.assert_array_equal(points, original_points)
        np.testing.assert_array_equal(faces, original_faces)

    def test_curved_sphere_patch_with_collar_assembles_with_exact_gates(self):
        points, faces, rows, bounds = perturbed_sphere_case()
        original_points, original_faces = points.copy(), faces.copy()
        patch_points, patch_faces, report = build_local_sdf_patch(
            points,
            faces,
            defect_face_rows=rows,
            roi_bounds=np.stack((bounds[0] - 0.12, bounds[1] + 0.12)),
            pitch=0.04,
            max_voxels=2_000_000,
            max_patch_area=2.0,
            max_error=0.07,
            max_bridge_length=0.5,
            smoothing_sigma_voxels=0.5,
        )

        self.assertEqual(report["status"], "ready")
        self.assertEqual(report["zipper"]["max_normal_angle"], 20.0)
        self.assertLessEqual(report["source_distance"]["max_error"], 0.07)
        self.assertLessEqual(report["grid"]["voxels"], 2_000_000)
        self.assertGreater(report["collar_faces_count"], 0)
        self.assertTrue(set(report["collar_source_face_rows"]).issubset(set(rows)))

        keep = np.setdiff1d(np.arange(len(faces)), rows)
        output_points = np.vstack((points, patch_points))
        output_faces = np.vstack(
            (
                faces[keep],
                patch_faces + len(points),
                np.asarray(report["zipper_faces"], dtype=np.int64),
            )
        )
        output = trimesh.Trimesh(vertices=output_points, faces=output_faces, process=False)
        topology, _edge_faces = edge_topology(output_faces)
        self.assertTrue(output.is_watertight)
        self.assertTrue(output.is_winding_consistent)
        self.assertEqual(topology["boundary_edges"], 0)
        self.assertEqual(topology["non_manifold_edges"], 0)
        self.assertEqual(topology["non_manifold_vertices"], 0)
        self.assertTrue(np.array_equal(output_points[: len(points)], original_points))
        self.assertTrue(np.array_equal(output_faces[: len(keep)], original_faces[keep]))
        exact = exact_self_intersection_report(output_points, output_faces)
        self.assertEqual(exact["status"], "computed")
        self.assertEqual(exact["intersection_pairs"], 0)

    def test_smooth_synthetic_ring_assembles_with_exact_closed_shell(self):
        points, faces, rows = smooth_box_ring_case()
        source_region = extract_defect_regions(
            points,
            faces,
            seed_faces=rows,
            halo_rings=0,
            max_region_faces=len(rows),
            max_total_faces=len(rows),
        )["regions"][0]
        source_loop = source_region["boundary_loops"][0]
        count = 20
        theta = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
        radius = 0.39 * (1.0 + 0.08 * np.cos(3.0 * theta) + 0.04 * np.sin(2.0 * theta))
        ring = np.column_stack(
            (
                0.08 + radius * np.cos(theta),
                -0.05 + 0.34 * np.sin(theta),
                np.full(count, 0.5),
            )
        )
        patch_points = np.vstack(([[0.08, -0.05, 0.5]], ring))
        patch_faces = np.asarray(
            [[0, 1 + index, 1 + (index + 1) % count] for index in range(count)],
            dtype=np.int64,
        )
        patch_region = extract_defect_regions(
            patch_points,
            patch_faces,
            seed_faces=np.arange(len(patch_faces)),
            halo_rings=0,
            max_region_faces=len(patch_faces),
            max_total_faces=len(patch_faces),
        )["regions"][0]
        patch_loop = patch_region["boundary_loops"][0]
        collar_points, collar_faces, connection = connect_source_collar(
            points,
            faces,
            rows,
            source_loop,
            patch_points,
            patch_faces,
            patch_loop,
            max_bridge_length=1.0,
        )
        self.assertEqual(connection["zipper"]["max_normal_angle"], 20.0)
        self.assertGreater(len(collar_faces), 0)
        np.testing.assert_allclose(np.linalg.norm(connection["collar_inner_edge_normals"], axis=1), 1.0)
        np.testing.assert_allclose(np.linalg.norm(connection["patch_boundary_edge_normals"], axis=1), 1.0)
        assembled_patch_points = np.vstack((collar_points, patch_points))
        assembled_patch_faces = patch_faces + len(collar_points)
        assembled_points = np.vstack((points, assembled_patch_points))
        keep = np.setdiff1d(np.arange(len(faces)), rows)
        assembled_faces = np.vstack(
            (
                faces[keep],
                assembled_patch_faces + len(points),
                np.asarray(connection["zipper_faces"], dtype=np.int64),
            )
        )
        assembled = trimesh.Trimesh(vertices=assembled_points, faces=assembled_faces, process=False)
        topology, _edge_faces = edge_topology(assembled_faces)
        self.assertTrue(assembled.is_watertight)
        self.assertTrue(assembled.is_winding_consistent)
        self.assertEqual(topology["boundary_edges"], 0)
        self.assertEqual(topology["non_manifold_edges"], 0)
        self.assertEqual(topology["non_manifold_vertices"], 0)
        intersections = exact_self_intersection_report(assembled_points, assembled_faces)
        self.assertEqual(intersections["status"], "computed")
        self.assertEqual(intersections["intersection_pairs"], 0)

    def test_voxel_budget_rejects_before_generating_a_patch(self):
        points, faces, rows, bounds = perturbed_sphere_case()
        patch_points, patch_faces, report = build_local_sdf_patch(
            points,
            faces,
            defect_face_rows=rows,
            roi_bounds=bounds,
            pitch=0.04,
            max_voxels=100,
            max_patch_area=2.0,
            max_error=0.07,
            max_bridge_length=0.5,
        )

        self.assertIsNone(patch_points)
        self.assertIsNone(patch_faces)
        self.assertEqual(report["status"], "budget_exceeded")
        self.assertEqual(report["reason"], "voxel_budget_exceeded")
        self.assertGreater(report["grid"]["voxels"], 100)

    def test_patch_area_budget_returns_no_candidate(self):
        points, faces, rows, bounds = perturbed_sphere_case()
        patch_points, patch_faces, report = build_local_sdf_patch(
            points,
            faces,
            defect_face_rows=rows,
            roi_bounds=bounds,
            pitch=0.04,
            max_voxels=100_000,
            max_patch_area=0.8,
            max_error=0.07,
            max_bridge_length=0.5,
        )

        self.assertIsNone(patch_points)
        self.assertIsNone(patch_faces)
        self.assertEqual(report["status"], "budget_exceeded")
        self.assertEqual(report["reason"], "source_patch_area_exceeded")
        self.assertGreater(report["source_patch_area"], 0.8)

    def test_source_region_area_budget_rejects_before_voxelization(self):
        points, faces, rows, bounds = perturbed_sphere_case()
        patch_points, patch_faces, report = build_local_sdf_patch(
            points,
            faces,
            defect_face_rows=rows,
            roi_bounds=bounds,
            pitch=0.04,
            max_voxels=100_000,
            max_patch_area=0.7,
            max_error=0.07,
            max_bridge_length=0.5,
        )
        self.assertIsNone(patch_points)
        self.assertIsNone(patch_faces)
        self.assertEqual(report["reason"], "source_patch_area_exceeded")
        self.assertNotIn("grid", report)

    def test_source_distance_budget_returns_no_candidate(self):
        points, faces, rows, bounds = perturbed_sphere_case()
        patch_points, patch_faces, report = build_local_sdf_patch(
            points,
            faces,
            defect_face_rows=rows,
            roi_bounds=bounds,
            pitch=0.04,
            max_voxels=100_000,
            max_patch_area=2.0,
            max_error=0.0001,
            max_bridge_length=0.5,
        )

        self.assertIsNone(patch_points)
        self.assertIsNone(patch_faces)
        self.assertEqual(report["status"], "rejected")
        self.assertEqual(report["reason"], "source_distance_exceeded")
        self.assertGreater(report["source_distance"]["max_error"], 0.0001)

    def test_multiple_source_loops_are_rejected_for_zipper_handling(self):
        points, faces, rows, _bounds = perturbed_sphere_case()
        centers = points[faces].mean(axis=1)
        south = np.flatnonzero(centers[:, 2] < -0.8)
        patch_points, patch_faces, report = build_local_sdf_patch(
            points,
            faces,
            defect_face_rows=np.r_[rows, south],
            roi_bounds=np.asarray([[-1.1, -1.1, -1.1], [1.1, 1.1, 1.1]]),
            pitch=0.04,
            max_voxels=100_000,
            max_patch_area=10.0,
            max_error=0.07,
            max_bridge_length=0.5,
        )

        self.assertIsNone(patch_points)
        self.assertIsNone(patch_faces)
        self.assertEqual(report["status"], "unsupported")
        self.assertEqual(report["reason"], "source_region_requires_one_simple_boundary_loop")

    def test_source_region_without_interior_face_ring_is_rejected(self):
        points, faces, rows, _bounds = perturbed_sphere_case()
        row = rows[:1]
        triangle = points[faces[row].ravel()]
        bounds = np.stack((triangle.min(axis=0) - 0.02, triangle.max(axis=0) + 0.02))
        patch_points, patch_faces, report = build_local_sdf_patch(
            points,
            faces,
            defect_face_rows=row,
            roi_bounds=bounds,
            pitch=0.04,
            max_voxels=100_000,
            max_patch_area=10.0,
            max_error=0.07,
            max_bridge_length=0.5,
        )

        self.assertIsNone(patch_points)
        self.assertIsNone(patch_faces)
        self.assertEqual(report["reason"], "source_roi_has_no_interior_faces")
        self.assertNotIn("grid", report)


if __name__ == "__main__":
    unittest.main()
