from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyvista as pv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from compare_surface_projection import compare, compare_pixels, render


def triangle_mesh(z: float = 0.0, normal=(0.0, 0.0, 1.0)) -> pv.PolyData:
    points = np.asarray([[-0.8, -0.8, z], [0.8, -0.8, z], [0.0, 0.8, z]])
    mesh = pv.PolyData(points, np.asarray([3, 0, 1, 2]))
    mesh.point_data["Normals"] = np.tile(normal, (3, 1))
    mesh.GetPointData().SetActiveNormals("Normals")
    return mesh


class ProjectionComparisonAuditTest(unittest.TestCase):
    def test_reversed_stored_normals_are_reported_and_change_front_face_shading(self):
        source = triangle_mesh(normal=(0.0, 0.0, 1.0))
        reversed_candidate = triangle_mesh(normal=(0.0, 0.0, -1.0))
        bounds = np.asarray([[-1.0, -1.0], [1.0, 1.0]])

        source_raster, source_normals, source_rgb = render(
            source, (0.0, 0.0, 1.0), (0.0, 1.0, 0.0), bounds, 65
        )
        candidate_raster, candidate_normals, candidate_rgb = render(
            reversed_candidate, (0.0, 0.0, 1.0), (0.0, 1.0, 0.0), bounds, 65
        )
        row = compare_pixels(
            source_raster,
            candidate_raster,
            source_normals,
            candidate_normals,
            np.ones(source_raster.depth.shape, dtype=bool),
            1.0,
        )

        self.assertAlmostEqual(row["native_normal_angle_deg"]["p50"], 180.0)
        self.assertTrue(np.any(source_rgb != candidate_rgb))
        reversed_candidate.faces = np.asarray([3, 2, 1, 0])
        _, _, reversed_winding_rgb = render(
            reversed_candidate, (0.0, 0.0, 1.0), (0.0, 1.0, 0.0), bounds, 65
        )
        np.testing.assert_array_equal(source_rgb, reversed_winding_rgb)

    def test_overlapping_layers_select_the_nearest_depth_and_its_normal(self):
        back = np.asarray([[-0.8, -0.8, 0.0], [0.8, -0.8, 0.0], [0.0, 0.8, 0.0]])
        front = back + np.asarray([0.0, 0.0, 0.5])
        points = np.vstack((back, front))
        faces = np.asarray([3, 0, 1, 2, 3, 3, 4, 5])
        mesh = pv.PolyData(points, faces)
        mesh.point_data["Normals"] = np.vstack(
            (np.tile([1.0, 0.0, 0.0], (3, 1)), np.tile([0.0, 0.0, 1.0], (3, 1)))
        )
        mesh.GetPointData().SetActiveNormals("Normals")

        raster, normals, _ = render(
            mesh,
            (0.0, 0.0, 1.0),
            (0.0, 1.0, 0.0),
            np.asarray([[-1.0, -1.0], [1.0, 1.0]]),
            65,
        )

        self.assertEqual(raster.first_hit_face[32, 32], 1)
        self.assertAlmostEqual(raster.depth[32, 32], -0.5)
        np.testing.assert_allclose(normals[32, 32], [0.0, 0.0, -1.0], atol=1e-7)

    def test_candidate_points_outside_registered_viewport_are_reported(self):
        source = triangle_mesh()
        candidate_points = np.asarray(
            [[-0.8, -0.8, 0.0], [2.0, -0.8, 0.0], [0.0, 0.8, 0.0]]
        )
        outside_candidate = pv.PolyData(candidate_points, np.asarray([3, 0, 1, 2]))
        outside_candidate.point_data["Normals"] = np.tile([0.0, 0.0, 1.0], (3, 1))
        outside_candidate.GetPointData().SetActiveNormals("Normals")

        recomputed_candidate = triangle_mesh()
        del recomputed_candidate.point_data["Normals"]

        corner_candidate = triangle_mesh()
        del corner_candidate.point_data["Normals"]
        corner_candidate.cell_data["corner_normals"] = np.asarray(
            [[0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0]]
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference_path = root / "reference.vtp"
            outside_path = root / "outside.vtp"
            recomputed_path = root / "recomputed.vtp"
            corner_path = root / "corner.vtp"
            source.save(reference_path)
            outside_candidate.save(outside_path)
            recomputed_candidate.save(recomputed_path)
            corner_candidate.save(corner_path)
            report = compare(
                reference_path,
                [outside_path, recomputed_path, corner_path],
                root / "report",
                size=33,
                views=["front"],
            )

        rows = {
            path.name: report["candidates"][str(path)][0]
            for path in (outside_path, recomputed_path, corner_path)
        }
        self.assertEqual(rows["outside.vtp"]["normal_field"], "stored_point")
        self.assertEqual(rows["recomputed.vtp"]["normal_field"], "geometry_recomputed")
        self.assertEqual(rows["corner.vtp"]["normal_field"], "stored_corner")
        outside = rows["outside.vtp"]["candidate_points_outside_viewport"]
        if isinstance(outside, dict):
            self.assertGreater(outside.get("count", outside.get("points", 0)), 0)
        else:
            self.assertGreater(outside, 0)

    def test_empty_wheel_region_is_unavailable(self):
        empty = SimpleNamespace(
            silhouette=np.zeros((4, 4), dtype=bool), depth=np.full((4, 4), np.inf)
        )
        row = compare_pixels(
            empty,
            empty,
            np.zeros((4, 4, 3)),
            np.zeros((4, 4, 3)),
            np.zeros((4, 4), dtype=bool),
            1.0,
        )
        self.assertIs(row["available"], False)


if __name__ == "__main__":
    unittest.main()
