from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pyvista as pv

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from fidelity_policy_baseline import (
    channel_rays_clear,
    sampled_distance,
    surface_samples,
    synthetic_cases,
)
from mesh_metrics import mesh_report


class FidelityPolicyBaselineTest(unittest.TestCase):
    def test_on_sphere_vertices_do_not_hide_planar_face_error(self) -> None:
        points = np.eye(3)
        faces = np.asarray([[0, 1, 2]])
        np.testing.assert_allclose(np.linalg.norm(points, axis=1), 1.0)
        measurement = sampled_distance(
            points,
            faces,
            pv.Sphere(radius=1.0, theta_resolution=100, phi_resolution=100),
        )
        self.assertGreater(measurement["p95_mm"], 0.3)
        self.assertGreater(measurement["sampled_max_mm"], 0.4)

    def test_surface_weights_integrate_triangle_area(self) -> None:
        points = np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
        samples, weights = surface_samples(points, np.asarray([[0, 1, 2]]))
        self.assertAlmostEqual(float(weights.sum()), 3.0)
        self.assertTrue(np.all(samples[:, 2] == 0.0))
        self.assertTrue(np.all(samples[:, 0] / 2 + samples[:, 1] / 3 <= 1 + 1e-12))

    def test_channel_probe_distinguishes_open_passage_from_blocked_passage(
        self,
    ) -> None:
        duct = synthetic_cases()["wide_duct"]["source"]
        packed = np.column_stack((np.full(len(duct.faces), 3), duct.faces)).ravel()
        mesh = pv.PolyData(duct.vertices, packed)
        self.assertEqual(channel_rays_clear(mesh, 2.0)["clear_rays"], 5)
        blocked = mesh.merge(pv.Plane(i_size=10.0, j_size=10.0).triangulate())
        self.assertEqual(channel_rays_clear(blocked, 2.0)["clear_rays"], 0)

    def test_references_have_declared_components_and_topology(self) -> None:
        for name, case in synthetic_cases().items():
            with self.subTest(case=name):
                mesh = case["reference"]
                report = mesh_report(mesh.vertices, mesh.faces)
                topology = report["topology"]
                self.assertEqual(topology["boundary_edges"], 0)
                self.assertEqual(topology["non_manifold_edges"], 0)
                self.assertEqual(topology["components"]["count"], case["expected_components"])
                self.assertEqual(
                    report["points"] - topology["edges"] + report["triangles"],
                    case["expected_euler"],
                )
                self.assertTrue(report["volume"]["reliable"])
        damaged = synthetic_cases()["missing_cover"]["source"]
        self.assertGreater(
            mesh_report(damaged.vertices, damaged.faces)["topology"]["boundary_edges"],
            0,
        )


if __name__ == "__main__":
    unittest.main()
