from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from continuous_surface_bound import source_to_candidate_bound  # noqa: E402


class ContinuousSurfaceBoundTest(unittest.TestCase):
    def test_common_candidate_triangle_bounds_the_whole_source_triangle(self):
        source = np.asarray([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
        target = source + np.asarray([0., 0., .005])
        result = source_to_candidate_bound(source, [[0, 1, 2]], target, [[0, 1, 2]],
                                          max_error=.01)
        self.assertEqual(result["status"], "within_budget")
        self.assertTrue(result["passed"])
        self.assertAlmostEqual(result["max_certified_upper_bound"], .005, places=10)

    def test_subdivision_and_lipschitz_fallback_bound_adjacent_candidate_faces(self):
        source = np.asarray([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.]])
        target = np.asarray([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]])
        result = source_to_candidate_bound(source, [[0, 1, 2]], target,
                                           [[0, 1, 2], [0, 2, 3]], max_error=.8,
                                           max_depth=2)
        self.assertEqual(result["status"], "within_budget")
        self.assertTrue(result["passed"])
        self.assertGreater(result["certified_subtriangles"], 1)

    def test_budget_exhaustion_is_unproven_and_never_reported_as_exceeded(self):
        source = np.asarray([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.]])
        target = np.asarray([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]])
        result = source_to_candidate_bound(source, [[0, 1, 2]], target,
                                           [[0, 1, 2], [0, 2, 3]], max_error=.8,
                                           max_depth=0)
        self.assertEqual(result["status"], "unproven")
        self.assertFalse(result["passed"])
        self.assertNotIn("witness_distance", result)

    def test_a_vertex_beyond_budget_proves_exceedance(self):
        source = np.asarray([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
        target = source + np.asarray([0., 0., .1])
        result = source_to_candidate_bound(source, [[0, 1, 2]], target, [[0, 1, 2]],
                                          max_error=.02)
        self.assertEqual(result["status"], "exceeded")
        self.assertFalse(result["passed"])
        self.assertGreater(result["witness_distance"], .02)

    def test_subdivision_budget_returns_unproven(self):
        source = np.asarray([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.]])
        target = np.asarray([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]])
        result = source_to_candidate_bound(source, [[0, 1, 2]], target,
                                           [[0, 1, 2], [0, 2, 3]], max_error=.8,
                                           max_depth=8, max_subtriangles=2)
        self.assertEqual(result["status"], "unproven")
        self.assertFalse(result["passed"])
        self.assertLessEqual(result["visited_subtriangles"], 2)


if __name__ == "__main__":
    unittest.main()
