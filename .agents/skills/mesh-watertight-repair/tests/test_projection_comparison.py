from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pyvista as pv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from compare_surface_projection import compare_pixels, frame, render, visible_points


class ProjectionComparisonTest(unittest.TestCase):
    def test_backprojected_pixels_lie_on_oblique_source_and_measure_known_offset(self):
        source = pv.Plane(i_resolution=1, j_resolution=1).triangulate()
        eye, up = (1., .4, 1.), (0., 1., 0.)
        coords, basis = frame(np.asarray(source.points), eye, up)
        bounds = np.array([coords[:, :2].min(axis=0)-.1, coords[:, :2].max(axis=0)+.1])
        raster, _, _ = render(source, eye, up, bounds, 64)
        points = visible_points(raster, bounds, basis)
        np.testing.assert_allclose(points[:, 2], 0., atol=1e-15)
        _, nearest = source.find_closest_cell(points, return_closest_point=True)
        self.assertLess(np.linalg.norm(points - nearest, axis=1).max(), 1e-14)
        target = pv.Plane(center=(0., 0., .1), i_size=3., j_size=3.).triangulate()
        _, nearest = target.find_closest_cell(points, return_closest_point=True)
        np.testing.assert_allclose(np.linalg.norm(points-nearest, axis=1), .1, atol=1e-8)

    def test_same_geometry_and_native_normal_field_are_identical_after_subdivision(self):
        source = pv.Plane(i_resolution=1, j_resolution=1).triangulate()
        candidate = source.subdivide(2, 'linear')
        bounds = np.array([[-.6, -.6], [.6, .6]])
        a, an, rgb_a = render(source, (0., 0., 1.), (0., 1., 0.), bounds, 64)
        b, bn, rgb_b = render(candidate, (0., 0., 1.), (0., 1., 0.), bounds, 64)
        row = compare_pixels(a, b, an, bn, np.ones(a.depth.shape, bool), 1.)
        self.assertEqual(row['silhouette_iou'], 1.)
        self.assertEqual(row['depth_mm_at_200mm']['max'], 0.)
        self.assertLess(row['native_normal_angle_deg']['max'], 1e-5)
        np.testing.assert_array_equal(rgb_a, rgb_b)

    def test_fixed_reference_viewport_does_not_hide_a_translated_candidate(self):
        source = pv.Plane(i_resolution=1, j_resolution=1).triangulate()
        candidate = source.translate([.3, 0., 0.])
        bounds = np.array([[-.6, -.6], [.6, .6]])
        a, an, _ = render(source, (0., 0., 1.), (0., 1., 0.), bounds, 64)
        b, bn, _ = render(candidate, (0., 0., 1.), (0., 1., 0.), bounds, 64)
        row = compare_pixels(a, b, an, bn, np.ones(a.depth.shape, bool), 1.)
        self.assertLess(row['silhouette_iou'], .8)
        self.assertGreater(row['lost_source_pixels'], 0)
        self.assertGreater(row['new_candidate_pixels'], 0)

    def test_side_camera_keeps_world_up_vertical(self):
        points = np.array([[0., 0., 0.], [0., 1., 0.]])
        camera, _ = frame(points, (1., 0., 0.), (0., 1., 0.))
        np.testing.assert_array_equal(camera[1]-camera[0], [0., 1., 0.])


if __name__ == '__main__':
    unittest.main()
