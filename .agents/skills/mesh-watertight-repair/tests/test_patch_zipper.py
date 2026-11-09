import unittest
from pathlib import Path
from sys import path as import_path

import numpy as np
import trimesh

import_path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from local_defect_regions import extract_defect_regions
from patch_zipper import stitch_boundary_loops
from source_projected_validation import exact_self_intersection_report


def rings(outer_count=8, inner_count=13, height=0.1):
    a = np.linspace(0.0, 2 * np.pi, outer_count, endpoint=False)
    b = np.linspace(0.11, 2 * np.pi + 0.11, inner_count, endpoint=False)
    points = np.vstack(
        (
            np.column_stack((np.cos(a), np.sin(a), np.zeros_like(a))),
            np.column_stack((0.8 * np.cos(b), 0.8 * np.sin(b), np.full_like(b, height))),
        )
    )
    return points, np.arange(outer_count), np.arange(outer_count, outer_count + inner_count)


def loop_normals_from_faces(points, faces, loop):
    result = []
    for left, right in zip(loop, np.roll(loop, -1)):
        matches = [
            face
            for face in faces
            if any((int(a), int(b)) == (int(left), int(right)) for a, b in zip(face, np.roll(face, -1)))
        ]
        if not matches:
            matches = [
                face
                for face in faces
                if any((int(a), int(b)) == (int(right), int(left)) for a, b in zip(face, np.roll(face, -1)))
            ]
        if len(matches) != 1:
            raise ValueError("each oriented loop edge must have one adjacent face")
        tri = points[matches[0]]
        normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        result.append(normal / np.linalg.norm(normal))
    return np.asarray(result)


class PatchZipperTest(unittest.TestCase):
    def test_different_loop_sampling_counts_stitch_with_directed_boundaries(self):
        points, outer, inner = rings()
        original = points.copy()
        faces = stitch_boundary_loops(points, outer, inner, max_bridge_length=1.0)
        self.assertEqual(len(faces), len(outer) + len(inner))
        for a, b in zip(outer, np.roll(outer, -1)):
            self.assertTrue(
                any(tuple(face_edge) == (a, b) for face in faces for face_edge in zip(face, np.roll(face, -1)))
            )
        for a, b in zip(inner, np.roll(inner, -1)):
            self.assertTrue(
                any(tuple(face_edge) == (b, a) for face in faces for face_edge in zip(face, np.roll(face, -1)))
            )
        np.testing.assert_array_equal(points, original)

    def test_cyclic_start_does_not_reverse_boundary_orientation(self):
        points, outer, inner = rings(9, 12)
        outer, inner = np.roll(outer, 3), np.roll(inner, 7)
        result = stitch_boundary_loops(points, outer, inner, max_bridge_length=1.0)
        self.assertEqual(len(result), len(outer) + len(inner))

    def test_equal_cost_symmetric_rings_return_deterministic_valid_strip(self):
        count = 8
        angles = np.linspace(0.0, 2 * np.pi, count, endpoint=False)
        inner_angles = angles + np.pi / count
        points = np.vstack(
            (
                np.column_stack((np.cos(angles), np.sin(angles), np.zeros(count))),
                np.column_stack((0.8 * np.cos(inner_angles), 0.8 * np.sin(inner_angles), np.full(count, 0.01))),
            )
        )
        outer, inner = np.arange(count), np.arange(count, 2 * count)
        indices = np.arange(count)
        alignment_costs = []
        for rotation in (0, count - 1):
            outer_bridges = points[outer] - points[inner[(indices + rotation) % count]]
            inner_bridges = points[inner] - points[outer[(indices - rotation) % count]]
            alignment_costs.append(float(np.sum(outer_bridges**2) + np.sum(inner_bridges**2)))
        self.assertAlmostEqual(alignment_costs[0], alignment_costs[1], places=12)

        faces = stitch_boundary_loops(points, outer, inner, max_bridge_length=2.0)
        repeated = stitch_boundary_loops(points, outer, inner, max_bridge_length=2.0)

        np.testing.assert_array_equal(faces, repeated)
        self.assertEqual(len(faces), len(outer) + len(inner))
        triangles = points[faces]
        areas = 0.5 * np.linalg.norm(
            np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1
        )
        self.assertTrue(np.all(areas > 0.0))
        for left, right in zip(outer, np.roll(outer, -1)):
            self.assertTrue(any((left, right) == (a, b) for face in faces for a, b in zip(face, np.roll(face, -1))))
        for left, right in zip(inner, np.roll(inner, -1)):
            self.assertTrue(any((right, left) == (a, b) for face in faces for a, b in zip(face, np.roll(face, -1))))

    def test_normal_gate_rejects_boundary_fold(self):
        points, outer, inner = rings(9, 12, height=0.02)
        baseline = stitch_boundary_loops(points, outer, inner, max_bridge_length=1.0)
        outer_normals = loop_normals_from_faces(points, baseline, outer)
        inner_normals = loop_normals_from_faces(points, baseline, inner[::-1])
        outer_normals[0] *= -1.0
        with self.assertRaisesRegex(ValueError, "zipper"):
            stitch_boundary_loops(
                points,
                outer,
                inner,
                max_bridge_length=1.0,
                outer_edge_normals=outer_normals,
                inner_edge_normals=inner_normals,
                max_normal_angle=20.0,
            )

    def test_far_rings_are_rejected_by_bridge_budget(self):
        points, outer, inner = rings(height=1.0)
        with self.assertRaisesRegex(ValueError, "bridge"):
            stitch_boundary_loops(points, outer, inner, max_bridge_length=0.2)

    def test_bridge_budget_checks_actual_triangles_at_coarse_source_edges(self):
        points, outer, inner = rings(4, 16, height=0.0)
        points[inner] *= 0.7
        normals_a = np.tile([0.0, 0.0, 2.0], (len(outer), 1))
        normals_b = np.tile([0.0, 0.0, 3.0], (len(inner), 1))
        before_a, before_b = normals_a.copy(), normals_b.copy()
        with self.assertRaisesRegex(ValueError, "zipper"):
            stitch_boundary_loops(
                points, outer, inner, max_bridge_length=0.5, outer_edge_normals=normals_a, inner_edge_normals=normals_b
            )
        np.testing.assert_array_equal(normals_a, before_a)
        np.testing.assert_array_equal(normals_b, before_b)

    def test_self_crossing_loop_is_rejected(self):
        points = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 0.5, 1.0],
                [0.5, 0.6, 1.0],
                [0.6, 0.6, 1.0],
                [0.6, 0.5, 1.0],
            ]
        )
        with self.assertRaisesRegex(ValueError, "not simple"):
            stitch_boundary_loops(points, [0, 1, 2, 3], [4, 5, 6, 7], max_bridge_length=2.0)

    def test_intersecting_loops_are_rejected(self):
        points, outer, inner = rings()
        points[inner[0]] = points[outer[0]]
        with self.assertRaisesRegex(ValueError, "intersect"):
            stitch_boundary_loops(points, outer, inner, max_bridge_length=1.0)

    def test_numerically_coincident_disjoint_loops_are_rejected(self):
        points, outer, inner = rings(8, 8, height=0.0)
        points[inner] = points[outer] * (1.0 - 1e-14)
        with self.assertRaisesRegex(ValueError, "numerically coincident"):
            stitch_boundary_loops(points, outer, inner, max_bridge_length=1.0)

    def test_sphere_cap_with_facet_normal_jump_over_twenty_degrees_is_rejected(self):
        source = trimesh.creation.icosphere(subdivisions=2)
        points, faces = np.asarray(source.vertices), np.asarray(source.faces)
        apex = int(np.argmax(points[:, 2]))
        removed = np.flatnonzero(np.any(faces == apex, axis=1))
        keep = np.setdiff1d(np.arange(len(faces)), removed)
        region = extract_defect_regions(
            points,
            faces[removed],
            seed_faces=np.arange(len(removed)),
            halo_rings=0,
            max_region_faces=100,
            max_total_faces=100,
        )
        outer = np.asarray(region["regions"][0]["boundary_loops"][0][:-1])
        center = points[outer].mean(axis=0)
        radial = points[outer] - center
        radial /= np.linalg.norm(radial, axis=1)[:, None]
        inner_points = points[outer] - radial * 5e-4
        patched_points = np.vstack((points, inner_points))
        patch_faces = faces[removed].copy()
        inner = np.arange(len(points), len(points) + len(outer))
        mapping = dict(zip(map(int, outer), map(int, inner)))
        patch_faces = np.asarray([[mapping.get(int(vertex), int(vertex)) for vertex in face] for face in patch_faces])
        source_normals = loop_normals_from_faces(points, faces[keep], outer[::-1])
        patch_normals = loop_normals_from_faces(patched_points, patch_faces, inner)
        max_bridge_length = float(
            np.linalg.norm(np.roll(points[outer], -1, axis=0) - points[outer], axis=1).max() * 1.01
        )
        geometric_diagnostic = stitch_boundary_loops(patched_points, outer, inner, max_bridge_length=max_bridge_length)
        self.assertGreater(len(geometric_diagnostic), 0)
        with self.assertRaisesRegex(ValueError, "zipper"):
            stitch_boundary_loops(
                patched_points,
                outer,
                inner,
                max_bridge_length=max_bridge_length,
                outer_edge_normals=source_normals,
                inner_edge_normals=patch_normals,
                max_normal_angle=20.0,
            )
        np.testing.assert_array_equal(patched_points[: len(points)], points)
        np.testing.assert_array_equal(patched_points[faces[keep]], points[faces[keep]])

    def test_smooth_sphere_cap_is_zipped_with_exact_global_acceptance(self):
        source = trimesh.creation.icosphere(subdivisions=3)
        points, faces = np.asarray(source.vertices), np.asarray(source.faces)
        apex = int(np.argmax(points[:, 2]))
        removed = np.flatnonzero(np.any(faces == apex, axis=1))
        keep = np.setdiff1d(np.arange(len(faces)), removed)
        region = extract_defect_regions(
            points,
            faces[removed],
            seed_faces=np.arange(len(removed)),
            halo_rings=0,
            max_region_faces=100,
            max_total_faces=100,
        )
        outer = np.asarray(region["regions"][0]["boundary_loops"][0][:-1])
        radial = points[outer] - points[outer].mean(axis=0)
        radial /= np.linalg.norm(radial, axis=1)[:, None]
        patched_points = np.vstack((points, points[outer] - radial * 1e-5))
        inner = np.arange(len(points), len(points) + len(outer))
        mapping = dict(zip(map(int, outer), map(int, inner)))
        patch_faces = np.asarray(
            [[mapping.get(int(vertex), int(vertex)) for vertex in face] for face in faces[removed]]
        )
        source_normals = loop_normals_from_faces(points, faces[keep], outer[::-1])
        patch_normals = loop_normals_from_faces(patched_points, patch_faces, inner)
        max_bridge_length = float(
            np.linalg.norm(np.roll(points[outer], -1, axis=0) - points[outer], axis=1).max() * 1.01
        )
        strip = stitch_boundary_loops(
            patched_points,
            outer,
            inner,
            max_bridge_length=max_bridge_length,
            outer_edge_normals=source_normals,
            inner_edge_normals=patch_normals,
            max_normal_angle=20.0,
        )
        result_faces = np.vstack((faces[keep], patch_faces, strip))
        result = trimesh.Trimesh(patched_points, result_faces, process=False)
        self.assertTrue(result.is_watertight)
        self.assertTrue(result.is_winding_consistent)
        np.testing.assert_array_equal(patched_points[: len(points)], points)
        np.testing.assert_array_equal(patched_points[result_faces[: len(keep)]], points[faces[keep]])
        exact = exact_self_intersection_report(patched_points, result_faces)
        self.assertTrue(exact["passed"], exact)


if __name__ == "__main__":
    unittest.main()
