import sys
import unittest
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from local_defect_regions import extract_defect_regions
from source_contour_stitch import (
    _pava,
    _project_to_contour,
    _single_boundary_loop,
    clip_source_contour,
    stitch_source_contour,
)
from source_patch_collar import build_source_collar


def planar_roi():
    mesh = trimesh.creation.box(extents=(2.0, 2.0, 1.0))
    for _ in range(3):
        mesh = mesh.subdivide()
    points = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    centers = points[faces].mean(axis=1)
    seeds = np.flatnonzero((centers[:, 2] > 0.49) & (np.linalg.norm(centers[:, :2], axis=1) < 0.72))
    packet = extract_defect_regions(
        points,
        faces,
        seed_faces=seeds,
        halo_rings=0,
        max_region_faces=2000,
        max_total_faces=2000,
    )
    if len(packet["regions"]) != 1 or packet["regions"][0]["status"] != "ready":
        raise AssertionError("synthetic planar ROI must have one closed component")
    region = packet["regions"][0]
    return mesh, points, faces, np.asarray(region["face_rows"], dtype=np.int64), np.asarray(region["boundary_loops"][0])


class SourceContourStitchTest(unittest.TestCase):
    def test_pava_merges_adjacent_blocks_without_losing_their_ranges(self):
        np.testing.assert_allclose(_pava(np.asarray([2.0, 1.0, 3.0])), [1.5, 1.5, 3.0])
        np.testing.assert_allclose(_pava(np.asarray([4.0, 1.0, 3.0, 2.0, 5.0])), [2.5, 2.5, 2.5, 2.5, 5.0])

    def test_scalar_clip_uses_the_source_collar_level(self):
        _, points, faces, rows, source_loop = planar_roi()
        patch_points, patch_faces, report = clip_source_contour(
            points,
            faces[rows],
            points,
            faces,
            rows,
            source_loop,
            fraction=0.25,
        )
        self.assertEqual(report["status"], "ready")
        self.assertEqual(report["boundary_loop_count"], 1)
        self.assertGreater(report["source_owner_rows_selected"], 0)
        self.assertTrue(np.isfinite(patch_points).all())
        self.assertEqual(patch_faces.shape[1], 3)
        self.assertEqual(report["clipped_scalar_tolerance"], 8 * np.finfo(np.float32).eps)
        self.assertGreaterEqual(report["clipped_scalar_min"], 0.25 - report["clipped_scalar_tolerance"])

        collar_points, _, inner_loop, _, _ = build_source_collar(points, faces, rows, fraction=0.25)
        contour = np.vstack((points, collar_points))[np.asarray(inner_loop, dtype=np.int64)]
        loop_ids = np.asarray(_single_boundary_loop(patch_faces), dtype=np.int64)
        projected, _, distances, _ = _project_to_contour(patch_points[loop_ids[:-1]], contour)
        self.assertLess(float(distances.max()), 1e-10)
        np.testing.assert_allclose(projected, patch_points[loop_ids[:-1]], atol=1e-10)

    def test_welding_inserts_source_corners_and_splits_the_collar(self):
        _, points, faces, rows, source_loop = planar_roi()
        collar_points, _, inner_loop, _, _ = build_source_collar(points, faces, rows, fraction=0.25)
        source_collar = np.vstack((points, collar_points))
        contour = source_collar[np.asarray(inner_loop, dtype=np.int64)]
        edge_midpoints = (contour[:-1] + contour[1:]) * 0.5
        center = edge_midpoints.mean(axis=0)
        patch_points = np.vstack((center, edge_midpoints))
        count = len(edge_midpoints)
        patch_faces = np.asarray([[0, i + 1, (i + 1) % count + 1] for i in range(count)], dtype=np.int64)
        patch_loop = np.r_[np.arange(1, count + 1), 1]

        joined_points, body_faces, report = stitch_source_contour(
            points,
            faces,
            rows,
            source_loop,
            patch_points,
            patch_faces,
            patch_loop,
            max_displacement=1e-8,
            fraction=0.25,
        )
        self.assertEqual(report["zipper"]["status"], "ready")
        self.assertEqual(report["zipper"]["contour_nodes"], count * 2)
        self.assertGreater(len(body_faces), len(patch_faces))
        self.assertGreater(report["collar_faces_count"], 0)
        self.assertEqual(report["collar_faces_count"], len(report["collar_source_face_rows"]))
        self.assertEqual(report["patch"]["triangles"], len(body_faces))
        self.assertEqual(report["topology"]["boundary_edges"], len(source_loop) - 1)
        np.testing.assert_allclose(joined_points[: len(collar_points)], collar_points)

    def test_boundary_displacement_budget_fails_closed(self):
        _, points, faces, rows, source_loop = planar_roi()
        collar_points, _, inner_loop, _, _ = build_source_collar(points, faces, rows, fraction=0.25)
        contour = np.vstack((points, collar_points))[np.asarray(inner_loop, dtype=np.int64)]
        boundary = (contour[:-1] + contour[1:]) * 0.5
        center = boundary.mean(axis=0)
        patch_points = np.vstack((center, boundary + np.array([0.0, 0.0, 0.01])))
        count = len(boundary)
        patch_faces = np.asarray([[0, i + 1, (i + 1) % count + 1] for i in range(count)], dtype=np.int64)
        patch_loop = np.r_[np.arange(1, count + 1), 1]
        with self.assertRaisesRegex(ValueError, "displacement budget"):
            stitch_source_contour(
                points,
                faces,
                rows,
                source_loop,
                patch_points,
                patch_faces,
                patch_loop,
                max_displacement=1e-4,
            )


if __name__ == "__main__":
    unittest.main()
