import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from local_defect_regions import extract_defect_regions


class LocalDefectRegionsTest(unittest.TestCase):
    def test_adjacent_intersection_seeds_form_one_simple_loop_region(self):
        points = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
        faces = np.asarray([[0, 1, 2], [0, 2, 3]])
        points_before, faces_before = points.copy(), faces.copy()
        report = extract_defect_regions(
            points, faces, intersection_pairs=[[0, 1]], source_face_ids=[27, 31], halo_rings=0
        )
        self.assertEqual(len(report["regions"]), 1)
        region = report["regions"][0]
        self.assertEqual(region["status"], "ready")
        self.assertEqual(region["source_face_ids"], [27, 31])
        self.assertEqual(region["boundary"]["status"], "closed_loops")
        self.assertEqual(len(region["boundary_loops"]), 1)
        self.assertEqual(len(region["boundary_loops"][0]), 5)
        np.testing.assert_array_equal(points, points_before)
        np.testing.assert_array_equal(faces, faces_before)

    def test_far_defects_remain_separate_and_roi_padding_is_explicit(self):
        points = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [10.0, 0.0, 0.0],
                [11.0, 0.0, 0.0],
                [11.0, 1.0, 0.0],
                [10.0, 1.0, 0.0],
            ]
        )
        faces = np.asarray([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]])
        report = extract_defect_regions(points, faces, seed_faces=[0, 2], halo_rings=0, roi_padding=0.25)
        self.assertEqual(len(report["regions"]), 2)
        self.assertEqual(report["regions"][0]["bbox"], [[-0.25, -0.25, -0.25], [1.25, 1.25, 0.25]])
        self.assertEqual(report["regions"][1]["bbox"], [[9.75, -0.25, -0.25], [11.25, 1.25, 0.25]])
        self.assertEqual(report["overlapping_region_face_ids"], [])

    def test_edge_halo_is_local_and_records_every_source_face(self):
        points = np.asarray([[x, y, 0.0] for y in range(4) for x in range(4)])
        faces = []
        for y in range(3):
            for x in range(3):
                a = y * 4 + x
                faces.extend(([a, a + 1, a + 5], [a, a + 5, a + 4]))
        report = extract_defect_regions(points, np.asarray(faces), seed_faces=[8], halo_rings=1)
        ids = report["regions"][0]["source_face_ids"]
        self.assertEqual(len(ids), 4)
        self.assertIn(8, ids)
        self.assertTrue(all(0 <= value < len(faces) for value in ids))
        self.assertGreater(report["regions"][0]["boundary"]["loop_count"], 0)

    def test_vertex_branch_is_flagged_without_inventing_a_boundary_loop(self):
        points = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
        faces = np.asarray([[0, 1, 2], [0, 3, 4]])
        report = extract_defect_regions(points, faces, intersection_pairs=[[0, 1]], halo_rings=0)
        region = report["regions"][0]
        self.assertEqual(region["status"], "review_required")
        self.assertEqual(region["boundary"]["status"], "branched")
        self.assertEqual(region["boundary_loops"], [])

    def test_face_budget_rejects_roi_without_returning_a_truncated_patch(self):
        points = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
        faces = np.asarray([[0, 1, 2], [0, 2, 3]])
        report = extract_defect_regions(
            points, faces, seed_faces=[0], halo_rings=2, max_region_faces=1, max_total_faces=2
        )
        region = report["regions"][0]
        self.assertEqual(region["status"], "face_budget_exceeded")
        self.assertNotIn("source_face_ids", region)
        self.assertEqual(report["region_faces"], 0)

    def test_overlap_created_by_halos_is_reported(self):
        points = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [3.0, 1.0, 0.0],
                [2.0, 1.0, 0.0],
            ]
        )
        faces = np.asarray([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]])
        report = extract_defect_regions(points, faces, seed_faces=[0, 2], halo_rings=1, roi_padding=0.5)
        self.assertTrue(report["overlapping_region_rois"])


if __name__ == "__main__":
    unittest.main()
