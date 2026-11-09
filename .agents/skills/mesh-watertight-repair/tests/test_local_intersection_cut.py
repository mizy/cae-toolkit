import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import trimesh

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import local_intersection_cut
from local_intersection_cut import repair_local_intersection_cut
from source_projected_validation import exact_self_intersection_report
from test_local_patch_repair import folded_box_case


class LocalIntersectionCutTest(unittest.TestCase):
    def test_first_only_and_pair_limit_do_not_claim_complete_intersection_inventory(self):
        points, faces, _ = folded_box_case()
        limited = exact_self_intersection_report(points, faces, max_reported_pairs=1)
        self.assertFalse(limited["passed"])
        self.assertTrue(limited["reported_pairs_truncated"])
        self.assertTrue(limited["intersection_pairs_is_lower_bound"])
        first = exact_self_intersection_report(points, faces, first_only=True)
        self.assertFalse(first["passed"])
        self.assertTrue(first["reported_pairs_truncated"])
        self.assertTrue(first["intersection_pairs_is_lower_bound"])
        full = exact_self_intersection_report(points, faces)
        self.assertFalse(full["reported_pairs_truncated"])

    def test_folded_box_cut_preserves_outside_and_unchanged_inside_faces(self):
        points, faces, rows = folded_box_case()
        original_points, original_faces = points.copy(), faces.copy()

        output_points, output_faces, lineage, report = repair_local_intersection_cut(
            points,
            faces,
            rows,
            max_error=0.07,
        )

        self.assertTrue(report["accepted"], report.get("failure_reason"))
        self.assertTrue(report["local_repair_accepted"])
        self.assertTrue(report["geometry"]["passed"])
        self.assertTrue(report["local_geometry"]["passed"])
        self.assertTrue(report["source_distance"]["source_to_candidate"]["passed"])
        self.assertTrue(report["source_distance"]["candidate_to_source"]["passed"])
        self.assertLessEqual(report["seam"]["max_increase_deg"], 20.0)
        self.assertEqual(report["reconstruction"]["method"], "exact_source_intersection_cut")
        self.assertEqual(report["reconstruction"]["collar_source_face_rows"], [])
        self.assertEqual(report["reconstruction"]["patch"]["triangles"], int(np.count_nonzero(lineage < 0)))

        retained = lineage >= 0
        np.testing.assert_array_equal(output_points[: len(points)], points)
        np.testing.assert_array_equal(output_faces[retained], faces[lineage[retained]])
        np.testing.assert_array_equal(points, original_points)
        np.testing.assert_array_equal(faces, original_faces)
        births = np.asarray(report["reconstruction"]["normal_source_face_rows"], dtype=np.int64)
        self.assertEqual(births.shape, (len(output_faces),))
        np.testing.assert_array_equal(births[retained], lineage[retained])

        preservation = report["face_preservation"]
        self.assertTrue(preservation["outside_roi"]["all_preserved"])
        self.assertEqual(
            preservation["outside_roi"]["source_faces"],
            preservation["outside_roi"]["exact_oriented_faces_preserved_once"],
        )
        self.assertEqual(
            report["replaced_source_faces"],
            len(preservation["inside_roi"]["changed_source_rows"]),
        )
        self.assertTrue(exact_self_intersection_report(output_points, output_faces)["passed"])

    def test_changed_face_outside_roi_rolls_back(self):
        box = trimesh.creation.box(extents=(2.0, 2.0, 1.0))
        points, faces = np.asarray(box.vertices), np.asarray(box.faces)
        selected = np.flatnonzero(box.face_normals[:, 2] > 0.5)
        outside_row = int(np.flatnonzero(~np.isin(np.arange(len(faces)), selected))[0])
        candidate_points = np.vstack((points, points[faces[outside_row]].mean(axis=0) + [0.0, 0.0, 0.01]))
        candidate_faces = faces.copy()
        candidate_faces[outside_row, 0] = len(points)
        births = np.arange(len(faces), dtype=np.int64)
        original_points, original_faces = points.copy(), faces.copy()

        with patch.object(
            local_intersection_cut,
            "_union",
            return_value=(candidate_points, candidate_faces, births),
        ):
            result_points, result_faces, lineage, report = repair_local_intersection_cut(
                points,
                faces,
                selected,
                max_error=0.1,
            )

        self.assertFalse(report["accepted"])
        self.assertTrue(report["rolled_back"])
        self.assertIn("outside the selected region", report["failure_reason"])
        self.assertEqual(report["face_preservation"]["outside_roi"]["failed_source_rows"], [outside_row])
        np.testing.assert_array_equal(result_points, original_points)
        np.testing.assert_array_equal(result_faces, original_faces)
        np.testing.assert_array_equal(lineage, np.arange(len(faces)))

    def test_tight_source_error_budget_rolls_back(self):
        points, faces, rows = folded_box_case()
        original_points, original_faces = points.copy(), faces.copy()

        result_points, result_faces, lineage, report = repair_local_intersection_cut(
            points,
            faces,
            rows,
            max_error=1e-8,
        )

        self.assertFalse(report["accepted"])
        self.assertTrue(report["rolled_back"])
        self.assertIn("distance", report["failure_reason"])
        np.testing.assert_array_equal(result_points, original_points)
        np.testing.assert_array_equal(result_faces, original_faces)
        np.testing.assert_array_equal(lineage, np.arange(len(faces)))

    def test_open_source_is_rejected_as_outside_operator_domain(self):
        box = trimesh.creation.box(extents=(2.0, 2.0, 1.0))
        points, faces = np.asarray(box.vertices), np.asarray(box.faces[:-1])

        result_points, result_faces, lineage, report = repair_local_intersection_cut(
            points,
            faces,
            np.asarray([0]),
            max_error=0.1,
        )

        self.assertFalse(report["accepted"])
        self.assertTrue(report["rolled_back"])
        self.assertIn("closed", report["failure_reason"])
        np.testing.assert_array_equal(result_points, points)
        np.testing.assert_array_equal(result_faces, faces)
        np.testing.assert_array_equal(lineage, np.arange(len(faces)))


if __name__ == "__main__":
    unittest.main()
