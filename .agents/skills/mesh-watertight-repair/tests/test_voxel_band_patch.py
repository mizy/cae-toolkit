import sys
import unittest
from pathlib import Path

import numpy as np
import trimesh

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from local_defect_regions import extract_defect_regions
from source_projected_validation import exact_self_intersection_report
from voxel_band_patch import reconstruct_voxel_band


def _box(*, folded):
    mesh = trimesh.creation.box(extents=(2.0, 2.0, 1.0))
    points, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    for _ in range(3):
        points, faces = trimesh.remesh.subdivide(points, faces)
    centers = points[faces].mean(axis=1)
    if folded:
        vertex = np.argmin(np.linalg.norm(points - [0.0, 0.0, 0.5], axis=1))
        points[vertex] += [0.43, 0.07, 0.0]
        intersections = exact_self_intersection_report(points, faces)
        defect = np.unique(np.asarray(intersections["reported_pairs"], dtype=np.int64))
    else:
        top = np.flatnonzero(
            (centers[:, 2] > 0.49) & (centers[:, 0] > 0.65) & (centers[:, 0] < 0.9)
        )
        defect = top[np.argmin(np.abs(centers[top, 1])) :][:1]
    packet = extract_defect_regions(
        points,
        faces,
        seed_faces=defect,
        halo_rings=4,
        max_region_faces=2000,
        max_total_faces=2000,
    )
    region = np.asarray(packet["regions"][0]["face_rows"], dtype=np.int64)
    return points, faces, defect, region


class VoxelBandPatchTest(unittest.TestCase):
    def test_sharp_box_reconstructs_both_strips_in_one_fixed_boundary_band(self):
        points, faces, defect, region = _box(folded=False)
        candidate_points, candidate_faces, lineage, report = reconstruct_voxel_band(
            points,
            faces,
            region,
            pitch=0.08,
            max_error=0.08,
            max_voxels=100_000,
            unreliable_source_face_rows=defect,
        )
        self.assertTrue(report.get("accepted"), report.get("failure_reason"))
        self.assertGreater(report["band"]["ring_face_counts"][0], 0)
        self.assertGreater(sum(report["band"]["ring_face_counts"][1:]), 0)
        self.assertEqual(report["band"]["reconstructed_faces"], len(region))
        self.assertEqual(report["grid"]["max_voxels"], 100_000)
        self.assertLessEqual(report["grid"]["voxels"], 100_000)
        self.assertEqual(report["boundary"]["max_displacement"], 0.0)
        self.assertEqual(report["reconstruction"]["separate_collar_or_zipper_faces"], 0)
        self.assertTrue(report["outside_source"]["exactly_preserved"])
        self.assertTrue(report["geometry"]["exact"]["passed"])
        self.assertEqual(len(lineage), len(candidate_faces))
        self.assertGreater(len(candidate_points), len(points))
        self.assertGreater(len(candidate_faces), len(faces))

    def test_folded_box_repair_must_certify_exact_intersections_and_source_bound(self):
        points, faces, defect, region = _box(folded=True)
        candidate_points, candidate_faces, lineage, report = reconstruct_voxel_band(
            points,
            faces,
            region,
            pitch=0.06,
            max_error=0.08,
            max_voxels=150_000,
            unreliable_source_face_rows=defect,
        )
        self.assertTrue(report.get("accepted"), report.get("failure_reason"))
        self.assertEqual(
            report["source_intersections"]["pairs"],
            len(exact_self_intersection_report(points, faces)["reported_pairs"]),
        )
        self.assertEqual(report["boundary"]["max_displacement"], 0.0)
        self.assertTrue(report["source_deviation"]["passed"])
        self.assertTrue(report["geometry"]["exact"]["passed"])
        self.assertEqual(report["geometry"]["exact"]["intersection_pairs"], 0)
        self.assertEqual(len(candidate_faces), report["output_faces"])
        self.assertEqual(len(lineage), len(candidate_faces))
        np.testing.assert_array_equal(candidate_points[: len(points)], points)


if __name__ == "__main__":
    unittest.main()
