import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyvista as pv
import trimesh

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from local_defect_regions import extract_defect_regions
from local_patch_repair import repair_local_voxel_patch, repair_mesh_patch
from local_patch_worker import run_file
from mesh_io import read_surface, triangle_faces
from source_projected_closure import _closest_source_points
from source_projected_validation import exact_self_intersection_report
from test_local_sdf_patch import perturbed_sphere_case


def folded_box_case():
    box = trimesh.creation.box(extents=(2.0, 2.0, 1.0))
    points, faces = np.asarray(box.vertices), np.asarray(box.faces)
    for _ in range(3):
        points, faces = trimesh.remesh.subdivide(points, faces)
    vertex = np.argmin(np.linalg.norm(points - [0.0, 0.0, 0.5], axis=1))
    points[vertex] += [0.43, 0.07, 0.0]
    centers = points[faces].mean(axis=1)
    rows = np.flatnonzero((centers[:, 2] > 0.49) & (np.abs(centers[:, 0]) < 0.76) & (np.abs(centers[:, 1]) < 0.76))
    return points, faces, rows


def double_fold_box_case():
    box = trimesh.creation.box(extents=(4.0, 2.0, 1.0))
    points, faces = np.asarray(box.vertices), np.asarray(box.faces)
    for _ in range(4):
        points, faces = trimesh.remesh.subdivide(points, faces)
    for x in (-1.0, 1.0):
        vertex = np.argmin(np.linalg.norm(points - [x, 0.0, 0.5], axis=1))
        points[vertex] += [0.4, 0.03, 0.0]
    centers = points[faces].mean(axis=1)
    rows = np.flatnonzero(
        (centers[:, 2] > 0.49) & (centers[:, 0] > -1.75) & (centers[:, 0] < -0.25) & (np.abs(centers[:, 1]) < 0.76)
    )
    return points, faces, rows


class LocalPatchRepairTest(unittest.TestCase):
    def test_dense_flat_patch_uses_a_shared_contour_without_relaxing_the_zipper_limit(self):
        points, faces, rows = folded_box_case()
        bad = np.unique(np.asarray(exact_self_intersection_report(points, faces)["reported_pairs"]))
        p, f, lineage, report = repair_local_voxel_patch(
            points, faces, rows, pitch=0.008, max_error=0.07, max_voxels=1000000, unreliable_source_face_rows=bad
        )
        self.assertTrue(report["accepted"], report.get("failure_reason"))
        reconstruction = report["reconstruction"]
        self.assertEqual(reconstruction["method"], "local_sdf_shared_source_contour")
        self.assertGreater(reconstruction["zipper"]["contour_nodes"], 512)
        self.assertEqual(reconstruction["zipper"]["max_normal_angle"], 20.0)
        self.assertEqual(reconstruction["zipper"]["max_seam_dihedral_deg"], 0.0)
        np.testing.assert_array_equal(p[f[lineage >= 0]], points[faces[lineage[lineage >= 0]]])

    def test_exact_cut_inherits_source_cell_fields_and_retains_native_normals(self):
        points, faces, rows = folded_box_case()
        source = pv.PolyData(points, np.column_stack((np.full(len(faces), 3), faces)).ravel())
        normals = source.compute_normals(cell_normals=False).point_data["Normals"].astype(np.float64)
        source.point_data["Normals"] = normals
        source.cell_data["corner_normals"] = normals[faces].reshape(-1, 9)
        source.cell_data["part_id"] = np.arange(len(faces), dtype=np.int64)
        candidate, report = repair_mesh_patch(source, rows, method="intersection-cut", max_error=0.08)
        self.assertTrue(report["accepted"], report.get("failure_reason"))
        lineage = np.asarray(candidate.cell_data["local_patch_source_face_row"])
        births = np.asarray(report["reconstruction"]["normal_source_face_rows"])
        np.testing.assert_array_equal(candidate.cell_data["part_id"], source.cell_data["part_id"][births])
        np.testing.assert_array_equal(
            candidate.cell_data["corner_normals"][lineage >= 0],
            source.cell_data["corner_normals"][lineage[lineage >= 0]],
        )
        self.assertGreater(np.count_nonzero(lineage < 0), 0)

    def test_unresolved_sharp_contour_rolls_back_without_relaxing_the_seam_limit(self):
        box = trimesh.creation.box(extents=(2.0, 2.0, 1.0))
        points, faces = np.asarray(box.vertices), np.asarray(box.faces)
        for _ in range(3):
            points, faces = trimesh.remesh.subdivide(points, faces)
        rows = np.flatnonzero(np.linalg.norm(points[faces].mean(axis=1) - [0.0, 1.0, 0.5], axis=1) < 0.7)
        p, f, lineage, report = repair_local_voxel_patch(
            points, faces, rows, pitch=0.04, max_error=0.07, max_voxels=300000, unreliable_source_face_rows=[]
        )
        self.assertFalse(report["accepted"])
        self.assertTrue(report["rolled_back"])
        self.assertIn("20 degree dihedral limit", report["failure_reason"])
        np.testing.assert_array_equal(p, points)
        np.testing.assert_array_equal(f, faces)
        np.testing.assert_array_equal(lineage, np.arange(len(faces)))

    def test_local_success_does_not_accept_unrepaired_defects_elsewhere(self):
        points, faces, rows = double_fold_box_case()
        p, f, lineage, report = repair_local_voxel_patch(
            points, faces, rows, pitch=0.06, max_error=0.08, max_voxels=100000
        )
        self.assertTrue(report["local_repair_accepted"], report.get("failure_reason"))
        self.assertFalse(report["accepted"])
        self.assertGreater(report["local_geometry"]["remaining_outside_intersection_pairs"], 0)
        self.assertEqual(report["local_geometry"]["generated_intersection_pairs"], 0)
        kept = lineage >= 0
        np.testing.assert_array_equal(p[f[kept]], points[faces[lineage[kept]]])

    def test_actual_source_self_intersection_is_removed_locally(self):
        points, faces, rows = folded_box_case()
        defects = exact_self_intersection_report(points, faces)
        self.assertFalse(defects["passed"])
        bad = np.unique(np.asarray(defects["reported_pairs"]))
        p, f, lineage, report = repair_local_voxel_patch(
            points, faces, rows, pitch=0.04, max_error=0.07, max_voxels=100000, unreliable_source_face_rows=bad
        )
        self.assertTrue(report["local_repair_accepted"], report.get("failure_reason"))
        self.assertTrue(report["accepted"])
        self.assertTrue(exact_self_intersection_report(p, f)["passed"])
        projection = report["reconstruction"]["source_projection"]
        self.assertTrue(projection["boundary_unchanged"])
        self.assertGreater(projection["projected_vertices"], 0)
        kept = lineage >= 0
        np.testing.assert_array_equal(p[f[kept]], points[faces[lineage[kept]]])

    def test_local_replacement_preserves_outside_geometry_and_passes_whole_mesh(self):
        points, faces, rows, _ = perturbed_sphere_case()
        before_p, before_f = points.copy(), faces.copy()
        result_p, result_f, lineage, report = repair_local_voxel_patch(
            points, faces, rows, pitch=0.04, max_error=0.07, max_voxels=100000
        )
        self.assertTrue(report["accepted"], report.get("failure_reason"))
        self.assertTrue(report["geometry"]["passed"])
        self.assertTrue(report["source_distance"]["candidate_to_source"]["passed"])
        self.assertTrue(report["source_distance"]["source_to_candidate"]["passed"])
        self.assertLessEqual(report["seam"]["max_increase_deg"], 20.0)
        np.testing.assert_array_equal(points, before_p)
        np.testing.assert_array_equal(faces, before_f)
        np.testing.assert_array_equal(result_p[: len(points)], points)
        retained = lineage >= 0
        np.testing.assert_array_equal(result_f[retained], faces[lineage[retained]])

    def test_budget_failure_returns_unchanged_source(self):
        points, faces, rows, _ = perturbed_sphere_case()
        result_p, result_f, lineage, report = repair_local_voxel_patch(
            points, faces, rows, pitch=0.04, max_error=0.07, max_voxels=10
        )
        self.assertFalse(report["accepted"])
        self.assertTrue(report["rolled_back"])
        np.testing.assert_array_equal(result_p, points)
        np.testing.assert_array_equal(result_f, faces)
        np.testing.assert_array_equal(lineage, np.arange(len(faces)))

    def test_native_corner_normals_are_identical_on_retained_faces(self):
        points, faces, rows, _ = perturbed_sphere_case()
        source = pv.PolyData(points, np.column_stack((np.full(len(faces), 3), faces)).ravel())
        normals = source.compute_normals(cell_normals=False).point_data["Normals"].astype(np.float64)
        source.point_data["Normals"] = normals
        source.cell_data["corner_normals"] = normals[faces].reshape(-1, 9)
        source.cell_data["part_id"] = np.arange(len(faces), dtype=np.int64)
        candidate, report = repair_mesh_patch(source, rows, pitch=0.04, max_error=0.07, max_voxels=100000)
        self.assertTrue(report["accepted"], report.get("failure_reason"))
        self.assertEqual(report["attributes"]["native_normal_transport"]["trusted_source_faces"], 0)
        lineage = np.asarray(candidate.cell_data["local_patch_source_face_row"])
        kept = lineage >= 0
        np.testing.assert_array_equal(
            candidate.cell_data["corner_normals"][kept], source.cell_data["corner_normals"][lineage[kept]]
        )
        np.testing.assert_array_equal(candidate.cell_data["part_id"][kept], lineage[kept])
        point_rows = np.asarray(candidate.point_data["local_patch_source_point_row"])
        known = point_rows >= 0
        np.testing.assert_array_equal(candidate.point_data["Normals"][known], normals[point_rows[known]])

    def test_generated_corner_normals_follow_trusted_native_field(self):
        points, faces, rows, _ = perturbed_sphere_case()
        native_offset = np.asarray([0.2, -0.1, 0.15])
        native = points + native_offset
        source = pv.PolyData(points, np.column_stack((np.full(len(faces), 3), faces)).ravel())
        selected = np.asarray(
            extract_defect_regions(
                points, faces, seed_faces=rows, halo_rings=0, max_region_faces=2000, max_total_faces=2000
            )["regions"][0]["face_rows"]
        )
        centers = points[faces[selected]].mean(axis=1)
        bad_face = int(selected[np.argmin(np.linalg.norm(centers - centers.mean(axis=0), axis=1))])
        sentinel = np.asarray([25.0, -31.0, 42.0])
        corner_field = native[faces].copy()
        corner_field[bad_face] = sentinel
        source.point_data["Normals"] = points / np.linalg.norm(points, axis=1, keepdims=True)
        source.cell_data["corner_normals"] = corner_field.reshape(-1, 9)
        candidate, report = repair_mesh_patch(
            source,
            rows,
            pitch=0.04,
            max_error=0.07,
            max_voxels=100000,
            unreliable_source_face_rows=[bad_face],
        )
        self.assertTrue(report["accepted"], report.get("failure_reason"))
        lineage = np.asarray(candidate.cell_data["local_patch_source_face_row"])
        replaced = np.setdiff1d(np.arange(len(faces)), lineage[lineage >= 0])
        np.testing.assert_array_equal(selected, replaced)
        trusted = np.setdiff1d(replaced, [bad_face])
        self.assertNotIn(bad_face, trusted)

        result_faces = triangle_faces(candidate)
        generated = np.flatnonzero(lineage < 0)
        reconstruction = report["reconstruction"]
        collar_count = len(reconstruction.get("collar_source_face_rows", []))
        collar_start = int(np.count_nonzero(lineage >= 0)) + reconstruction["patch"]["triangles"]
        collar_rows = np.arange(collar_start, collar_start + collar_count)
        transport_rows = np.setdiff1d(generated, collar_rows, assume_unique=True)
        queries = candidate.points[result_faces[transport_rows]].reshape(-1, 3)
        closest, local_ids, distances = _closest_source_points(points, faces[trusted], queries)
        generated_triangles = candidate.points[result_faces[transport_rows]]
        generated_normals = np.cross(
            generated_triangles[:, 1] - generated_triangles[:, 0], generated_triangles[:, 2] - generated_triangles[:, 0]
        )
        generated_normals /= np.maximum(np.linalg.norm(generated_normals, axis=1)[:, None], 1e-20)
        support_normals = np.cross(
            points[faces[trusted[local_ids], 1]] - points[faces[trusted[local_ids], 0]],
            points[faces[trusted[local_ids], 2]] - points[faces[trusted[local_ids], 0]],
        )
        support_normals /= np.maximum(np.linalg.norm(support_normals, axis=1)[:, None], 1e-20)
        alignment = np.einsum("ij,ij->i", np.repeat(generated_normals, 3, axis=0), support_normals)
        supported = np.isfinite(queries).all(axis=1) & np.isfinite(closest).all(axis=1) & np.isfinite(distances)
        supported &= distances <= min(2 * 0.04, 0.07)
        supported &= alignment >= np.cos(np.deg2rad(20.0))
        expected = closest + native_offset
        actual = np.asarray(candidate.cell_data["corner_normals"])[transport_rows].reshape(-1, 3)
        self.assertGreater(int(supported.sum()), 0)
        np.testing.assert_allclose(actual[supported], expected[supported], atol=1e-6)
        self.assertTrue(np.all(np.linalg.norm(actual - sentinel, axis=1) > 1.0))

        bad_closest, _, bad_distances = _closest_source_points(points, faces[[bad_face]], queries)
        bad_triangle = points[faces[bad_face]]
        bad_normal = np.cross(bad_triangle[1] - bad_triangle[0], bad_triangle[2] - bad_triangle[0])
        bad_normal /= np.linalg.norm(bad_normal)
        bad_alignment = np.einsum("ij,j->i", np.repeat(generated_normals, 3, axis=0), bad_normal)
        near_bad = np.isfinite(bad_closest).all(axis=1) & (bad_distances <= min(2 * 0.04, 0.07))
        near_bad &= bad_alignment >= np.cos(np.deg2rad(20.0))
        self.assertGreater(int(near_bad.sum()), 0)
        self.assertTrue(np.all(np.linalg.norm(actual[near_bad] - sentinel, axis=1) > 1.0))
        transport_report = report["attributes"]["native_normal_transport"]
        self.assertEqual(transport_report["trusted_source_faces"], len(trusted))
        self.assertEqual(transport_report["generated_corners_supported_by_trusted_faces"], int(supported.sum()))
        self.assertEqual(transport_report["collar_birth_corners_transported"], 3 * collar_count)

    def test_region_packet_is_bound_to_current_geometry(self):
        points, faces, rows, _ = perturbed_sphere_case()
        packet = extract_defect_regions(points, faces, seed_faces=rows, halo_rings=0)
        self.assertIn("source_geometry_sha256", packet)
        self.assertIn("face_rows", packet["regions"][0])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            changed = points.copy()
            changed[0] += 0.1
            source = pv.PolyData(changed, np.column_stack((np.full(len(faces), 3), faces)).ravel())
            source.save(root / "source.vtp")
            (root / "regions.json").write_text(json.dumps(packet))
            args = SimpleNamespace(input=root / "source.vtp", regions=root / "regions.json")
            with self.assertRaisesRegex(ValueError, "does not match"):
                run_file(args)

    def test_cli_deadline_never_leaves_an_accepted_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "run"
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS / "local_patch_worker.py"),
                    str(root / "source.vtp"),
                    "--regions",
                    str(root / "regions.json"),
                    "--region",
                    "unused",
                    "--pitch",
                    ".04",
                    "--max-error",
                    ".07",
                    "--output-dir",
                    str(output),
                    "--timeout-seconds",
                    ".001",
                ],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            self.assertEqual(result.returncode, 124, result.stderr)
            report = json.loads((output / "local_patch_report.json").read_text())
            self.assertFalse(report["accepted"])
            self.assertIsNone(report["outputs"]["accepted_mesh_vtp"])

    def test_file_worker_reports_partial_result_with_distinct_exit_code(self):
        points, faces, rows = double_fold_box_case()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = pv.PolyData(points, np.column_stack((np.full(len(faces), 3), faces)).ravel())
            source.save(root / "source.vtp")
            loaded = read_surface(root / "source.vtp")
            packet = extract_defect_regions(loaded.points, triangle_faces(loaded), seed_faces=rows, halo_rings=0)
            (root / "regions.json").write_text(json.dumps(packet))
            args = SimpleNamespace(
                input=root / "source.vtp",
                regions=root / "regions.json",
                method="voxel",
                region=packet["regions"][0]["id"],
                output_dir=root,
                pitch=0.06,
                max_error=0.08,
                max_voxels=100000,
                max_seam_angle_increase=20.0,
            )
            self.assertEqual(run_file(args), 3)
            report = json.loads((root / "local_patch_report.json").read_text())
            self.assertTrue(report["local_repair_accepted"])
            self.assertFalse(report["accepted"])
            self.assertIsNone(report["outputs"]["accepted_mesh_vtp"])
            self.assertTrue(Path(report["outputs"]["accepted_local_candidate_vtp"]).exists())


if __name__ == "__main__":
    unittest.main()
