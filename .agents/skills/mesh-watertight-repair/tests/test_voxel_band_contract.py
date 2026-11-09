import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyvista as pv
import trimesh

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from local_defect_regions import extract_defect_regions
from mesh_metrics import edge_topology
from local_patch_repair import repair_mesh_patch
from mesh_io import read_surface, triangle_faces
from source_projected_validation import exact_self_intersection_report
from test_local_patch_repair import folded_box_case
from voxel_band_patch import reconstruct_voxel_band


def _source_with_native_normals(points, faces):
    source = pv.PolyData(points, np.column_stack((np.full(len(faces), 3), faces)).ravel())
    normals = source.compute_normals(cell_normals=False, split_vertices=False).point_data["Normals"].copy()
    face_normals = np.cross(
        points[faces[:, 1]] - points[faces[:, 0]],
        points[faces[:, 2]] - points[faces[:, 0]],
    )
    face_normals /= np.maximum(np.linalg.norm(face_normals, axis=1)[:, None], 1e-20)
    source.point_data["Normals"] = normals
    source.cell_data["corner_normals"] = np.repeat(face_normals[:, None, :], 3, axis=1).reshape(-1, 9)
    source.cell_data["part_id"] = np.arange(len(faces), dtype=np.int32)
    return source


def _sharp_edge_case():
    box = trimesh.creation.box(extents=(2.0, 2.0, 1.0))
    points, faces = np.asarray(box.vertices), np.asarray(box.faces)
    for _ in range(3):
        points, faces = trimesh.remesh.subdivide(points, faces)
    centers = points[faces].mean(axis=1)
    selected = np.flatnonzero(np.linalg.norm(centers - [0.0, 1.0, 0.5], axis=1) < 0.7)
    seed = np.asarray([selected[np.argmin(np.linalg.norm(centers[selected] - [0.0, 1.0, 0.5], axis=1))]])
    return points, faces, selected, seed


def _crease_angles(points, faces):
    normals = np.cross(
        points[faces[:, 1]] - points[faces[:, 0]],
        points[faces[:, 2]] - points[faces[:, 0]],
    )
    normals /= np.maximum(np.linalg.norm(normals, axis=1)[:, None], 1e-20)
    _, edge_faces = edge_topology(faces)
    angles = []
    for (left, right), incident in edge_faces.items():
        endpoints = points[[left, right]]
        on_original_crease = (
            np.all(np.abs(endpoints[:, 1] - 1.0) < 1e-8)
            and np.all(np.abs(endpoints[:, 2] - 0.5) < 1e-8)
            and np.max(np.abs(endpoints[:, 0])) < 0.7
        )
        if on_original_crease and len(incident) == 2:
            dot = float(np.clip(np.dot(normals[incident[0]], normals[incident[1]]), -1.0, 1.0))
            angles.append(float(np.degrees(np.arccos(dot))))
    return np.asarray(angles)


class VoxelBandContractTest(unittest.TestCase):
    def test_folded_box_repair_preserves_native_data_and_passes_separate_gates(self):
        points, faces, region = folded_box_case()
        source = _source_with_native_normals(points, faces)
        original_points = np.asarray(source.points).copy()
        original_faces = triangle_faces(source).copy()
        before = exact_self_intersection_report(original_points, original_faces)
        self.assertFalse(before["passed"])
        bad_rows = np.unique(np.asarray(before["reported_pairs"], dtype=np.int64))

        candidate, report = repair_mesh_patch(
            source,
            region,
            method="voxel-band",
            pitch=0.06,
            max_error=0.08,
            max_voxels=150_000,
            unreliable_source_face_rows=bad_rows,
        )

        self.assertTrue(report["accepted"], report.get("failure_reason"))
        self.assertTrue(report["source_deviation"]["passed"])
        self.assertTrue(report["geometry"]["passed"])
        topology = report["geometry"]["metrics"]["topology"]
        self.assertEqual(topology["boundary_edges"], 0)
        self.assertEqual(topology["non_manifold_edges"], 0)
        self.assertEqual(topology["non_manifold_vertices"], 0)
        self.assertTrue(report["geometry"]["exact"]["passed"])
        self.assertEqual(report["geometry"]["exact"]["intersection_pairs"], 0)
        self.assertLessEqual(report["grid"]["voxels"], report["grid"]["max_voxels"])
        self.assertGreater(report["grid"]["voxels"], 0)
        full_shape = np.ceil((np.ptp(original_points, axis=0) + 4 * 0.06) / 0.06).astype(int) + 1
        self.assertLess(report["grid"]["voxels"], int(np.prod(full_shape)))
        self.assertEqual(report["boundary"]["max_displacement"], 0.0)

        lineage = np.asarray(candidate.cell_data["local_patch_source_face_row"], dtype=np.int64)
        retained = lineage >= 0
        expected_outside = np.setdiff1d(np.arange(len(faces)), region)
        np.testing.assert_array_equal(np.sort(lineage[retained]), expected_outside)
        candidate_faces = triangle_faces(candidate)
        np.testing.assert_array_equal(
            candidate.points[candidate_faces[retained]], original_points[original_faces[lineage[retained]]]
        )
        np.testing.assert_array_equal(
            candidate.cell_data["corner_normals"][retained],
            source.cell_data["corner_normals"][lineage[retained]],
        )
        np.testing.assert_array_equal(
            candidate.cell_data["part_id"][retained], source.cell_data["part_id"][lineage[retained]]
        )
        point_rows = np.asarray(candidate.point_data["local_patch_source_point_row"], dtype=np.int64)
        known = point_rows >= 0
        np.testing.assert_array_equal(candidate.points[known], original_points[point_rows[known]])
        np.testing.assert_array_equal(
            candidate.point_data["Normals"][known], source.point_data["Normals"][point_rows[known]]
        )
        after = exact_self_intersection_report(candidate.points, candidate_faces)
        self.assertTrue(after["passed"], after)
        np.testing.assert_array_equal(source.points, original_points)
        np.testing.assert_array_equal(triangle_faces(source), original_faces)

    def test_sharp_box_edge_region_is_accepted(self):
        points, faces, region, seed = _sharp_edge_case()
        source = _source_with_native_normals(points, faces)
        candidate, report = repair_mesh_patch(
            source,
            region,
            method="voxel-band",
            pitch=0.04,
            max_error=0.07,
            max_voxels=300_000,
            unreliable_source_face_rows=seed,
        )

        self.assertTrue(report["accepted"], report.get("failure_reason"))
        self.assertTrue(report["source_deviation"]["passed"])
        self.assertTrue(report["geometry"]["exact"]["passed"])
        self.assertEqual(report["geometry"]["metrics"]["topology"]["boundary_edges"], 0)
        self.assertLessEqual(report["grid"]["voxels"], 300_000)
        self.assertGreater(len(triangle_faces(candidate)), len(faces))
        crease = _crease_angles(np.asarray(candidate.points), triangle_faces(candidate))
        self.assertGreater(len(crease), 0, "the reconstructed band must retain the box crease line")
        self.assertGreater(float(crease.max()), 65.0, "the band rounded away the sharp box edge")

    def test_voxel_budget_rejection_returns_unchanged_source(self):
        points, faces, region = folded_box_case()
        before = exact_self_intersection_report(points, faces)
        bad_rows = np.unique(np.asarray(before["reported_pairs"], dtype=np.int64))

        result_points, result_faces, lineage, report = reconstruct_voxel_band(
            points,
            faces,
            region,
            pitch=0.06,
            max_error=0.08,
            max_voxels=1,
            unreliable_source_face_rows=bad_rows,
        )

        self.assertFalse(report["accepted"])
        self.assertIn("voxel_budget_exceeded", report["failure_reason"])
        np.testing.assert_array_equal(result_points, points)
        np.testing.assert_array_equal(result_faces, faces)
        np.testing.assert_array_equal(lineage, np.arange(len(faces)))

    def test_worker_deadline_revokes_voxel_band_acceptance(self):
        points, faces, region = folded_box_case()
        before = exact_self_intersection_report(points, faces)
        bad_rows = np.unique(np.asarray(before["reported_pairs"], dtype=np.int64))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = _source_with_native_normals(points, faces)
            source_path = root / "source.vtp"
            source.save(source_path)
            loaded = read_surface(source_path)
            loaded_faces = triangle_faces(loaded)
            packet = extract_defect_regions(
                loaded.points,
                loaded_faces,
                seed_faces=region,
                halo_rings=0,
                max_region_faces=len(region),
                max_total_faces=len(region),
            )
            packet["regions"][0]["defect_face_rows"] = bad_rows.tolist()
            packet_path = root / "regions.json"
            packet_path.write_text(json.dumps(packet))
            output = root / "run"
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS / "local_patch_worker.py"),
                    str(source_path),
                    "--regions",
                    str(packet_path),
                    "--region",
                    packet["regions"][0]["id"],
                    "--method",
                    "voxel-band",
                    "--pitch",
                    ".06",
                    "--max-error",
                    ".08",
                    "--max-voxels",
                    "150000",
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
            self.assertIsNone(report["outputs"].get("accepted_mesh_vtp"))
            self.assertIn("exceeded", report["failure_reason"])


if __name__ == "__main__":
    unittest.main()
