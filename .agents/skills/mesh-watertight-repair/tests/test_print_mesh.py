from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from base64 import b64encode
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
from zipfile import ZipFile

import numpy as np
import pyvista as pv
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from gltf_surface import gltf_accessor
from print_export import run_slicer
from print_geometry import construct_wall, union_parts
from print_mesh import certify_print_candidate, print_candidate, print_source
from repair_inventory import build_inventory
from repair_policy import policy_item_id
from two_stage_watertight_remesh import main


class PrintMeshTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def run_worker(self, mesh, name, *flags):
        source = self.root / "input.vtp"
        pv.wrap(mesh).save(source)
        directory = self.root / name
        command = [
            "worker",
            str(source),
            "--purpose",
            "print",
            "--print-units",
            "mm",
            "--output-dir",
            str(directory),
            *flags,
        ]
        with (
            mock.patch.object(sys, "argv", command),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(main(), 0)
        return json.loads((directory / "print_report.json").read_text())

    def write_gltf_triangles(self, name, triangles, normals):
        points = np.asarray(triangles, dtype="<f4").reshape(-1, 3)
        vectors = np.asarray(normals, dtype="<f4").reshape(-1, 3)
        binary = points.tobytes() + vectors.tobytes()
        document = {
            "asset": {"version": "2.0"},
            "buffers": [{
                "byteLength": len(binary),
                "uri": "data:application/octet-stream;base64," + b64encode(binary).decode(),
            }],
            "bufferViews": [
                {"buffer": 0, "byteOffset": 0, "byteLength": points.nbytes},
                {"buffer": 0, "byteOffset": points.nbytes, "byteLength": vectors.nbytes},
            ],
            "accessors": [
                {"bufferView": 0, "componentType": 5126, "type": "VEC3", "count": len(points)},
                {"bufferView": 1, "componentType": 5126, "type": "VEC3", "count": len(vectors)},
            ],
            "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "NORMAL": 1}}]}],
            "nodes": [{"mesh": 0}],
            "scenes": [{"nodes": [0]}],
        }
        path = self.root / name
        path.write_text(json.dumps(document))
        return path

    def print_args(self, path):
        return SimpleNamespace(
            input=path,
            print_units=None,
            print_size_mm=20.0,
            print_up_axis="y",
            print_openings="cap",
            print_mode="preserve",
            print_parts="preserve",
            print_wall_mm=0.8,
            print_min_wall_mm=0.8,
            print_max_error_mm=0.1,
            print_max_voxels=1000,
            print_process="fff",
        )

    def test_print_source_keeps_hard_corner_normals_through_clean_and_axis_scale(self):
        triangles = np.asarray([
            [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
            [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
            [[1, 0, 0], [0, 0, 0], [0, 0, 1]],
        ], dtype=float)
        normals = np.asarray([
            [[0, 0, 1]] * 3,
            [[1, 0, 0]] * 3,
            [[0, 1, 0]] * 3,
        ], dtype=float)
        path = self.write_gltf_triangles("crease.gltf", triangles, normals)

        source, factor, _ = print_source(self.print_args(path))

        self.assertEqual(factor, 20.0)
        self.assertEqual(len(source.vertices), 4)
        stored = np.asarray(source.face_attributes["corner_normals"])
        self.assertEqual(stored.shape, (2, 3, 3))
        for row in stored:
            np.testing.assert_allclose(row, np.repeat(row[:1], 3, axis=0), atol=1e-7)
        actual = {tuple(np.round(row[0], 7)) for row in stored}
        self.assertEqual(actual, {(0.0, -1.0, 0.0), (0.0, 0.0, 1.0)})
        np.testing.assert_allclose(np.ptp(source.vertices, axis=0).max(), 20.0)

    def test_print_source_transports_vtp_corner_normals_and_keeps_their_magnitude(self):
        points = np.asarray(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float
        )
        faces = np.asarray([[0, 1, 2], [1, 0, 3]], dtype=np.int64)
        packed = np.column_stack((np.full(len(faces), 3), faces)).ravel()
        mesh = pv.PolyData(points, packed)
        mesh.cell_data["corner_normals"] = np.asarray(
            [[0, 0, 2] * 3, [0, 3, 0] * 3], dtype=np.float32
        )
        paths = {}
        for point_normals in (False, True):
            source_mesh = mesh.copy()
            if point_normals:
                source_mesh.point_data["Normals"] = np.tile(
                    [1.0, 0.0, 0.0], (len(points), 1)
                )
            path = self.root / f"corner-normal-source-{point_normals}.vtp"
            source_mesh.save(path)
            paths[point_normals] = path
            source, _, _ = print_source(self.print_args(path))

            normals = np.asarray(source.face_attributes["corner_normals"])
            actual = {tuple(np.round(row[0], 6)) for row in normals}
            self.assertEqual(actual, {(0.0, -2.0, 0.0), (0.0, 0.0, 3.0)})
            self.assertNotIn((1.0, 0.0, 0.0), actual)

        output = self.root / "print-corner-normal-vtp"
        with (
            mock.patch.object(
                sys,
                "argv",
                [
                    "worker",
                    str(paths[False]),
                    "--purpose",
                    "print",
                    "--print-size-mm",
                    "20",
                    "--print-up-axis",
                    "y",
                    "--output-dir",
                    str(output),
                ],
            ),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(main(), 0)
        report = json.loads((output / "print_report.json").read_text())
        saved_source = pv.read(report["stages"]["print_source"]["path"])
        saved_candidate = pv.read(report["outputs"]["print_candidate_vtp"])
        for saved in (saved_source, saved_candidate):
            np.testing.assert_allclose(
                np.asarray(saved.cell_data["corner_normals"]).reshape(-1, 3, 3)[:, 0],
                [[0, -2, 0], [0, 0, 3]],
                atol=1e-6,
            )

    def test_print_candidate_uses_geometry_normals_for_added_cap_faces(self):
        quads = [
            ([(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)], [0, 0, -1]),
            ([(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)], [-1, 0, 0]),
            ([(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)], [1, 0, 0]),
            ([(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)], [0, -1, 0]),
            ([(0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)], [0, 1, 0]),
        ]
        triangles, normals = [], []
        for corners, normal in quads:
            corners = np.asarray(corners, dtype=float)
            if np.dot(np.cross(corners[1] - corners[0], corners[2] - corners[0]), normal) < 0:
                corners = corners[::-1]
            triangles.extend((corners[[0, 1, 2]], corners[[0, 2, 3]]))
            normals.extend((np.tile(normal, (3, 1)), np.tile(normal, (3, 1))))
        path = self.write_gltf_triangles("open-box.gltf", triangles, normals)
        args = self.print_args(path)
        source, _, _ = print_source(args)
        inventory = build_inventory(
            source.vertices,
            source.faces,
            np.arange(len(source.faces)),
            "print-normal-test",
            max_items=16,
        )
        region = inventory["boundary_regions"]["items"][0]
        decisions = [{
            "item_id": policy_item_id(region["id"]),
            "status": "decided",
            "decision": "cap",
            "shape_prior": "planar",
        }]

        candidate, generated, _, ready, _, normal_source_face_ids = print_candidate(
            source, args, inventory, decisions
        )
        candidate = certify_print_candidate(
            candidate, source, generated, args, ready, normal_source_face_ids
        )[0]

        self.assertTrue(ready)
        self.assertGreater(int(generated.sum()), 0)
        corner_normals = np.asarray(candidate.face_attributes["corner_normals"])
        self.assertEqual(corner_normals.shape, (len(candidate.faces), 3, 3))
        np.testing.assert_allclose(
            corner_normals[: len(source.faces)],
            np.asarray(source.face_attributes["corner_normals"]),
            atol=1e-7,
        )
        generated_ids = np.flatnonzero(generated)
        triangles_out = candidate.vertices[candidate.faces[generated_ids]]
        geometric = np.cross(
            triangles_out[:, 1] - triangles_out[:, 0],
            triangles_out[:, 2] - triangles_out[:, 0],
        )
        geometric /= np.linalg.norm(geometric, axis=1)[:, None]
        np.testing.assert_allclose(
            corner_normals[generated_ids],
            np.repeat(geometric[:, None, :], 3, axis=1),
            atol=1e-7,
        )

        output = self.root / "print-with-native-normals"
        with (
            mock.patch.object(
                sys,
                "argv",
                [
                    "worker",
                    str(path),
                    "--purpose",
                    "print",
                    "--print-size-mm",
                    "20",
                    "--print-up-axis",
                    "y",
                    "--print-openings",
                    "cap",
                    "--output-dir",
                    str(output),
                ],
            ),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(main(), 0)
        report = json.loads((output / "print_report.json").read_text())
        saved_source = pv.read(report["stages"]["print_source"]["path"])
        saved_candidate = pv.read(report["outputs"]["print_candidate_vtp"])
        self.assertEqual(saved_source.cell_data["corner_normals"].shape, (10, 9))
        self.assertEqual(
            saved_candidate.cell_data["corner_normals"].shape,
            (saved_candidate.n_cells, 9),
        )
        saved_generated = np.asarray(saved_candidate.cell_data["print_generated_face"], dtype=bool)
        saved_corner_normals = np.asarray(
            saved_candidate.cell_data["corner_normals"]
        ).reshape(-1, 3, 3)
        saved_faces = np.asarray(saved_candidate.faces).reshape(-1, 4)[:, 1:]
        saved_triangles = np.asarray(saved_candidate.points)[saved_faces[saved_generated]]
        saved_geometry = np.cross(
            saved_triangles[:, 1] - saved_triangles[:, 0],
            saved_triangles[:, 2] - saved_triangles[:, 0],
        )
        saved_geometry /= np.linalg.norm(saved_geometry, axis=1)[:, None]
        np.testing.assert_allclose(
            saved_corner_normals[saved_generated],
            np.repeat(saved_geometry[:, None, :], 3, axis=1),
            atol=1e-6,
        )

    def test_closed_duct_keeps_its_bore_and_exports_mm_files(self):
        mesh = trimesh.creation.annulus(r_min=0.6, r_max=3, height=4, sections=24)
        report = self.run_worker(mesh, "duct")
        self.assertEqual(report["decision"]["status"], "accepted")
        saved = pv.read(report["outputs"]["accepted_mesh_vtp"])
        self.assertEqual(len(saved.ray_trace((0, 0, -3), (0, 0, 3))[0]), 0)
        self.assertEqual(report["printing"]["layer_check"]["invalid_contours"], 0)
        self.assertFalse(report["printing"]["printer_ready"])
        self.assertTrue(all(row["passed"] for row in report["outputs"]["print_files"]))
        with ZipFile(self.root / "duct/print_candidate.3mf") as archive:
            self.assertIn(
                'unit="millimeter"', archive.read("3D/3dmodel.model").decode()
            )

    def test_reviewed_cap_and_preserve_have_different_geometry_and_acceptance(self):
        closed = trimesh.creation.box([4, 4, 4])
        opened = trimesh.Trimesh(
            closed.vertices,
            closed.faces[closed.face_normals[:, 2] < 0.5],
            process=False,
        )
        pending = self.run_worker(opened, "pending")
        self.assertIsNone(pending["outputs"]["accepted_mesh_vtp"])
        item = pending["unresolved_policy_packet"]["items"][0]
        results = {}
        for action in ("cap", "preserve"):
            decision = {
                **item["decision_template"],
                "status": "decided",
                "decision": action,
                "shape_prior": "planar",
                "reason": "Injected geometry contract test; no AI model evaluated",
            }
            path = self.root / f"{action}.json"
            path.write_text(json.dumps({"decisions": [decision]}))
            results[action] = self.run_worker(
                opened, action, "--policy-decisions", str(path)
            )
        self.assertEqual(results["cap"]["decision"]["status"], "accepted")
        self.assertIsNone(results["preserve"]["outputs"]["accepted_mesh_vtp"])
        self.assertEqual(
            results["cap"]["opening_transactions"][0]["generated_faces"], 2
        )
        self.assertIn(
            "needs a wall", results["preserve"]["opening_transactions"][0]["reason"]
        )

    def test_nested_void_orientation_survives_print_files_and_sla_rejects_sealed_void(
        self,
    ):
        material = trimesh.util.concatenate(
            [trimesh.creation.box([4, 4, 4]), trimesh.creation.box([2, 2, 2])]
        )
        report = self.run_worker(material, "void")
        self.assertEqual(report["decision"]["status"], "accepted")
        self.assertEqual([row["depth"] for row in report["printing"]["shells"]], [0, 1])
        self.assertLess(report["printing"]["shells"][1]["signed_volume_mm3"], 0)
        sla = self.run_worker(material, "sla", "--print-process", "sla")
        self.assertFalse(sla["gates"]["sla_closed_cavities"]["passed"])
        solid = self.run_worker(
            material, "solid", "--print-process", "sla", "--print-mode", "solid"
        )
        self.assertEqual(solid["decision"]["status"], "accepted")
        self.assertEqual(len(solid["printing"]["shells"]), 1)

    def test_separate_parts_stay_separate_and_union_requires_actual_contact(self):
        a = trimesh.creation.box([2, 2, 2])
        b = a.copy()
        b.apply_translation([2.8, 0, 0])
        source = trimesh.util.concatenate([a, b])
        report = self.run_worker(source, "parts")
        self.assertEqual(report["decision"]["status"], "accepted")
        self.assertEqual(len(report["printing"]["shells"]), 2)
        merged = union_parts(source)
        self.assertEqual(len(merged.split()), 2)
        b.apply_translation([-0.9, 0, 0])
        merged = union_parts(trimesh.util.concatenate([a, b]))
        self.assertEqual(len(merged.split()), 1)

    def test_sheet_thickening_and_explicit_memory_budget(self):
        plane = pv.Plane(
            i_size=2, j_size=2, i_resolution=1, j_resolution=1
        ).triangulate()
        report = self.run_worker(
            plane,
            "sheet",
            "--print-mode",
            "thicken",
            "--print-wall-mm",
            "1",
            "--print-max-error-mm",
            ".2",
        )
        self.assertEqual(report["decision"]["status"], "accepted")
        self.assertTrue(report["gates"]["sampled_minimum_wall"]["passed"])
        with self.assertRaises(MemoryError):
            construct_wall(
                trimesh.creation.box([10, 10, 10]),
                mode="hollow",
                thickness=1,
                error=0.01,
                max_voxels=100,
            )
        limited = self.run_worker(
            trimesh.creation.box([4, 4, 4]),
            "grid-budget",
            "--print-mode",
            "hollow",
            "--print-max-voxels",
            "100",
        )
        self.assertTrue(limited["gates"]["opening_decisions_applied"]["passed"])
        self.assertFalse(limited["gates"]["geometry_operation_completed"]["passed"])
        self.assertIsNone(limited["outputs"]["accepted_mesh_vtp"])

    def test_normalized_strided_and_sparse_gltf_accessors(self):
        packed = np.array(
            [[32767, 0, -32768, 99], [0, 32767, 0, 99]], dtype="<i2"
        ).tobytes()
        document = {
            "bufferViews": [{"buffer": 0, "byteLength": len(packed), "byteStride": 8}],
            "accessors": [
                {
                    "bufferView": 0,
                    "componentType": 5122,
                    "normalized": True,
                    "type": "VEC3",
                    "count": 2,
                }
            ],
        }
        actual = gltf_accessor(document, [packed], 0)
        np.testing.assert_array_equal(actual, [[1, 0, -1], [0, 1, 0]])
        sparse = bytes([1]) + np.array([[2, 3, 4]], dtype="<f4").tobytes()
        document = {
            "bufferViews": [
                {"buffer": 0, "byteLength": 1},
                {"buffer": 0, "byteOffset": 1, "byteLength": 12},
            ],
            "accessors": [
                {
                    "componentType": 5126,
                    "type": "VEC3",
                    "count": 2,
                    "sparse": {
                        "count": 1,
                        "indices": {"bufferView": 0, "componentType": 5121},
                        "values": {"bufferView": 1},
                    },
                }
            ],
        }
        np.testing.assert_array_equal(
            gltf_accessor(document, [sparse], 0), [[0, 0, 0], [2, 3, 4]]
        )

    def test_thin_material_is_not_promoted_and_hollowing_adds_a_material_void(self):
        thin = self.run_worker(trimesh.creation.box([2, 2, 0.4]), "thin")
        self.assertFalse(thin["gates"]["sampled_minimum_wall"]["passed"])
        self.assertEqual(thin["outputs"]["accepted_print_files"], [])
        hollow = self.run_worker(
            trimesh.creation.box([4, 4, 4]),
            "hollow",
            "--print-mode",
            "hollow",
            "--print-max-error-mm",
            ".2",
            "--print-min-wall-mm",
            ".6",
        )
        self.assertEqual(hollow["decision"]["status"], "accepted")
        self.assertEqual(len(hollow["printing"]["shells"]), 2)

    def test_stale_policy_is_not_overridden_by_whole_model_cap(self):
        closed = trimesh.creation.box([4, 4, 4])
        opened = trimesh.Trimesh(
            closed.vertices,
            closed.faces[closed.face_normals[:, 2] < 0.5],
            process=False,
        )
        initial = self.run_worker(opened, "initial")
        item = initial["unresolved_policy_packet"]["items"][0]
        decision = {
            **item["decision_template"],
            "decision": "cap",
            "run_fingerprint": "stale",
        }
        path = self.root / "stale.json"
        path.write_text(json.dumps([decision]))
        stale = self.run_worker(
            opened, "stale", "--print-openings", "cap", "--policy-decisions", str(path)
        )
        self.assertTrue(stale["policy_decisions"][0]["stale_policy_decision"])
        self.assertFalse(stale["gates"]["opening_decisions_applied"]["passed"])

    def test_authorized_cap_can_be_followed_by_hollow_wall_construction(self):
        box = trimesh.creation.box([4, 4, 4])
        opened = trimesh.Trimesh(
            box.vertices, box.faces[box.face_normals[:, 2] < 0.5], process=False
        )
        report = self.run_worker(
            opened,
            "cap-hollow",
            "--print-mode",
            "hollow",
            "--print-openings",
            "cap",
            "--print-max-error-mm",
            ".2",
            "--print-min-wall-mm",
            ".6",
        )
        self.assertEqual(report["decision"]["status"], "accepted")
        self.assertGreater(
            report["printing"]["fidelity"]["generated_faces_excluded_forward"], 0
        )

    def test_thickening_does_not_silently_merge_separate_sheets(self):
        first = pv.Plane(
            i_size=2, j_size=2, i_resolution=1, j_resolution=1
        ).triangulate()
        second = first.copy()
        second.points[:, 2] += 0.4
        source = first.merge(second, merge_points=False)
        kept = self.run_worker(
            source,
            "keep-sheets",
            "--print-mode",
            "thicken",
            "--print-wall-mm",
            "1",
            "--print-max-error-mm",
            ".2",
        )
        self.assertFalse(kept["gates"]["protected_topology"]["passed"])
        merged = self.run_worker(
            source,
            "merge-sheets",
            "--print-mode",
            "thicken",
            "--print-parts",
            "merge",
            "--print-wall-mm",
            "1",
            "--print-max-error-mm",
            ".2",
        )
        self.assertEqual(merged["decision"]["status"], "accepted")

    def test_solid_mode_does_not_imply_permission_to_fuse_overlapping_parts(self):
        left = trimesh.creation.box([2, 2, 2])
        right = left.copy()
        right.apply_translation([1.5, 0, 0])
        report = self.run_worker(
            trimesh.util.concatenate([left, right]),
            "overlap-solid",
            "--print-mode",
            "solid",
        )
        self.assertFalse(report["gates"]["exact_self_intersection"]["passed"])
        self.assertEqual(report["outputs"]["accepted_print_files"], [])

    def test_existing_shell_worker_executes_supplied_cap_and_reaudits_inventory(self):
        box = trimesh.creation.box([4, 4, 4])
        opened = trimesh.Trimesh(
            box.vertices, box.faces[box.face_normals[:, 2] < 0.5], process=False
        )
        source = self.root / "shell-source.vtp"
        pv.wrap(opened).save(source)
        flags = [
            "--visibility-grid",
            "64",
            "--outside-flood-grid",
            "24",
            "--sealed-exterior-grid",
            "24",
            "--voxel-pitch",
            ".2",
            "--preview-size",
            "256",
            "--skip-previews",
        ]
        first = self.root / "shell-inventory"
        # This checks worker exceptions; the process supervisor has separate tests.
        flags.append("--_projected-shell-worker")
        with (
            mock.patch.object(
                sys, "argv", ["worker", str(source), "--output-dir", str(first), *flags]
            ),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(main(), 0)
        packet = json.loads((first / "ai_policy_packet.json").read_text())
        decisions = [
            {**row["decision_template"], "decision": "cap", "shape_prior": "planar"}
            for row in packet["items"]
        ]
        self.assertEqual(len(decisions), 1)
        path = self.root / "shell-cap.json"
        path.write_text(json.dumps(decisions))
        second = self.root / "shell-cap"
        with (
            mock.patch.object(
                sys,
                "argv",
                [
                    "worker",
                    str(source),
                    "--output-dir",
                    str(second),
                    *flags,
                    "--policy-decisions",
                    str(path),
                ],
            ),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(main(), 0)
        report = json.loads((second / "two_stage_report.json").read_text())
        self.assertTrue(report["opening_transactions"][0]["applied"])
        self.assertTrue(report["gates"]["opening_policy_resolved"]["passed"])
        self.assertEqual(
            report["inventory_after"]["boundary_regions"]["total_regions"], 0
        )
        for row in decisions:
            row["decision"] = "preserve"
        path.write_text(json.dumps(decisions))
        blocked = self.root / "shell-preserve"
        with (
            mock.patch.object(
                sys,
                "argv",
                [
                    "worker",
                    str(source),
                    "--output-dir",
                    str(blocked),
                    *flags,
                    "--policy-decisions",
                    str(path),
                ],
            ),
            contextlib.redirect_stdout(io.StringIO()),
            self.assertRaisesRegex(ValueError, "opening decisions"),
        ):
            main()
        status = json.loads((blocked / "run_status.json").read_text())
        self.assertIsNone(status["accepted_mesh_vtp"])

    def test_print_units_are_required_and_print_settings_cannot_be_silently_ignored(
        self,
    ):
        from two_stage_watertight_remesh import parse_args

        for flags in (
            ("--purpose", "print"), ("--print-units", "mm"),
            ("--purpose", "print", "--print-units", "mm", "--scaffold-smoothing-iterations", "30"),
            ("--purpose", "print", "--print-units", "mm", "--cross-component-jump-ratio", "0.5"),
            ("--purpose", "print", "--print-units", "mm", "--projection-target", "reviewed.vtp"),
        ):
            with (
                self.subTest(flags=flags),
                mock.patch.object(
                    sys,
                    "argv",
                    ["worker", "input.stl", "--output-dir", str(self.root), *flags],
                ),
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                parse_args()

    def test_external_slicer_protocol_checks_extrusion_and_retains_timeout_diagnostics(
        self,
    ):
        slicer = self.root / "slicer"
        slicer.write_text(
            f'#!{sys.executable}\nimport sys\nfrom pathlib import Path\nPath(sys.argv[sys.argv.index("--output")+1]).write_text("G1 X1 Y1 E1\\n")\n'
        )
        slicer.chmod(0o700)
        profile = self.root / "printer.ini"
        profile.write_text("; synthetic protocol fixture")
        result = run_slicer(slicer, profile, self.root / "part.3mf", self.root)
        self.assertTrue(result["passed"])
        self.assertIn("--no-binary-gcode", result["command"])
        import subprocess

        with mock.patch(
            "print_export.subprocess.run",
            side_effect=subprocess.TimeoutExpired("synthetic", 300),
        ):
            result = run_slicer(slicer, profile, self.root / "part.3mf", self.root)
        self.assertFalse(result["passed"])
        self.assertEqual(result["status"], "failed")


if __name__ == "__main__":
    unittest.main()
