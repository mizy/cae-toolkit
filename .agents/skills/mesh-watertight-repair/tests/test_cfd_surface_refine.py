from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pyvista as pv
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from cfd_surface_refine import refine_cfd_surface
from mesh_io import read_surface, write_vtp
from two_stage_contract import record_cfd_refinement
from two_stage_watertight_remesh import parse_args


class CfdSurfaceRefineTest(unittest.TestCase):
    def run_refinement(self, directory: Path, mesh: trimesh.Trimesh, size: float, budget: int = 2_000_000) -> dict:
        path = directory / "shell.vtp"
        write_vtp(path, mesh.vertices, mesh.faces, {"region_id": np.arange(len(mesh.faces)) % 3})
        return refine_cfd_surface(path, path, directory / "cfd", max_edge_length=size, max_output_faces=budget)

    def test_conforming_refinement_keeps_shape_closure_and_region_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = trimesh.creation.icosphere(subdivisions=1)
            report = self.run_refinement(root, original, 0.45)
            self.assertEqual(report["decision"]["status"], "accepted", report["gates"])
            output = pv.read(report["outputs"]["accepted_mesh_vtp"])
            self.assertGreater(output.n_cells, len(original.faces))
            normalized = read_surface(root / "shell.vtp")
            np.testing.assert_array_equal(output.points[: normalized.n_points], normalized.points)
            parents = output["watertight_parent_face_id"]
            np.testing.assert_array_equal(output["region_id"], parents % 3)
            self.assertTrue(all(row["passed"] for row in report["gates"].values()))

    def test_refinement_preserves_a_real_through_passage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            mesh = trimesh.creation.annulus(r_min=0.3, r_max=1, height=1, sections=12)
            report = self.run_refinement(Path(tmp), mesh, 0.5)
            self.assertEqual(report["decision"]["status"], "accepted", report["gates"])
            result = pv.read(report["outputs"]["accepted_mesh_vtp"])
            self.assertEqual(len(result.ray_trace((0, 0, -2), (0, 0, 2))[0]), 0)
            self.assertEqual(report["gates"]["euler_preserved"]["value"], 0)

    def test_new_vertices_project_to_source_while_input_vertices_stay_fixed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shell = trimesh.creation.icosphere(subdivisions=1)
            write_vtp(root / "shell.vtp", shell.vertices, shell.faces)
            write_vtp(root / "source.vtp", shell.vertices * 1.005, shell.faces)
            report = refine_cfd_surface(root / "shell.vtp", root / "source.vtp", root / "cfd", max_edge_length=0.45)
            result = pv.read(report["outputs"]["cfd_surface_candidate_vtp"])
            normalized = read_surface(root / "shell.vtp")
            np.testing.assert_array_equal(result.points[: normalized.n_points], normalized.points)
            applied = np.asarray(result["cfd_source_projection_applied"]).astype(bool)
            self.assertFalse(np.any(applied[: normalized.n_points]))
            self.assertGreater(np.count_nonzero(applied), 0)
            self.assertTrue(report["gates"]["source_projection_bounded"]["passed"])

    def test_face_budget_cannot_silently_accept_unmet_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = self.run_refinement(Path(tmp), trimesh.creation.icosphere(subdivisions=1), 0.1, 100)
            self.assertEqual(report["decision"]["status"], "rejected")
            self.assertFalse(report["gates"]["target_edge_length"]["passed"])
            self.assertIsNone(report["outputs"]["accepted_mesh_vtp"])
            self.assertTrue(Path(report["outputs"]["watertight_mesh_vtp"]).exists())

    def test_inherited_sliver_is_reported_instead_of_called_cfd_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = self.run_refinement(Path(tmp), trimesh.creation.box(extents=(2, 0.001, 1)), 10)
            self.assertTrue(report["gates"]["watertight"]["passed"])
            self.assertFalse(report["gates"]["aspect_ratio_max"]["passed"])
            self.assertEqual(report["decision"]["status"], "rejected")

    def test_profile_default_and_cfd_dimensions_are_explicit(self) -> None:
        command = ["worker", "source.vtp", "--output-dir", "output"]
        with mock.patch.object(sys, "argv", command):
            self.assertEqual(parse_args().purpose, "watertight")
        for extra in (
            ["--purpose", "cfd"],
            ["--cfd-max-edge-length", ".1"],
            ["--purpose", "cfd", "--cfd-max-edge-length", "nan"],
        ):
            with (
                mock.patch.object(sys, "argv", command + extra),
                mock.patch("sys.stderr"),
                self.assertRaises(SystemExit),
            ):
                parse_args()
        with mock.patch.object(sys, "argv", command + ["--purpose", "cfd", "--cfd-max-edge-length", ".1"]):
            self.assertEqual(parse_args().cfd_max_edge_length, 0.1)

    def test_final_output_routes_to_cfd_only_when_all_profile_gates_pass(self) -> None:
        for budget in (100, 2_000_000):
            with self.subTest(budget=budget), tempfile.TemporaryDirectory() as tmp:
                result = self.run_refinement(Path(tmp), trimesh.creation.icosphere(subdivisions=1), 0.45, budget)
                shell_path = result["outputs"]["watertight_mesh_vtp"]
                report = {
                    "outputs": {"accepted_mesh_vtp": shell_path},
                    "stages": {"source_projected_watertight_candidate": {"accepted_final_geometry": True}},
                    "gates": {},
                    "ignored_outputs": [],
                    "output_contract": {},
                    "input_truth": {"derived_artifacts": []},
                    "limitations": [],
                    "repair_report": {
                        "geometry_to_mesh_trace": [],
                        "change_summary": {"source_projected_watertight_shell": {}},
                        "defect_matrix": {},
                        "requested_capabilities": {},
                    },
                }
                record_cfd_refinement(report, result)
                self.assertEqual(report["outputs"]["watertight_mesh_vtp"], shell_path)
                self.assertEqual(report["decision"]["final_output_path"], result["decision"]["final_output_path"])
                self.assertEqual(report["outputs"]["accepted_mesh_vtp"], result["decision"]["final_output_path"])
                self.assertFalse(report["outputs"]["mesh_result"]["engineering_ready"])
                if budget == 100:
                    self.assertEqual(report["outputs"]["mesh_result"]["path"], shell_path)
                    self.assertEqual(report["outputs"]["mesh_result"]["role"], "accepted_watertight_fallback")


if __name__ == "__main__":
    unittest.main()
