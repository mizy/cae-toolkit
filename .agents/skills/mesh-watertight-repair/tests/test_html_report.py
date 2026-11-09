from __future__ import annotations

import os
import sys
import tempfile
import unittest
import urllib.request
from pathlib import Path
from unittest import mock

import numpy as np
import pyvista as pv


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import html_report  # noqa: E402
import html_mesh_preview  # noqa: E402
import watertight_issue_html  # noqa: E402
import watertight_issue_markers  # noqa: E402


def write_grid_mesh(
    path: Path,
    cells_per_side: int = 8,
    with_proxy_weight: bool = False,
    with_issue_mask: bool = False,
) -> int:
    points = []
    for y in range(cells_per_side + 1):
        for x in range(cells_per_side + 1):
            points.append((float(x), float(y), 0.0))

    faces = []
    row = cells_per_side + 1
    for y in range(cells_per_side):
        for x in range(cells_per_side):
            a = y * row + x
            b = a + 1
            c = a + row
            d = c + 1
            faces.extend([3, a, b, d, 3, a, d, c])

    mesh = pv.PolyData(np.asarray(points, dtype=np.float64), np.asarray(faces, dtype=np.int64))
    if with_proxy_weight:
        mesh.cell_data["proxy_weight"] = np.linspace(0.0, 1.0, cells_per_side * cells_per_side * 2)
    if with_issue_mask:
        issue_mask = np.zeros(cells_per_side * cells_per_side * 2, dtype=np.uint8)
        issue_mask[::2] = 1
        mesh.cell_data["watertight_audit_issue_mask"] = issue_mask
    mesh.save(path)
    return cells_per_side * cells_per_side * 2


class HtmlReportMeshPreviewTest(unittest.TestCase):
    def test_corner_normal_display_preserves_physical_mesh_and_face_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corner_normals.vtp"
            mesh = pv.PolyData(np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [1., 1., 0.]]),
                               np.array([3, 0, 1, 2, 3, 1, 3, 2]))
            mesh.cell_data["corner_normals"] = np.array([[0., 0., 1.] * 3, [0., 1., 0.] * 3])
            mesh.cell_data["print_generated_face"] = [0, 1]
            mesh.save(path)
            preview = html_mesh_preview.mesh_preview("native seams", path)
            faces = np.array(preview["polys"]).reshape(-1, 4)[:, 1:]
            corners = np.array(preview["points"])[faces]
            np.testing.assert_array_equal(corners, mesh.points[np.array([[0, 1, 2], [1, 3, 2]])])
            np.testing.assert_array_equal(np.array(preview["normals"])[faces], mesh.cell_data["corner_normals"].reshape(-1, 3, 3))
            np.testing.assert_array_equal(preview["geometry_normals"], np.tile([0., 0., 1.], (6, 1)))
            self.assertEqual(preview["fields"]["print_generated_face"], [0, 1])
            self.assertEqual(pv.read(path).n_points, 4)
            self.assertEqual(preview["triangles"], 2)
            mesh.cell_data["corner_normals"] = np.tile([0., 0., 1.] * 3, (2, 1))
            mesh.save(path)
            shared = html_mesh_preview.mesh_preview("continuous normals", path)
            self.assertEqual(shared["points_count"], 4)
            np.testing.assert_array_equal(np.array(shared["points"])[np.array(shared["polys"]).reshape(-1, 4)[:, 1:]], corners)

    def test_preview_preserves_saved_point_normals_without_changing_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source_normals.vtp"
            mesh = pv.Plane(i_resolution=1, j_resolution=1).triangulate()
            mesh.point_data["Normals"] = np.tile([0., 1., 0.], (mesh.n_points, 1))
            mesh.GetPointData().SetActiveNormals("Normals")
            mesh.save(path)
            preview = html_mesh_preview.mesh_preview("source normals", path)
            np.testing.assert_array_equal(preview["points"], mesh.points)
            np.testing.assert_array_equal(preview["normals"], mesh.point_data["Normals"])
            self.assertIn("vtk-source-normals-0", html_report.viewer_section([preview]))

    def test_issue_comparison_report_uses_closure_first_change_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source = root / "source.vtp"
            candidate = root / "candidate.vtp"
            write_grid_mesh(source)
            write_grid_mesh(candidate)
            marker_report = watertight_issue_markers.generate_issue_report(
                source,
                candidate,
                root / "issues",
            )
            marker_path = Path(marker_report["report_path"])

            report = watertight_issue_html.build_comparison_report(
                marker_report,
                marker_path,
                {},
                None,
                [],
            )

        change_summary = report["repair_report"]["change_summary"]
        self.assertNotIn("patch_transactions", change_summary)
        self.assertIn("watertight_issue_delta", change_summary)
        self.assertFalse(change_summary["accepted_final_geometry"])

    def test_vtk_js_source_defaults_to_bundled_offline_renderer(self) -> None:
        with (
            mock.patch.dict(os.environ, {"VTK_JS_PATH": ""}),
            mock.patch.object(urllib.request, "urlopen") as urlopen,
            mock.patch.object(Path, "home") as home,
            mock.patch.object(Path, "mkdir") as mkdir,
            mock.patch.object(Path, "write_text") as write_text,
        ):
            source = html_mesh_preview.vtk_js_source()

        bundled = SCRIPTS_DIR.parent / "assets/vtk-32.12.0/vtk.js"
        self.assertTrue(source.endswith(bundled.read_text(encoding="utf-8")))
        self.assertIn("Redistribution and use in source and binary forms", source)
        self.assertIn("Ken Martin, Will Schroeder, Bill Lorensen", source)
        self.assertIn("window.vtk=", source)
        urlopen.assert_not_called()
        home.assert_not_called()
        mkdir.assert_not_called()
        write_text.assert_not_called()

    def test_vtk_js_source_with_invalid_path_is_offline_and_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            missing = Path(tmp_dir) / "missing-vtk.js"
            with (
                mock.patch.dict(os.environ, {"VTK_JS_PATH": str(missing)}),
                mock.patch.object(urllib.request, "urlopen") as urlopen,
                mock.patch.object(Path, "home") as home,
                mock.patch.object(Path, "mkdir") as mkdir,
                mock.patch.object(Path, "write_text") as write_text,
            ):
                source = html_mesh_preview.vtk_js_source()

        self.assertEqual(source, html_mesh_preview.VTK_JS_UNAVAILABLE_STUB)
        urlopen.assert_not_called()
        home.assert_not_called()
        mkdir.assert_not_called()
        write_text.assert_not_called()

    def test_vtk_js_source_reads_only_explicit_local_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "vtk.js"
            expected = "window.vtk = { local: true };"
            path.write_text(expected, encoding="utf-8")
            with (
                mock.patch.dict(os.environ, {"VTK_JS_PATH": str(path)}),
                mock.patch.object(urllib.request, "urlopen") as urlopen,
                mock.patch.object(Path, "home") as home,
                mock.patch.object(Path, "mkdir") as mkdir,
                mock.patch.object(Path, "write_text") as write_text,
            ):
                source = html_mesh_preview.vtk_js_source()

        self.assertEqual(source, expected)
        urlopen.assert_not_called()
        home.assert_not_called()
        mkdir.assert_not_called()
        write_text.assert_not_called()

    def test_preview_specs_prioritizes_original_processed_issue_comparison(self) -> None:
        report = {
            "input": {"path": "/models/input.stl"},
            "outputs": {
                "original_mesh_vtp": "outputs/report/original_surface.vtp",
                "processed_mesh_vtp": "outputs/report/processed_surface.vtp",
                "original_issue_faces_vtp": "outputs/report/original_issue_faces.vtp",
                "processed_issue_faces_vtp": "outputs/report/processed_issue_faces.vtp",
                "rejected_candidate_vtp": "outputs/run/candidate.vtp",
            },
        }

        specs = html_report.preview_specs(report)

        self.assertEqual(
            [label for label, _ in specs],
            [
                "Original mesh with watertight issue arrays",
                "Processed candidate with watertight issue arrays",
                "Original issue-adjacent faces",
                "Processed issue-adjacent faces",
            ],
        )

    def test_workflow_can_select_labeled_3d_stages_without_promoting_candidates(self) -> None:
        report = {"decision": {"status": "rejected"}, "outputs": {
            "accepted_mesh_vtp": None,
            "mesh_previews": [
                {"label": "Original physical assembly", "path": "source.vtp"},
                {"label": "Unaccepted closure candidate", "path": "candidate.vtp"},
            ],
        }}
        self.assertEqual(html_report.preview_specs(report), [
            ("Original physical assembly", "source.vtp"),
            ("Unaccepted closure candidate", "candidate.vtp"),
        ])
        self.assertIsNone(report["outputs"]["accepted_mesh_vtp"])

    def test_rejected_projected_candidate_is_visible_without_proxy(self) -> None:
        report = {"input": {"path": "source.vtp"}, "outputs": {
            "accepted_mesh_vtp": None,
            "source_projected_watertight_candidate_vtp": "candidate.vtp",
        }}
        self.assertEqual(html_report.preview_specs(report), [
            ("Source exterior", "source.vtp"),
            ("Unaccepted candidate — diagnostic only", "candidate.vtp"),
        ])

    def test_preview_specs_labels_closure_proxy_as_diagnostic(self) -> None:
        report = {
            "input": {"path": "/models/input.stl"},
            "outputs": {
                "rejected_candidate_vtp": "outputs/run/source_preserving_candidate.vtp",
                "closure_proxy_vtp": "outputs/run/closure_proxy.vtp",
            }
        }

        specs = html_report.preview_specs(report)

        self.assertEqual(specs[0][0], "Rejected candidate")
        self.assertEqual(specs[1][0], "Diagnostic closure proxy")

    def test_preview_specs_orders_accepted_source_projected_contract(self) -> None:
        projected = "outputs/run/source_projected_watertight_candidate.vtp"
        report = {
            "input": {"path": "/models/input.stl"},
            "outputs": {
                "stage1_exterior_candidate_vtp": "outputs/run/stage1_exterior_candidate.vtp",
                "accepted_mesh_vtp": projected,
                "source_projected_watertight_candidate_vtp": projected,
                "closure_proxy_vtp": "outputs/run/closure_proxy.vtp",
            },
        }

        specs = html_report.preview_specs(report)

        self.assertEqual([label for label, _ in specs[:3]], [
            "Source exterior",
            "Accepted source-projected watertight shell",
            "Diagnostic closure proxy",
        ])

    def test_preview_specs_uses_accepted_projected_mesh(self) -> None:
        projected = "outputs/run/source_projected_watertight_candidate.vtp"
        report = {
            "input": {"path": "/models/input.stl"},
            "outputs": {
                "stage1_exterior_candidate_vtp": "outputs/run/stage1_exterior_candidate.vtp",
                "accepted_mesh_vtp": projected,
                "source_projected_watertight_candidate_vtp": projected,
                "closure_proxy_vtp": "outputs/run/closure_proxy.vtp",
            },
        }

        specs = html_report.preview_specs(report)

        self.assertEqual(specs[1], ("Accepted source-projected watertight shell", projected))

    def test_mesh_preview_defaults_to_full_resolution_without_a_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "source.vtp"
            original_triangles = write_grid_mesh(path)
            with mock.patch.dict(os.environ, {"CAD_SURFACE_MESHER_VIEWER_TRIANGLES": ""}):
                preview = html_report.mesh_preview("dirty input", path)

        self.assertEqual(preview["triangles"], original_triangles)
        self.assertEqual(preview["original_triangles"], original_triangles)
        self.assertTrue(preview["full_resolution"])
        self.assertFalse(preview["downsampled"])
        self.assertIsNone(preview["viewer_triangle_limit"])

    def test_mesh_preview_downsamples_when_limit_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "source.vtp"
            original_triangles = write_grid_mesh(path)
            with mock.patch.dict(os.environ, {"CAD_SURFACE_MESHER_VIEWER_TRIANGLES": "24"}):
                preview = html_report.mesh_preview("dirty input", path)

        self.assertLess(preview["triangles"], original_triangles)
        self.assertEqual(preview["original_triangles"], original_triangles)
        self.assertFalse(preview["full_resolution"])
        self.assertTrue(preview["downsampled"])
        self.assertEqual(preview["viewer_triangle_limit"], 24)

    def test_mesh_preview_exposes_binary_issue_cell_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "issues.vtp"
            triangle_count = write_grid_mesh(path, with_issue_mask=True)
            with mock.patch.dict(os.environ, {"CAD_SURFACE_MESHER_VIEWER_TRIANGLES": "0"}):
                preview = html_report.mesh_preview("issue mesh", path)

        self.assertEqual(len(preview["issue_values"]), triangle_count)
        self.assertEqual(set(preview["issue_values"]), {0.0, 1.0})

    def test_preview_preserves_degenerate_triangles_and_issue_cells_for_inspection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "dirty.vtp"
            mesh = pv.PolyData(
                np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]),
                np.array([3, 0, 1, 2, 3, 0, 0, 1]),
            )
            mesh.cell_data["watertight_audit_issue_mask"] = np.array([0, 1])
            mesh.save(path)
            with mock.patch.dict(os.environ, {"CAD_SURFACE_MESHER_VIEWER_TRIANGLES": "0"}):
                preview = html_report.mesh_preview("dirty input", path)
        self.assertEqual(preview["triangles"], 2)
        self.assertTrue(preview["full_resolution"])
        self.assertEqual(preview["issue_values"], [0., 1.])

    def test_print_scalar_fields_keep_uint8_generated_faces_and_both_3d_stages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "print.vtp"
            count = write_grid_mesh(path)
            mesh = pv.read(path)
            mesh.cell_data["print_generated_face"] = np.zeros(count, dtype=np.uint8)
            mesh.cell_data["print_source_distance_mm"] = np.zeros(count)
            mesh.cell_data["print_thickness_mm"] = np.full(count, -1.)
            mesh.save(path)
            report = {"outputs": {"mesh_previews": [{"label": "Source", "path": str(path)}, {"label": "Candidate", "path": str(path)}]}}
            previews = html_mesh_preview.mesh_previews(report, Path(tmp_dir))
        self.assertEqual(len(previews), 2)
        self.assertTrue(all("error" not in mesh for mesh in previews))
        self.assertEqual(previews[1]["fields"]["print_generated_face"], [0.] * count)

    def test_missing_comparison_geometry_is_visible_instead_of_silently_dropped(self) -> None:
        report = {"outputs": {"mesh_previews": [{"label": "Missing candidate", "path": "/missing-model.vtp"}]}}
        html = html_report.render_html(report, "Report", Path("."))
        self.assertIn("3D model unavailable", html)
        self.assertIn("Missing candidate", html)
        self.assertIn("mesh file is missing", html)

    def test_decimated_source_cannot_reuse_original_opening_face_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "source.vtp"
            write_grid_mesh(path)
            report = {"target": {"purpose": "print"}, "outputs": {"mesh_previews": [{"label": "Source", "path": str(path)}]}, "unresolved_policy_packet": {"items": [{"id": "opening_1", "source_triangle_ids": [100]}]}}
            with mock.patch.dict(os.environ, {"CAD_SURFACE_MESHER_VIEWER_TRIANGLES": "24"}):
                preview = html_mesh_preview.mesh_previews(report, Path(tmp_dir))[0]
        self.assertNotIn("regions", preview)
        self.assertIn("full-resolution", preview["opening_review_unavailable"])

    def test_projection_fallback_heatmap_comes_from_saved_projection_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "projected.vtp"
            count = write_grid_mesh(path)
            mesh = pv.read(path)
            mesh.cell_data["source_projected_vertex_fraction"] = np.linspace(1., 0., count)
            mesh.save(path)
            preview = html_report.mesh_preview("Projected", path)
        values = preview["fields"]["projection_fallback_fraction"]
        self.assertEqual(len(values), count)
        self.assertEqual(values[0], 0.)
        self.assertEqual(values[-1], 1.)
        self.assertIn("projection_fallback_fraction", html_report.viewer_section([preview]))

    def test_render_html_includes_contract_sections_and_diagnostic_proxy_label(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            source = Path(tmp_dir) / "source.vtp"
            candidate = Path(tmp_dir) / "candidate.vtp"
            proxy = Path(tmp_dir) / "proxy.vtp"
            write_grid_mesh(source)
            write_grid_mesh(candidate)
            write_grid_mesh(proxy)
            report = {
                "input": {"path": str(source)},
                "decision": {"status": "rejected"},
                "outputs": {
                    "rejected_candidate_vtp": str(candidate),
                    "closure_proxy_vtp": str(proxy),
                },
            }

            with mock.patch.object(html_report, "vtk_js_source", return_value=""):
                html = html_report.render_html(report, "Report", Path(tmp_dir))

        self.assertIn("Rejected candidate", html)
        self.assertIn("Diagnostic closure proxy", html)
        self.assertIn("Summary", html)
        self.assertIn("Outputs", html)
        self.assertIn("Full JSON", html)
        self.assertLess(html.index("<h2>3D Mesh Evidence</h2>"), html.index("<h2>Summary</h2>"))
        self.assertIn('id="vtk-reset-0"', html)
        self.assertIn('id="vtk-edges-1"', html)
        self.assertIn('id="vtk-opacity-1"', html)

    def test_render_html_displays_source_projected_stage(self) -> None:
        report = {
            "stages": {
                "source_projected_watertight_candidate": {
                    "path": "source_projected_watertight_candidate.vtp",
                    "source": "closure_proxy_plus_source_surface",
                    "status": "accepted",
                    "accepted_final_geometry": True,
                    "metrics": {
                        "triangles": 42,
                        "topology": {
                            "boundary_edges": 0,
                            "non_manifold_edges": 0,
                            "components": {"count": 1},
                        },
                    },
                }
            }
        }

        with mock.patch.object(html_report, "vtk_js_source", return_value=""):
            html = html_report.render_html(report, "Report", Path("."))

        self.assertIn("<h2>Stages</h2>", html)
        self.assertIn("source_projected_watertight_candidate", html)
        self.assertIn("source_projected_watertight_candidate.vtp", html)

    def test_render_html_links_full_resolution_watertight_vtp_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            original = root / "original.vtp"
            processed = root / "processed.vtp"
            marker = root / "processed_boundary_edges.vtp"
            write_grid_mesh(original, with_issue_mask=True)
            write_grid_mesh(processed, with_issue_mask=True)
            marker.write_text("marker", encoding="utf-8")
            report = {
                "input": {"path": str(original)},
                "decision": {"status": "rejected"},
                "outputs": {
                    "original_mesh_vtp": str(original),
                    "processed_mesh_vtp": str(processed),
                    "watertight_issue_artifacts": [
                        {
                            "stage": "processed",
                            "kind": "boundary_edges",
                            "count": 7,
                            "regions": 2,
                            "representation": "line_cells",
                            "path": str(marker),
                        }
                    ],
                },
            }

            with mock.patch.object(html_report, "vtk_js_source", return_value=""):
                html = html_report.render_html(report, "Report", root)

        self.assertIn("Watertight Issue VTP Artifacts", html)
        self.assertIn('href="processed_boundary_edges.vtp"', html)
        self.assertIn("red cells are adjacent to a recorded watertightness defect", html)


if __name__ == "__main__":
    unittest.main()
