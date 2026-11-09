from __future__ import annotations

import sys
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import pyvista as pv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from mesh_io import triangle_faces
from source_fidelity_recovery import recover_source_fidelity
from source_fidelity_sampling import (
    _closed_material_winding_soup,
    _closed_soup_interior_witness_mask,
    _closed_material_reference,
    _interior_witness_mask,
    _pixel_samples,
    select_missing_source_facets,
)
from source_material_repair import repair_source_mesh
import igl


class SourceFidelityRecoveryTest(unittest.TestCase):
    def test_recovery_facet_budget_stops_before_patch_construction(self):
        source = pv.Cube().triangulate()
        candidate = source.copy(deep=True)
        with patch("source_fidelity_recovery.select_missing_source_facets",
                   return_value=(np.asarray([0, 1]), {"source_facets_selected": 2})), \
             patch("source_fidelity_recovery._minkowski_facets") as make_patches:
            output, report = recover_source_fidelity(source, candidate, max_error=.01,
                                                     grid_size=32, max_recovery_facets=1)
        self.assertIs(output, candidate)
        self.assertEqual(report["status"], "recovery_facet_budget_exceeded")
        self.assertFalse(report["acceptance_ready"])
        self.assertIn("2 source facets", report["failure_reason"])
        make_patches.assert_not_called()

    def test_fast_winding_is_only_a_conservative_exact_query_filter(self):
        box = pv.Cube().triangulate()
        points = np.asarray(box.points, dtype=np.float64)
        faces = triangle_faces(box)
        query = np.asarray([[0., 0., 0.], [.25, .1, -.2], [3., 0., 0.],
                            [0., 0., 1.01], [-2., 1., 0.]])
        exact = np.asarray(igl.winding_number(points, faces, query)) > .5
        actual, detail = _interior_witness_mask(points, faces, query)
        np.testing.assert_array_equal(actual, exact)
        self.assertLess(detail["direct_winding_queries"], detail["winding_queries"])
        self.assertIn("not an exact point-side proof", detail["winding_numeric_method"])
        self.assertGreater(detail["fast_winding_negative_queries_kept"], 0)

    def test_fast_winding_false_negative_cannot_hide_an_interior_witness(self):
        box = pv.Cube().triangulate()
        points = np.asarray(box.points, dtype=np.float64)
        faces = triangle_faces(box)
        query = np.asarray([[0., 0., 0.], [3., 0., 0.]])
        with patch("source_fidelity_sampling.igl.fast_winding_number",
                   return_value=np.asarray([-1., -1.])):
            actual, detail = _interior_witness_mask(points, faces, query)
        np.testing.assert_array_equal(actual, [False, False])
        self.assertEqual(detail["direct_winding_queries"], 0)
        # Leaving a truly interior sample visible is conservative and cannot
        # turn a source-fidelity failure into an accepted candidate.

    def test_uncertain_soup_winding_is_not_filtered(self):
        box = pv.Cube().triangulate()
        query = np.asarray([[0., 0., 0.], [.1, 0., 0.]])
        with patch("source_fidelity_sampling.igl.fast_winding_number", return_value=np.ones(2)), \
             patch("source_fidelity_sampling.igl.winding_number", return_value=np.asarray([.6, 1.])):
            actual, _ = _interior_witness_mask(np.asarray(box.points), triangle_faces(box), query,
                                               interior_threshold=.5001, integer_margin=1e-4)
        np.testing.assert_array_equal(actual, [False, True])

    def test_identical_source_returns_original_candidate_without_union(self):
        source = pv.Plane(i_resolution=1, j_resolution=1).triangulate()
        candidate = source.copy(deep=True)

        output, report = recover_source_fidelity(
            source, candidate, max_error=1.0e-4, grid_size=48
        )

        self.assertIs(output, candidate)
        self.assertEqual(report["status"], "within_budget")
        self.assertEqual(report["source_facets_selected"], 0)

    def test_visible_error_selects_source_rows_and_transports_corner_normals(self):
        source = pv.Sphere(radius=1.0, theta_resolution=12, phi_resolution=12).triangulate()
        source_faces = triangle_faces(source)
        source.cell_data["source_facet_id"] = np.full(source.n_cells, 900_000)
        source.cell_data["corner_normals"] = source.point_data["Normals"][
            source_faces
        ].reshape(source.n_cells, 9)
        candidate = pv.Sphere(radius=0.9, theta_resolution=12, phi_resolution=12).triangulate()

        output, report = recover_source_fidelity(
            source, candidate, max_error=0.05, grid_size=48
        )

        self.assertEqual(report["status"], "union_proposal_created")
        self.assertFalse(report["acceptance_ready"])
        generated = np.asarray(output.cell_data["source_fidelity_recovery_generated"])
        self.assertTrue(generated.any())
        lineage = np.asarray(output.cell_data["source_facet_id"])
        self.assertTrue(np.all((lineage[generated] >= 0) & (lineage[generated] < source.n_cells)))
        self.assertTrue(np.all(lineage[~generated] == -1))
        normals = np.asarray(output.cell_data["corner_normals"])
        self.assertEqual(normals.shape, (output.n_cells, 9))
        self.assertTrue(np.isfinite(normals).all())

    def test_invalid_candidate_returns_the_original_with_failure_diagnostics(self):
        source = pv.Plane(i_resolution=1, j_resolution=1).triangulate()
        candidate = pv.PolyData()

        output, report = recover_source_fidelity(
            source, candidate, max_error=0.01, grid_size=48
        )

        self.assertIs(output, candidate)
        self.assertEqual(report["status"], "failed_original_candidate_returned")
        self.assertIn("failure_reason", report)

    def test_nonfinite_candidate_distance_fails_closed(self):
        source = pv.Plane(i_resolution=1, j_resolution=1).triangulate()
        with patch("source_fidelity_sampling.igl.point_mesh_squared_distance",
                   return_value=(np.asarray([np.nan]), np.asarray([0]), np.zeros((1, 3)))):
            with self.assertRaisesRegex(ValueError, "candidate distance returned invalid"):
                select_missing_source_facets(
                    np.asarray(source.points), triangle_faces(source),
                    np.asarray(source.points), triangle_faces(source),
                    max_error=1e-4, grid_size=32,
                )

    def test_source_row_lineage_ignores_unrelated_source_ids_and_keeps_receiver_ids(self):
        source = pv.Sphere(radius=1.0, theta_resolution=10, phi_resolution=10).triangulate()
        source.cell_data["source_facet_id"] = np.full(source.n_cells, 900_000, dtype=np.int64)
        candidate = pv.Sphere(radius=0.9, theta_resolution=10, phi_resolution=10).triangulate()
        candidate_ids = np.arange(candidate.n_cells, dtype=np.int64) + 700_000
        candidate.cell_data["source_facet_id"] = candidate_ids
        candidate.field_data["coordinate_scale"] = np.asarray([3.0])

        output, report = recover_source_fidelity(
            source, candidate, max_error=0.05, grid_size=48
        )

        self.assertEqual(report["status"], "union_proposal_created")
        generated = np.asarray(output.cell_data["source_fidelity_recovery_generated"], dtype=bool)
        birth = np.asarray(output.cell_data["source_fidelity_recovery_cgal_birth_face_index"])
        source_rows = np.asarray(output.cell_data["source_fidelity_recovery_source_face_row"])
        facet_ids = np.asarray(output.cell_data["source_facet_id"])
        np.testing.assert_array_equal(facet_ids[~generated], candidate_ids[birth[~generated]])
        np.testing.assert_array_equal(source_rows[generated], facet_ids[generated])
        self.assertTrue(np.all((source_rows[generated] >= 0) & (source_rows[generated] < source.n_cells)))
        np.testing.assert_array_equal(output.field_data["coordinate_scale"], [3.0])

    def test_unsupported_candidate_point_data_fails_closed_without_losing_fields(self):
        source = pv.Sphere(radius=1.0, theta_resolution=10, phi_resolution=10).triangulate()
        candidate = pv.Sphere(radius=0.9, theta_resolution=10, phi_resolution=10).triangulate()
        candidate.point_data["temperature"] = np.linspace(1.0, 2.0, candidate.n_points)

        output, report = recover_source_fidelity(
            source, candidate, max_error=0.05, grid_size=48
        )

        self.assertIs(output, candidate)
        self.assertEqual(report["status"], "failed_original_candidate_returned")
        self.assertIn("temperature", report["failure_reason"])
        self.assertEqual(output.point_data["temperature"].shape, (candidate.n_points,))

    def test_recovery_and_selected_source_rows_are_invariant_to_rigid_scale_and_translation(self):
        source = pv.Sphere(radius=1.0, theta_resolution=10, phi_resolution=10).triangulate()
        candidate = pv.Sphere(radius=0.9, theta_resolution=10, phi_resolution=10).triangulate()
        outputs = []
        reports = []
        for scale, translation in ((1.0, np.zeros(3)), (1000.0, np.array([10_000., -20_000., 30_000.]))):
            src = source.copy(deep=True)
            dst = candidate.copy(deep=True)
            src.points = src.points * scale + translation
            dst.points = dst.points * scale + translation
            output, report = recover_source_fidelity(
                src, dst, max_error=0.05 * scale, grid_size=48
            )
            outputs.append(output)
            reports.append(report)
        self.assertEqual(reports[0]["status"], "union_proposal_created")
        self.assertEqual(reports[1]["status"], "union_proposal_created")
        self.assertEqual(reports[0]["selected_source_face_rows"], reports[1]["selected_source_face_rows"])
        self.assertAlmostEqual(
            reports[0]["patch_radius_input_units"],
            reports[1]["patch_radius_input_units"] / 1000.0,
            places=7,
        )
        scaled_points = (np.asarray(outputs[1].points) - np.array([10_000., -20_000., 30_000.])) / 1000.
        first_distance2, _, _ = igl.point_mesh_squared_distance(
            np.asarray(outputs[0].points), scaled_points, triangle_faces(outputs[1])
        )
        second_distance2, _, _ = igl.point_mesh_squared_distance(
            scaled_points, np.asarray(outputs[0].points), triangle_faces(outputs[0])
        )
        self.assertLess(np.sqrt(np.maximum(first_distance2, 0.)).max(), 1e-5)
        self.assertLess(np.sqrt(np.maximum(second_distance2, 0.)).max(), 1e-5)

    def test_unannotated_mesh_recovers_geometry_without_inventing_native_normals(self):
        source = pv.Sphere(radius=1.0, theta_resolution=10, phi_resolution=10).triangulate()
        candidate = pv.Sphere(radius=0.9, theta_resolution=10, phi_resolution=10).triangulate()
        source.point_data.clear()
        candidate.point_data.clear()

        output, report = recover_source_fidelity(
            source, candidate, max_error=0.05, grid_size=48
        )

        self.assertEqual(report["status"], "union_proposal_created")
        self.assertNotIn("corner_normals", output.cell_data)
        self.assertEqual(output.n_cells, len(output.cell_data["source_fidelity_recovery_generated"]))

    def test_protected_torus_aperture_is_not_patched_when_within_budget(self):
        major, minor, around, tube = 1.0, 0.25, 32, 12
        points = []
        for i in range(around):
            u = 2 * np.pi * i / around
            for j in range(tube):
                v = 2 * np.pi * j / tube
                radius = major + minor * np.cos(v)
                points.append((radius * np.cos(u), radius * np.sin(u), minor * np.sin(v)))
        faces = []
        for i in range(around):
            for j in range(tube):
                a = i * tube + j
                b = ((i + 1) % around) * tube + j
                c = ((i + 1) % around) * tube + (j + 1) % tube
                d = i * tube + (j + 1) % tube
                faces.extend(((a, b, c), (a, c, d)))
        raw_faces = np.column_stack((np.full(len(faces), 3), np.asarray(faces))).ravel()
        source = pv.PolyData(np.asarray(points), raw_faces)
        candidate = source.copy(deep=True)

        output, report = recover_source_fidelity(
            source, candidate, max_error=1e-4, grid_size=48
        )

        self.assertIs(output, candidate)
        self.assertEqual(report["status"], "within_budget")
        self.assertEqual(report["source_facets_selected"], 0)
        self.assertEqual(output.n_points - len(points), 0)
        # A triangulated torus has Euler characteristic zero; its through-hole stays open.
        edges = {tuple(sorted(edge)) for face in faces for edge in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0]))}
        self.assertEqual(len(points) - len(edges) + len(faces), 0)

    def test_subpixel_external_wing_is_found_by_triangle_witnesses_at_low_view_grid(self):
        body = pv.Cube().triangulate()
        wing_points = np.asarray(
            [[1.2, .013, .027], [1.2, .023, .027], [1.2, .013, .037]]
        )
        body_faces = triangle_faces(body)
        faces = np.vstack(
            (body_faces, np.asarray([[body.n_points, body.n_points + 1, body.n_points + 2]]))
        )
        points = np.vstack((body.points, wing_points))
        source = pv.PolyData(
            points, np.column_stack((np.full(len(faces), 3), faces)).ravel()
        )
        candidate = body.copy(deep=True)
        source.point_data.clear()
        candidate.point_data.clear()

        pixels, pixel_rows = _pixel_samples(points, faces, 32)
        self.assertNotIn(len(faces) - 1, np.concatenate(pixel_rows))

        output, report = recover_source_fidelity(
            source, candidate, max_error=.01, grid_size=32
        )

        self.assertEqual(report["status"], "union_proposal_created")
        self.assertEqual(report["witnesses_per_source_face"], 7)
        self.assertTrue(report["geometry_witness_samples"] >= 7 * len(faces))
        self.assertIn(len(faces) - 1, report["selected_source_face_rows"])
        self.assertGreater(output.n_points, candidate.n_points)

    def test_overlapping_closed_components_filter_only_witnesses_inside_union_material(self):
        outer = pv.Cube(center=(0., 0., 0.), x_length=2., y_length=2., z_length=2.).triangulate()
        inner = pv.Cube(center=(.7, 0., 0.), x_length=.8, y_length=.8, z_length=.8).triangulate()
        outer_faces = triangle_faces(outer)
        inner_faces = triangle_faces(inner) + outer.n_points
        points = np.vstack((outer.points, inner.points))
        faces = np.vstack((outer_faces, inner_faces))
        inner_start = len(outer_faces)
        source = pv.PolyData(
            points, np.column_stack((np.full(len(faces), 3), faces)).ravel()
        )
        candidate = outer.copy(deep=True)
        source.point_data.clear()
        candidate.point_data.clear()
        inner_rows = np.arange(inner_start, len(faces))
        partial_inner_rows = inner_rows[
            (np.max(points[faces[inner_rows], 0], axis=1) > 1.05)
            & (np.min(points[faces[inner_rows], 0], axis=1) < 0.5)
            & (np.max(np.abs(points[faces[inner_rows], 1] - 0.4), axis=1) < 1e-6)
        ]
        self.assertGreater(len(partial_inner_rows), 0)
        buried_inner = inner_rows[
            np.all(np.abs(points[faces[inner_rows]]) < 1.0 - 1e-8, axis=(1, 2))
        ]

        output, report = recover_source_fidelity(
            source, candidate, max_error=.02, grid_size=48
        )

        self.assertGreater(report["source_material_interior_witnesses_filtered"], 0)
        self.assertGreater(report["raw_sample_distance_input_units"]["max"],
                           report["sample_distance_input_units"]["max"])
        selected = np.asarray(report["selected_source_face_rows"], dtype=np.int64)
        self.assertTrue(np.any(np.isin(selected, inner_rows)))
        self.assertTrue(np.all(np.isin(partial_inner_rows, selected)))
        self.assertEqual(np.intersect1d(selected, buried_inner).size, 0)
        self.assertEqual(
            report["closed_material_reference"]["status"],
            "validated_closed_source_material",
        )
        self.assertGreater(output.n_points, candidate.n_points)

    def test_negative_inner_shell_is_a_cavity_not_a_patchable_hidden_solid(self):
        outer = pv.Cube(x_length=2., y_length=2., z_length=2.).triangulate()
        inner = pv.Cube(x_length=.6, y_length=.6, z_length=.6).triangulate()
        outer_faces = triangle_faces(outer)
        inner_faces = triangle_faces(inner)[:, ::-1] + outer.n_points
        points = np.vstack((outer.points, inner.points))
        faces = np.vstack((outer_faces, inner_faces))
        source = pv.PolyData(
            points, np.column_stack((np.full(len(faces), 3), faces)).ravel()
        )
        candidate = outer.copy(deep=True)
        source.point_data.clear()
        candidate.point_data.clear()

        output, report = recover_source_fidelity(
            source, candidate, max_error=.02, grid_size=48
        )

        self.assertIs(output, candidate)
        self.assertEqual(report["status"], "failed_original_candidate_returned")
        self.assertFalse(report["acceptance_ready"])
        self.assertIn("semantic Boolean difference", report["failure_reason"])

    def test_closed_component_winding_soup_matches_union_for_material_relations(self):
        from source_material_closure import build_source_material

        cases = []
        outer = pv.Cube(x_length=2., y_length=2., z_length=2.).triangulate()
        cavity = pv.Cube(x_length=.6, y_length=.6, z_length=.6).triangulate()
        for reverse_root in (False, True):
            outer_faces = triangle_faces(outer)
            cavity_faces = triangle_faces(cavity)[:, ::-1] + outer.n_points
            if reverse_root:
                outer_faces = outer_faces[:, ::-1]
                cavity_faces = cavity_faces[:, ::-1]
            cases.append(("nested_cavity" if not reverse_root else "reversed_root_cavity",
                          np.vstack((outer.points, cavity.points)),
                          np.vstack((outer_faces, cavity_faces)),
                          np.asarray([[.8, 0., 0.], [0., 0., 0.], [2., 0., 0.]])))

        left = pv.Cube(x_length=2., y_length=2., z_length=2.).triangulate()
        right = pv.Cube(center=(.8, 0., 0.), x_length=1.2, y_length=1.2, z_length=1.2).triangulate()
        cases.append(("overlapping_positive", np.vstack((left.points, right.points)),
                      np.vstack((triangle_faces(left), triangle_faces(right) + left.n_points)),
                      np.asarray([[.5, 0., 0.], [1.2, 0., 0.], [2., 0., 0.]])))

        torus = pv.ParametricTorus(ringradius=1., crosssectionradius=.25,
                                   u_res=28, v_res=14).triangulate()
        crossing = pv.Cube(center=(.9, 0., 0.), x_length=.8, y_length=.8, z_length=.8).triangulate()
        cases.append(("torus_crossing_box", np.vstack((torus.points, crossing.points)),
                      np.vstack((triangle_faces(torus), triangle_faces(crossing) + torus.n_points)),
                      np.asarray([[0., 0., 0.], [.9, 0., 0.], [1.28, .3, .2]])))

        for name, points, faces, query in cases:
            with self.subTest(case=name):
                soup = _closed_material_winding_soup(points, faces, .005)
                self.assertIsNotNone(soup)
                soup_points, soup_faces, detail = soup
                self.assertIn("floating-point", detail["winding_numeric_method"])
                union_points, union_faces, _, _ = build_source_material(points, faces, max_error=.005)
                soup_winding = np.asarray(igl.winding_number(soup_points, soup_faces, query))
                union_winding = np.asarray(igl.winding_number(union_points, union_faces, query))
                np.testing.assert_array_equal(soup_winding > .5, union_winding > .5)

    def test_point_touch_winding_soup_filters_only_far_interior_witnesses(self):
        left = pv.Cube(x_length=2., y_length=2., z_length=2.).triangulate()
        right = pv.Cube(center=(2., 2., 2.), x_length=2., y_length=2., z_length=2.).triangulate()
        points = np.vstack((left.points, right.points))
        faces = np.vstack((triangle_faces(left), triangle_faces(right) + left.n_points))
        query = np.asarray([[0., 0., 0.], [2., 2., 2.], [1., 1., 1.], [4., 4., 4.]])
        soup, details = _closed_soup_interior_witness_mask(points, faces, query, .01)
        np.testing.assert_array_equal(soup, [True, True, False, False])
        self.assertEqual(details["status"], "validated_closed_component_winding_soup")
        self.assertEqual(details["source_closed_components"], 2)
        self.assertIn("not an exact point-side proof", details["winding_numeric_method"])

    def test_positive_shell_face_can_be_buried_inside_another_shell(self):
        left = pv.Cube(x_length=2., y_length=2., z_length=2.).triangulate()
        right = pv.Cube(center=(.6, 0., 0.), x_length=2., y_length=2., z_length=2.).triangulate()
        points = np.vstack((left.points, right.points))
        left_faces = triangle_faces(left)
        faces = np.vstack((left_faces, triangle_faces(right) + left.n_points))
        triangles = points[left_faces]
        inside_row = int(np.flatnonzero(np.all(np.isclose(triangles[:, :, 0], 1.), axis=1))[0])
        outside_row = int(np.flatnonzero(np.all(np.isclose(triangles[:, :, 0], -1.), axis=1))[0])
        query = np.asarray([triangles[inside_row].mean(axis=0), triangles[outside_row].mean(axis=0)])
        buried, detail = _closed_soup_interior_witness_mask(
            points, faces, query, .01, query_face_rows=np.asarray([inside_row, outside_row])
        )
        np.testing.assert_array_equal(buried, [True, False])
        self.assertEqual(detail["witnesses_inside_other_closed_material"], 1)

    def test_negative_cavity_face_is_never_filtered_as_other_shell_interior(self):
        outer = pv.Cube(x_length=2., y_length=2., z_length=2.).triangulate()
        inner = pv.Cube(x_length=.6, y_length=.6, z_length=.6).triangulate()
        points = np.vstack((outer.points, inner.points))
        outer_faces = triangle_faces(outer)
        inner_faces = triangle_faces(inner)[:, ::-1] + outer.n_points
        faces = np.vstack((outer_faces, inner_faces))
        row = len(outer_faces)
        query = points[faces[row]].mean(axis=0).reshape(1, 3)
        buried, _ = _closed_soup_interior_witness_mask(points, faces, query, .01,
                                                       query_face_rows=np.asarray([row]))
        self.assertFalse(buried[0])

    def test_other_shell_containment_uses_numerical_clearance_not_shape_budget(self):
        first = pv.Cube(x_length=2., y_length=2., z_length=2.).triangulate()
        inside = pv.Cube(center=(.002, 0., 0.), x_length=2., y_length=2., z_length=2.).triangulate()
        points = np.vstack((first.points, inside.points))
        first_faces = triangle_faces(first)
        faces = np.vstack((first_faces, triangle_faces(inside) + first.n_points))
        triangles = points[first_faces]
        row = int(np.flatnonzero(np.all(np.isclose(triangles[:, :, 0], 1.), axis=1))[0])
        query = triangles[row].mean(axis=0).reshape(1, 3)
        buried, detail = _closed_soup_interior_witness_mask(points, faces, query, .01,
                                                            query_face_rows=np.asarray([row]))
        self.assertTrue(buried[0])
        self.assertLess(detail["other_shell_clearance_input_units"], .002)

        coincident_points = np.vstack((first.points, first.points))
        coincident_faces = np.vstack((first_faces, first_faces + first.n_points))
        boundary, _ = _closed_soup_interior_witness_mask(coincident_points, coincident_faces, query, .01,
                                                         query_face_rows=np.asarray([row]))
        self.assertFalse(boundary[0])

    def test_fast_reference_keeps_exterior_faces_missing_from_candidate(self):
        left = pv.Cube(x_length=2., y_length=2., z_length=2.).triangulate()
        right = pv.Cube(center=(.6, 0., 0.), x_length=2., y_length=2., z_length=2.).triangulate()
        left_faces = triangle_faces(left)
        right_faces = triangle_faces(right) + left.n_points
        points = np.vstack((left.points, right.points))
        faces = np.vstack((left_faces, right_faces))
        selected, report = select_missing_source_facets(
            points, faces, np.asarray(left.points), left_faces,
            max_error=.01, grid_size=32, prefer_soup=True,
        )
        right_triangles = points[right_faces]
        outside_rows = np.flatnonzero(np.all(np.isclose(right_triangles[:, :, 0], 1.6), axis=1)) + len(left_faces)
        self.assertGreater(len(outside_rows), 0)
        self.assertTrue(np.all(np.isin(outside_rows, selected)))
        self.assertGreater(report["source_material_interior_witnesses_filtered"], 0)

    def test_union_reference_failure_uses_closed_soup_for_buried_open_sheet(self):
        left = pv.Cube(x_length=2., y_length=2., z_length=2.).triangulate()
        right = pv.Cube(center=(2., 2., 2.), x_length=2., y_length=2., z_length=2.).triangulate()
        plane = pv.Plane(center=(0., 0., 0.), i_size=.2, j_size=.2,
                         i_resolution=1, j_resolution=1).triangulate()
        points = np.vstack((left.points, right.points, plane.points))
        left_faces = triangle_faces(left)
        right_faces = triangle_faces(right) + left.n_points
        plane_rows = np.arange(len(left_faces) + len(right_faces),
                               len(left_faces) + len(right_faces) + plane.n_cells)
        faces = np.vstack((left_faces, right_faces, triangle_faces(plane) + left.n_points + right.n_points))
        candidate_faces = np.vstack((left_faces, right_faces))
        self.assertIsNone(_closed_material_reference(points, faces, .01))
        selected, report = select_missing_source_facets(
            points, faces, points[:left.n_points + right.n_points], candidate_faces,
            max_error=.01, grid_size=32,
        )
        self.assertEqual(report["closed_material_reference"]["status"],
                         "validated_closed_component_winding_soup")
        self.assertGreater(report["source_material_interior_witnesses_filtered"], 0)
        self.assertFalse(np.any(np.isin(selected, plane_rows)))

    def test_invalid_closed_shell_fails_open_for_winding_soup(self):
        box = pv.Cube().triangulate()
        points = np.asarray(box.points, dtype=np.float64)
        faces = triangle_faces(box).copy()
        faces[0] = faces[1]
        result = _closed_material_winding_soup(points, faces, .01)
        self.assertIsNone(result)

    def test_intersecting_negative_shell_fails_closed_in_repair_and_witness_filtering(self):
        outer = pv.Cube(x_length=2., y_length=2., z_length=2.).triangulate()
        negative = pv.Cube(center=(.51, 0., 0.), x_length=1., y_length=1., z_length=1.).triangulate()
        plane = pv.Plane(center=(.76, 0., 0.), i_size=.2, j_size=.2,
                         i_resolution=1, j_resolution=1).triangulate()
        points = np.vstack((outer.points, negative.points, plane.points))
        outer_faces = triangle_faces(outer)
        negative_faces = triangle_faces(negative)[:, ::-1] + outer.n_points
        plane_faces = triangle_faces(plane) + outer.n_points + negative.n_points
        faces = np.vstack((outer_faces, negative_faces, plane_faces))
        source = pv.PolyData(
            points, np.column_stack((np.full(len(faces), 3), faces)).ravel()
        )

        with self.assertRaisesRegex(ValueError, "opposite-oriented closed components.*intersect"):
            _closed_material_reference(points, faces, .02)
        with self.assertRaisesRegex(ValueError, "opposite-oriented closed components.*intersect"):
            _closed_material_winding_soup(points, faces, .02)
        with self.assertRaisesRegex(ValueError, "opposite-oriented closed components.*intersect"):
            _closed_soup_interior_witness_mask(points, faces, np.asarray([[.76, 0., 0.]]), .02)

        selected, detail = select_missing_source_facets(
            points, faces, np.asarray(outer.points), outer_faces,
            max_error=.02, grid_size=32,
        )
        self.assertEqual(detail["closed_material_reference"]["status"],
                         "failed_open_no_samples_excluded")
        self.assertEqual(detail["source_material_interior_witnesses_filtered"], 0)
        plane_rows = np.arange(len(outer_faces) + len(negative_faces), len(faces))
        np.testing.assert_array_equal(np.intersect1d(selected, plane_rows), plane_rows)
        self.assertGreater(detail["raw_sample_distance_input_units"]["max"], .02)
        self.assertGreater(detail["sample_distance_input_units"]["max"], .02)

        proposal, recovery = recover_source_fidelity(
            source, outer, max_error=.02, grid_size=32
        )
        self.assertNotEqual(recovery["status"], "within_budget")
        self.assertFalse(recovery["acceptance_ready"])
        self.assertEqual(recovery["source_material_interior_witnesses_filtered"], 0)
        self.assertIsInstance(proposal, pv.PolyData)

        candidate, report = repair_source_mesh(source, max_error=.02, boundary_mode="wall")
        self.assertFalse(report["accepted"])
        self.assertIn("opposite-oriented closed components", report["failure_reason"])
        self.assertIsInstance(candidate, pv.PolyData)

    def test_subpixel_feature_is_found_without_any_first_hit_pixel(self):
        body = pv.Cube(x_length=2., y_length=2., z_length=2.).triangulate()
        wing = np.asarray([[1.2, .013, .027], [1.2, .023, .027], [1.2, .013, .037]])
        body_faces = triangle_faces(body)
        wing_row = len(body_faces)
        faces = np.vstack((body_faces, [[body.n_points, body.n_points + 1, body.n_points + 2]]))
        points = np.vstack((np.asarray(body.points), wing))
        source = pv.PolyData(points, np.column_stack((np.full(len(faces), 3), faces)).ravel())
        candidate = body.copy(deep=True)
        source.point_data.clear()
        candidate.point_data.clear()

        _, pixel_faces = _pixel_samples(points, faces, 32)
        self.assertFalse(any(np.any(rows == wing_row) for rows in pixel_faces))

        output, report = recover_source_fidelity(
            source, candidate, max_error=.01, grid_size=32
        )

        self.assertEqual(report["status"], "union_proposal_created")
        self.assertIn(wing_row, report["selected_source_face_rows"])
        self.assertGreater(output.n_points, candidate.n_points)


if __name__ == "__main__":
    unittest.main()
