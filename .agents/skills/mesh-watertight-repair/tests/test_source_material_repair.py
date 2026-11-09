from __future__ import annotations

import sys
import json
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import pyvista as pv
import trimesh
import igl

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from mesh_io import triangle_faces  # noqa: E402
from mesh_metrics import mesh_report  # noqa: E402
from source_projected_validation import exact_self_intersection_report  # noqa: E402
from source_material_repair import repair_source_mesh  # noqa: E402


def surface(points: np.ndarray, faces: np.ndarray) -> pv.PolyData:
    packed = np.column_stack((np.full(len(faces), 3, dtype=np.int64), faces)).ravel()
    return pv.PolyData(points, packed)


def box(center=(0., 0., 0.), size=2.) -> pv.PolyData:
    x, y, z = np.asarray(center, dtype=float)
    h = size / 2
    points = np.asarray([
        [x-h, y-h, z-h], [x+h, y-h, z-h], [x+h, y+h, z-h], [x-h, y+h, z-h],
        [x-h, y-h, z+h], [x+h, y-h, z+h], [x+h, y+h, z+h], [x-h, y+h, z+h],
    ])
    faces = np.asarray([
        [0, 3, 2], [0, 2, 1], [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4], [3, 7, 6], [3, 6, 2],
        [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5],
    ], dtype=np.int64)
    return surface(points, faces)


def triangle_coordinate_keys(points: np.ndarray, faces: np.ndarray) -> set[tuple]:
    return {
        tuple(sorted(tuple(np.round(point, 10)) for point in triangle))
        for triangle in points[faces]
    }


def two_open_boxes() -> pv.PolyData:
    points, faces = [], []
    for center in ((-2., 0., 0.), (2., 0., 0.)):
        mesh = box(center=center)
        mesh_points = np.asarray(mesh.points)
        mesh_faces = triangle_faces(mesh)
        mesh_faces = mesh_faces[mesh_points[mesh_faces, 2].min(axis=1) < .99]
        faces.append(mesh_faces + sum(len(block) for block in points))
        points.append(mesh_points)
    return surface(np.vstack(points), np.vstack(faces))


class SourceMaterialRepairTest(unittest.TestCase):
    def test_open_surface_defaults_to_intent_review_without_geometry_execution(self):
        source = surface(np.asarray([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]),
                         np.asarray([[0, 1, 2]]))
        with patch("source_material_repair.build_source_material") as build:
            candidate, report = repair_source_mesh(source, max_error=.01)
        self.assertFalse(report["accepted"])
        self.assertEqual(report["boundary_mode"], "review")
        self.assertGreater(report["unresolved_policy_packet"]["item_count_total"], 0)
        self.assertIn("functional correctness", report["limitations"][0])
        np.testing.assert_array_equal(candidate.points, source.points)
        np.testing.assert_array_equal(triangle_faces(candidate), triangle_faces(source))
        build.assert_not_called()

    def test_rejected_source_witnesses_are_identified_in_acceptance_checks(self):
        source = surface(np.asarray([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]),
                         np.asarray([[0, 1, 2]]))
        distances = {"p50": 0., "p95": 0., "p99": 0., "max": 0.1}
        measurement = {"sample_distance_input_units": distances,
                       "raw_sample_distance_input_units": distances}
        with patch("source_material_repair.select_missing_source_facets",
                   return_value=(np.asarray([0]), measurement)):
            _, report = repair_source_mesh(source, max_error=.01, boundary_mode="wall",
                                           max_recovery_rounds=0)
        self.assertFalse(report["accepted"])
        self.assertTrue(report["acceptance_checks"]["geometry"])
        self.assertFalse(report["acceptance_checks"]["source_witnesses_within_budget"])
        self.assertEqual(report["acceptance_checks"]["source_facets_over_budget"], 1)

    def assert_accepted_mesh(self, candidate: pv.PolyData, report: dict) -> dict:
        self.assertIsInstance(candidate, pv.PolyData)
        self.assertIs(report.get("accepted"), True, report)
        points = np.asarray(candidate.points, dtype=np.float64)
        faces = triangle_faces(candidate)
        self.assertTrue(np.isfinite(points).all())
        self.assertGreater(len(faces), 0)
        metrics = mesh_report(points, faces)
        topology = metrics["topology"]
        self.assertEqual(topology["boundary_edges"], 0)
        self.assertEqual(topology["non_manifold_edges"], 0)
        self.assertEqual(topology["non_manifold_vertices"], 0)
        self.assertEqual(topology["inconsistent_winding_edges"], 0)
        self.assertEqual(metrics["quality"]["degenerate_faces"], 0)
        self.assertTrue(metrics["volume"]["reliable"])
        self.assertTrue(metrics["volume"]["orientation_consistent"])
        exact = exact_self_intersection_report(points, faces)
        self.assertIs(exact["passed"], True, exact)
        return metrics

    def test_closed_box_and_two_disjoint_solids_preserve_geometry_and_components(self):
        source = box()
        candidate, report = repair_source_mesh(source, max_error=0.01)
        metrics = self.assert_accepted_mesh(candidate, report)
        self.assertTrue(report["acceptance_checks"]["source_witnesses_within_budget"])
        self.assertEqual(report["acceptance_checks"]["source_facets_over_budget"], 0)
        self.assertTrue(report["acceptance_checks"]["output_frame_geometry"])
        self.assertEqual(metrics["topology"]["components"]["count"], 1)
        self.assertEqual(triangle_coordinate_keys(candidate.points, triangle_faces(candidate)),
                         triangle_coordinate_keys(source.points, triangle_faces(source)))

        two = pv.MultiBlock([box(center=(-3., 0., 0.)), box(center=(3., 0., 0.))]).combine()
        candidate, report = repair_source_mesh(two, max_error=0.01)
        metrics = self.assert_accepted_mesh(candidate, report)
        self.assertEqual(metrics["topology"]["components"]["count"], 2)

    def test_exact_source_geometry_skips_sampling_and_recovery(self):
        with patch("source_material_repair.recover_source_fidelity") as recover, \
             patch("source_material_repair.select_missing_source_facets") as select, \
             patch("source_material_repair.candidate_to_source_bound") as bound:
            candidate, report = repair_source_mesh(box(), max_error=.01)
        self.assert_accepted_mesh(candidate, report)
        self.assertEqual(report["source_fidelity_final"]["status"], "exact_geometry_match")
        self.assertTrue(report["candidate_surface_distance"]["reverse_continuous_certificate"])
        self.assertEqual(report["candidate_surface_distance"]["continuous_one_sided_bound"], 0)
        recover.assert_not_called()
        select.assert_not_called()
        bound.assert_not_called()

    def test_normalized_identity_accounts_for_world_coordinate_roundtrip(self):
        source = box(center=(.17, -.23, .31), size=1.13)
        candidate, report = repair_source_mesh(source, max_error=.01)
        self.assert_accepted_mesh(candidate, report)
        self.assertTrue(report["candidate_surface_distance"]["normalized_geometry_match"])
        source_points = np.asarray(source.points)
        candidate_points = np.asarray(candidate.points)
        left = source_points[np.lexsort((source_points[:, 2], source_points[:, 1], source_points[:, 0]))]
        right = candidate_points[np.lexsort((candidate_points[:, 2], candidate_points[:, 1], candidate_points[:, 0]))]
        actual = float(np.linalg.norm(left - right, axis=1).max())
        self.assertLessEqual(actual, report["candidate_surface_distance"]["continuous_one_sided_bound"])

    def test_interpenetrating_closed_solids_are_boolean_unioned(self):
        source = pv.MultiBlock([
            box(center=(-0.25, 0., 0.)),
            box(center=(0.25, 0., 0.)),
        ]).combine()
        candidate, report = repair_source_mesh(source, max_error=0.01)
        metrics = self.assert_accepted_mesh(candidate, report)
        self.assertEqual(metrics["topology"]["components"]["count"], 1)
        # Each 2×2×2 box has volume 8; their 1.5×2×2 overlap must be counted once.
        self.assertAlmostEqual(metrics["volume"]["signed_sum"], 10.0, places=6)

    def test_manifold_backend_uses_the_same_final_geometry_and_fidelity_gates(self):
        source = pv.MultiBlock([box(center=(-.25, 0., 0.)), box(center=(.25, 0., 0.))]).combine()
        candidate, report = repair_source_mesh(source, max_error=.01, union_backend="manifold", grid_size=32)
        metrics = self.assert_accepted_mesh(candidate, report)
        self.assertEqual(report["construction"]["union_backend"], "manifold")
        self.assertAlmostEqual(metrics["volume"]["signed_sum"], 10., places=6)

    def test_manifold_backend_rejects_failed_geometry_without_global_cleanup(self):
        source = surface(np.asarray([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]),
                         np.asarray([[0, 1, 2]]))
        with patch("source_material_repair._geometry_gate", return_value={"passed": False}), \
             patch("source_material_repair._cleanup") as cleanup, \
             patch("source_material_repair.recover_source_fidelity") as recovery:
            candidate, report = repair_source_mesh(source, max_error=.01, union_backend="manifold", boundary_mode="wall")
        self.assertFalse(report["accepted"])
        self.assertEqual(report["candidate_role"], "unmodified_reference_after_failure")
        np.testing.assert_array_equal(candidate.points, source.points)
        cleanup.assert_not_called()
        recovery.assert_not_called()

    def test_closed_solids_touching_at_one_vertex_are_repaired(self):
        source = pv.MultiBlock([box(size=1.), box(center=(1., 1., 1.), size=1.)]).combine()
        candidate, report = repair_source_mesh(source, max_error=.01)
        self.assert_accepted_mesh(candidate, report)
        self.assertTrue(report["pre_recovery_geometry"]["passed"])
        self.assertTrue(any(action["operator"] == "bounded_contact_neighborhood"
                            for action in report["cleanup"]))
        self.assertLessEqual(report["source_fidelity_final"]["sample_distance_original_units"]["max"], .01)

    def test_invalid_initial_material_never_enters_fidelity_recovery(self):
        invalid = surface(np.asarray([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]),
                          np.asarray([[0, 1, 2]]))
        invalid.cell_data["source_facet_id"] = np.asarray([0])
        with patch("source_material_repair._cleanup", return_value=(invalid, [])), \
             patch("source_material_repair.recover_source_fidelity") as recover:
            _, report = repair_source_mesh(box(), max_error=.01)
        self.assertFalse(report["accepted"])
        self.assertFalse(report["pre_recovery_geometry"]["passed"])
        self.assertIn("before fidelity recovery", report["failure_reason"])
        recover.assert_not_called()

    def test_cleanup_pins_exact_source_vertices_only(self):
        from source_material_repair import _cleanup
        points = np.asarray([[1., 0., 0.], [1. + 2*np.finfo(float).eps, 0., 0.],
                             [-1., 0., 0.], [0., -1., 0.], [0., 0., 1.], [0., 0., -1.]])
        faces = []
        for x in (0, 2):
            for y in (1, 3):
                for z in (4, 5):
                    face = [x, y, z]
                    triangle = points[face]
                    if np.cross(triangle[1]-triangle[0], triangle[2]-triangle[0]) @ triangle.mean(axis=0) < 0:
                        face[1], face[2] = face[2], face[1]
                    faces.append(face)
        candidate = surface(points, np.asarray(faces))
        candidate.cell_data["source_facet_id"] = np.arange(len(faces))
        with patch("material_cleanup.exact_self_intersection_report", return_value={"passed": True}):
            cleaned, actions = _cleanup(candidate, points[[0, 2, 3, 4, 5]], .01)
        self.assertTrue(any(action.get("pinned_source_endpoint") for action in actions))
        self.assertTrue(np.any(np.all(np.asarray(cleaned.points) == points[0], axis=1)))
        metrics = mesh_report(np.asarray(cleaned.points), triangle_faces(cleaned))
        self.assertEqual(metrics["quality"]["degenerate_faces"], 0)
        self.assertTrue(metrics["topology"]["closed_manifold"])
        self.assertTrue(exact_self_intersection_report(np.asarray(cleaned.points), triangle_faces(cleaned))["passed"])

    def test_invalid_recovery_stops_before_another_fidelity_check(self):
        invalid = surface(np.asarray([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]),
                          np.asarray([[0, 1, 2]]))
        invalid.cell_data["source_facet_id"] = np.asarray([0])
        calls = []

        def cleanup(candidate, *_args):
            calls.append(1)
            if len(calls) == 1:
                moved = candidate.copy(deep=True)
                moved.points = np.asarray(moved.points) + [0.0001, 0, 0]
                return moved, []
            return invalid, []

        with patch("source_material_repair._cleanup", side_effect=cleanup), \
             patch("source_material_repair.recover_source_fidelity", side_effect=lambda source, candidate, **kwargs:
                   (candidate, {"status": "union_proposal_created"})) as recover, \
             patch("source_material_repair.select_missing_source_facets") as final_check:
            _, report = repair_source_mesh(box(), max_error=.01)
        self.assertFalse(report["accepted"])
        self.assertIn("after fidelity recovery", report["failure_reason"])
        self.assertFalse(report["recovery_geometry"]["passed"])
        recover.assert_called_once()
        final_check.assert_not_called()

    def test_intersection_gate_reports_first_pair_as_lower_bound(self):
        from source_material_repair import _geometry_gate
        source = pv.MultiBlock([box(center=(-.25, 0, 0)), box(center=(.25, 0, 0))]).combine().extract_surface(algorithm="dataset_surface").triangulate()
        gate = _geometry_gate(source)
        self.assertFalse(gate["passed"])
        self.assertTrue(gate["exact"]["first_only"])
        self.assertTrue(gate["exact"]["intersection_pairs_is_lower_bound"])
        self.assertGreater(gate["exact"]["intersection_pairs"], 0)

    def test_basic_geometry_failure_skips_expensive_gates(self):
        from source_material_repair import _geometry_gate
        source = surface(np.asarray([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]),
                         np.asarray([[0, 1, 2]]))
        with patch("material_cleanup.exact_self_intersection_report") as exact, \
             patch("material_cleanup.orient_material") as orient:
            gate = _geometry_gate(source)
        self.assertFalse(gate["passed"])
        self.assertEqual(gate["exact"]["status"], "not_checked")
        self.assertEqual(gate["material_orientation_status"], "not_checked")
        exact.assert_not_called()
        orient.assert_not_called()

    def test_material_orientation_gate_does_not_depend_on_component_order(self):
        from material_cleanup import _geometry_gate, orient_material
        inner = box(size=1.)
        outer = box(size=2.)

        def ordered_shells(mesh):
            oriented, shells = orient_material(mesh)
            return oriented, list(reversed(shells))

        for is_cavity in (True, False):
            inner_faces = triangle_faces(inner)
            if is_cavity:
                inner_faces = inner_faces[:, ::-1]
            combined = pv.MultiBlock([surface(np.asarray(inner.points), inner_faces), outer]).combine()
            combined = combined.extract_surface(algorithm="dataset_surface").triangulate()
            with patch("material_cleanup.orient_material", side_effect=ordered_shells):
                gate = _geometry_gate(combined)
            self.assertEqual(gate["material_orientation"], is_cavity)
            self.assertEqual(gate["passed"], is_cavity)

    def test_closed_torus_preserves_the_functional_through_hole(self):
        source = pv.ParametricTorus(ringradius=1.0, crosssectionradius=0.25,
                                    u_res=36, v_res=18).triangulate()
        candidate, report = repair_source_mesh(source, max_error=0.005)
        self.assert_accepted_mesh(candidate, report)
        self.assertEqual(trimesh.Trimesh(candidate.points, triangle_faces(candidate),
                                         process=False).euler_number, 0)

    def test_open_triangle_wall_and_explicit_planar_cap_are_geometrically_closed(self):
        plane = surface(np.asarray([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]),
                        np.asarray([[0, 1, 2]]))
        candidate, report = repair_source_mesh(plane, max_error=0.01, boundary_mode="wall")
        self.assert_accepted_mesh(candidate, report)
        self.assertGreater(candidate.n_cells, 1)

        open_box = box()
        source_faces = triangle_faces(open_box)
        source_faces = source_faces[np.asarray(open_box.points)[source_faces, 2].min(axis=1) < 0.99]
        cap_source = surface(np.asarray(open_box.points), source_faces)
        candidate, report = repair_source_mesh(cap_source, max_error=0.01, boundary_mode="cap")
        metrics = self.assert_accepted_mesh(candidate, report)
        self.assertEqual(metrics["topology"]["components"]["count"], 1)
        np.testing.assert_allclose(np.ptp(candidate.points, axis=0), [2., 2., 2.], atol=1e-9)

    def test_per_region_cap_and_functional_opening_preserve_are_both_applied(self):
        source = two_open_boxes()
        _, review = repair_source_mesh(source, max_error=.01, boundary_mode="review")
        packet = review["unresolved_policy_packet"]
        self.assertEqual(len(packet["items"]), 2)
        decisions = []
        for item in packet["items"]:
            decision = dict(item["decision_template"])
            cap = item["centroid"][0] < 0
            decision.update(
                decision="cap" if cap else "preserve",
                semantic_label="missing_cover" if cap else "functional_opening",
                reason="synthetic geometry policy fixture",
            )
            decisions.append(decision)

        candidate, report = repair_source_mesh(
            source, max_error=.01, boundary_mode="wall", policy_decisions=decisions, grid_size=32
        )

        self.assert_accepted_mesh(candidate, report)
        self.assertEqual([row["action"] for row in report["preparation"]["opening_transactions"]],
                         ["cap", "preserve"])
        self.assertTrue(report["functional_openings"]["passed"], report["functional_openings"])
        distances_squared, _, _ = igl.point_mesh_squared_distance(
            np.asarray([[-2., 0., 1.], [2., 0., 1.]]),
            np.asarray(candidate.points), triangle_faces(candidate),
        )
        # The left decision creates a cap; the right functional opening remains clear.
        np.testing.assert_allclose(distances_squared, [0., 1.], atol=1e-10)

    def test_stale_per_region_decisions_do_not_inherit_global_cap_mode(self):
        source = two_open_boxes()
        _, review = repair_source_mesh(source, max_error=.01, boundary_mode="review")
        stale_decisions = []
        for item in review["unresolved_policy_packet"]["items"]:
            decision = dict(item["decision_template"])
            decision.update(decision="cap", semantic_label="missing_cover")
            stale_decisions.append(decision)

        changed_points = np.asarray(source.points).copy()
        changed_points[0, 0] -= .01
        changed_source = surface(changed_points, triangle_faces(source))
        _, report = repair_source_mesh(
            changed_source, max_error=.01, boundary_mode="cap", policy_decisions=stale_decisions,
            grid_size=32,
        )

        self.assertFalse(report["accepted"])
        self.assertEqual(report["failure_reason"],
                         "boundary semantic review or a supported cap is required")
        self.assertTrue(all(row["status"] == "pending_review" for row in report["policy_decisions"]))
        self.assertTrue(all(row["stale_policy_decision"] for row in report["policy_decisions"]))
        self.assertTrue(all("mismatched_run_fingerprint" in row["stale_reasons"]
                            for row in report["policy_decisions"]))

    def test_three_face_non_manifold_edge_is_resolved_without_model_names_or_ids(self):
        points = np.asarray([
            [0., 0., -1.], [0., 0., 1.],
            [1., 0., 0.], [0., 1., 0.], [-1., 0., 0.],
        ])
        faces = np.asarray([[0, 1, 2], [1, 0, 3], [0, 1, 4]], dtype=np.int64)
        candidate, report = repair_source_mesh(surface(points, faces), max_error=0.02,
                                                boundary_mode="wall")
        self.assert_accepted_mesh(candidate, report)

    def test_four_face_non_manifold_edge_is_resolved_geometry_only(self):
        points = np.asarray([
            [0., 0., -1.], [0., 0., 1.],
            [1., 0., 0.], [0., 1., 0.], [-1., 0., 0.], [0., -1., 0.],
        ])
        faces = np.asarray([
            [0, 1, 2], [1, 0, 3], [0, 1, 4], [1, 0, 5],
        ], dtype=np.int64)
        candidate, report = repair_source_mesh(surface(points, faces), max_error=.16,
                                                boundary_mode="wall")
        self.assert_accepted_mesh(candidate, report)

    def test_units_and_large_translation_do_not_change_normalized_result(self):
        source = surface(np.asarray([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]),
                         np.asarray([[0, 1, 2]]))
        outputs = []
        for scale, offset in ((1., np.zeros(3)),
                              (1000., np.asarray([1.0e6, -2.0e6, 3.0e6]))):
            transformed = surface(np.asarray(source.points) * scale + offset,
                                  triangle_faces(source))
            candidate, report = repair_source_mesh(transformed, max_error=0.01 * scale,
                                                    boundary_mode="wall")
            self.assert_accepted_mesh(candidate, report)
            fidelity = report["source_fidelity_final"]
            self.assertEqual(fidelity["coordinate_frame"], "normalized")
            self.assertAlmostEqual(fidelity["sample_distance_original_units"]["max"],
                                   fidelity["sample_distance_input_units"]["max"] * report["normalization"]["scale"])
            span = np.ptp(candidate.points, axis=0)
            outputs.append(span / scale)
        np.testing.assert_allclose(outputs[0], outputs[1], rtol=0., atol=1e-8)

    def test_native_corner_normal_seams_and_magnitudes_are_preserved(self):
        source = surface(
            np.asarray([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [1., 1., 0.]]),
            np.asarray([[0, 1, 2], [1, 3, 2]], dtype=np.int64),
        )
        source.cell_data["corner_normals"] = np.asarray(
            [[0., 0., 2.] * 3, [0., 5., 0.] * 3], dtype=np.float32
        )
        candidate, report = repair_source_mesh(source, max_error=0.01,
                                                boundary_mode="wall")
        self.assert_accepted_mesh(candidate, report)
        self.assertIn("corner_normals", candidate.cell_data)
        self.assertEqual(candidate.cell_data["corner_normals"].shape,
                         (candidate.n_cells, 9))
        rows = np.asarray(candidate.cell_data["source_facet_id"])
        normals = np.asarray(candidate.cell_data["corner_normals"], dtype=np.float64)
        candidate_faces = triangle_faces(candidate)
        candidate_points = np.asarray(candidate.points, dtype=np.float64)
        source_points = np.asarray(source.points, dtype=np.float64)
        source_faces = triangle_faces(source)
        source_normal_vectors = {
            0: np.asarray([0., 0., 2.]),
            1: np.asarray([0., 5., 0.]),
        }
        for source_row, expected in source_normal_vectors.items():
            source_triangle = source_points[source_faces[source_row]]
            edge_a, edge_b = source_triangle[1:] - source_triangle[0]
            normal = np.cross(edge_a, edge_b)
            triangle_points = candidate_points[candidate_faces[rows == source_row]]
            plane_error = np.abs(np.einsum("fci,i->fc", triangle_points-source_triangle[0], normal)) / np.linalg.norm(normal)
            mapped = normals[rows == source_row].reshape(-1, 3, 3)
            # Back skins and edge walls inherit a source face ID as birth
            # lineage; only the actual source-plane faces must carry its field.
            mapped = mapped[np.all(plane_error <= 1e-10, axis=1)]
            self.assertGreater(len(mapped), 0, f"source face row {source_row} lost lineage")
            np.testing.assert_allclose(
                mapped, np.broadcast_to(expected, mapped.shape), rtol=1e-6, atol=1e-6
            )

    def test_invalid_error_or_degenerate_input_never_returns_success(self):
        source = box()
        for error in (0.0, -1.0, np.inf, np.nan):
            with self.subTest(max_error=error), self.assertRaises(ValueError):
                repair_source_mesh(source, max_error=error)
        degenerate = surface(np.asarray([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]]),
                             np.asarray([[0, 1, 2]], dtype=np.int64))
        with self.assertRaises(ValueError):
            repair_source_mesh(degenerate, max_error=1e-3)

    def test_invalid_candidate_distance_fails_closed(self):
        from source_material_repair import _candidate_distance
        with patch("source_material_repair.igl.point_mesh_squared_distance",
                   return_value=(np.asarray([np.nan]), np.asarray([0]), np.zeros((1, 3)))):
            with self.assertRaisesRegex(ValueError, "invalid values"):
                _candidate_distance(box(), box())

    def test_invalid_source_lineage_fails_closed(self):
        from source_distance_bound import candidate_to_source_bound
        candidate = box()
        candidate.cell_data["source_facet_id"] = np.full(candidate.n_cells, -1, dtype=np.int64)
        with patch("source_distance_bound.igl.point_mesh_squared_distance",
                   return_value=(np.zeros(candidate.n_cells), np.full(candidate.n_cells, -1),
                                 np.zeros((candidate.n_cells, 3)))):
            with self.assertRaisesRegex(ValueError, "lookup is invalid"):
                candidate_to_source_bound(box(), candidate, max_error=0.01)

    def test_skinny_birth_triangle_rechecks_a_false_fast_overestimate_exactly(self):
        from source_distance_bound import candidate_to_source_bound
        source = surface(
            np.asarray([[0., 0., 0.], [1., 0., 0.], [1., 1.0e-12, 0.]]),
            np.asarray([[0, 1, 2]], dtype=np.int64),
        )
        candidate = surface(np.asarray(source.points), triangle_faces(source))
        candidate.cell_data["source_facet_id"] = np.asarray([0], dtype=np.int64)
        bound = candidate_to_source_bound(source, candidate, max_error=5.0e-13)
        self.assertLessEqual(bound, 5.0e-13)

    def test_external_point_on_skinny_triangle_is_never_underbounded(self):
        from source_distance_bound import (
            candidate_to_source_bound,
            _point_triangle_distance2,
            _sqrt_upper,
        )
        source_points = np.asarray([[0., 0., 0.], [1., 0., 0.], [1., 1.0e-12, 0.]])
        source = surface(source_points, np.asarray([[0, 1, 2]], dtype=np.int64))
        candidate = surface(
            np.asarray([source_points[0], source_points[1], [1., 1.0e-12, 1.0e-6]]),
            np.asarray([[0, 1, 2]], dtype=np.int64),
        )
        candidate.cell_data["source_facet_id"] = np.asarray([0], dtype=np.int64)
        exact = _sqrt_upper(
            _point_triangle_distance2(candidate.points[2], source_points)
        )
        bound = candidate_to_source_bound(source, candidate, max_error=1.0e-7)
        self.assertGreaterEqual(bound, exact)
        self.assertGreater(bound, 1.0e-7)

    def test_candidate_bound_strictly_rejects_a_distance_over_budget(self):
        from source_distance_bound import candidate_to_source_bound
        points = np.asarray([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
        source = surface(points, np.asarray([[0, 1, 2]], dtype=np.int64))
        candidate = surface(points + np.asarray([0., 0., .010000000001]),
                            np.asarray([[0, 1, 2]], dtype=np.int64))
        candidate.cell_data["source_facet_id"] = np.asarray([0], dtype=np.int64)
        self.assertGreater(candidate_to_source_bound(source, candidate, max_error=.01), .01)

    def test_exact_triangle_distance_handles_interior_edge_and_vertex_regions(self):
        from fractions import Fraction
        from source_distance_bound import _point_triangle_distance2, _sqrt_upper
        triangle = np.asarray([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
        cases = [
            ([.25, .25, .1], .1),       # perpendicular projection inside
            ([.5, -.2, .1], np.sqrt(.05)),  # nearest point lies on edge AB
            ([1.1, -.1, .1], np.sqrt(.03)),  # nearest point is vertex B
        ]
        for point, expected in cases:
            with self.subTest(point=point):
                squared = _point_triangle_distance2(point, triangle)
                upper = _sqrt_upper(squared)
                self.assertGreaterEqual(Fraction.from_float(upper) ** 2, squared)
                self.assertAlmostEqual(upper, expected, places=14)

    def test_exact_distance_sqrt_rejects_negative_and_preserves_zero(self):
        from fractions import Fraction
        from source_distance_bound import _sqrt_upper
        self.assertEqual(_sqrt_upper(Fraction(0)), 0.)
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            _sqrt_upper(Fraction(-1))

    def test_well_conditioned_triangle_uses_the_verified_fast_path(self):
        from source_distance_bound import candidate_to_source_bound
        source = surface(
            np.asarray([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]),
            np.asarray([[0, 1, 2]], dtype=np.int64),
        )
        candidate = surface(
            np.asarray([[0., 0., .001], [1., 0., .001], [0., 1., .001]]),
            np.asarray([[0, 1, 2]], dtype=np.int64),
        )
        candidate.cell_data["source_facet_id"] = np.asarray([0], dtype=np.int64)
        with patch("source_distance_bound._point_triangle_distance2") as exact_distance:
            bound = candidate_to_source_bound(source, candidate, max_error=.01)
        self.assertLessEqual(bound, .01)
        exact_distance.assert_not_called()

    def test_out_of_range_birth_face_is_rejected(self):
        from source_distance_bound import candidate_to_source_bound
        candidate = box()
        candidate.cell_data["source_facet_id"] = np.full(candidate.n_cells, 999, dtype=np.int64)
        with self.assertRaisesRegex(ValueError, "lineage is invalid"):
            candidate_to_source_bound(box(), candidate, max_error=.01)

    def test_unmapped_faces_do_not_choose_degenerate_source_as_support(self):
        from source_distance_bound import candidate_to_source_bound
        points = np.asarray([[0., 0., 0.], [1., 0., 0.], [.5, 0., 0.],
                             [0., 0., .1], [1., 0., .1], [0., 1., .1]])
        source = surface(points, np.asarray([[0, 1, 2], [3, 4, 5]]))
        candidate = surface(np.asarray([[.1, .01, 0.], [.2, .01, 0.], [.1, .02, 0.]]),
                            np.asarray([[0, 1, 2]]))
        candidate.cell_data["source_facet_id"] = np.asarray([-1], dtype=np.int64)
        bound = candidate_to_source_bound(source, candidate, max_error=.2)
        self.assertGreaterEqual(bound, .1)
        self.assertLess(bound, .2)
        candidate.cell_data["source_facet_id"] = np.asarray([0], dtype=np.int64)
        with self.assertRaisesRegex(ValueError, "exactly degenerate"):
            candidate_to_source_bound(source, candidate, max_error=.2)

    def test_output_frame_normalized_geometry_must_pass(self):
        from mesh_io import same_oriented_geometry
        with patch("source_material_repair.same_oriented_geometry",
                   side_effect=[True, True, False]):
            _, report = repair_source_mesh(box(), max_error=0.01)
        self.assertFalse(report["accepted"])
        self.assertFalse(report["output_frame_normalized_geometry_preserved"])
        self.assertIn("output-frame", report["failure_reason"])

        source = box(center=(1.0e6, -2.0e6, 3.0e6))
        candidate, report = repair_source_mesh(source, max_error=0.01)
        self.assert_accepted_mesh(candidate, report)
        cleaned = candidate.clean(lines_to_points=False, polys_to_lines=False,
                                  strips_to_polys=False)
        self.assertTrue(same_oriented_geometry(candidate, cleaned))
        self.assertTrue(report["output_frame_normalized_read"]["passed"])

    def test_optional_reverse_bound_covers_the_whole_reference(self):
        candidate, report = repair_source_mesh(box(), max_error=0.01,
                                               certify_reverse_fidelity=True)
        self.assert_accepted_mesh(candidate, report)
        self.assertEqual(report["reverse_surface_bound"]["status"], "within_budget")
        self.assertTrue(report["candidate_surface_distance"]["reverse_continuous_certificate"])
        self.assertIn("internal material surfaces", report["reverse_surface_bound"]["scope"])

    def test_report_identifies_actual_reference_and_pipeline_input(self):
        from source_material_repair import repair_source_file
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original, reference = root / "original.vtp", root / "reference.vtp"
            box().save(original)
            box(size=1.0).save(reference)
            report = repair_source_file(reference, root / "out", max_error=.01,
                                        original_input=original)
            self.assertTrue(report["accepted"], report)
            self.assertEqual(report["fidelity_reference"]["path"], str(reference.resolve()))
            self.assertEqual(report["original_pipeline_input"], str(original.resolve()))
            self.assertEqual(report["outputs"]["mesh_previews"][0]["label"],
                             "Fidelity reference input")

    def test_main_cli_preserves_per_boundary_ai_review_and_emits_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            open_box = box()
            open_faces = triangle_faces(open_box)
            open_faces = open_faces[np.asarray(open_box.points)[open_faces, 2].min(axis=1) < .99]
            source_path = root / "open_mesh.vtp"
            surface(np.asarray(open_box.points), open_faces).save(source_path)
            output = root / "output"
            command = [sys.executable, str(SCRIPTS_DIR / "two_stage_watertight_remesh.py"),
                       str(source_path), "--purpose", "watertight", "--surface-strategy",
                       "source-material", "--source-max-error", ".01", "--output-dir", str(output)]
            completed = subprocess.run(command, cwd=SCRIPTS_DIR.parents[3], capture_output=True,
                                       text=True, timeout=30)
            self.assertEqual(completed.returncode, 2, completed.stderr)
            report = json.loads((output / "source_material_report.json").read_text())
            self.assertFalse(report["accepted"])
            self.assertEqual(report["target"]["name"], "watertight-material")
            self.assertEqual(report["unresolved_policy_packet"]["item_count_total"], 1)
            self.assertIsNone(report["outputs"]["accepted_mesh_vtp"])
            self.assertTrue((output / "source_material_report.html").is_file())


if __name__ == "__main__":
    unittest.main()
