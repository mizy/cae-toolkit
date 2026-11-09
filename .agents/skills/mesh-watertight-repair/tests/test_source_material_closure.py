from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import trimesh
import pyvista as pv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from source_material_closure import _correct_offset_directions, _sheet_shell, build_source_material
from source_material_repair import repair_source_mesh
from mesh_io import triangle_faces
from source_projected_validation import exact_self_intersection_report


def join(meshes: list[trimesh.Trimesh]) -> tuple[np.ndarray, np.ndarray]:
    points, faces, offset = [], [], 0
    for mesh in meshes:
        points.append(np.asarray(mesh.vertices))
        faces.append(np.asarray(mesh.faces) + offset)
        offset += len(mesh.vertices)
    return np.vstack(points), np.vstack(faces)


def report_mesh(points: np.ndarray, faces: np.ndarray) -> trimesh.Trimesh:
    return trimesh.Trimesh(vertices=points, faces=faces, process=False)


def high_valence_edge(valence: int) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(
        [[0., 0., -1.], [0., 0., 1.]]
        + [[np.cos(t), np.sin(t), 0.] for t in np.arange(valence) * 2*np.pi/valence]
    )
    faces = np.asarray(
        [[0, 1, 2+i] if i % 2 == 0 else [1, 0, 2+i] for i in range(valence)],
        dtype=np.int64,
    )
    return points, faces


class SourceMaterialClosureTest(unittest.TestCase):
    def test_manifold_backend_unions_positive_solids_and_preserves_face_ids(self):
        first = trimesh.creation.box(extents=[2., 2., 2.])
        second = trimesh.creation.box(extents=[2., 2., 2.])
        second.apply_translation([.5, 0., 0.])
        points, faces = join([first, second])
        result_points, result_faces, rows, details = build_source_material(
            points, faces, max_error=.01, union_backend="manifold"
        )
        result = report_mesh(result_points, result_faces)
        self.assertEqual(details["union_backend"], "manifold")
        self.assertTrue(result.is_watertight)
        self.assertAlmostEqual(result.volume, 10., places=8)
        self.assertTrue(np.all((rows >= 0) & (rows < len(faces))))
        self.assertTrue(exact_self_intersection_report(result_points, result_faces)["passed"])

    def test_manifold_backend_rejects_negative_cavity_shell(self):
        outer = trimesh.creation.box(extents=[2., 2., 2.])
        inner = trimesh.creation.box(extents=[1., 1., 1.])
        inner.faces = inner.faces[:, ::-1]
        points, faces = join([outer, inner])
        with self.assertRaisesRegex(ValueError, "positive material shells"):
            build_source_material(points, faces, max_error=.01, union_backend="manifold")

    def test_folded_sheet_offsets_stay_inward_and_keep_source_geometry(self):
        points = np.asarray([
            [0., 0., 0.], [.5627152529143113, 0., -3.5883168786426074],
            [.17393199558753625, .30125905341945847, 1.3452477264680462],
            [-.27591690816165676, .4779021036033056, -2.509142794661634],
            [-.5839053284656452, 0., -.8842562819672661],
            [-.8692838848329144, -1.5056438547314586, 3.0729156677971377],
            [.8184661482628339, -1.4176249530664295, 3.6077004448366736],
        ])
        faces = np.asarray([[0, 1, 2], [0, 2, 3], [0, 3, 4],
                            [0, 4, 5], [0, 5, 6], [0, 6, 1]])
        points_out, faces_out, lineage, corrected, splits = _sheet_shell(points, faces, np.arange(6), .03)
        self.assertGreater(corrected, 0)
        self.assertEqual(splits, 0)
        np.testing.assert_array_equal(points_out[:len(points)], points)
        self.assertLessEqual(np.linalg.norm(points_out[len(points):] - points, axis=1).max(), .03 + 1e-12)
        self.assertEqual(len(lineage), len(faces_out))
        self.assertTrue(exact_self_intersection_report(points_out, faces_out)["passed"])

    def test_incompatible_incident_normals_require_feature_split(self):
        faces = np.asarray([[0, 1, 2], [0, 3, 4]])
        normals = np.asarray([[1., 0., 0.], [-1., 0., 0.]])
        directions = np.asarray([[1., 0., 0.], [1., 0., 0.], [1., 0., 0.],
                                 [-1., 0., 0.], [-1., 0., 0.]])
        _, _, unresolved = _correct_offset_directions(faces, normals, directions)
        self.assertEqual(unresolved, [0])

    def test_feature_split_preserves_every_source_triangle_before_union(self):
        points = np.asarray([
            [0., 0., 0.], [-.3293531604958654, -2.1737409405112955, -1.2514578970135206],
            [.7599285532796657, 1.9982579627400703, .42292706648690026],
            [-1.7034457349739551, .3226950623669582, .0840747464483445],
            [-.7249716258946706, -.43279454491534336, 2.6474285751017184],
            [1.7806339724253888, 1.9474317020500351, -.2785484427804372],
            [1.2594825540191819, -.009442091740037933, .899090444588229],
        ])
        faces = np.asarray([[0, i + 1, (i + 1) % 6 + 1] for i in range(6)])
        result_points, result_faces, rows, _, splits = _sheet_shell(points, faces, np.arange(6), .002)
        self.assertGreater(splits, 0)
        for index in range(len(faces)):
            before = sorted(map(tuple, points[faces[index]]))
            after = sorted(map(tuple, result_points[result_faces[index]]))
            self.assertEqual(before, after)
        np.testing.assert_array_equal(rows[:len(faces)], np.arange(len(faces)))
        # The duplicated seam is only a construction intermediate and needs a union/audit.
        self.assertFalse(exact_self_intersection_report(result_points, result_faces, first_only=True)["passed"])

    def test_repair_preserves_corner_normal_lineage_after_source_rows_are_removed(self):
        box = trimesh.creation.box(extents=[2., 2., 2.])
        native_faces = np.asarray(box.faces)
        # Put an exact duplicate and an indexed-degenerate face before later source rows.
        raw_faces = np.vstack((native_faces[:4], native_faces[[0]], [0, 0, 0], native_faces[4:]))
        source_rows = [0, 1, 2, 3, *range(6, len(raw_faces))]
        source = pv.PolyData(box.vertices, np.column_stack((np.full(len(raw_faces), 3), raw_faces)).ravel())
        normal_rows = np.asarray([[[0., 0., 1.]] * 3 for _ in raw_faces], dtype=np.float32)
        normal_rows[0] = np.asarray([[0., 1., 0.]] * 3, dtype=np.float32)
        normal_rows[4] = np.asarray([[1., 0., 0.]] * 3, dtype=np.float32)
        source.cell_data["corner_normals"] = normal_rows.reshape(-1, 9)
        source.point_data["Normals"] = np.tile([1., 0., 0.], (source.n_points, 1))

        candidate, report = repair_source_mesh(
            source, max_error=.01, boundary_mode="wall"
        )

        self.assertTrue(report["accepted"], report.get("failure_reason"))
        self.assertEqual(len(triangle_faces(candidate)), len(native_faces))
        np.testing.assert_array_equal(candidate.cell_data["source_facet_id"], source_rows)
        np.testing.assert_array_equal(
            np.asarray(candidate.cell_data["corner_normals"]).reshape(-1, 3, 3),
            normal_rows[source_rows],
        )
        self.assertFalse(np.array_equal(candidate.cell_data["corner_normals"][0], source.point_data["Normals"][:3].reshape(-1)))

    def test_closed_component_is_preserved_and_inward_root_is_reoriented(self):
        source = trimesh.creation.box(extents=[2., 1., 0.5])
        points, faces = np.asarray(source.vertices), np.asarray(source.faces)
        for source_faces in (faces, faces[:, ::-1]):
            out_points, out_faces, source_ids, details = build_source_material(
                points, source_faces, max_error=1e-3
            )
            np.testing.assert_array_equal(out_points, points)
            output = report_mesh(out_points, out_faces)
            self.assertTrue(output.is_watertight)
            self.assertTrue(output.is_winding_consistent)
            self.assertGreater(output.volume, 0.)
            self.assertEqual(details["method"], "closed source components retained without Boolean")
            self.assertEqual(sorted(source_ids.tolist()), list(range(len(source_faces))))

    def test_closed_torus_keeps_its_functional_through_hole(self):
        source = trimesh.creation.torus(1., .25, major_sections=20, minor_sections=12)
        points, faces = np.asarray(source.vertices), np.asarray(source.faces)
        out_points, out_faces, source_ids, details = build_source_material(
            points, faces, max_error=1e-4
        )
        output = report_mesh(out_points, out_faces)
        self.assertTrue(output.is_watertight)
        self.assertEqual(output.euler_number, 0)
        self.assertTrue(np.array_equal(out_points, points))
        self.assertEqual(details["boolean_applied"], False)
        self.assertEqual(len(np.unique(source_ids)), len(faces))

    def test_nested_closed_hollow_source_keeps_cavity_orientation_and_volume(self):
        outer = trimesh.creation.box(extents=[2., 2., 2.])
        inner = trimesh.creation.box(extents=[1., 1., 1.])
        inner.faces = inner.faces[:, ::-1]
        points, faces = join([outer, inner])
        out_points, out_faces, source_ids, details = build_source_material(
            points, faces, max_error=1e-3
        )
        output = report_mesh(out_points, out_faces)
        self.assertTrue(output.is_watertight)
        self.assertAlmostEqual(output.volume, 7., places=8)
        self.assertEqual(details["nested_closed_component_relations"][0]["input_orientation_relation"], "cavity_boundary")
        self.assertEqual(set(source_ids.tolist()), set(range(len(faces))))

        # An entirely inward-oriented nested tree is reversed as one unit;
        # the inner boundary stays opposite to its enclosing solid.
        outer.faces = outer.faces[:, ::-1]
        inner.faces = inner.faces[:, ::-1]
        points, faces = join([outer, inner])
        out_points, out_faces, _, details = build_source_material(
            points, faces, max_error=1e-3
        )
        self.assertAlmostEqual(report_mesh(out_points, out_faces).volume, 7., places=8)
        self.assertTrue(all(row["whole_tree_orientation_reversed"] for row in details["components"]))

    def test_nested_same_orientation_shell_is_unioned_into_material(self):
        outer = trimesh.creation.box(extents=[2., 2., 2.])
        inner = trimesh.creation.box(extents=[.5, .5, .5])
        points, faces = join([outer, inner])
        out_points, out_faces, source_ids, details = build_source_material(
            points, faces, max_error=1e-3
        )
        self.assertTrue(details["boolean_applied"])
        self.assertEqual(details["nested_closed_component_relations"][0]["input_orientation_relation"], "same_material_orientation")
        self.assertAlmostEqual(report_mesh(out_points, out_faces).volume, 8., places=8)
        self.assertEqual(source_ids.min(), 0)
        self.assertLess(source_ids.max(), len(faces))

    def test_overlapping_closed_solids_are_boolean_unioned(self):
        boxes = [
            trimesh.creation.box(extents=[2., 2., 2.]),
            trimesh.creation.box(extents=[2., 2., 2.], transform=trimesh.transformations.translation_matrix([1., 0., 0.])),
        ]
        points, faces = join(boxes)
        out_points, out_faces, source_ids, details = build_source_material(
            points, faces, max_error=1e-3
        )
        output = report_mesh(out_points, out_faces)
        self.assertTrue(details["boolean_applied"])
        self.assertTrue(output.is_watertight)
        self.assertAlmostEqual(output.volume, 12., places=8)
        self.assertTrue(np.all((source_ids >= 0) & (source_ids < len(faces))))

    def test_opposite_oriented_intersecting_closed_shells_fail_semantics_closed(self):
        outer = trimesh.creation.box(extents=[2., 2., 2.])
        cavity = trimesh.creation.box(
            extents=[1., 1., 1.],
            transform=trimesh.transformations.translation_matrix([.51, 0., 0.]),
        )
        cavity.faces = cavity.faces[:, ::-1]
        points, faces = join([outer, cavity])

        with self.assertRaisesRegex(ValueError, "opposite-oriented closed components.*intersect"):
            build_source_material(points, faces, max_error=.02)

    def test_disjoint_reversed_root_remains_an_independent_solid(self):
        outer = trimesh.creation.box(extents=[2., 2., 2.])
        reversed_part = trimesh.creation.box(
            extents=[1., 1., 1.],
            transform=trimesh.transformations.translation_matrix([4., 0., 0.]),
        )
        reversed_part.faces = reversed_part.faces[:, ::-1]
        points, faces = join([outer, reversed_part])
        out_points, out_faces, _, details = build_source_material(
            points, faces, max_error=.02
        )

        self.assertFalse(details["boolean_applied"])
        self.assertAlmostEqual(report_mesh(out_points, out_faces).volume, 9., places=8)

    def test_reversed_closed_shell_inside_torus_hole_is_not_bbox_intersection(self):
        torus = trimesh.creation.torus(1., .25, major_sections=20, minor_sections=12)
        box = trimesh.creation.box(extents=[.1, .1, .1])
        box.faces = box.faces[:, ::-1]
        points, faces = join([torus, box])
        out_points, out_faces, _, details = build_source_material(
            points, faces, max_error=.001
        )

        self.assertEqual(details["nested_closed_component_relations"], [])
        self.assertEqual(len(report_mesh(out_points, out_faces).split(only_watertight=False)), 2)
        self.assertAlmostEqual(report_mesh(out_points, out_faces).volume,
                               torus.volume + abs(box.volume), places=7)

    def test_face_touching_closed_solids_are_not_returned_as_unmerged(self):
        boxes = [
            trimesh.creation.box(extents=[2., 2., 2.]),
            trimesh.creation.box(extents=[2., 2., 2.], transform=trimesh.transformations.translation_matrix([2., 0., 0.])),
        ]
        points, faces = join(boxes)
        out_points, out_faces, _, details = build_source_material(
            points, faces, max_error=1e-3
        )
        output = report_mesh(out_points, out_faces)
        self.assertTrue(details["boolean_applied"])
        self.assertTrue(output.is_watertight)
        self.assertEqual(len(output.split(only_watertight=False)), 1)
        self.assertAlmostEqual(output.volume, 16., places=8)

    def test_duplicate_closed_surfaces_are_union_deduplicated(self):
        source = trimesh.creation.box(extents=[2., 2., 2.])
        points = np.vstack((source.vertices, source.vertices))
        faces = np.vstack((source.faces, source.faces + len(source.vertices)))
        out_points, out_faces, source_ids, details = build_source_material(
            points, faces, max_error=1e-3
        )
        self.assertTrue(details["boolean_applied"])
        self.assertAlmostEqual(report_mesh(out_points, out_faces).volume, 8., places=8)
        self.assertEqual(len(out_faces), len(source.faces))
        self.assertTrue(np.all((source_ids >= 0) & (source_ids < len(faces))))

    def test_box_inside_torus_hole_is_not_misclassified_as_cavity(self):
        torus = trimesh.creation.torus(1., .25, major_sections=20, minor_sections=12)
        cube = trimesh.creation.box(extents=[.1, .1, .1])
        points, faces = join([torus, cube])
        out_points, out_faces, _, details = build_source_material(
            points, faces, max_error=1e-3
        )
        output = report_mesh(out_points, out_faces)
        self.assertTrue(details["boolean_applied"])
        self.assertEqual(details["nested_closed_component_relations"], [])
        self.assertEqual(len(output.split(only_watertight=False)), 2)
        self.assertAlmostEqual(output.volume, report_mesh(*join([torus, cube])).volume, places=7)

    def test_reversed_box_crossing_torus_fails_as_ambiguous_material(self):
        torus = trimesh.creation.torus(1., .25, major_sections=20, minor_sections=12)
        box = trimesh.creation.box(
            extents=[1.5, .2, .2],
            transform=trimesh.transformations.translation_matrix([.35, 0., 0.]),
        )
        # Start the box at a vertex inside the torus tube. Its remaining volume
        # crosses the torus boundary into the central hole, while its bbox is
        # fully contained by the torus bbox. This defeats point-only nesting.
        first_vertex = int(
            np.argmin(np.linalg.norm(box.vertices - [1.1, -.1, -.1], axis=1))
        )
        order = np.r_[first_vertex, np.delete(np.arange(len(box.vertices)), first_vertex)]
        inverse = np.empty_like(order)
        inverse[order] = np.arange(len(order))
        box.vertices = box.vertices[order]
        box.faces = inverse[box.faces][:, ::-1]
        self.assertTrue(np.all(box.bounds[0] >= torus.bounds[0]))
        self.assertTrue(np.all(box.bounds[1] <= torus.bounds[1]))
        self.assertAlmostEqual(box.vertices[0, 0], 1.1)

        points, faces = join([torus, box])
        with self.assertRaisesRegex(ValueError, "opposite-oriented closed components.*intersect"):
            build_source_material(points, faces, max_error=1e-3)

    def test_same_oriented_box_crossing_torus_is_unioned(self):
        torus = trimesh.creation.torus(1., .25, major_sections=20, minor_sections=12)
        box = trimesh.creation.box(
            extents=[1.5, .2, .2],
            transform=trimesh.transformations.translation_matrix([.35, 0., 0.]),
        )
        points, faces = join([torus, box])
        out_points, out_faces, _, details = build_source_material(
            points, faces, max_error=1e-3
        )
        output = report_mesh(out_points, out_faces)
        self.assertTrue(details["boolean_applied"])
        self.assertEqual(details["nested_closed_component_relations"], [])
        self.assertTrue(output.is_watertight)
        self.assertEqual(len(output.split(only_watertight=False)), 1)
        self.assertGreater(output.volume, torus.volume)

    def test_unit_scaling_preserves_open_sheet_topology_and_lineage(self):
        base = np.asarray([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
        faces = np.asarray([[0, 1, 2]])
        for scale in (1e-3, 1., 1e3):
            points, error = base * scale, 1e-3 * scale
            out_points, out_faces, source_ids, details = build_source_material(
                points, faces, max_error=error
            )
            output = report_mesh(out_points, out_faces)
            self.assertTrue(output.is_watertight)
            self.assertTrue(output.is_winding_consistent)
            self.assertEqual(set(source_ids.tolist()), {0})
            self.assertEqual(details["max_error"], error)
            expected_volume = 0.5 * scale**2 * error
            self.assertAlmostEqual(output.volume / expected_volume, 1., places=8)

    def test_mixed_open_and_closed_parts_preserve_input_face_lineage(self):
        box = trimesh.creation.box(extents=[1., 1., 1.])
        triangle = np.asarray([[2., 0., 0.], [3., 0., 0.], [2., 1., 0.]])
        points, faces = join([box, trimesh.Trimesh(vertices=triangle, faces=[[0,1,2]], process=False)])
        out_points, out_faces, source_ids, details = build_source_material(
            points, faces, max_error=1e-3
        )
        self.assertEqual(len(source_ids), len(out_faces))
        self.assertTrue(np.all((source_ids >= 0) & (source_ids < len(faces))))
        self.assertTrue(any(row["mode"] == "preserved_closed_component" for row in details["components"]))
        self.assertTrue(any(row["mode"] == "inward_offset_sheet_shell" for row in details["components"]))
        self.assertTrue(report_mesh(out_points, out_faces).is_watertight)

    def test_three_and_four_face_nonmanifold_edges_remain_diagnostic_until_gate(self):
        for valence in (3, 4):
            with self.subTest(valence=valence):
                points, faces = high_valence_edge(valence)
                out_points, out_faces, source_ids, details = build_source_material(
                    points, faces, max_error=1e-3
                )
                self.assertGreater(details["fan_split_points"], 0)
                self.assertEqual(set(source_ids.tolist()), set(range(valence)))
                self.assertEqual(len(source_ids), len(out_faces))
                topology = __import__("mesh_metrics").mesh_report(out_points, out_faces)["topology"]
                intersections = exact_self_intersection_report(out_points, out_faces)
                self.assertTrue(
                    topology["non_manifold_edges"]
                    or topology["non_manifold_vertices"]
                    or topology["boundary_edges"]
                    or intersections["intersection_pairs"]
                )
                self.assertEqual(details["acceptance"], "not performed by this geometry builder")

    def test_invalid_coordinates_indices_and_error_fail_closed(self):
        points, faces = np.asarray([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]), np.asarray([[0,1,2]])
        for max_error in (0., -1., np.inf, np.nan):
            with self.subTest(max_error=max_error), self.assertRaises(ValueError):
                build_source_material(points, faces, max_error=max_error)
        with self.assertRaises(ValueError):
            build_source_material(points, faces.astype(float), max_error=.01)
        with self.assertRaises(ValueError):
            build_source_material(points, np.asarray([[0,1,3]]), max_error=.01)
        invalid = points.copy(); invalid[0, 0] = np.nan
        with self.assertRaises(ValueError):
            build_source_material(invalid, faces, max_error=.01)
        distant = points + np.asarray([1e12, 1e12, 1e12])
        with self.assertRaisesRegex(ValueError, "too small to move some inner-shell vertices"):
            build_source_material(distant, faces, max_error=1e-6)


if __name__ == "__main__":
    unittest.main()
