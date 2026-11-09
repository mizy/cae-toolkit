from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from material_cleanup import _surface, _geometry_gate
from mesh_io import triangle_faces
from mesh_topology_repair import round_degenerate_vertices
from source_material_closure import _manifold_union_blocks
from source_material_local import repair_material_component
from source_material_repair import repair_source_mesh


def overlapping_boxes():
    a = trimesh.creation.box()
    b = trimesh.creation.box()
    b.apply_translation([.2, .1, .15])
    points = np.vstack((a.vertices, b.vertices))
    faces = np.vstack((a.faces, b.faces + len(a.vertices)))
    return points, faces, np.arange(len(faces))


class LocalMaterialTest(unittest.TestCase):
    def test_healthy_component_is_returned_without_mutation(self):
        mesh = trimesh.creation.icosphere(subdivisions=1)
        block = (np.asarray(mesh.vertices), np.asarray(mesh.faces), np.arange(len(mesh.faces)))
        result, report = repair_material_component(block, _surface(*block[:2]), max_error=.001)
        self.assertIs(result, block)
        self.assertTrue(report["passed"])
        self.assertFalse(report["changed"])
        self.assertEqual(report["attempts"], [])

    def test_intersecting_component_is_locally_repaired_with_lineage(self):
        block = overlapping_boxes()
        result, report = repair_material_component(block, _surface(*block[:2]), max_error=.5)
        self.assertTrue(report["passed"], report)
        self.assertTrue(report["changed"])
        self.assertTrue(all(values.flags.owndata for values in result))
        self.assertTrue(_geometry_gate(_surface(*result))["passed"])
        self.assertTrue(np.all((result[2] >= -1) & (result[2] < len(block[1]))))

    def test_fidelity_failure_rolls_back_whole_component(self):
        block = overlapping_boxes()
        before = [a.copy() for a in block]
        result, report = repair_material_component(block, _surface(*block[:2]), max_error=1e-5)
        self.assertFalse(report["passed"])
        self.assertIs(result, block)
        for actual, expected in zip(block, before):
            np.testing.assert_array_equal(actual, expected)
        self.assertGreater(report["attempts"][0]["source_sample_max"], 1e-5)

    def test_rounding_only_moves_degenerate_face_vertices_within_budget(self):
        points = np.array([[0., 0., 0.], [1., 0., 0.], [1., 1e-17, 0.],
                           [0., 1., 0.], [9., 9., 9.]])
        faces = np.array([[0, 1, 2], [0, 2, 3]])
        result, output_faces, kept, report = round_degenerate_vertices(
            points, faces, max_displacement=1e-8)
        self.assertTrue(report["applied"])
        self.assertLessEqual(report["max_displacement"], 1e-8)
        self.assertEqual(len(output_faces), 1)
        np.testing.assert_array_equal(kept, [1])
        self.assertTrue(any(np.array_equal(p, points[4]) for p in result))
        np.testing.assert_array_equal(points[2], [1., 1e-17, 0.])

    def test_generated_face_lineage_survives_fast_assembly(self):
        mesh = trimesh.creation.box()
        rows = np.arange(len(mesh.faces)); rows[0] = -1
        _, _, output_rows, _ = _manifold_union_blocks([(mesh.vertices, mesh.faces, rows)])
        self.assertEqual(set(output_rows), set(rows))

    def test_local_backend_keeps_closed_functional_hole(self):
        mesh = trimesh.creation.torus(1., .25, major_sections=16, minor_sections=8)
        source = _surface(mesh.vertices, mesh.faces)
        candidate, report = repair_source_mesh(source, max_error=.001, union_backend="local", grid_size=24)
        self.assertTrue(report["accepted"], report)
        self.assertEqual(report["construction"]["repaired_components"], 0)
        self.assertEqual(trimesh.Trimesh(candidate.points, triangle_faces(candidate), process=False).euler_number, 0)

    def test_local_backend_is_independent_of_axis_and_units(self):
        mesh = trimesh.creation.torus(1., .25, major_sections=12, minor_sections=8)
        mesh.apply_transform(trimesh.transformations.rotation_matrix(.73, [1., 2., 3.]))
        for scale in (.001, 1000.):
            with self.subTest(scale=scale):
                source = _surface(np.asarray(mesh.vertices) * scale + [3., -5., 7.], mesh.faces)
                candidate, report = repair_source_mesh(source, max_error=.001 * scale, union_backend="local")
                self.assertTrue(report["accepted"], report)
                self.assertEqual(trimesh.Trimesh(candidate.points, triangle_faces(candidate), process=False).euler_number, 0)

    def test_oversized_faulty_component_is_rejected_before_boolean(self):
        points, faces, _ = overlapping_boxes()
        faces = np.tile(faces, (420, 1))
        block = (points, faces, np.arange(len(faces)))
        with patch("source_material_local._union") as union:
            result, report = repair_material_component(block, _surface(points, faces), max_error=.001)
        self.assertIs(result, block)
        self.assertEqual(report["failure_reason"], "local_component_face_budget_exceeded")
        union.assert_not_called()

    def test_local_backend_keeps_unknown_openings_in_review(self):
        source = _surface(np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]), np.array([[0, 1, 2]]))
        candidate, report = repair_source_mesh(source, max_error=.01, union_backend="local")
        self.assertFalse(report["accepted"])
        self.assertNotIn("construction", report)
        np.testing.assert_array_equal(candidate.points, source.points)


if __name__ == "__main__":
    unittest.main()
