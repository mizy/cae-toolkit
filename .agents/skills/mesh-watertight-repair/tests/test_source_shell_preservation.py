from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pyvista as pv
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import two_stage_watertight_remesh as worker
from mesh_io import read_surface, triangle_faces
from opening_repair import apply_opening_decisions, stitch_boundary_seams
from repair_inventory import build_inventory
from source_projected_closure import project_closure_to_source
from source_projected_validation import exact_self_intersection_report
from source_shell_preservation import select_preserved_shells


def mixed_source():
    good = trimesh.creation.icosphere(subdivisions=1)
    good.apply_translation([-4., 0., 0.])
    broken = trimesh.creation.box()
    broken.apply_translation([4., 0., 0.])
    faces = np.asarray(broken.faces)
    faces = faces[~np.all(np.asarray(broken.vertices)[faces, 2] > .49, axis=1)]
    points = np.vstack((good.vertices, broken.vertices))
    combined = np.vstack((good.faces, faces + len(good.vertices)))
    return points, combined, good, broken


def arguments(directory):
    return SimpleNamespace(output_dir=Path(directory), voxel_pitch=.1, sealed_exterior_grid=64,
        sealed_exterior_radius_voxels=1, scaffold_smoothing_iterations=0)


class SourceShellPreservationTest(unittest.TestCase):
    def test_valid_torus_never_enters_voxel_reconstruction(self):
        mesh = trimesh.creation.torus(1., .25, major_sections=16, minor_sections=8)
        with tempfile.TemporaryDirectory() as directory, patch.object(worker, "update_run_status"), \
             patch.object(worker, "voxel_watertight_remesh") as voxel:
            p, f, report, _, _, point_ids, face_ids = worker.build_closure_proxy_artifact(
                arguments(directory), {}, mesh.vertices, mesh.faces)
        voxel.assert_not_called()
        np.testing.assert_array_equal(p[f], np.asarray(mesh.vertices)[mesh.faces])
        np.testing.assert_array_equal(p, np.asarray(mesh.vertices)[point_ids])
        self.assertTrue(np.all(face_ids >= 0))
        self.assertFalse(report["voxel_reconstruction_applied"])
        projected, _, provenance, _ = project_closure_to_source(
            p, f, mesh.vertices, mesh.faces, preserved_vertex_mask=point_ids >= 0)
        np.testing.assert_array_equal(projected, p)
        self.assertTrue(np.all(provenance["point_data"]["source_preserved"]))

    def test_only_broken_component_is_sent_to_voxels(self):
        points, faces, good, broken = mixed_source()
        def rebuild(p, f, *_args, **_kwargs):
            self.assertEqual(len(f), 10)
            self.assertGreater(p[:, 0].min(), 3.)
            return np.asarray(broken.vertices), np.asarray(broken.faces), {"exterior_volume": {}}
        with tempfile.TemporaryDirectory() as directory, patch.object(worker, "update_run_status"), \
             patch.object(worker, "voxel_watertight_remesh", side_effect=rebuild):
            p, f, report, _, _, point_ids, face_ids = worker.build_closure_proxy_artifact(
                arguments(directory), {}, points, faces)
        keep = face_ids >= 0
        self.assertEqual(int(keep.sum()), len(good.faces))
        np.testing.assert_array_equal(p[f[keep]], points[faces[face_ids[keep]]])
        self.assertEqual(report["source_preservation"]["repair_source_faces"], 10)
        self.assertTrue(exact_self_intersection_report(p, f)["passed"])

    def test_reviewed_local_cap_preserves_every_original_triangle(self):
        points, faces, good, _ = mixed_source()
        inventory = build_inventory(points, faces, np.arange(len(faces)), "source", max_items=100)
        capped, generated, _, ready = apply_opening_decisions(points, faces, inventory, [], default_action="cap")
        self.assertTrue(ready)
        self.assertEqual(int(generated.sum()), 2)
        np.testing.assert_array_equal(capped[:len(faces)], faces)
        with tempfile.TemporaryDirectory() as directory, patch.object(worker, "update_run_status"), \
             patch.object(worker, "voxel_watertight_remesh") as voxel:
            p, f, report, *_ = worker.build_closure_proxy_artifact(arguments(directory), {}, points, capped)
        voxel.assert_not_called()
        np.testing.assert_array_equal(p[f[:len(faces)]], points[faces])
        self.assertEqual(report["source_preservation"]["preserved_components"], 2)

    def test_closed_overlapping_solids_are_not_certified_as_preservable(self):
        first = trimesh.creation.box()
        second = trimesh.creation.box(); second.apply_translation([.25, .25, .25])
        points = np.vstack((first.vertices, second.vertices))
        faces = np.vstack((first.faces, second.faces + len(first.vertices)))
        keep, report = select_preserved_shells(points, faces)
        self.assertFalse(np.any(keep))
        self.assertEqual(report["intersecting_closed_components"], 2)

    def test_overlapping_bounds_alone_do_not_discard_valid_shells(self):
        torus = trimesh.creation.torus(1., .25, major_sections=16, minor_sections=8)
        box = trimesh.creation.box(extents=[.2, .2, .2])
        broken = trimesh.creation.box(); broken.apply_translation([4., 0., 0.])
        points = np.vstack((torus.vertices, box.vertices, broken.vertices))
        faces = np.vstack((torus.faces, box.faces + len(torus.vertices),
                           broken.faces[:-1] + len(torus.vertices) + len(box.vertices)))
        keep, report = select_preserved_shells(points, faces)
        self.assertEqual(int(keep.sum()), len(torus.faces) + len(box.faces))
        self.assertEqual(report["preserved_components"], 2)

    def test_fixed_shell_must_exist_on_projection_source(self):
        mesh = trimesh.creation.icosphere(subdivisions=1)
        with self.assertRaisesRegex(ValueError, "absent"):
            project_closure_to_source(mesh.vertices, mesh.faces, mesh.vertices * 2, mesh.faces,
                                     preserved_vertex_mask=np.ones(len(mesh.vertices), bool))

    def test_import_cleaning_keeps_native_normal_seams(self):
        box = trimesh.creation.box()
        indices = np.arange(len(box.faces) * 3).reshape(-1, 3)
        source = pv.PolyData(np.asarray(box.vertices)[box.faces].reshape(-1, 3),
                             np.column_stack((np.full(len(indices), 3), indices)).ravel())
        source.point_data["Normals"] = np.repeat(box.face_normals, 3, axis=0)
        expected = np.asarray(source.point_data["Normals"])[triangle_faces(source)].reshape(-1, 9)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.vtp"; source.save(path)
            cleaned = read_surface(path)
        self.assertLess(cleaned.n_points, source.n_points)
        np.testing.assert_array_equal(cleaned.cell_data["corner_normals"], expected)

    def test_bounded_seam_stitches_matching_rim_and_leaves_closed_shell_unchanged(self):
        cube = trimesh.creation.box()
        vertices, faces = np.asarray(cube.vertices), np.asarray(cube.faces)
        top = np.all(vertices[faces, 2] > .49, axis=1)
        body, lid = faces[~top], faces[top]
        lid_vertices, inverse = np.unique(lid, return_inverse=True)
        points = np.vstack((vertices, vertices[lid_vertices] + [0., 0., .0001]))
        faces = np.vstack((body, inverse.reshape(-1, 3) + len(vertices)))
        sphere = trimesh.creation.icosphere(subdivisions=1); sphere.apply_translation([3., 0., 0.])
        protected = np.asarray(sphere.faces) + len(points)
        points = np.vstack((points, sphere.vertices)); faces = np.vstack((faces, protected))
        before = points.copy()
        result, report = stitch_boundary_seams(points, faces, tolerance=.0002)
        self.assertTrue(report["applied"], report)
        self.assertEqual(report["boundary_edges_after"], 0)
        self.assertLessEqual(report["max_surface_displacement"], .0002)
        np.testing.assert_array_equal(result[-len(protected):], protected)
        np.testing.assert_array_equal(points, before)
        self.assertTrue(exact_self_intersection_report(points, result)["passed"])
        # A visible gap is not silently bridged under a smaller declared tolerance.
        unchanged, rejected = stitch_boundary_seams(points, faces, tolerance=.00001)
        np.testing.assert_array_equal(unchanged, faces)
        self.assertFalse(rejected["applied"])

    def test_sewing_retains_functional_open_boundary_without_a_matching_partner(self):
        mesh = trimesh.creation.box()
        faces = np.asarray(mesh.faces)[:-2]
        result, report = stitch_boundary_seams(mesh.vertices, faces, tolerance=.001)
        np.testing.assert_array_equal(result, faces)
        self.assertFalse(report["applied"])


if __name__ == "__main__":
    unittest.main()
