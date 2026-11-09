import sys
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyvista as pv
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from modeled_patch import attach_modeled_patch
from mesh_io import triangle_faces


def part(points, faces):
    ids, inv = np.unique(faces, return_inverse=True)
    f = inv.reshape(-1, 3)
    return pv.PolyData(np.asarray(points)[ids], np.column_stack((np.full(len(f), 3), f)).ravel())


def box_parts():
    box = trimesh.creation.box()
    p, f = np.asarray(box.vertices), np.asarray(box.faces)
    top = (p[f][:, :, 2] > 0).all(axis=1)
    return part(p, f[~top]), part(p, f[top])


class ModeledPatchTest(unittest.TestCase):
    def test_cli_serializes_and_audits_the_joined_mesh(self):
        source, patch = box_parts()
        source.point_data['Normals'] = np.tile([0.0, 0.0, 1.0], (source.n_points, 1))
        source.cell_data['source_triangle_index'] = np.arange(source.n_cells, dtype=np.int64)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source.save(root / 'source.vtp')
            patch.save(root / 'patch.vtp')
            command = [sys.executable, str(Path(__file__).resolve().parents[1] / 'scripts' / 'modeled_patch.py'),
                       str(root / 'source.vtp'), '--patch', str(root / 'patch.vtp'), '--seam-tolerance', '0',
                       '--max-seam-angle-deg', '100', '--output-dir', str(root / 'result')]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads((root / 'result' / 'modeled_patch_report.json').read_text())
            self.assertTrue(report['geometry_accepted'])
            self.assertFalse(report['accepted'])
            self.assertIsNone(report['outputs']['accepted_mesh_vtp'])
            self.assertIsNotNone(report['outputs']['geometry_accepted_candidate_vtp'])
            self.assertTrue(report['serialized_readback']['passed'])
            self.assertTrue(report['source_attributes_readback']['point_data_preserved'])
            self.assertTrue(report['source_attributes_readback']['cell_data_preserved'])
            self.assertFalse(report['engineering_ready'])
            result_mesh = pv.read(report['outputs']['geometry_accepted_candidate_vtp'])
            np.testing.assert_array_equal(result_mesh.cell_data['source_triangle_index'][source.n_cells:], [-1, -1])
            self.assertTrue((root / 'result' / 'modeled_patch_report.html').is_file())

    def test_join_preserves_source_and_accepts_reversed_patch(self):
        source, patch = box_parts()
        source.cell_data['source_tag'] = np.arange(source.n_cells)
        # Reversal exercises automatic orientation rather than relying on modeling orientation.
        patch = pv.PolyData(patch.points, np.column_stack((np.full(patch.n_cells, 3), triangle_faces(patch)[:, ::-1])).ravel())
        candidate, report = attach_modeled_patch(source, patch, seam_tolerance=1e-6, max_seam_angle_deg=100)
        self.assertTrue(report['geometry_accepted'], report)
        self.assertTrue(report['patch_orientation_flipped'])
        self.assertTrue(report['source_faces_unchanged'])
        self.assertTrue(report['geometry']['exact']['passed'])
        np.testing.assert_array_equal(candidate.cell_data['source_tag'][:source.n_cells], source.cell_data['source_tag'])
        np.testing.assert_array_equal(triangle_faces(candidate)[:source.n_cells], triangle_faces(source))
        self.assertEqual(int(candidate.cell_data['modeled_patch_face'].sum()), patch.n_cells)

    def test_rejects_mismatch_and_seam_angle(self):
        source, patch = box_parts()
        moved = patch.copy(); moved.points = moved.points + [0, 0, 0.01]
        with self.assertRaisesRegex(ValueError, 'within tolerance'):
            attach_modeled_patch(source, moved, seam_tolerance=0.001, max_seam_angle_deg=100)
        _, report = attach_modeled_patch(source, patch, seam_tolerance=1e-6, max_seam_angle_deg=20)
        self.assertFalse(report['geometry_accepted'])
        self.assertTrue(report['geometry']['passed'])

    def test_snap_within_budget_preserves_source_coordinates(self):
        source, patch = box_parts()
        patch.points = patch.points + [0, 0, 0.0001]
        candidate, report = attach_modeled_patch(source, patch, seam_tolerance=0.0002, max_seam_angle_deg=100)
        self.assertTrue(report['geometry_accepted'], report)
        self.assertAlmostEqual(report['max_boundary_snap'], 0.0001)
        np.testing.assert_array_equal(candidate.points[:source.n_points], source.points)


if __name__ == '__main__':
    unittest.main()
