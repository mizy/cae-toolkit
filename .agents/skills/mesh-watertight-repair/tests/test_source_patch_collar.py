import sys
import unittest
from itertools import pairwise
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from mesh_metrics import edge_topology
from source_patch_collar import build_source_collar


class SourcePatchCollarTest(unittest.TestCase):
    def test_curved_source_collar_keeps_original_cut_edge_normals(self):
        mesh = trimesh.creation.icosphere(subdivisions=3)
        points, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
        rows = np.flatnonzero(points[faces].mean(axis=1)[:, 2] > 0.75)
        before_p, before_f = points.copy(), faces.copy()
        new, collar, inner, normals, births = build_source_collar(points, faces, rows)
        combined = np.vstack((points, new))
        tri = combined[collar]
        actual = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        source_tri = points[faces[births]]
        expected = np.cross(source_tri[:, 1] - source_tri[:, 0], source_tri[:, 2] - source_tri[:, 0])
        cosine = np.einsum("ij,ij->i", actual, expected) / (
            np.linalg.norm(actual, axis=1) * np.linalg.norm(expected, axis=1)
        )
        self.assertTrue(np.all(cosine > 1 - 1e-12))
        self.assertTrue(all(value >= len(points) for value in inner))
        self.assertEqual(len(normals), len(inner) - 1)
        topology, edges = edge_topology(collar)
        self.assertEqual(topology["non_manifold_edges"], 0)
        for left, right in pairwise(inner):
            ids = edges[tuple(sorted((left, right)))]
            self.assertEqual(len(ids), 1)
            # The returned loop is directed opposite to the retained collar.
            face = collar[ids[0]]
            self.assertTrue(any(a == right and b == left for a, b in zip(face, np.roll(face, -1))))
        np.testing.assert_array_equal(points, before_p)
        np.testing.assert_array_equal(faces, before_f)

    def test_full_source_without_cut_loop_is_refused(self):
        mesh = trimesh.creation.icosphere(subdivisions=1)
        with self.assertRaisesRegex(ValueError, "simple source cut loop"):
            build_source_collar(mesh.vertices, mesh.faces, np.arange(len(mesh.faces)))


if __name__ == "__main__":
    unittest.main()
