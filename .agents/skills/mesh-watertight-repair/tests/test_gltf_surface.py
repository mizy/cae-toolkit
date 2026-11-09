from __future__ import annotations

import base64
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from gltf_surface import read_gltf_surface
from mesh_io import triangle_faces


class GltfNativeNormalsTest(unittest.TestCase):
    def load_triangle(self, root: Path, *, normals: bool, mixed: bool = False):
        points = np.asarray([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]], dtype='<f4')
        native = np.tile(np.asarray([73, 73, 73], dtype='i1'), (3, 1))
        blob = points.tobytes() + native.tobytes()
        attributes = {'POSITION': 0, **({'NORMAL': 1} if normals else {})}
        primitives = [{'attributes': attributes}]
        if mixed:
            primitives.append({'attributes': {'POSITION': 0}})
        doc = {
            'asset': {'version': '2.0'},
            'buffers': [{'byteLength': len(blob), 'uri': 'data:application/octet-stream;base64,' + base64.b64encode(blob).decode()}],
            'bufferViews': [{'buffer': 0, 'byteOffset': 0, 'byteLength': points.nbytes}, {'buffer': 0, 'byteOffset': points.nbytes, 'byteLength': native.nbytes}],
            'accessors': [{'bufferView': 0, 'componentType': 5126, 'type': 'VEC3', 'count': 3}, {'bufferView': 1, 'componentType': 5120, 'normalized': True, 'type': 'VEC3', 'count': 3}],
            'meshes': [{'primitives': primitives}],
            'nodes': [{'mesh': 0, 'scale': [-2., 3., 1.], 'translation': [4., 5., 6.]}],
            'scenes': [{'nodes': [0]}],
        }
        path = root / 'native.gltf'
        path.write_text(json.dumps(doc))
        return read_gltf_surface(path)

    def test_quantized_normals_use_inverse_transpose_and_preserve_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            native = self.load_triangle(root, normals=True)
            geometry = self.load_triangle(root, normals=False)
        expected = np.asarray([-0.5, 1./3., 1.])
        expected /= np.linalg.norm(expected)
        np.testing.assert_allclose(native.point_data['Normals'], np.tile(expected, (3, 1)), atol=1e-7)
        np.testing.assert_array_equal(native.points, geometry.points)
        np.testing.assert_array_equal(triangle_faces(native), [[2, 1, 0]])
        self.assertIsNotNone(native.GetPointData().GetNormals())
        self.assertNotIn('Normals', geometry.point_data)

    def test_missing_primitive_normals_do_not_discard_other_native_normals(self):
        with tempfile.TemporaryDirectory() as directory:
            mesh = self.load_triangle(Path(directory), normals=True, mixed=True)
        normals = np.asarray(mesh.point_data['Normals'])
        self.assertEqual(normals.shape, (6, 3))
        self.assertTrue(np.isfinite(normals).all())
        np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1., atol=1e-7)
        np.testing.assert_allclose(normals[3:], [[0., 0., 1.]] * 3, atol=1e-7)


if __name__ == '__main__':
    unittest.main()
