from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from mesh_metrics import edge_topology
from source_preserving_trial import collapse_nanoface_edges, face_keys


def octahedron_with_short_edge():
    points = np.array(
        [[1., 0., 0.], [1. + 1e-13, 0., 0.], [-1., 0., 0.], [0., -1., 0.], [0., 0., 1.], [0., 0., -1.]]
    )
    faces = []
    for x in (0, 2):
        for y in (1, 3):
            for z in (4, 5):
                face = [x, y, z]
                triangle = points[face]
                if np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0]) @ triangle.mean(axis=0) < 0:
                    face[1], face[2] = face[2], face[1]
                faces.append(face)
    return points, np.asarray(faces)


def tetrahedron_with_short_edge():
    points = np.array([[0., 0., 0.], [1e-13, 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    center = points.mean(axis=0)
    faces = []
    for face in ([0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]):
        triangle = points[face]
        if np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0]) @ (triangle.mean(axis=0) - center) < 0:
            face[1], face[2] = face[2], face[1]
        faces.append(face)
    return points, np.asarray(faces)


def octahedron_with_degenerate_edge(edge_length_m=1e-7):
    offset = 1e-11
    points = np.array(
        [
            [1., 0., 0.], [1. + edge_length_m, 0., 0.], [-1., 0., 0.], [0., -1., 0.],
            [1. + edge_length_m / 2, offset, offset], [1. + edge_length_m / 2, -offset, -offset],
        ]
    )
    faces = []
    for x in (0, 2):
        for y in (1, 3):
            for z in (4, 5):
                face = [x, y, z]
                triangle = points[face]
                if np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0]) @ triangle.mean(axis=0) < 0:
                    face[1], face[2] = face[2], face[1]
                faces.append(face)
    return points, np.asarray(faces)


def test_face_identity_preserves_xyz_pairing_and_ignores_corner_order():
    # These two triangles have identical independent x/y/z coordinate multisets.
    points = np.array([[0., 0., 0.], [1., 1., 0.], [2., 0., 1.],
                       [0., 1., 0.], [1., 0., 0.], [2., 0., 1.]])
    keys = face_keys(points, np.array([[0, 1, 2], [2, 0, 1], [3, 4, 5]]))
    assert keys[0] == keys[1]
    assert keys[0] != keys[2]


def test_nanoface_collapse_keeps_closed_manifold_and_face_provenance():
    points, faces = octahedron_with_short_edge()
    out_points, out_faces, kept_ids, actions = collapse_nanoface_edges(points, faces)
    topology, _ = edge_topology(out_faces)

    assert len(actions) == 1
    assert actions[0]["removed_input_face_ids"] == [0, 1]
    assert kept_ids.tolist() == [2, 3, 4, 5, 6, 7]
    assert topology["boundary_edges"] == 0
    assert topology["non_manifold_edges"] == 0
    assert topology["non_manifold_vertices"] == 0
    assert np.array_equal(out_points, points)


def test_nanoface_collapse_pins_source_endpoint():
    points, faces = octahedron_with_short_edge()
    pinned = np.array([False, True, False, False, False, False])
    _, out_faces, kept_ids, actions = collapse_nanoface_edges(points, faces, pinned_points=pinned)
    topology, _ = edge_topology(out_faces)

    assert actions[0]["kept_point_id"] == 1
    assert actions[0]["collapsed_point_id"] == 0
    assert actions[0]["pinned_source_endpoint"] is True
    assert kept_ids.tolist() == [2, 3, 4, 5, 6, 7]
    assert topology["boundary_edges"] == 0
    assert topology["non_manifold_edges"] == 0
    assert topology["non_manifold_vertices"] == 0


def test_nanoface_collapse_rejects_duplicate_keep_star_faces():
    points, faces = tetrahedron_with_short_edge()
    _, out_faces, kept_ids, actions = collapse_nanoface_edges(points, faces)
    topology, _ = edge_topology(out_faces)

    assert actions == []
    assert kept_ids.tolist() == [0, 1, 2, 3]
    assert np.array_equal(out_faces, faces)
    assert topology["boundary_edges"] == 0
    assert topology["non_manifold_edges"] == 0
    assert topology["non_manifold_vertices"] == 0


def test_nanoface_edge_limit_defaults_to_10nm_and_allows_bounded_override():
    points, faces = octahedron_with_degenerate_edge()
    _, default_faces, _, default_actions = collapse_nanoface_edges(points, faces)
    _, extended_faces, kept_ids, extended_actions = collapse_nanoface_edges(
        points, faces, max_edge_length_m=1e-6
    )
    topology, _ = edge_topology(extended_faces)

    assert default_actions == []
    assert np.array_equal(default_faces, faces)
    assert extended_actions
    assert all(10e-9 < action["length_m"] <= 1e-6 for action in extended_actions)
    assert len(kept_ids) == len(extended_faces)
    assert topology["boundary_edges"] == 0
    assert topology["non_manifold_edges"] == 0
    assert topology["non_manifold_vertices"] == 0


def test_nanoface_edge_limit_never_collapses_pinned_endpoints():
    points, faces = octahedron_with_degenerate_edge()
    pinned = np.array([True, True, False, False, True, True])
    _, out_faces, kept_ids, actions = collapse_nanoface_edges(
        points, faces, pinned_points=pinned, max_edge_length_m=1e-6
    )

    assert actions == []
    assert np.array_equal(out_faces, faces)
    assert kept_ids.tolist() == list(range(len(faces)))


@pytest.mark.parametrize("edge_limit", [0., -1., np.inf, np.nan])
def test_nanoface_edge_limit_must_be_finite_and_positive(edge_limit):
    points, faces = octahedron_with_degenerate_edge()

    with pytest.raises(ValueError, match="finite and positive"):
        collapse_nanoface_edges(points, faces, max_edge_length_m=edge_limit)
