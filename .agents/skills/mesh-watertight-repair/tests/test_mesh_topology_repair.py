from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from mesh_metrics import edge_topology, mesh_report
from mesh_topology_repair import collapse_independent_short_edges, collapse_short_manifold_edges


def octahedron_with_short_edge(scale=1.0):
    points = np.array(
        [[1., 0., 0.], [1. + 1e-13, 0., 0.], [-1., 0., 0.], [0., -1., 0.], [0., 0., 1.], [0., 0., -1.]]
    ) * scale
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


def test_short_edge_collapse_is_unit_independent_and_preserves_face_lineage():
    small_points, input_faces = octahedron_with_short_edge()
    large_points, _ = octahedron_with_short_edge(1000)
    _, small_faces, small_ids, small_actions = collapse_short_manifold_edges(
        small_points, input_faces, max_edge_length=1e-9
    )
    _, large_faces, large_ids, large_actions = collapse_short_manifold_edges(
        large_points, input_faces, max_edge_length=1e-6
    )

    assert small_ids.tolist() == large_ids.tolist() == [2, 3, 4, 5, 6, 7]
    assert np.array_equal(small_faces, large_faces)
    assert small_actions[0]["removed_input_face_ids"] == large_actions[0]["removed_input_face_ids"] == [0, 1]
    assert np.isclose(large_actions[0]["length"], small_actions[0]["length"] * 1000)
    for faces in (small_faces, large_faces):
        topology, _ = edge_topology(faces)
        assert topology["boundary_edges"] == 0
        assert topology["non_manifold_edges"] == 0
        assert topology["non_manifold_vertices"] == 0


def test_short_edge_collapse_requires_input_unit_length_limit():
    points, faces = octahedron_with_short_edge()

    for limit in (0, -1, np.inf, np.nan):
        try:
            collapse_short_manifold_edges(points, faces, max_edge_length=limit)
        except ValueError as error:
            assert "finite and positive" in str(error)
        else:
            raise AssertionError("invalid edge length limit was accepted")


def test_independent_local_collapses_preserve_source_anchors_and_topology():
    first, faces = octahedron_with_short_edge()
    first[1, 0] = np.nextafter(np.nextafter(first[0, 0], np.inf), np.inf)
    second = first + [4., 0., 0.]
    second[1, 0] = np.nextafter(np.nextafter(second[0, 0], np.inf), np.inf)
    points = np.vstack((first, second))
    input_faces = np.vstack((faces, faces + len(first)))
    pins = np.ones(len(points), dtype=bool)
    pins[[1, 7]] = False
    output_points, output_faces, rows, actions, summary = collapse_independent_short_edges(
        points, input_faces, pinned_points=pins, max_edge_length=1e-8
    )
    assert summary["collapsed_edges"] == 2
    assert len(actions) == 2
    assert all(action["pinned_source_endpoint"] for action in actions)
    assert np.array_equal(output_points, points)
    assert len(rows) == len(output_faces) == len(input_faces) - 4
    metrics = mesh_report(output_points, output_faces)
    assert metrics["topology"]["closed_manifold"]
    assert metrics["quality"]["degenerate_faces"] == 0


def test_local_collapse_stops_when_bad_face_budget_is_exceeded():
    points, faces = octahedron_with_short_edge()
    points[1, 0] = np.nextafter(np.nextafter(points[0, 0], np.inf), np.inf)
    _, result, rows, actions, summary = collapse_independent_short_edges(
        points, faces, pinned_points=np.zeros(len(points), dtype=bool),
        max_edge_length=1e-8, max_bad_faces=1,
    )
    assert summary["status"] == "bad_face_budget_exceeded"
    assert not actions
    assert np.array_equal(result, faces)
    assert np.array_equal(rows, np.arange(len(faces)))


def test_exact_collinear_face_is_retriangulated_without_moving_vertices():
    points = np.asarray([[0., 0., 0.], [2., 0., 0.], [0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])
    faces = np.asarray([[0, 2, 4], [4, 2, 1], [0, 4, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]])
    output_points, output_faces, rows, actions, summary = collapse_independent_short_edges(
        points, faces, pinned_points=np.ones(len(points), dtype=bool), max_edge_length=.01
    )
    assert summary["collapsed_edges"] == 0
    assert summary["collinear_flips"] == 1
    assert actions[0]["operator"] == "exact_collinear_edge_flip"
    assert np.array_equal(output_points, points)
    assert rows[2] == rows[3] == 3
    metrics = mesh_report(output_points, output_faces)
    assert metrics["topology"]["closed_manifold"]
    assert metrics["topology"]["inconsistent_winding_edges"] == 0
    assert metrics["quality"]["degenerate_faces"] == 0
    assert np.isclose(metrics["volume"]["signed_sum"], 1. / 3.)


def test_nearly_collinear_face_is_not_treated_as_exactly_collinear():
    points = np.asarray([[0., 0., 0.], [2., 0., 0.], [0., 1., 0.], [0., 0., 1.], [1., 1e-18, 0.]])
    faces = np.asarray([[0, 2, 4], [4, 2, 1], [0, 4, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]])
    _, output_faces, rows, actions, summary = collapse_independent_short_edges(
        points, faces, pinned_points=np.ones(len(points), dtype=bool), max_edge_length=.01
    )
    assert summary["collinear_flips"] == 0
    assert not actions
    assert np.array_equal(output_faces, faces)
    assert np.array_equal(rows, np.arange(len(faces)))


def test_unpinned_nearly_collinear_face_has_bounded_retriangulation():
    points = np.asarray([[0., 0., 0.], [2., 0., 0.], [0., 1., 0.], [0., 0., 1.], [1., 1e-18, 0.]])
    faces = np.asarray([[0, 2, 4], [4, 2, 1], [0, 4, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]])
    _, output_faces, _, actions, summary = collapse_independent_short_edges(
        points, faces, pinned_points=np.zeros(len(points), dtype=bool), max_edge_length=.01
    )
    assert summary["bounded_near_collinear_flips"] == 1
    assert actions[0]["operator"] == "bounded_near_collinear_edge_flip"
    assert 2e-18 <= actions[0]["bidirectional_bound"] <= .01
    metrics = mesh_report(points, output_faces)
    assert metrics["topology"]["closed_manifold"]
    assert metrics["topology"]["inconsistent_winding_edges"] == 0
    assert metrics["quality"]["degenerate_faces"] == 0

    _, unchanged, _, actions, _ = collapse_independent_short_edges(
        points, faces, pinned_points=np.zeros(len(points), dtype=bool), max_edge_length=1e-20
    )
    assert not actions
    assert np.array_equal(unchanged, faces)
