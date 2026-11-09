import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyvista as pv
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from assembly_intersections import compare_assembly_intersections


def surface(points, faces):
    faces = np.asarray(faces, dtype=np.int64)
    return pv.PolyData(np.asarray(points, dtype=float),
                       np.column_stack((np.full(len(faces), 3), faces)).ravel())


def test_new_replacement_collision_is_rejected_even_when_original_is_clear():
    original = surface([[-1, -1, 0], [1, -1, 0], [0, 1, 0]], [[0, 1, 2]])
    replacement = surface([[-1, -1, .5], [1, -1, .5], [0, 1, .5]], [[0, 1, 2]])
    context = surface([[-.2, -.2, .4], [.2, -.2, .6], [0, .2, .5]], [[0, 1, 2]])
    report = compare_assembly_intersections(original, replacement, context)
    assert report['status'] == 'computed'
    assert report['before']['cross_pairs'] == 0
    assert report['after']['cross_pairs'] == 1
    assert report['newly_intersected_neighbor_face_rows'] == [0]
    assert not report['passed']


def test_replacement_that_removes_a_cross_intersection_passes_context_check():
    original = surface([[-1, -1, .5], [1, -1, .5], [0, 1, .5]], [[0, 1, 2]])
    replacement = surface([[-1, -1, 0], [1, -1, 0], [0, 1, 0]], [[0, 1, 2]])
    context = surface([[-.2, -.2, .4], [.2, -.2, .6], [0, .2, .5]], [[0, 1, 2]])
    report = compare_assembly_intersections(original, replacement, context)
    assert report['before']['cross_pairs'] == 1
    assert report['after']['cross_pairs'] == 0
    assert report['passed']


def test_budget_or_missing_exact_certification_is_not_a_pass(monkeypatch):
    part = surface([[-1, -1, 0], [1, -1, 0], [0, 1, 0]], [[0, 1, 2]])
    context = surface([[-.2, -.2, -.1], [.2, -.2, .1], [0, .2, 0],
                       [-.3, -.3, -.1], [.3, -.3, .1], [0, .3, 0]], [[0, 1, 2], [3, 4, 5]])
    budget = compare_assembly_intersections(part, part, context, max_context_faces=1)
    assert budget['status'] == 'context_budget_exceeded'
    assert not budget['passed']
    monkeypatch.setattr('assembly_intersections.exact_self_intersection_report',
                        lambda *a, **k: {'status': 'dependency_missing', 'reported_pairs': []})
    missing = compare_assembly_intersections(part, part, context)
    assert missing['status'] == 'incomplete'
    assert not missing['passed']


def test_modeled_patch_rejects_cross_component_collision(tmp_path):
    box = trimesh.creation.box()
    points, faces = np.asarray(box.vertices), np.asarray(box.faces)
    top = (points[faces][:, :, 2] > 0).all(axis=1)
    source, patch = surface(points, faces[~top]).remove_unused_points(), surface(points, faces[top]).remove_unused_points()
    obstacle = trimesh.creation.box(extents=(.2, .2, .2))
    obstacle.apply_translation((0, 0, .5))
    context = surface(obstacle.vertices, obstacle.faces)
    source.save(tmp_path / 'source.vtp'); patch.save(tmp_path / 'patch.vtp'); context.save(tmp_path / 'context.vtp')
    script = Path(__file__).resolve().parents[1] / 'scripts' / 'modeled_patch.py'
    proc = subprocess.run([sys.executable, str(script), str(tmp_path/'source.vtp'), '--patch', str(tmp_path/'patch.vtp'),
                           '--context', str(tmp_path/'context.vtp'), '--seam-tolerance', '0',
                           '--max-seam-angle-deg', '100', '--output-dir', str(tmp_path/'result')], capture_output=True, text=True)
    assert proc.returncode == 2, proc.stderr
    report = json.loads((tmp_path/'result/modeled_patch_report.json').read_text())
    assert not report['geometry_accepted']
    assert report['assembly_context']['after']['cross_pairs'] > 0
    assert report['outputs']['geometry_accepted_candidate_vtp'] is None


def test_unreadable_requested_context_is_not_accepted(tmp_path):
    box = trimesh.creation.box()
    points, faces = np.asarray(box.vertices), np.asarray(box.faces)
    top = (points[faces][:, :, 2] > 0).all(axis=1)
    surface(points, faces[~top]).remove_unused_points().save(tmp_path / 'source.vtp')
    surface(points, faces[top]).remove_unused_points().save(tmp_path / 'patch.vtp')
    script = Path(__file__).resolve().parents[1] / 'scripts' / 'modeled_patch.py'
    proc = subprocess.run([sys.executable, str(script), str(tmp_path/'source.vtp'), '--patch', str(tmp_path/'patch.vtp'),
                           '--context', str(tmp_path/'missing.vtp'), '--seam-tolerance', '0',
                           '--max-seam-angle-deg', '100', '--output-dir', str(tmp_path/'result')], capture_output=True, text=True)
    assert proc.returncode == 2
    report = json.loads((tmp_path/'result/modeled_patch_report.json').read_text())
    assert not report['geometry_accepted']
    assert report['assembly_context']['status'] == 'not_completed'
    assert report['outputs']['geometry_accepted_candidate_vtp'] is None
