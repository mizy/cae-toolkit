"""Attach one modeled triangle patch to one reviewed open source boundary."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree

from html_report import write_html_report
from local_defect_regions import _boundary
from material_cleanup import geometry_gate_arrays
from mesh_io import read_surface, triangle_faces
from mesh_metrics import edge_topology, face_component_labels
from source_projected_validation import roundtrip_validation


def _single_boundary(points, faces, label):
    _, edges = edge_topology(faces)
    if len(np.unique(face_component_labels(len(faces), edges))) != 1:
        raise ValueError(f'{label} must contain one face-connected component')
    _, loops, info = _boundary(faces, np.arange(len(faces)), edges)
    if info['status'] != 'closed_loops' or len(loops) != 1:
        raise ValueError(f'{label} must have exactly one simple boundary loop')
    return np.asarray(loops[0][:-1], dtype=np.int64), edges


def _match_boundary(source_points, source_faces, patch_points, patch_faces, tolerance):
    source_loop, source_edges = _single_boundary(source_points, source_faces, 'source')
    patch_loop, _ = _single_boundary(patch_points, patch_faces, 'patch')
    if len(source_loop) != len(patch_loop):
        raise ValueError('source and patch boundary vertex counts differ')
    distance, nearest = cKDTree(source_points[source_loop]).query(patch_points[patch_loop], k=1)
    if not np.isfinite(distance).all() or np.any(distance > tolerance) or len(np.unique(nearest)) != len(source_loop):
        raise ValueError('patch boundary does not match source boundary within tolerance')
    source_ids = source_loop[nearest]
    source_set = {tuple(sorted((int(a), int(b)))) for a, b in zip(source_loop, np.roll(source_loop, -1))}
    mapped_set = {tuple(sorted((int(a), int(b)))) for a, b in zip(source_ids, np.roll(source_ids, -1))}
    if mapped_set != source_set:
        raise ValueError('matched boundary vertices do not preserve boundary edges')
    mapping = np.full(len(patch_points), -1, dtype=np.int64)
    mapping[patch_loop] = source_ids
    interior = np.flatnonzero(mapping < 0)
    mapping[interior] = len(source_points) + np.arange(len(interior))
    mapped_faces = mapping[patch_faces]
    a, b = map(int, source_ids[:2])
    source_face = source_faces[source_edges[tuple(sorted((a, b)))][0]]
    source_direction = any(int(x) == a and int(y) == b for x, y in zip(source_face, np.roll(source_face, -1)))
    patch_direction = any(int(x) == a and int(y) == b for face in mapped_faces for x, y in zip(face, np.roll(face, -1)))
    flipped = source_direction == patch_direction
    if flipped:
        mapped_faces = mapped_faces[:, ::-1]
    return np.vstack((source_points, patch_points[interior])), mapped_faces, interior, source_loop, float(distance.max()), flipped


def _seam_angle(points, faces, source_loop):
    _, edges = edge_topology(faces)
    triangle = points[faces]
    normals = np.cross(triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    if np.any(lengths == 0):
        raise ValueError('assembled candidate contains degenerate faces')
    normals /= lengths[:, None]
    angles = []
    for a, b in zip(source_loop, np.roll(source_loop, -1)):
        incident = edges.get(tuple(sorted((int(a), int(b)))), [])
        if len(incident) != 2:
            raise ValueError('source boundary is not shared by exactly two faces after assembly')
        angles.append(float(np.degrees(np.arccos(np.clip(normals[incident[0]] @ normals[incident[1]], -1, 1)))))
    return max(angles)


def _same_data(a, b):
    if np.asarray(a).dtype.kind in 'fc':
        return np.array_equal(a, b, equal_nan=True)
    return np.array_equal(a, b)


# @entry: one explicitly reviewed patch whose perimeter already follows the source boundary.
def attach_modeled_patch(source, patch, *, seam_tolerance, max_seam_angle_deg):
    if not np.isfinite(seam_tolerance) or seam_tolerance < 0:
        raise ValueError('seam tolerance must be finite and nonnegative')
    if not np.isfinite(max_seam_angle_deg) or not 0 <= max_seam_angle_deg <= 180:
        raise ValueError('maximum seam angle must be between 0 and 180 degrees')
    sp, sf = np.asarray(source.points, dtype=np.float64), triangle_faces(source)
    pp, pf = np.asarray(patch.points, dtype=np.float64), triangle_faces(patch)
    if not np.isfinite(sp).all() or not np.isfinite(pp).all():
        raise ValueError('mesh coordinates must be finite')
    if source.n_cells != len(sf) or patch.n_cells != len(pf):
        raise ValueError('source and patch must contain only triangle cells')
    if 'modeled_patch_face' in source.cell_data:
        raise ValueError('source already contains reserved modeled_patch_face provenance')
    points, added_faces, interior, loop, displacement, flipped = _match_boundary(sp, sf, pp, pf, seam_tolerance)
    faces = np.vstack((sf, added_faces))
    seam_angle = _seam_angle(points, faces, loop)
    geometry = geometry_gate_arrays(points, faces)
    accepted = bool(geometry['passed'] and seam_angle <= max_seam_angle_deg)
    packed = np.column_stack((np.full(len(faces), 3, dtype=np.int64), faces))
    candidate = pv.PolyData(points, packed.ravel())
    tri = points[added_faces]
    patch_face_normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    patch_face_normals /= np.maximum(np.linalg.norm(patch_face_normals, axis=1)[:, None], np.finfo(float).tiny)
    patch_point_normals = np.zeros_like(points)
    for corner in range(3):
        np.add.at(patch_point_normals, added_faces[:, corner], patch_face_normals)
    patch_point_normals /= np.maximum(np.linalg.norm(patch_point_normals, axis=1)[:, None], np.finfo(float).tiny)
    # Copy existing data exactly; generated values are explicitly distinguished by provenance.
    for name in source.point_data:
        values = np.asarray(source.point_data[name])
        if name == 'vtkOriginalPointIds' and values.dtype.kind == 'u':
            values = values.astype(np.int64)
        extra = np.zeros((len(interior), *values.shape[1:]), dtype=values.dtype)
        if name == 'vtkOriginalPointIds':
            extra.fill(-1)
        if name == 'Normals':
            extra = patch_point_normals[len(sp):].astype(values.dtype, copy=False)
        candidate.point_data[name] = np.concatenate((values, extra))
    for name in source.cell_data:
        values = np.asarray(source.cell_data[name])
        source_id = name in ('source_triangle_index', 'local_patch_source_face_row', 'vtkOriginalCellIds')
        if source_id and values.dtype.kind == 'u':
            values = values.astype(np.int64)
        extra = np.zeros((len(pf), *values.shape[1:]), dtype=values.dtype)
        if source_id:
            extra.fill(-1)
        if name == 'corner_normals' and values.ndim == 2 and values.shape[1] == 9:
            extra = np.repeat(patch_face_normals[:, None, :], 3, axis=1).reshape(-1, 9).astype(values.dtype)
        candidate.cell_data[name] = np.concatenate((values, extra))
    candidate.cell_data['modeled_patch_face'] = np.r_[np.zeros(len(sf), dtype=np.uint8), np.ones(len(pf), dtype=np.uint8)]
    report = {'accepted': False, 'geometry_accepted': accepted, 'engineering_ready': False, 'source_faces': len(sf), 'patch_faces': len(pf),
              'source_faces_unchanged': bool(np.array_equal(points[faces[:len(sf)]], sp[sf])),
              'boundary_edges': len(loop), 'max_boundary_snap': displacement, 'seam_tolerance': seam_tolerance,
              'max_seam_angle_deg': seam_angle, 'seam_angle_limit_deg': max_seam_angle_deg,
              'patch_orientation_flipped': flipped, 'geometry': geometry,
              'generated_attributes': 'Source arrays preserved; generated source IDs are -1, normals geometric, and other attributes zero. modeled_patch_face distinguishes generated values.',
              'source_geometry_sha256': hashlib.sha256(sp.tobytes() + sf.tobytes()).hexdigest(),
              'patch_geometry_sha256': hashlib.sha256(pp.tobytes() + pf.tobytes()).hexdigest(),
              'limitations': ['Patch shape and boundary intent require independent review; geometry acceptance does not establish original shape, CFD or print readiness.',
                              'One simple boundary per mesh, one-to-one matching vertices and matching boundary edges are required; no intersection trimming is performed.']}
    return candidate, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--patch', type=Path, required=True)
    parser.add_argument('--context', type=Path, help='Unchanged complete assembly excluding this source component')
    parser.add_argument('--seam-tolerance', type=float, required=True, help='Maximum patch boundary snap in input units')
    parser.add_argument('--max-seam-angle-deg', type=float, required=True, help='Maximum resulting dihedral angle at the seam')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {'accepted': False, 'geometry_accepted': False, 'engineering_ready': False, 'input': {'path': str(args.input.resolve())},
              'outputs': {'accepted_mesh_vtp': None}}
    try:
        source, patch = read_surface(args.input, clean=False), read_surface(args.patch, clean=False)
        source_unused = source.n_points
        source = source.remove_unused_points()
        report['source_unused_points_removed'] = source_unused - source.n_points
        patch = patch.remove_unused_points()
        candidate, result = attach_modeled_patch(source, patch, seam_tolerance=args.seam_tolerance,
                                                 max_seam_angle_deg=args.max_seam_angle_deg)
        report.update(result)
        path = args.output_dir / 'modeled_patch_candidate.vtp'
        candidate.save(path)
        if report['geometry_accepted']:
            report['serialized_readback'] = roundtrip_validation(path, np.asarray(candidate.points), triangle_faces(candidate))
            saved = pv.read(path)
            point_ok = all(name in saved.point_data and _same_data(np.asarray(saved.point_data[name])[:source.n_points], np.asarray(source.point_data[name])) for name in source.point_data)
            cell_ok = all(name in saved.cell_data and _same_data(np.asarray(saved.cell_data[name])[:source.n_cells], np.asarray(source.cell_data[name])) for name in source.cell_data)
            report['source_attributes_readback'] = {'point_data_preserved': bool(point_ok), 'cell_data_preserved': bool(cell_ok)}
            report['geometry_accepted'] = bool(report['serialized_readback']['passed'] and point_ok and cell_ok)
        report['assembly_context'] = {'status': 'not_checked', 'passed': False,
                                      'reason': 'No assembly context was supplied; cross-component interactions remain unverified.'}
        if args.context is not None:
            from assembly_intersections import compare_assembly_intersections

            report['assembly_context'] = {'status': 'not_completed', 'passed': False,
                                          'reason': 'The supplied assembly context was not checked.'}
            context = read_surface(args.context, clean=False).remove_unused_points()
            report['assembly_context'] = compare_assembly_intersections(source, candidate, context)
            report['assembly_context']['context_path'] = str(args.context.resolve())
            report['geometry_accepted'] = bool(report['geometry_accepted'] and report['assembly_context']['passed'])
        report.setdefault('diagnostics', {})['assembly_context'] = report['assembly_context']
        report['outputs'] = {'accepted_mesh_vtp': None,
                             'geometry_accepted_candidate_vtp': str(path.resolve()) if report['geometry_accepted'] else None,
                             'candidate_vtp': str(path.resolve()), 'mesh_previews': [
                                 {'label': 'Original source', 'path': str(args.input.resolve())},
                                 {'label': 'Modeled patch', 'path': str(args.patch.resolve())},
                                 {'label': 'Geometry accepted; shape review required' if report['geometry_accepted'] else 'Rejected candidate', 'path': str(path.resolve())}]}
    except (ValueError, RuntimeError, OSError, IndexError) as exc:
        report['geometry_accepted'] = False
        report['outputs'] = {'accepted_mesh_vtp': None, 'geometry_accepted_candidate_vtp': None}
        report['failure_reason'] = f'{type(exc).__name__}: {exc}'
    report['elapsed_seconds'] = time.monotonic() - started
    report['decision'] = {'status': 'geometry_candidate' if report['geometry_accepted'] else 'rejected',
                          'reason': 'geometry passed; shape and opening intent require review' if report['geometry_accepted'] else report.get('failure_reason', 'geometry, seam, readback or supplied assembly-context check failed')}
    (args.output_dir / 'modeled_patch_report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    write_html_report(report, args.output_dir / 'modeled_patch_report.html', 'Modeled patch geometry assembly')
    return 0 if report['geometry_accepted'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
