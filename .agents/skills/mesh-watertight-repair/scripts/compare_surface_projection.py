"""Registered software depth/normal comparisons; no display server or geometry mutation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyvista as pv
from PIL import Image, ImageDraw

from mesh_io import read_surface, triangle_faces
from solid_triangle_raster import ORTHOGRAPHIC_FRAMES, ViewSpec, projection_frame as frame, rasterize_solid_view


VIEWS = dict(zip(('side', 'opposite-side', 'front', 'rear', 'top', 'bottom', 'oblique', 'rear-oblique'), ORTHOGRAPHIC_FRAMES))


def point_normals(mesh: pv.PolyData) -> np.ndarray:
    normals = mesh.point_data.get('Normals')
    if normals is None:
        normals = mesh.compute_normals(cell_normals=False, split_vertices=False).point_data['Normals']
        mesh.point_data['Normals'] = normals
    return np.asarray(normals, dtype=float)


def visible_normals(mesh: pv.PolyData, raster, coords: np.ndarray, basis: np.ndarray, bounds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = raster.depth.shape
    y, x = np.nonzero(raster.silhouette)
    ids = raster.first_hit_face[y, x]
    faces = triangle_faces(mesh)[ids]
    triangles = coords[faces, :2]
    samples = bounds[0] + np.column_stack((x / (cols - 1), y / (rows - 1))) * (bounds[1] - bounds[0])
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    v0, v1, v2 = b - a, c - a, samples - a
    det = v0[:, 0] * v1[:, 1] - v0[:, 1] * v1[:, 0]
    u = np.divide(v2[:, 0] * v1[:, 1] - v2[:, 1] * v1[:, 0], det, out=np.zeros(len(det)), where=np.abs(det) > 1e-24)
    v = np.divide(v0[:, 0] * v2[:, 1] - v0[:, 1] * v2[:, 0], det, out=np.zeros(len(det)), where=np.abs(det) > 1e-24)
    weights = np.clip(np.column_stack((1-u-v, u, v)), 0., 1.)
    weights /= np.maximum(weights.sum(axis=1)[:, None], 1e-20)
    corner = mesh.cell_data.get('corner_normals')
    values = np.asarray(corner).reshape(-1, 3, 3)[ids] if corner is not None else point_normals(mesh)[faces]
    normals = np.einsum('ij,ijk->ik', weights, values) @ basis
    normals /= np.maximum(np.linalg.norm(normals, axis=1)[:, None], 1e-20)
    field = np.zeros((*raster.depth.shape, 3))
    field[y, x] = normals
    backfaces = np.zeros(raster.depth.shape, dtype=bool)
    backfaces[y, x] = det < 0
    return field, backfaces


def render(mesh: pv.PolyData, eye: tuple, up: tuple, bounds: np.ndarray, size: int):
    coords, basis = frame(np.asarray(mesh.points), eye, up)
    raster = rasterize_solid_view(
        coords, triangle_faces(mesh), ViewSpec('registered', (0, 1), (0., 0., 1.)),
        grid_size=size, depth_tolerance=0., collect_face_evidence=True,
        projection_bounds=bounds, sample_centers=True,
    )
    normals, backfaces = visible_normals(mesh, raster, coords, basis, bounds)
    light = np.asarray([-.35, .45, -1.]); light /= np.linalg.norm(light)
    facing_normals = normals.copy()
    # Match vtk.js: flip for back-facing triangle winding, not the stored normal direction.
    facing_normals[backfaces] *= -1
    shade = .25 + .75 * np.maximum(0., facing_normals @ light)
    rgb = np.full((*raster.depth.shape, 3), [238, 242, 245], dtype=np.uint8)
    rgb[raster.silhouette] = np.clip(np.rint(shade[raster.silhouette, None] * [140, 158, 177]), 0, 255).astype(np.uint8)
    return raster, normals, rgb


def stats(values: np.ndarray) -> dict:
    if not values.size:
        return {'samples': 0}
    return {'samples': int(values.size), **dict(zip(('p50', 'p95', 'p99', 'max'), map(float, np.percentile(values, [50, 95, 99, 100]))))}


def visible_points(raster, bounds: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Back-project actual first-hit pixel centers, rather than visible-face centroids."""
    y, x = np.nonzero(raster.silhouette)
    rows, cols = raster.depth.shape
    xy = bounds[0] + np.column_stack((x / (cols - 1), y / (rows - 1))) * (bounds[1] - bounds[0])
    return np.column_stack((xy, raster.depth[y, x])) @ basis.T


def wheel_mask(source: pv.PolyData, coords: np.ndarray, bounds: np.ndarray, shape: tuple) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    if 'review_category' not in source.cell_data or 'gltf_node_index' not in source.cell_data:
        return mask
    faces = triangle_faces(source)
    nodes = np.asarray(source.cell_data['gltf_node_index'])
    category = np.asarray(source.cell_data['review_category'])
    for node in np.unique(nodes[category == 3]):
        uv = coords[np.unique(faces[(nodes == node) & (category == 3)]), :2]
        box = (np.array([uv.min(axis=0), uv.max(axis=0)]) - bounds[0]) / (bounds[1] - bounds[0])
        lo = np.maximum(0, np.floor(box[0] * [shape[1]-1, shape[0]-1]).astype(int))
        hi = np.minimum([shape[1]-1, shape[0]-1], np.ceil(box[1] * [shape[1]-1, shape[0]-1]).astype(int))
        mask[lo[1]:hi[1]+1, lo[0]:hi[0]+1] = True
    return mask


def compare_pixels(source, candidate, source_normals, candidate_normals, roi, scale: float) -> dict:
    if not np.any(roi):
        return {'available': False}
    a, b = source.silhouette, candidate.silhouette
    common = a & b & roi
    union = (a | b) & roi
    delta = np.abs(source.depth[common] - candidate.depth[common]) * scale
    angle = np.rad2deg(np.arccos(np.clip(np.einsum('ij,ij->i', source_normals[common], candidate_normals[common]), -1., 1.)))
    return {'available': True, 'silhouette_iou': float(common.sum() / max(union.sum(), 1)),
            'lost_source_pixels': int((a & ~b & roi).sum()),
            'new_candidate_pixels': int((~a & b & roi).sum()),
            'depth_mm_at_200mm': stats(delta), 'native_normal_angle_deg': stats(angle)}


def compare(reference: Path, candidates: list[Path], output: Path, size: int = 720, views: list[str] | None = None, mm_per_input_unit: float | None = None, surface_distances: bool = False) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    source = read_surface(reference, clean=False)
    meshes = [read_surface(path, clean=False) for path in candidates]
    normal_fields = ['stored_corner' if 'corner_normals' in mesh.cell_data else 'stored_point' if 'Normals' in mesh.point_data else 'geometry_recomputed' for mesh in meshes]
    scale = 200. / np.ptp(np.asarray(source.points), axis=0).max() if mm_per_input_unit is None else mm_per_input_unit
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('mm_per_input_unit must be finite and positive')
    report = {'reference': str(reference), 'candidates': {str(p): [] for p in candidates},
              'size': size, 'scale_mm_per_input_unit': float(scale), 'status': 'diagnostic',
              'method': 'same_reference_bounds_pixel_center_first_hit_depth_and_native_normal_interpolation',
              'limitations': ['Uniform diffuse software shading; not original material/texture rendering.',
                              'Pixel-center coverage can miss subpixel features; use higher resolution for protected apertures.',
                              'This report does not certify topology or intersections.']}
    for name in views or list(VIEWS):
        eye, up = VIEWS[name]
        coords, basis = frame(np.asarray(source.points), eye, up)
        lo, hi = coords[:, :2].min(axis=0), coords[:, :2].max(axis=0)
        padding = np.maximum((hi-lo) * .04, np.ptp(coords[:, :2], axis=0).max() * .01)
        bounds = np.array([lo-padding, hi+padding])
        sr, sn, rgb = render(source, eye, up, bounds, size)
        wheels = wheel_mask(source, coords, bounds, sr.depth.shape)
        for path, mesh, normal_field in zip(candidates, meshes, normal_fields):
            cr, cn, other = render(mesh, eye, up, bounds, size)
            candidate_coords, _ = frame(np.asarray(mesh.points), eye, up)
            outside = np.any((candidate_coords[:, :2] < bounds[0]) | (candidate_coords[:, :2] > bounds[1]), axis=1)
            row = {'view': name, 'projection_bounds': bounds.tolist(), 'grid': list(sr.depth.shape),
                   'normal_field': normal_field, 'candidate_points_outside_viewport': int(outside.sum()),
                   'all_visible': compare_pixels(sr, cr, sn, cn, np.ones(sr.depth.shape, bool), scale),
                   'wheel_boxes': compare_pixels(sr, cr, sn, cn, wheels, scale)}
            if surface_distances:
                directed = {}
                for label, raster, target in (('source_to_candidate', sr, mesh), ('candidate_to_source', cr, source)):
                    samples = visible_points(raster, bounds, basis)
                    _, nearest = target.find_closest_cell(samples, return_closest_point=True)
                    directed[label] = stats(np.linalg.norm(samples - nearest, axis=1) * scale)
                row['visible_point_surface_distance_mm'] = directed
            common = sr.silhouette & cr.silhouette
            row['rgb_abs_difference'] = stats(np.abs(rgb[common].astype(float)-other[common]).mean(axis=1))
            stem = path.parent.name + '-' + path.stem + '-' + name
            images = [Image.fromarray(np.flipud(x)) for x in (rgb, other)]
            panel = Image.new('RGB', (images[0].width*2, images[0].height+28), 'white')
            panel.paste(images[0], (0, 28)); panel.paste(images[1], (images[0].width, 28))
            draw = ImageDraw.Draw(panel); draw.text((8, 7), 'Native source / '+name, fill='black')
            draw.text((images[0].width+8, 7), path.name, fill='black')
            panel.save(output/(stem+'.png'))
            report['candidates'][str(path)].append(row)
            (output/'comparison.json').write_text(json.dumps(report, indent=2))
            print(json.dumps({'candidate': path.name, 'view': name, 'depth': row['all_visible']['depth_mm_at_200mm'], 'normal': row['all_visible']['native_normal_angle_deg']}), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference', type=Path)
    parser.add_argument('candidates', type=Path, nargs='+')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--size', type=int, default=720)
    parser.add_argument('--views', nargs='+', choices=list(VIEWS))
    parser.add_argument('--mm-per-input-unit', type=float, help='Keep the whole-model physical scale for a local part comparison.')
    parser.add_argument('--surface-distances', action='store_true', help='Measure both directions from actual visible pixel samples to the other surface.')
    args = parser.parse_args()
    compare(args.reference, args.candidates, args.output_dir, args.size, args.views, args.mm_per_input_unit, args.surface_distances)
