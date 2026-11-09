"""Model-3-specific mirror-height sensitivity test; geometry proxy, not CFD truth."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from vehicle_cd_learning.render import _rasterize, canonical_points
from vehicle_cd_similarity_lib.artifacts import (
    rasterize_side,
    save_artifacts,
    triangle_faces,
)
from vehicle_cd_similarity_lib.mesh_io import read_mesh
from vehicle_cd_similarity_lib.normalization import normalize_vehicle_mesh
from vehicle_cd_similarity_lib.scoring import compare_artifacts, score_library, softmax


def flatten(mesh, ratio):
    """Edit only the visually reviewed mirror region of the canonical Model 3."""
    points = np.asarray(mesh.points)
    selected = (
        (points[:, 0] > -0.85)
        & (points[:, 0] < -0.35)
        & (points[:, 2] > 0.75)
        & (points[:, 2] < 1.2)
        & (np.abs(points[:, 1]) > 0.92)
    )
    if selected.sum() < 100:
        raise ValueError("This experiment requires the reviewed Model 3 geometry")
    # Root vertices at |Y|<=.92 remain fixed; smoothly reach full compression at .96.
    t = np.clip((np.abs(points[:, 1]) - 0.92) / 0.04, 0, 1)
    weight = (t * t * (3 - 2 * t)) * selected
    out = mesh.copy(deep=True)
    changed = points.copy()
    details = []
    for sign in (-1, 1):
        core = selected & (sign * points[:, 1] >= 0.96)
        side = selected & (sign * points[:, 1] > 0)
        z = points[core, 2]
        anchor = float((z.min() + z.max()) / 2)
        changed[side, 2] = points[side, 2] + (ratio - 1) * weight[side] * (
            points[side, 2] - anchor
        )
        details.append(
            {
                "side": sign,
                "core_vertices": int(core.sum()),
                "core_height_before_mm": float(np.ptp(z) * 1000),
                "core_height_after_mm": float(np.ptp(changed[core, 2]) * 1000),
                "anchor_z_m": anchor,
            }
        )
    out.points = changed
    assert np.array_equal(points[~selected], changed[~selected])
    assert np.array_equal(points[:, :2], changed[:, :2])
    assert np.array_equal(np.asarray(mesh.faces), np.asarray(out.faces))
    assert np.array_equal(np.asarray(mesh.bounds), np.asarray(out.bounds))
    return out, {
        "ratio": ratio,
        "selected_vertex_ids": np.flatnonzero(selected).tolist(),
        "moved_vertices": int(np.count_nonzero(np.any(changed != points, axis=1))),
        "max_displacement_mm": float(
            np.linalg.norm(changed - points, axis=1).max() * 1000
        ),
        "outside_region_exactly_unchanged": True,
        "all_xy_exactly_unchanged": True,
        "whole_car_bounds_exactly_unchanged": True,
        "mirror_details": details,
    }


def geometry_metrics(mesh):
    faces = triangle_faces(mesh)
    p = np.asarray(mesh.points)
    areas = (
        np.linalg.norm(
            np.cross(p[faces[:, 1]] - p[faces[:, 0]], p[faces[:, 2]] - p[faces[:, 0]]),
            axis=1,
        )
        / 2
    )
    boundary = mesh.extract_feature_edges(
        boundary_edges=True,
        non_manifold_edges=False,
        feature_edges=False,
        manifold_edges=False,
    ).n_cells
    nonmanifold = mesh.extract_feature_edges(
        boundary_edges=False,
        non_manifold_edges=True,
        feature_edges=False,
        manifold_edges=False,
    ).n_cells
    return {
        "points": mesh.n_points,
        "triangles": mesh.n_cells,
        "boundary_edges": boundary,
        "non_manifold_edges": nonmanifold,
        "degenerate_faces": int((areas < 1e-12).sum()),
        "components": int(np.unique(mesh.connectivity().cell_data["RegionId"]).size),
        "surface_area_m2": float(areas.sum()),
        "bbox": list(mesh.bounds),
        "volume_m3": float(mesh.volume) if boundary == 0 and nonmanifold == 0 else None,
        "self_intersections": "not evaluated; this is not CFD mesh acceptance",
    }


def predict_legacy(variants, library, directory):
    scored = []
    for orientation, artifact in variants.items():
        scores = score_library(artifact, library, directory, query_category=None)
        scored.append((orientation, scores))
    orientation, scores = max(scored, key=lambda pair: pair[1][0]["score"])
    top = scores[:3]
    weights = softmax([0.08 / (max(0, 1 - row["score"]) + 1e-6) for row in top])
    return {
        "cd": float(sum(w * row["cd"] for w, row in zip(weights, top))),
        "orientation": orientation,
        "top_k": [{**row, "weight": w} for row, w in zip(top, weights)],
    }


def preview(meshes, output):
    rendered = {}
    frames = {}
    length = float(np.ptp(meshes[0].points[:, 0]))
    for index, mesh in enumerate(meshes):
        points, _ = canonical_points(mesh.points, unit_scale_to_m=1)
        faces = triangle_faces(mesh)
        for view in ("right", "front"):
            art = _rasterize(points, faces, view=view, size=2048, window_l=1.3)
            rendered[index, view] = art
    canvas = Image.new("RGB", (3 * 384, 840), "white")
    draw = ImageDraw.Draw(canvas)
    for index, label in enumerate(
        ("Original", "Mirror outer height -25%", "Mirror outer height -50%")
    ):
        draw.text((index * 384 + 8, 8), label, fill="black")
        art = rendered[index, "right"]
        rgb = np.repeat(
            np.uint8(np.clip((art.depth - 0.2) / 0.3, 0, 1) * 255)[..., None], 3, axis=2
        )
        rgb[~art.mask] = 245
        rows, cols = np.nonzero(art.mask)
        whole = Image.fromarray(rgb).crop(
            (cols.min() - 10, rows.min() - 10, cols.max() + 10, rows.max() + 10)
        )
        whole.thumbnail((376, 180))
        canvas.paste(whole, (index * 384 + 4, 30))
        for row, view in enumerate(("front", "right")):
            art = rendered[index, view]
            h = (1.0 if view == "front" else -0.65) / length
            z = 0.977 / length
            scale = (2048 - 8) / 1.3
            px = (h + 1.3 / 2) * scale + 4
            py = 2048 - 1 - (z * scale + 4)
            half = 0.23 / length * scale
            box = tuple(int(v) for v in (px - half, py - half, px + half, py + half))
            if view not in frames:
                original = rendered[0, view]
                region = original.depth[box[1] : box[3], box[0] : box[2]]
                mask = original.mask[box[1] : box[3], box[0] : box[2]]
                frames[view] = (float(region[mask].min()), float(region[mask].max()))
            lo, hi = frames[view]
            rgb = np.repeat(
                np.uint8(np.clip((art.depth - lo) / max(hi - lo, 1e-8), 0, 1) * 255)[
                    ..., None
                ],
                3,
                axis=2,
            )
            rgb[~art.mask] = 245
            image = Image.fromarray(rgb).crop(box).resize((300, 300))
            y = 220 + row * 310
            canvas.paste(image, (index * 384 + 40, y))
            draw.text(
                (index * 384 + 45, y + 5), view + " detail (display only)", fill="red"
            )
    canvas.save(output)


def run(mesh_path, library_path, output):
    output.mkdir(parents=True, exist_ok=False)
    original = read_mesh(mesh_path)
    library = json.loads(library_path.read_text())
    filtered = {**library, "cars": [c for c in library["cars"] if c["id"] != "model3"]}
    variants = []
    meshes = []
    for ratio in (1.0, 0.75, 0.5):
        mesh, edit = flatten(original, ratio)
        path = output / f"model3-mirror-height-{ratio:.2f}.vtp"
        mesh.save(path)
        restored = read_mesh(path)
        np.testing.assert_array_equal(restored.points, mesh.points)
        np.testing.assert_array_equal(restored.faces, mesh.faces)
        variants.append(
            {
                "height_ratio": ratio,
                "mesh": str(path.resolve()),
                "geometry": geometry_metrics(mesh),
                "edit": edit,
            }
        )
        meshes.append(mesh)
    preview(meshes, output / "before-after.png")
    baseline = None
    for record, mesh in zip(variants, meshes):
        images = {}
        normalizations = {}
        for name, rotate in (
            ("front_as_normalized", False),
            ("front_rotated_180_z", True),
        ):
            normalized = normalize_vehicle_mesh(mesh, rotate_180_z=rotate)
            print("render legacy", record["height_ratio"], name, flush=True)
            artifact = rasterize_side(
                normalized.mesh,
                width=library["image_width"],
                height=library["image_height"],
            )
            images[name] = artifact
            normalizations[name] = normalized.info
        if baseline is None:
            baseline = images
        record["normalization"] = normalizations
        record["with_self_reference"] = predict_legacy(
            images, library, library_path.parent
        )
        record["without_self_reference"] = predict_legacy(
            images, filtered, library_path.parent
        )
        record["image_changes"] = {}
        for name, art in images.items():
            base = baseline[name]
            overlap = base.mask & art.mask
            record["image_changes"][name] = {
                "mask_changed_pixels": int(np.count_nonzero(base.mask != art.mask)),
                "depth_changed_overlap_pixels": int(
                    np.count_nonzero(
                        np.abs(base.depth[overlap] - art.depth[overlap]) > 1e-6
                    )
                ),
                "max_depth_difference_overlap": float(
                    np.max(np.abs(base.depth[overlap] - art.depth[overlap]))
                ),
                "self_similarity": compare_artifacts(base, art),
            }
        folder = output / f"height-{record['height_ratio']:.2f}"
        save_artifacts(
            images[record["with_self_reference"]["orientation"]], folder, "query"
        )
        print(
            "result",
            record["height_ratio"],
            record["with_self_reference"]["cd"],
            record["without_self_reference"]["cd"],
            flush=True,
        )
    report = {
        "input_kind": "mesh",
        "output_kind": "modified surface mesh",
        "input": str(mesh_path.resolve()),
        "source_sha256": hashlib.sha256(mesh_path.read_bytes()).hexdigest(),
        "target": {
            "purpose": "legacy predictor sensitivity",
            "modify": "both mirror outer shells in Z only; smooth blend to fixed mounting region",
        },
        "decision": {"status": "sensitivity experiment only; no CFD drag validation"},
        "method": "Exact legacy side renderer, both pose candidates, original scoring, top-k=3, temperature=.08, epsilon=1e-6, no category bonus",
        "limitations": [
            "No CFD was run; numerical Cd changes are not measured aerodynamic improvements",
            "Original surfaces are not certified CFD-ready",
        ],
        "variants": variants,
        "outputs": {
            "mesh_previews": [
                {
                    "label": f"Height ratio {r['height_ratio']} (sensitivity only)",
                    "path": r["mesh"],
                }
                for r in variants
            ]
        },
        "visual_artifacts": [
            {
                "label": "Before / after and magnified details",
                "path": str((output / "before-after.png").resolve()),
            }
        ],
        "change_summary": {
            "body_and_xy": "bitwise unchanged",
            "topology": "faces and counts preserved",
            "mirror_mount": "fixed below |Y|=.92; smooth transition to full compression at .96",
        },
        "requested_capabilities": {
            "mirror_flattening": "performed",
            "CFD_delta_cd": "not performed",
        },
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    scripts = Path(__file__).resolve().parents[2] / "mesh-watertight-repair/scripts"
    sys.path.insert(0, str(scripts))
    from html_report import write_html_report

    write_html_report(
        report,
        output / "report.html",
        "Model 3 mirror flattening: geometry and legacy predictor sensitivity",
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mesh", type=Path)
    parser.add_argument("library", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.mesh, args.library, args.output_dir)


if __name__ == "__main__":
    main()
