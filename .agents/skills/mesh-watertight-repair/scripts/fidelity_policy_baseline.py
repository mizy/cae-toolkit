"""Measure synthetic fidelity and opening invariants; never certify a mesher output."""

from __future__ import annotations

import argparse
import base64
import html
import json
import platform
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pyvista as pv
import trimesh
from agent_repair_contract import geometry_hash_from_file
from mesh_io import triangle_faces, write_vtp
from mesh_metrics import mesh_report
from sdf_closure import SdfClosureConfig, build_tsdf_closure
from source_projected_closure import ProjectionThresholds, project_closure_to_source
from source_projected_validation import (
    exact_self_intersection_report,
    repair_projected_self_intersections,
    roundtrip_validation,
    watertight_metrics,
)


def synthetic_cases() -> dict[str, dict]:
    """All coordinates are millimetres. References encode the intended shape."""
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=10.0)
    missing_cover = sphere.copy()
    missing_cover.update_faces(missing_cover.triangles_center[:, 2] < 9.8)
    missing_cover.remove_unreferenced_vertices()
    left = trimesh.creation.box(extents=(8.0, 8.0, 8.0))
    right = left.copy()
    left.apply_translation((-4.4, 0.0, 0.0))
    right.apply_translation((4.4, 0.0, 0.0))
    meshes = {
        "sphere": sphere,
        "sharp_box": trimesh.creation.box(extents=(20.0, 20.0, 20.0)),
        "missing_cover": missing_cover,
        "wide_duct": trimesh.creation.annulus(r_min=2.0, r_max=5.0, height=12.0, sections=48),
        "narrow_duct": trimesh.creation.annulus(r_min=0.6, r_max=5.0, height=12.0, sections=48),
        "separate_parts": trimesh.util.concatenate((left, right)),
        "thin_plate": trimesh.creation.box(extents=(10.0, 0.4, 10.0)),
    }
    return {
        name: {
            "source": mesh,
            "reference": sphere if name == "missing_cover" else mesh,
            "expected_components": 2 if name == "separate_parts" else 1,
            "expected_euler": 0 if "duct" in name else (4 if name == "separate_parts" else 2),
            "intent": "cap_missing_cover" if name == "missing_cover" else "preserve",
            "channel_radius_mm": {"wide_duct": 2.0, "narrow_duct": 0.6}.get(name),
        }
        for name, mesh in meshes.items()
    }


def surface_samples(points: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Seven fixed samples per triangle, with face-area weights; not Hausdorff certification."""
    triangles = points[faces]
    barycentric = np.asarray(
        [
            [1 / 3, 1 / 3, 1 / 3],
            [0.6, 0.2, 0.2],
            [0.2, 0.6, 0.2],
            [0.2, 0.2, 0.6],
            [0.5, 0.5, 0.0],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
        ]
    )
    samples = np.einsum("ki,fij->fkj", barycentric, triangles).reshape(-1, 3)
    areas = (
        np.linalg.norm(
            np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
            axis=1,
        )
        / 2
    )
    return samples, np.repeat(areas / len(barycentric), len(barycentric))


def sampled_distance(points: np.ndarray, faces: np.ndarray, target: pv.PolyData) -> dict:
    samples, weights = surface_samples(points, faces)
    _, closest = target.find_closest_cell(samples, return_closest_point=True)
    distances = np.linalg.norm(samples - closest, axis=1)
    order = np.argsort(distances)
    cumulative = np.cumsum(weights[order])
    index = min(int(np.searchsorted(cumulative, cumulative[-1] * 0.95)), len(order) - 1)
    return {
        "p95_mm": float(distances[order[index]]),
        "sampled_max_mm": float(distances.max()),
        "samples": len(samples),
    }


def channel_rays_clear(mesh: pv.PolyData, radius: float) -> dict:
    offsets = [
        (0.0, 0.0),
        (radius / 2, 0.0),
        (-radius / 2, 0.0),
        (0.0, radius / 2),
        (0.0, -radius / 2),
    ]
    clear = sum(len(mesh.ray_trace((x, y, -7.0), (x, y, 7.0))[0]) == 0 for x, y in offsets)
    return {
        "clear_rays": clear,
        "total_rays": len(offsets),
        "passed": clear == len(offsets),
    }


def run_case(
    name: str,
    case: dict,
    pitch: float,
    seal_mm: float,
    tolerance: float,
    directory: Path,
) -> dict:
    started = time.monotonic()
    directory.mkdir()
    source, reference = case["source"], case["reference"]
    write_vtp(directory / "source.vtp", source.vertices, source.faces)
    write_vtp(directory / "reference.vtp", reference.vertices, reference.faces)
    row = {
        "case": name,
        "pitch_mm": pitch,
        "requested_seal_radius_mm": seal_mm,
        "effective_seal_radius_mm": max(1, int(np.ceil(seal_mm / pitch))) * pitch,
        "intent": case["intent"],
        "intent_applied_by_operator": False,
        "scope": "TSDF/projection kernels; bypasses Stage 1 and production acceptance contract",
        "end_to_end_accepted": False,
        "output_dir": str(directory),
    }
    try:
        points, faces, _ = build_tsdf_closure(
            source.vertices,
            source.faces,
            config=SdfClosureConfig(
                pitch=pitch,
                seal_radius_voxels=max(1, int(np.ceil(seal_mm / pitch))),
                max_memory_gb=1.0,
                max_projection_distance=3 * pitch,
            ),
        )
        proxy_exact = exact_self_intersection_report(points, faces)
        if proxy_exact["passed"] is not True:
            raise RuntimeError(f"closure proxy failed exact self-intersection check: {proxy_exact['status']}")
        proxy_points = points.copy()
        points, faces, provenance, _ = project_closure_to_source(
            points,
            faces,
            source.vertices,
            source.faces,
            thresholds=ProjectionThresholds(
                max_projection_distance=3 * pitch,
                source_edge_barycentric_margin=0.0,
                orientation_vote_min_margin=0.02,
                collision_tolerance_bbox_ratio=1e-8,
            ),
        )
        points, provenance, exact = repair_projected_self_intersections(
            points,
            proxy_points,
            faces,
            provenance,
            certified_proxy_report=proxy_exact,
        )
        write_vtp(
            directory / "candidate.vtp",
            points,
            faces,
            provenance["cell_data"],
            point_data=provenance["point_data"],
        )
        candidate = pv.read(directory / "candidate.vtp")
        readback_faces = triangle_faces(candidate)
        readback_exact = exact_self_intersection_report(np.asarray(candidate.points), readback_faces)
        metrics = mesh_report(np.asarray(candidate.points), readback_faces)
        forward = sampled_distance(candidate.points, readback_faces, pv.read(directory / "reference.vtp"))
        backward = sampled_distance(reference.vertices, reference.faces, candidate)
        euler = metrics["points"] - metrics["topology"]["edges"] + metrics["triangles"]
        roundtrip = roundtrip_validation(directory / "candidate.vtp", points, faces)
        gates = {
            "watertight_oriented_nondegenerate": watertight_metrics(metrics),
            "exact_self_intersection_free": exact["passed"] is True and readback_exact["passed"] is True,
            "roundtrip": roundtrip["passed"] is True,
            "component_count_preserved": metrics["topology"]["components"]["count"] == case["expected_components"],
            "euler_characteristic_preserved": euler == case["expected_euler"],
            "sampled_bidirectional_p95": max(forward["p95_mm"], backward["p95_mm"]) <= tolerance,
            "sampled_bidirectional_max": max(forward["sampled_max_mm"], backward["sampled_max_mm"]) <= tolerance * 3,
        }
        channel = None
        if case["channel_radius_mm"] is not None:
            channel = channel_rays_clear(candidate, case["channel_radius_mm"])
            gates["channel_probe_rays_clear"] = channel["passed"]
        row.update(
            {
                "status": "measured",
                "metrics": metrics,
                "euler_characteristic": euler,
                "candidate_to_reference": forward,
                "reference_to_candidate": backward,
                "source_projection_ratio": float(np.mean(provenance["point_data"]["source_projection_applied"])),
                "channel": channel,
                "gates": gates,
                "baseline_geometry_pass": all(gates.values()),
                "exact_self_intersection_pairs": readback_exact["intersection_pairs"],
            }
        )
    except (ValueError, RuntimeError, MemoryError) as exc:
        row.update(
            {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "baseline_geometry_pass": False,
            }
        )
    row["elapsed_seconds"] = round(time.monotonic() - started, 3)
    (directory / "measurement.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    return row


def run_policy_ab(directory: Path, pitch: float) -> dict:
    """Inject decisions into the full worker; this does not evaluate an AI model."""
    source = directory / f"missing_cover-pitch-{pitch:g}" / "source.vtp"
    worker = Path(__file__).with_name("two_stage_watertight_remesh.py")
    base = [
        sys.executable,
        str(worker),
        str(source),
        "--visibility-grid",
        "128",
        "--outside-flood-grid",
        "48",
        "--sealed-exterior-grid",
        "48",
        "--voxel-pitch",
        str(pitch),
        "--preview-size",
        "256",
        "--skip-previews",
    ]
    results = {}
    decisions = {}
    for action in ("unreviewed", "cap", "preserve"):
        output = directory / f"full-worker-{action}"
        command = [*base, "--output-dir", str(output)]
        if action != "unreviewed":
            command.extend(["--policy-decisions", str(decisions[action])])
        process = subprocess.run(command, capture_output=True, text=True, check=False)
        (directory / f"full-worker-{action}.log").write_text(process.stdout + process.stderr, encoding="utf-8")
        if process.returncode != 0:
            return {
                "status": "worker_error",
                "action": action,
                "returncode": process.returncode,
                "passed": False,
            }
        report = json.loads((output / "two_stage_report.json").read_text())
        if action == "unreviewed":
            packet = json.loads((output / "ai_policy_packet.json").read_text())
            if len(packet["items"]) != 1:
                return {
                    "status": "expected_one_semantic_region",
                    "regions": len(packet["items"]),
                    "passed": False,
                }
            for choice in ("cap", "preserve"):
                decision = dict(packet["items"][0]["decision_template"])
                decision.update(
                    status="decided",
                    decision=choice,
                    semantic_label="unknown",
                    shape_prior="none",
                    confidence=1.0,
                    reason="Injected operator test input, not an AI judgment.",
                )
                path = directory / f"policy-{choice}.json"
                path.write_text(json.dumps({"decisions": [decision]}, indent=2), encoding="utf-8")
                decisions[choice] = path
        candidate = report["outputs"]["source_projected_watertight_candidate_vtp"]
        results[action] = {
            "candidate_geometry_hash": geometry_hash_from_file(Path(candidate)),
            "worker_decision": report["decision"],
            "resolved_policy": report["policy_decisions"],
        }
    same = results["cap"]["candidate_geometry_hash"] == results["preserve"]["candidate_geometry_hash"]
    resolved = all(
        len(results[action]["resolved_policy"]) == 1
        and results[action]["resolved_policy"][0].get("status") == "decided"
        and results[action]["resolved_policy"][0].get("decision") == action
        for action in ("cap", "preserve")
    )
    return {
        "status": "measured",
        "ai_model_evaluated": False,
        "decisions_resolved": resolved,
        "candidate_geometry_identical": same,
        "passed": resolved and not same,
        "limitation": "Different geometry alone would still need aperture, fidelity, and acceptance checks.",
        "results": results,
    }


def write_report(report: dict, directory: Path) -> None:
    rows = []
    for row in report["results"]:
        forward, backward = (
            row.get("candidate_to_reference", {}),
            row.get("reference_to_candidate", {}),
        )
        failed = ", ".join(key for key, passed in row.get("gates", {}).items() if not passed) or row.get("error", "—")
        rows.append(
            "<tr>"
            + "".join(
                f"<td>{html.escape(str(value))}</td>"
                for value in (
                    row["case"],
                    row["pitch_mm"],
                    row["status"],
                    row["baseline_geometry_pass"],
                    forward.get("p95_mm", "—"),
                    backward.get("p95_mm", "—"),
                    failed,
                )
            )
            + "</tr>"
        )
    pictures = []
    for name in dict.fromkeys(row["case"] for row in report["results"]):
        candidates = [row for row in report["results"] if row["case"] == name and row["status"] == "measured"]
        if not candidates:
            continue
        row = min(candidates, key=lambda item: item["pitch_mm"])
        path = Path(row["output_dir"])
        plotter = pv.Plotter(shape=(1, 2), off_screen=True, window_size=(1000, 420))
        for index, filename in enumerate(("source.vtp", "candidate.vtp")):
            plotter.subplot(0, index)
            plotter.set_background("white")
            plotter.add_mesh(pv.read(path / filename), color="lightgray", smooth_shading=False)
            plotter.add_text(filename, font_size=11, color="black")
            plotter.view_isometric()
        plotter.link_views()
        plotter.reset_camera()
        image_path = path / "comparison.png"
        plotter.screenshot(image_path)
        plotter.close()
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        pictures.append(
            f'<h2>{html.escape(name)} — pitch {row["pitch_mm"]} mm</h2><img alt="Source and candidate" src="data:image/png;base64,{encoded}">'
        )
    policy = html.escape(json.dumps(report["policy_ab"], indent=2))
    document = (
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>Fidelity and opening baseline</title>'
        "<style>body{font-family:system-ui;margin:32px;max-width:1200px}table{border-collapse:collapse;width:100%;table-layout:fixed}td,th{padding:8px;border:1px solid #ccc;overflow-wrap:anywhere}img{max-width:100%}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style>"
        "<h1>Fidelity and opening baseline</h1><p>Kernel matrix plus a separate full-worker policy A/B. "
        "No AI classification was evaluated. Kernel intent labels are expectations, not executed policy. "
        "The kernel matrix bypasses production acceptance; full-worker decisions are reported separately.</p>"
        "<p>Coordinates: mm. Area-weighted p95 ≤ 0.1 mm and sampled max ≤ 0.3 mm in both directions. "
        "Seven samples per triangle; this is not a certified Hausdorff bound. Images are inspection evidence, not a scored visual pass.</p>"
        "<table><tr><th>Case</th><th>Pitch mm</th><th>Status</th><th>Baseline pass</th><th>Output → reference p95 mm</th>"
        "<th>Reference → output p95 mm</th><th>Failed checks</th></tr>" + "".join(rows) + "</table>"
        "<h2>Full-worker cap / preserve A/B</h2><p>Injected policy inputs, not an AI accuracy test. "
        "Identical candidate hashes mean no geometric effect was observed for this fixture.</p><pre>"
        + policy
        + "</pre>"
        + "".join(pictures)
        + "</html>"
    )
    (directory / "baseline.html").write_text(document, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pitch", type=float, nargs="+", default=[1.0, 0.5])
    args = parser.parse_args()
    if any(not np.isfinite(pitch) or pitch <= 0 for pitch in args.pitch) or len(set(args.pitch)) != len(args.pitch):
        parser.error("pitches must be unique, finite positive millimetre values")
    directory = args.output_dir.expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": "fidelity_policy_baseline/v1",
        "units": "mm",
        "seed": "deterministic_synthetic_geometry",
        "scope": "kernel matrix plus separate full-worker policy A/B; not an AI benchmark or CFD acceptance",
        "tolerances": {
            "sampled_bidirectional_p95_mm": 0.1,
            "sampled_bidirectional_max_mm": 0.3,
        },
        "environment": {
            "python": platform.python_version(),
            **{
                name: version(name)
                for name in (
                    "numpy",
                    "pyvista",
                    "vtk",
                    "trimesh",
                    "libigl",
                    "scipy",
                    "scikit-image",
                )
            },
        },
        "results": [],
    }
    for name, case in synthetic_cases().items():
        for pitch in args.pitch:
            row = run_case(name, case, pitch, 1.0, 0.1, directory / f"{name}-pitch-{pitch:g}")
            report["results"].append(row)
            (directory / "baseline.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(
                f"{name} pitch={pitch:g}: {row['status']}, geometry_pass={row['baseline_geometry_pass']}",
                flush=True,
            )
    report["policy_ab"] = run_policy_ab(directory, min(args.pitch))
    (directory / "baseline.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"policy A/B: {report['policy_ab']['status']}, passed={report['policy_ab']['passed']}",
        flush=True,
    )
    write_report(report, directory)
    print(directory / "baseline.html")


if __name__ == "__main__":
    main()
