"""Candidate print files, full geometry readback, and explicit external slicer execution."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import trimesh
from mesh_metrics import mesh_report
from print_checks import face_distances
from print_geometry import material_valid, orient_material
from source_projected_validation import exact_self_intersection_report


def export_print_files(
    mesh: trimesh.Trimesh, directory: Path, tolerance: float
) -> list[dict]:
    rows = []
    mesh = mesh.copy()
    mesh.units = "mm"
    for extension in ("stl", "3mf"):
        path = directory / f"print_candidate.{extension}"
        mesh.export(path)
        loaded = trimesh.load(path, process=False)
        if isinstance(loaded, trimesh.Scene):
            loaded = loaded.to_geometry()
        loaded.merge_vertices()
        # Audit stored orientations before any normal repair.
        metrics = mesh_report(loaded.vertices, loaded.faces)
        _, shells = orient_material(loaded)
        exact = exact_self_intersection_report(loaded.vertices, loaded.faces)
        _, forward = face_distances(loaded, mesh)
        _, backward = face_distances(mesh, loaded)
        displacement = max(forward["sampled_max_mm"], backward["sampled_max_mm"])
        signs = metrics["volume"]["component_signed"]
        expected_signs = [row["signed_volume_mm3"] for row in shells]
        orientation = len(signs) == len(expected_signs) and np.array_equal(
            np.sign(signs), np.sign(expected_signs)
        )
        passed = (
            material_valid(metrics, shells)
            and orientation
            and exact.get("passed") is True
            and displacement <= tolerance
            and len(loaded.faces) == len(mesh.faces)
        )
        rows.append(
            {
                "path": str(path),
                "format": extension,
                "coordinate_units": "mm",
                "unit_metadata": "mm"
                if extension == "3mf"
                else "STL has no unit metadata; import as mm",
                "passed": bool(passed),
                "max_sampled_readback_error_mm": displacement,
                "faces": len(loaded.faces),
                "exact_intersection_pairs": exact.get("intersection_pairs"),
                "stored_material_orientation": bool(orientation),
            }
        )
    return rows


def run_slicer(
    executable: Path | None, profile: Path | None, mesh: Path, directory: Path
) -> dict:
    if executable is None:
        return {
            "status": "not_configured",
            "passed": False,
            "machine_toolpath_validated": False,
        }
    if not executable.is_file() or profile is None or not profile.is_file():
        raise ValueError(
            "external PrusaSlicer-compatible CLI requires an executable and an explicit printer profile"
        )
    output = directory / "print_candidate.gcode"
    output.unlink(missing_ok=True)
    command = [
        str(executable.resolve()),
        "--export-gcode",
        "--no-binary-gcode",
        "--load",
        str(profile.resolve()),
        "--output",
        str(output),
        str(mesh),
    ]
    try:
        process = subprocess.run(
            command, capture_output=True, text=True, timeout=300, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        (directory / "slicer.log").write_text(str(exc), encoding="utf-8")
        return {
            "status": "failed",
            "passed": False,
            "machine_toolpath_validated": False,
            "reason": str(exc),
            "command": command,
        }
    (directory / "slicer.log").write_text(
        process.stdout + process.stderr, encoding="utf-8"
    )
    extrusion = 0
    if output.is_file():
        with output.open(encoding="utf-8", errors="replace") as stream:
            extrusion = sum(
                line.startswith(("G1 ", "G01 ")) and " E" in line for line in stream
            )
    passed = process.returncode == 0 and extrusion > 0
    return {
        "status": "completed",
        "passed": passed,
        "command": command,
        "returncode": process.returncode,
        "extrusion_commands": extrusion,
        "gcode": str(output) if output.exists() else None,
        "machine_toolpath_validated": passed,
        "scope": "CLI completion and nonempty extrusion toolpath; no physical print certification",
    }
