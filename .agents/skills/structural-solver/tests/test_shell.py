"""Thin flat shell analytical checks, including the real CLI and VTP output."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pyvista as pv
import pytest

from assembly import Material
from shell import read_midsurface, solve_shell


def panel(length, width, nx, ny):
    x, y = np.meshgrid(np.linspace(0, length, nx + 1), np.linspace(0, width, ny + 1))
    points = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
    i = np.arange(y.size).reshape(y.shape)[:-1, :-1].ravel()
    a, b, c, d = i, i + 1, i + (nx + 1), i + (nx + 2)
    cells = np.vstack((np.column_stack((a, b, d)), np.column_stack((a, d, c))))
    return pv.PolyData(points, np.column_stack((np.full(len(cells), 3), cells)).ravel())


def test_thin_shell_cli_cantilever(tmp_path):
    # nu=0 gives cylindrical beam-strip bending. The small tip force keeps w << t.
    length, width, thickness = 200, 20, .4  # mm
    modulus, force = 2000, .000004  # MPa, N
    src = tmp_path / "midsurface.vtp"
    panel(length, width, 60, 6).save(src)
    cli = Path(__file__).resolve().parents[1] / "scripts" / "solve.py"
    subprocess.run(
        [sys.executable, str(cli), "shell", "--mesh", str(src), "--thickness", str(thickness),
         "--units", "mm", "--young-modulus", str(modulus), "--poisson-ratio", "0",
         "--fix", "x_min", "--load", f"x_max: fz={-force}", "--output-dir", str(tmp_path)],
        check=True,
    )
    report = json.loads((tmp_path / "shell_report.json").read_text())
    expected = force * length**3 / (3 * modulus * width * thickness**3 / 12)
    assert report["results"]["field_summary"]["load_edge_mean_displacement"] == pytest.approx(expected, rel=.05)
    assert report["results"]["reaction_sum"] == pytest.approx([0, 0, force], abs=1e-8)
    assert all(gate["passed"] for gate in report["gates"].values())
    mesh = pv.read(tmp_path / "shell_mesh.vtp")
    deformed = pv.read(tmp_path / "shell_deformed.vtp")
    assert deformed.points == pytest.approx(mesh.points + mesh["displacement"])
    assert np.max(mesh.cell_data["von_mises_top"]) > 0
    assert mesh.cell_data["von_mises_top"] == pytest.approx(mesh.cell_data["von_mises_bottom"])


def test_shell_membrane_on_rotated_surface(tmp_path):
    # A rotated rectangular patch, with explicit point masks identifying edges.
    length, width, thickness = .2, .04, .0004
    mat = Material("test", 2e9, 0)
    surface = panel(length, width, 20, 4)
    start = np.isclose(surface.points[:, 0], 0)
    tip = np.isclose(surface.points[:, 0], length)
    surface.point_data["clamp"] = start.astype(np.uint8)
    surface.point_data["pull"] = tip.astype(np.uint8)
    surface.points[:, [0, 1, 2]] = surface.points[:, [2, 1, 0]]
    mesh = tmp_path / "rotated.vtp"
    surface.save(mesh)
    poly, points, triangles = read_midsurface(mesh)
    force = 3.0
    result = solve_shell(poly, points, triangles, thickness, mat, ["clamp"], [f"pull: fz={force}"])
    expected = force * length / (mat.E * width * thickness)
    assert result["displacement"][tip, 2] == pytest.approx(expected, rel=1e-7)
    assert result["reaction_N"] == pytest.approx([0, 0, -force], abs=1e-8)
    assert result["element_stress"] == pytest.approx(force / (width * thickness), rel=1e-6)
    assert result["energy_balance"] < 1e-8


def test_shell_rejects_nonplanar_surface(tmp_path):
    surface = panel(.2, .04, 4, 2)
    surface.points[-1, 2] = .005
    mesh = tmp_path / "bent.vtp"
    surface.save(mesh)
    poly, points, triangles = read_midsurface(mesh)
    with pytest.raises(ValueError, match="must be planar"):
        solve_shell(poly, points, triangles, .001, Material("test", 2e9, .3), ["x_min"], ["x_max: fz=-1"])
