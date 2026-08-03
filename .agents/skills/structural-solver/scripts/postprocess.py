#!/usr/bin/env python3
"""Post-processing: fields, static stiffness, VTP outputs, offscreen renders."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvista as pv

PV_QUAD_TET = getattr(pv.CellType, "QUADRATIC_TETRA", 21)


def to_vtk_grid(nodes: np.ndarray, elements: np.ndarray, order: int) -> pv.UnstructuredGrid:
    """Build a pyvista UnstructuredGrid (T4 or T10) from our arrays."""
    if order == 2:
        cells = np.hstack([np.full((len(elements), 1), 10), elements]).astype(np.int64)
        ctype = np.full(len(elements), PV_QUAD_TET, dtype=np.uint8)
    else:
        cells = np.hstack([np.full((len(elements), 1), 4), elements[:, :4]]).astype(np.int64)
        ctype = np.full(len(elements), pv.CellType.TETRA, dtype=np.uint8)
    return pv.UnstructuredGrid(cells.ravel(), ctype, nodes)


def compute_stiffness(u: np.ndarray, load_nodes: np.ndarray, force: np.ndarray) -> tuple[float, float]:
    """Load-point displacement delta along force direction and stiffness k = |F|/delta."""
    n = force / np.linalg.norm(force)
    delta = float(np.mean(u[load_nodes] @ n))
    if abs(delta) < 1e-30:
        return 0.0, float("inf")
    return delta, float(np.linalg.norm(force) / delta)


def write_vtp_surface(grid: pv.UnstructuredGrid, path: Path) -> None:
    """Surface PolyData as .vtp (ParaView-friendly); keeps all point arrays."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    grid.extract_surface(algorithm="dataset_surface").save(str(path))


def write_vtu(grid: pv.UnstructuredGrid, path: Path) -> None:
    """Full-volume .vtu (tetrahedra preserved)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(str(path))


def read_back_audit(path: Path, expected_arrays: list[str], order: int, expect_volume: bool) -> dict:
    """Read artifact back and verify geometry/topology/fields survived serialization."""
    grid = pv.read(str(path))
    npts, ncell = grid.n_points, grid.n_cells
    missing = [a for a in expected_arrays if a not in grid.array_names]
    if expect_volume:
        cell_type_ok = ncell > 0 and grid.celltypes[0] == (PV_QUAD_TET if order == 2 else pv.CellType.TETRA)
    else:
        cell_type_ok = ncell > 0 and isinstance(grid, pv.PolyData) and grid.n_faces > 0
    return {
        "points": int(npts),
        "cells": int(ncell),
        "arrays_present": all(a in grid.array_names for a in expected_arrays),
        "missing_arrays": missing,
        "cell_type_preserved": cell_type_ok,
        "passed": npts > 0 and ncell > 0 and all(a in grid.array_names for a in expected_arrays) and cell_type_ok,
    }


def render_overlay(original: pv.UnstructuredGrid, deformed: pv.UnstructuredGrid, out_png: Path,
                   scale: float, title: str = "") -> None:
    """Original wireframe + deformed mesh colored by displacement magnitude."""
    p = pv.Plotter(off_screen=True, window_size=(1500, 1000))
    p.add_mesh(original, style="wireframe", color="lightgray", opacity=0.35, line_width=0.6)
    p.add_mesh(deformed, scalars="displacement_magnitude", cmap="jet",
               show_edges=False, scalar_bar_args={"title": "|u| (scaled)"})
    if title:
        p.add_text(title, font_size=12)
    p.screenshot(str(out_png))
    p.close()


def render_field(grid: pv.UnstructuredGrid, out_png: Path, field: str, title: str) -> None:
    p = pv.Plotter(off_screen=True, window_size=(1500, 1000))
    p.add_mesh(grid, scalars=field, cmap="jet", show_edges=False,
               scalar_bar_args={"title": field.replace("_", " ")})
    p.add_text(title, font_size=12)
    p.screenshot(str(out_png))
    p.close()
