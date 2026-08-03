#!/usr/bin/env python3
"""Boundary conditions: axis-aligned face selection, force loads."""
from __future__ import annotations

import numpy as np

FACE_AXIS = {"x": 0, "y": 1, "z": 2}


def face_nodes(nodes: np.ndarray, face: str) -> np.ndarray:
    """Node indices on an axis-aligned face: x_min/x_max/y_min/y_max/z_min/z_max."""
    if len(face) < 2 or face[0] not in FACE_AXIS or face[1:] not in ("_min", "_max"):
        raise ValueError(f"bad face spec {face!r}, use x_min|x_max|y_min|y_max|z_min|z_max")
    axis = FACE_AXIS[face[0]]
    coords = nodes[:, axis]
    tol = 1e-7 * float(np.ptp(coords)) + 1e-14
    if face.endswith("_min"):
        mask = coords <= coords.min() + tol
    else:
        mask = coords >= coords.max() - tol
    return np.flatnonzero(mask)


def parse_load(spec: str) -> tuple[str, np.ndarray]:
    """Parse 'FACE: fx=..,fy=..,fz=..' e.g. 'x_max: fz=-100' (total force)."""
    face, _, body = spec.partition(":")
    face = face.strip()
    if not face or not body:
        raise ValueError(f"bad load spec {spec!r}, use 'FACE: fx=0,fy=0,fz=-100'")
    force = np.zeros(3)
    for tok in body.split(","):
        tok = tok.strip()
        key, _, val = tok.partition("=")
        if key not in ("fx", "fy", "fz"):
            raise ValueError(f"unknown load component {key!r} in {spec!r}")
        force[{"fx": 0, "fy": 1, "fz": 2}[key]] = float(val)
    if not np.any(force):
        raise ValueError(f"zero force in load spec {spec!r}")
    return face, force


def build_load_vector(nodes: np.ndarray, ndof: int, loads: list[str]) -> tuple[np.ndarray, list[dict]]:
    """Assemble nodal force vector; forces on a face are split evenly among its nodes."""
    b = np.zeros(ndof)
    records = []
    for spec in loads:
        face, force = parse_load(spec)
        idx = face_nodes(nodes, face)
        if len(idx) == 0:
            raise RuntimeError(f"load face {face!r} has no nodes")
        per = force / len(idx)
        for k in range(3):
            b[3 * idx + k] += per[k]
        records.append({"face": face, "force_total": force.tolist(), "nodes": int(len(idx))})
    return b, records
