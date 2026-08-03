#!/usr/bin/env python3
"""Volume meshing via gmsh OCC: parametric geometry (box, hinge) -> tetrahedra.

Reads/writes gmsh MSH 2.2 text so element ordering is fully controlled:
TET10 (type 11) node order 0..3 corners, 4..9 edge midpoints in gmsh/VTK order
(0-1, 1-2, 2-0, 0-3, 1-3, 2-3), which matches our shape functions.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

import gmsh

TET4_TYPE = 4
TET10_TYPE = 11


@dataclass
class TetMesh:
    nodes: np.ndarray  # (n, 3) float, meters
    elements: np.ndarray  # (m, 10) int; order 1 uses first 4 columns
    order: int
    units: str
    geometry: str = ""


def build_box(gmsh, dims: tuple[float, float, float]) -> list:
    """Axis-aligned box [0..Lx]x[0..Ly]x[0..Lz]."""
    vol = gmsh.model.occ.addBox(0.0, 0.0, 0.0, dims[0], dims[1], dims[2])
    return [(3, vol)]


def build_hinge(gmsh, *, W=30.0, blade=30.0, thick=2.0, gap=2.0,
                knuckle_r=6.0, hole_r=1.5) -> list:
    """Door hinge: two blades joined by overlapping solid knuckles (one body).

    The two knuckles overlap axially so boolean fragment yields a single
    continuous solid. No pin and no screw holes are modeled: curved-hole/cut
    surfaces combined with knuckle cylinders make OCC boolean intersection
    effectively hang, so v1 keeps the simplest robust geometry.
    """
    occ = gmsh.model.occ
    x2 = blade + gap  # blade 2 starts here
    b1 = occ.addBox(0.0, 0.0, 0.0, blade, W, thick)
    b2 = occ.addBox(x2, 0.0, 0.0, blade, W, thick)
    zc = thick / 2.0
    k1 = (3, occ.addCylinder(blade, 0.0, zc, 0.0, W, 0.0, knuckle_r))
    k2 = (3, occ.addCylinder(x2, 0.0, zc, 0.0, W, 0.0, knuckle_r))
    fragments = occ.fragment([(3, b1), (3, b2), k1, k2], [])[0]
    return list(fragments)


def build_hinge_manual(*, mesh_size: float, order: int, units: str) -> TetMesh:
    """Hinge volume mesh built directly in numpy (no gmsh boolean).

    Cross-section (x-z) is a union of rectangles: blade 1, rectangular
    knuckle block, blade 2. The section is meshed with quads, extruded along
    y, and each prism split into 3 tets (order 2 adds mid-edge nodes with
    global dedup). gmsh OCC boolean meshing of knuckle cylinders hangs, so
    the v1 hinge uses this robust deterministic mesher.
    """
    if units != "mm":
        raise ValueError("hinge manual mesher supports --units mm only")
    scale = 1e-3
    size = mesh_size * scale
    rects = [
        (0.0, 26.0, 0.0, 2.0),      # blade 1
        (26.0, 36.0, -4.0, 0.0),    # knuckle block lower
        (26.0, 36.0, 0.0, 2.0),     # knuckle block middle (blade layer)
        (26.0, 36.0, 2.0, 6.0),     # knuckle block upper
        (36.0, 62.0, 0.0, 2.0),     # blade 2
    ]
    tris2d: list[tuple[int, int, int]] = []
    pts2d: list[tuple[float, float]] = []
    for x0, x1, z0, z1 in rects:
        nx = max(1, int(round((x1 - x0) / mesh_size)))
        nz = max(1, int(round((z1 - z0) / mesh_size)))
        gx = np.linspace(x0, x1, nx + 1)
        gz = np.linspace(z0, z1, nz + 1)
        base = len(pts2d)
        idx = (np.arange((nx + 1) * (nz + 1)) + base).reshape(nz + 1, nx + 1)
        for q in gz:
            for p in gx:
                pts2d.append((float(p), float(q)))
        for j in range(nz):
            for i in range(nx):
                a = idx[j, i]; b = idx[j, i + 1]
                c = idx[j + 1, i]; d = idx[j + 1, i + 1]
                tris2d.append((int(a), int(b), int(d)))
                tris2d.append((int(a), int(d), int(c)))
    ny = max(1, int(round(30.0 / mesh_size)))
    gy = np.linspace(0.0, 30.0, ny + 1)
    nodes: list[tuple[float, float, float]] = []
    node_ids: dict[tuple, int] = {}

    def nid(p):
        key = tuple(np.round(p, 12))
        if key not in node_ids:
            node_ids[key] = len(nodes)
            nodes.append(key)
        return node_ids[key]

    layer_ids = [[nid((x * scale, y * scale, z * scale)) for x, z in pts2d] for y in gy]
    elements: list[list[int]] = []
    for ly in range(ny):
        lo, hi = layer_ids[ly], layer_ids[ly + 1]
        for (a, b, c) in tris2d:
            base = [lo[a], lo[b], lo[c], hi[a], hi[b], hi[c]]
            pr = np.array([nodes[i] for i in base])
            tets = [[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5]]
            for t in tets:
                c4 = pr[t]
                d = np.linalg.det(np.column_stack([c4[1] - c4[0], c4[2] - c4[0], c4[3] - c4[0]]))
                if d < 0:
                    t = [t[0], t[2], t[1], t[3]]
                conn = [base[i] for i in t]
                edges = [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)]
                mid_ids = []
                for e in edges:
                    mid = (np.array(nodes[conn[e[0]]]) + np.array(nodes[conn[e[1]]])) / 2.0
                    mid_ids.append(nid(tuple(np.round(mid, 12))))
                elements.append(conn + mid_ids)
    return TetMesh(nodes=np.array(nodes), elements=np.array(elements),
                   order=order, units=units, geometry="hinge")


def build_tet_mesh(geometry: str, mesh_size: float, order: int, units: str) -> TetMesh:
    """Build a tetrahedral mesh for a parametric geometry spec.

    geometry: "box:Lx,Ly,Lz" or "hinge" (dimensions in `units`).
    """
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        name, _, params = geometry.partition(":")
        if name == "box":
            dims = tuple(float(v) for v in params.split(","))
            if len(dims) != 3 or min(dims) <= 0:
                raise ValueError(f"bad box spec: {geometry!r}, need Lx,Ly,Lz > 0")
            bodies = build_box(gmsh, dims)
        elif name == "hinge":
            return build_hinge_manual(mesh_size=mesh_size, order=order, units=units)
        else:
            raise ValueError(f"unknown geometry: {name!r} (box:Lx,Ly,Lz | hinge)")
        gmsh.model.occ.synchronize()
        # scale user units to SI meters
        scale = 1.0 if units == "m" else 1e-3 if units == "mm" else None
        if scale is None:
            raise ValueError(f"unsupported units: {units!r}")
        gmsh.model.occ.synchronize()
        gmsh.option.setNumber("Mesh.MeshSizeFactor", 1.0)
        for dim, tag in gmsh.model.getEntities(0):
            gmsh.model.mesh.setSize([(0, tag)], mesh_size * scale)
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.generate(3)
        if order == 2:
            gmsh.model.mesh.setOrder(2)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        import tempfile
        tmp = Path(tempfile.gettempdir()) / f"structural_solver_{abs(hash(geometry))}.msh"
        gmsh.write(str(tmp))
        nodes, elements, el_order = read_msh22(tmp)
        return TetMesh(nodes=nodes, elements=elements, order=el_order, units=units, geometry=geometry)
    finally:
        gmsh.finalize()


def read_msh22(path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    """Parse gmsh MSH 2.2: collect TET4 (type 4) and TET10 (type 11) elements."""
    nodes: list[tuple[int, float, float, float]] = []
    tets4: list[list[int]] = []
    tets10: list[list[int]] = []
    with open(path, encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line == "$Nodes":
            n = int(lines[i + 1]); i += 2
            for k in range(n):
                pid, x, y, z = lines[i + k].split()
                nodes.append((int(pid), float(x), float(y), float(z)))
            i += n
        elif line == "$Elements":
            n = int(lines[i + 1]); i += 2
            for k in range(n):
                parts = lines[i + k].split()
                el_type = int(parts[1])
                ntags = int(parts[2])
                conn = [int(v) for v in parts[3 + ntags:]]
                if el_type == TET10_TYPE:
                    tets10.append(conn)
                elif el_type == TET4_TYPE:
                    tets4.append(conn)
            i += n
        i += 1
    node_array = np.zeros((len(nodes), 3), dtype=float)
    id_map: dict[int, int] = {}
    for j, (pid, x, y, z) in enumerate(nodes):
        node_array[j] = (x, y, z)
        id_map[pid] = j
    if tets10:
        order, raw = 2, tets10
    elif tets4:
        order, raw = 1, tets4
    else:
        raise RuntimeError(f"no tetrahedra found in {path}")
    elements = np.zeros((len(raw), 10), dtype=np.int64)
    for j, conn in enumerate(raw):
        elements[j, : len(conn)] = [id_map[p] for p in conn]
    if order == 2:
        # gmsh orders tet10 edge midpoints (0-1,1-2,2-0,0-3,2-3,1-3);
        # solver/VTK convention is (0-1,1-2,2-0,0-3,1-3,2-3) -> swap last two.
        elements[:, [8, 9]] = elements[:, [9, 8]]
    return node_array, elements, order


def write_msh22(nodes: np.ndarray, elements: np.ndarray, order: int, path) -> None:
    """Serialize (nodes, elements) to gmsh MSH 2.2 with TET10 (type 11).

    Our elements use solver/VTK mid-edge order (0-1,1-2,2-0,0-3,1-3,2-3);
    gmsh wants (0-1,1-2,2-0,0-3,2-3,1-3) so the last two are swapped back.
    """
    lines = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat", "$Nodes", str(len(nodes))]
    for i, p in enumerate(nodes, 1):
        lines.append(f"{i} {p[0]:.17g} {p[1]:.17g} {p[2]:.17g}")
    lines += ["$EndNodes", "$Elements", str(len(elements))]
    for i, e in enumerate(elements, 1):
        ee = e.copy()
        if order == 2:
            ee[[8, 9]] = ee[[9, 8]]
        ntag = 2
        lines.append(f"{i} {11 if order == 2 else 4} {ntag} 1 1 " + " ".join(str(int(j) + 1) for j in ee))
    lines += ["$EndElements"]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
