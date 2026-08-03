#!/usr/bin/env python3
"""Linear elastic FE assembly and solve for T4/T10 tetrahedra (small strain)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

# --- materials (SI: Pa) -------------------------------------------------------
@dataclass(frozen=True)
class Material:
    name: str
    E: float
    nu: float


MATERIALS: dict[str, Material] = {
    "steel": Material("steel", 210.0e9, 0.30),
    "aluminum": Material("aluminum", 69.0e9, 0.33),
    "titanium": Material("titanium", 116.0e9, 0.34),
    "cast_iron": Material("cast_iron", 110.0e9, 0.26),
}

# --- quadrature -----------------------------------------------------------------
# 4-point symmetric rule on the reference tetrahedron (sum w = 1).
_SQRT5 = np.sqrt(5.0)
GAUSS4_RST = np.array(
    [
        [(5.0 - _SQRT5) / 20.0] * 3,
        [(5.0 + 3.0 * _SQRT5) / 20.0, (5.0 - _SQRT5) / 20.0, (5.0 - _SQRT5) / 20.0],
        [(5.0 - _SQRT5) / 20.0, (5.0 + 3.0 * _SQRT5) / 20.0, (5.0 - _SQRT5) / 20.0],
        [(5.0 - _SQRT5) / 20.0, (5.0 - _SQRT5) / 20.0, (5.0 + 3.0 * _SQRT5) / 20.0],
    ]
)
GAUSS4_W = np.full(4, 0.25)

# T4 shape function gradients in reference coords (constant).
TET4_DN = np.array(
    [[-1.0, -1.0, -1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
)


def elasticity_D(E: float, nu: float) -> np.ndarray:
    """Isotropic elasticity matrix (Voigt: exx, eyy, ezz, gxy, gyz, gzx)."""
    c = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
    D = np.zeros((6, 6))
    D[0, 0] = D[1, 1] = D[2, 2] = 1.0 - nu
    D[0, 1] = D[0, 2] = D[1, 0] = D[1, 2] = D[2, 0] = D[2, 1] = nu
    D[3, 3] = D[4, 4] = D[5, 5] = (1.0 - 2.0 * nu) / 2.0
    return c * D


def tet10_dN_ref(r: float, s: float, t: float) -> np.ndarray:
    """(10,3) gradients of quadratic tet shape functions wrt reference coords.

    Node order (gmsh/VTK): corners 0..3, edge midpoints 4..9 on edges
    (0-1, 1-2, 2-0, 0-3, 1-3, 2-3).
    """
    l0 = 1.0 - r - s - t
    dN = np.zeros((10, 3))
    # corner nodes: N_i = L_i * (2 L_i - 1)
    dN[0] = (1.0 - 4.0 * l0) * np.array([1.0, 1.0, 1.0])  # dL0/dr,s,t = -1
    dN[1] = (4.0 * r - 1.0, 0.0, 0.0)
    dN[2] = (0.0, 4.0 * s - 1.0, 0.0)
    dN[3] = (0.0, 0.0, 4.0 * t - 1.0)
    # edge midpoints: N = 4 L_i L_j
    dN[4] = (4.0 * (l0 - r), -4.0 * r, -4.0 * r)  # edge 0-1
    dN[5] = (4.0 * s, 4.0 * r, 0.0)  # edge 1-2
    dN[6] = (-4.0 * s, 4.0 * (l0 - s), -4.0 * s)  # edge 2-0
    dN[7] = (-4.0 * t, -4.0 * t, 4.0 * (l0 - t))  # edge 0-3
    dN[8] = (4.0 * t, 0.0, 4.0 * r)  # edge 1-3
    dN[9] = (0.0, 4.0 * t, 4.0 * s)  # edge 2-3
    return dN


def _build_B(dNdx: np.ndarray) -> np.ndarray:
    """(6, 3*nn) strain-displacement matrix from (nn,3) physical gradients."""
    nn = dNdx.shape[0]
    B = np.zeros((6, 3 * nn))
    for i in range(nn):
        n = dNdx[i]
        B[0, 3 * i] = n[0]
        B[1, 3 * i + 1] = n[1]
        B[2, 3 * i + 2] = n[2]
        B[3, 3 * i] = n[1]
        B[3, 3 * i + 1] = n[0]
        B[4, 3 * i + 1] = n[2]
        B[4, 3 * i + 2] = n[1]
        B[5, 3 * i] = n[2]
        B[5, 3 * i + 2] = n[0]
    return B


def element_stiffness(nodes: np.ndarray, D: np.ndarray, order: int) -> np.ndarray:
    """(3n,3n) element stiffness; nodes is (10,3) for order 2 or (4,3) for order 1."""
    nn = nodes.shape[0]
    if order == 2:
        Ke = np.zeros((3 * nn, 3 * nn))
        for (r, s, t), w in zip(GAUSS4_RST, GAUSS4_W):
            dN = tet10_dN_ref(r, s, t)  # (10,3)
            J = dN.T @ nodes  # J[i,j] = dx_j/dr_i
            detJ = np.linalg.det(J)
            if detJ <= 0:
                raise RuntimeError(f"non-positive Jacobian {detJ:.3e} in order-2 element")
            dNdx = dN @ np.linalg.inv(J).T  # physical gradients
            B = _build_B(dNdx)
            Ke += w * detJ * (B.T @ D @ B)
        return Ke / 6.0
    J = TET4_DN.T @ nodes
    detJ = np.linalg.det(J)
    if detJ <= 0:
        raise RuntimeError(f"non-positive Jacobian {detJ:.3e} in order-1 element")
    dNdx = TET4_DN @ np.linalg.inv(J).T
    B = _build_B(dNdx)
    return detJ / 6.0 * (B.T @ D @ B)


def assemble(nodes: np.ndarray, elements: np.ndarray, D: np.ndarray, order: int):
    """Global stiffness matrix (CSR) and per-element volumes."""
    ndof = 3 * len(nodes)
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    volumes = np.zeros(len(elements))
    for e, conn in enumerate(elements):
        nn = 10 if order == 2 else 4
        xe = nodes[conn[:nn]]
        Ke = element_stiffness(xe, D, order)
        dofs = (3 * conn[:nn][:, None] + np.arange(3)[None, :]).ravel()
        if order == 2:
            r0 = s0 = t0 = 0.25
            dN = tet10_dN_ref(r0, s0, t0)
            J = dN.T @ xe
            volumes[e] = np.linalg.det(J) / 6.0
        else:
            volumes[e] = np.linalg.det(TET4_DN.T @ xe) / 6.0
        for a in range(3 * nn):
            for b in range(3 * nn):
                if Ke[a, b] != 0.0:
                    rows.append(dofs[a]); cols.append(dofs[b]); vals.append(Ke[a, b])
    K = coo_matrix((vals, (rows, cols)), shape=(ndof, ndof)).tocsr()
    return K, volumes


def apply_dirichlet(K, b: np.ndarray, fixed: np.ndarray) -> tuple:
    """Zero fixed DOFs (u=0) by row/col elimination."""
    fixed = np.unique(fixed)
    Kl = K.tolil()
    Kl[fixed, :] = 0.0
    Kl[:, fixed] = 0.0
    Kl[fixed, fixed] = 1.0
    b = b.copy()
    b[fixed] = 0.0
    return Kl.tocsr(), b


def solve_linear(nodes, elements, D, order, load_vector, fixed_dofs, mat):
    """Full linear static solve. Returns result dict."""
    ndof = 3 * len(nodes)
    K, volumes = assemble(nodes, elements, D, order)
    Kc, bc = apply_dirichlet(K, load_vector.copy(), fixed_dofs)
    u = spsolve(Kc, bc)
    # reaction forces at fixed DOFs (sign: force applied BY supports)
    r_full = K @ u - load_vector
    strain_energy = 0.5 * float(u @ (K @ u))
    return {
        "u": u,
        "K": K,
        "reactions": r_full,
        "volumes": volumes,
        "strain_energy": strain_energy,
        "fixed_dofs": fixed_dofs,
    }


def element_stress(nodes, elements, order, D, u, volumes) -> np.ndarray:
    """Node-averaged Voigt stress (6,) per node, volume-weighted."""
    n = len(nodes)
    acc = np.zeros((n, 6))
    wacc = np.zeros(n)
    for e, conn in enumerate(elements):
        nn = 10 if order == 2 else 4
        xe = nodes[conn[:nn]]
        ue = u[3 * conn[:nn][:, None] + np.arange(3)[None, :]].ravel()
        vol = volumes[e]
        sig = None
        if order == 2:
            sig = np.zeros(6)
            for (r, s, t), w in zip(GAUSS4_RST, GAUSS4_W):
                dN = tet10_dN_ref(r, s, t)
                J = dN.T @ xe
                dNdx = dN @ np.linalg.inv(J).T
                B = _build_B(dNdx)
                sig += w * (D @ (B @ ue))
        else:
            J = TET4_DN.T @ xe
            dNdx = TET4_DN @ np.linalg.inv(J).T
            B = _build_B(dNdx)
            sig = D @ (B @ ue)
        acc[conn[:nn]] += vol * sig
        wacc[conn[:nn]] += vol
    return acc / wacc[:, None]


def von_mises(sigma: np.ndarray) -> np.ndarray:
    sxx, syy, szz = sigma[:, 0], sigma[:, 1], sigma[:, 2]
    txy, tyz, tzx = sigma[:, 3], sigma[:, 4], sigma[:, 5]
    return np.sqrt(
        0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
        + 3.0 * (txy**2 + tyz**2 + tzx**2)
    )
