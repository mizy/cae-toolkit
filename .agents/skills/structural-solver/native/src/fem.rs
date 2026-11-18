//! Linear elastic FEM: T10/T4 element stiffness, assembly, faer sparse LU solve,
//! stress recovery. Mirrors the Python reference solver exactly (same quadrature,
//! same node ordering) so results agree to floating point precision.

use faer::dyn_stack::MemStack;
use faer::prelude::*;
use faer::linalg::lu::partial_pivoting::factor::PartialPivLuParams;
use faer::sparse::linalg::lu::{factorize_symbolic_lu, LuSymbolicParams, NumericLu};
use faer::sparse::{SparseColMat, Triplet};
use faer::{Auto, Spec};
use rayon::prelude::*;

use crate::mesh::TetMesh;

const SQRT5: f64 = 2.23606797749979_f64;
const GAUSS4_RST: [[f64; 3]; 4] = [
    [(5.0 - SQRT5) / 20.0; 3],
    [(5.0 + 3.0 * SQRT5) / 20.0, (5.0 - SQRT5) / 20.0, (5.0 - SQRT5) / 20.0],
    [(5.0 - SQRT5) / 20.0, (5.0 + 3.0 * SQRT5) / 20.0, (5.0 - SQRT5) / 20.0],
    [(5.0 - SQRT5) / 20.0, (5.0 - SQRT5) / 20.0, (5.0 + 3.0 * SQRT5) / 20.0],
];

const TET4_DN: [[f64; 3]; 4] = [
    [-1.0, -1.0, -1.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
];

/// (10,3) gradients of quadratic tet shape functions at (r,s,t).
/// Node order: corners 0..3, edge midpoints 4..9 on edges
/// (0-1, 1-2, 2-0, 0-3, 1-3, 2-3) — VTK/solver convention.
#[inline]
fn tet10_dn_ref(r: f64, s: f64, t: f64) -> [[f64; 3]; 10] {
    let l0 = 1.0 - r - s - t;
    [
        [(1.0 - 4.0 * l0); 3],                  // 0
        [4.0 * r - 1.0, 0.0, 0.0],              // 1
        [0.0, 4.0 * s - 1.0, 0.0],              // 2
        [0.0, 0.0, 4.0 * t - 1.0],              // 3
        [4.0 * (l0 - r), -4.0 * r, -4.0 * r],   // 4: edge 0-1
        [4.0 * s, 4.0 * r, 0.0],                // 5: edge 1-2
        [-4.0 * s, 4.0 * (l0 - s), -4.0 * s],   // 6: edge 2-0
        [-4.0 * t, -4.0 * t, 4.0 * (l0 - t)],   // 7: edge 0-3
        [4.0 * t, 0.0, 4.0 * r],                // 8: edge 1-3
        [0.0, 4.0 * t, 4.0 * s],                // 9: edge 2-3
    ]
}

/// Isotropic elasticity matrix (Voigt: exx, eyy, ezz, gxy, gyz, gzx).
fn elasticity_d(e: f64, nu: f64) -> [[f64; 6]; 6] {
    let c = e / ((1.0 + nu) * (1.0 - 2.0 * nu));
    let mut d = [[0.0; 6]; 6];
    for i in 0..3 {
        for j in 0..3 {
            d[i][j] = if i == j { 1.0 - nu } else { nu };
        }
    }
    for i in 3..6 {
        d[i][i] = (1.0 - 2.0 * nu) / 2.0;
    }
    for i in 0..6 {
        for j in 0..6 {
            d[i][j] *= c;
        }
    }
    d
}

/// (6, 3n) strain-displacement matrix from physical gradients (n,3).
fn build_b(dndx: &[[f64; 3]], nn: usize) -> Vec<[f64; 30]> {
    let mut b = vec![[0.0; 30]; 6];
    for i in 0..nn {
        let n = dndx[i];
        b[0][3 * i] = n[0];
        b[1][3 * i + 1] = n[1];
        b[2][3 * i + 2] = n[2];
        b[3][3 * i] = n[1];
        b[3][3 * i + 1] = n[0];
        b[4][3 * i + 1] = n[2];
        b[4][3 * i + 2] = n[1];
        b[5][3 * i] = n[2];
        b[5][3 * i + 2] = n[0];
    }
    b
}

/// (3n,3n) element stiffness; xe is (10,3) for order 2, (4,3) for order 1.
pub fn element_stiffness_pub(xe: &[[f64; 3]], e: f64, nu: f64, order: u8) -> (Vec<[f64; 30]>, f64) {
    let d = elasticity_d(e, nu);
    element_stiffness(xe, &d, order)
}

fn element_stiffness(xe: &[[f64; 3]], d: &[[f64; 6]; 6], order: u8) -> (Vec<[f64; 30]>, f64) {
    let _nn = if order == 2 { 10 } else { 4 };
    let mut ke = vec![[0.0; 30]; 30];
    let mut vol = 0.0;
    if order == 2 {
        for (gp, w) in GAUSS4_RST.iter().zip([0.25; 4]) {
            let dn = tet10_dn_ref(gp[0], gp[1], gp[2]);
            let mut j = [[0.0; 3]; 3];
            for k in 0..10 {
                for i in 0..3 {
                    for m in 0..3 {
                        j[i][m] += dn[k][i] * xe[k][m];
                    }
                }
            }
            let detj = det3(&j);
            if detj <= 0.0 {
                panic!("non-positive Jacobian {detj:.3e}");
            }
            let inv = inv3(&j);
            let mut dndx = [[0.0; 3]; 10];
            for k in 0..10 {
                for m in 0..3 {
                    // physical gradient = dN @ inv(J).T
                    dndx[k][m] = dn[k][0] * inv[m][0] + dn[k][1] * inv[m][1] + dn[k][2] * inv[m][2];
                }
            }
            let b = build_b(&dndx, 10);
            let mut bt_d = [[0.0; 6]; 30];
            for a in 0..30 {
                for q in 0..6 {
                    let mut acc = 0.0;
                    for p in 0..6 {
                        acc += b[p][a] * d[p][q];
                    }
                    bt_d[a][q] = acc;
                }
            }
            for a in 0..30 {
                for j in 0..30 {
                    let mut acc = 0.0;
                    for q in 0..6 {
                        acc += bt_d[a][q] * b[q][j];
                    }
                    ke[a][j] += w * detj * acc;
                }
            }
            vol += w * detj;
        }
        vol /= 6.0;
        for a in 0..30 {
            for b in 0..30 {
                ke[a][b] /= 6.0;
            }
        }
    } else {
        let mut j = [[0.0; 3]; 3];
        for k in 0..4 {
            for i in 0..3 {
                for m in 0..3 {
                    j[i][m] += TET4_DN[k][i] * xe[k][m];
                }
            }
        }
        let detj = det3(&j);
        if detj <= 0.0 {
            panic!("non-positive Jacobian {detj:.3e}");
        }
        let inv = inv3(&j);
        let mut dndx = [[0.0; 3]; 4];
        for k in 0..4 {
            for m in 0..3 {
                dndx[k][m] = TET4_DN[k][0] * inv[m][0] + TET4_DN[k][1] * inv[m][1] + TET4_DN[k][2] * inv[m][2];
            }
        }
        let b = build_b(&dndx, 4);
        let mut bt_d = [[0.0; 6]; 12];
        for a in 0..12 {
            for q in 0..6 {
                let mut acc = 0.0;
                for p in 0..6 {
                    acc += b[p][a] * d[p][q];
                }
                bt_d[a][q] = acc;
            }
        }
        for a in 0..12 {
            for j in 0..12 {
                let mut acc = 0.0;
                for q in 0..6 {
                    acc += bt_d[a][q] * b[q][j];
                }
                ke[a][j] = detj * acc / 6.0;
            }
        }
        vol = detj / 6.0;
    }
    (ke, vol)
}

fn det3(m: &[[f64; 3]; 3]) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

fn inv3(m: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let det = det3(m);
    let inv_det = 1.0 / det;
    let mut r = [[0.0; 3]; 3];
    r[0][0] = (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv_det;
    r[0][1] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv_det;
    r[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_det;
    r[1][0] = (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv_det;
    r[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_det;
    r[1][2] = (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv_det;
    r[2][0] = (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv_det;
    r[2][1] = (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv_det;
    r[2][2] = (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv_det;
    r
}

pub struct SolveResult {
    pub u: Vec<f64>,           // ndof, meters
    pub sigma: Vec<[f64; 6]>,  // (n,6) node-averaged Voigt stress, Pa
    pub volumes: Vec<f64>,
    pub strain_energy: f64,
    pub reaction_sum: [f64; 3],
    pub backend: String,
    pub time_assembly_s: f64,
    pub time_lu_s: f64,
    pub time_stress_s: f64,
}

pub fn solve_linear(
    mesh: &TetMesh,
    e: f64,
    nu: f64,
    fixed_dofs: &std::collections::HashSet<usize>,
    load_vector: &[f64],
    backend: &str,
) -> SolveResult {
    let n = mesh.nodes.len();
    let ndof = 3 * n;
    let d = elasticity_d(e, nu);
    let t0 = std::time::Instant::now();

    // parallel element stiffness -> triplets
    let nn = if mesh.order == 2 { 10 } else { 4 };
    let rows: Vec<[usize; 10]> = mesh.elements.clone();
    let cell_trips: Vec<Vec<(usize, usize, f64)>> = rows
        .par_iter()
        .map(|conn| {
            let xe: Vec<[f64; 3]> = conn[..nn].iter().map(|&i| mesh.nodes[i]).collect();
            let (ke, _) = element_stiffness(&xe, &d, mesh.order);
            let dofs: Vec<usize> = conn[..nn]
                .iter()
                .flat_map(|&i| [3 * i, 3 * i + 1, 3 * i + 2])
                .collect();
            let mut trips = Vec::with_capacity(3 * nn * 3 * nn);
            for a in 0..3 * nn {
                for b in 0..3 * nn {
                    let v = ke[a][b];
                    if v != 0.0 {
                        trips.push((dofs[a], dofs[b], v));
                    }
                }
            }
            trips
        })
        .collect();
    let mut trips: Vec<(usize, usize, f64)> = cell_trips.into_iter().flatten().collect();
    // sort + merge duplicates (row-major); keep full K for reactions/energy
    trips.sort_unstable_by_key(|a| (a.0, a.1));
    let mut merged: Vec<(usize, usize, f64)> = Vec::with_capacity(trips.len());
    for (r, c, v) in trips {
        if let Some(last) = merged.last_mut()
            && last.0 == r && last.1 == c {
                last.2 += v;
                continue;
            }
        merged.push((r, c, v));
    }
    // modified matrix for the solve: zero fixed rows/cols, identity on diagonal
    let mut trips_mod: Vec<(usize, usize, f64)> = merged
        .iter()
        .filter(|(r, c, _)| !fixed_dofs.contains(r) && !fixed_dofs.contains(c))
        .cloned()
        .collect();
    for &fd in fixed_dofs.iter() {
        trips_mod.push((fd, fd, 1.0));
    }
    trips_mod.sort_unstable_by_key(|a| (a.0, a.1));
    let time_assembly_s = t0.elapsed().as_secs_f64();

    // solve
    let t1 = std::time::Instant::now();
    let u: Vec<f64> = if backend == "gpu" {
        let mut row_ptr = vec![0u32; ndof+1];
        for &(r, _, _) in &trips_mod { row_ptr[r+1] += 1; }
        for i in 0..ndof { row_ptr[i+1] += row_ptr[i]; }
        let mut rhs = load_vector.to_vec();
        for &fd in fixed_dofs { rhs[fd] = 0.0; }
        let system = structural_core::LinearSystem { row_ptr, col_idx: trips_mod.iter().map(|t|t.1 as u32).collect(), values: trips_mod.iter().map(|t|t.2).collect(), rhs };
        let result = pollster::block_on(structural_core::solve(&system,"gpu",1e-5,6000)).expect("GPU solve failed");
        assert!(result.converged,"GPU did not converge");
        result.solution
    } else {
    let faer_trips: Vec<Triplet<usize, usize, f64>> = trips_mod
        .iter()
        .map(|(r, c, v)| Triplet::new(*r, *c, *v))
        .collect();
    let mat = SparseColMat::try_new_from_triplets(ndof, ndof, &faer_trips)
        .map_err(|e| format!("faer triplet error {e:?}"))
        .expect("matrix build failed");
    let symbolic = factorize_symbolic_lu(mat.symbolic(), LuSymbolicParams::default())
        .expect("symbolic LU failed");
    let mut numeric = NumericLu::<usize, f64>::new();
    let mut mem = vec![std::mem::MaybeUninit::<u8>::uninit(); 256 * 1024 * 1024];
    let mut stack = MemStack::new(&mut mem);
    let lu = symbolic
        .factorize_numeric_lu(&mut numeric, mat.as_ref(), faer::Par::Seq, stack,
                              Spec::<PartialPivLuParams, f64>::new(<PartialPivLuParams as Auto<f64>>::auto()))
        .expect("numeric LU failed");
    let mut rhs = Mat::<f64>::zeros(ndof, 1);
    for i in 0..ndof {
        rhs[(i, 0)] = if fixed_dofs.contains(&i) { 0.0 } else { load_vector[i] };
    }
    lu.solve_in_place_with_conj(faer::Conj::No, rhs.as_mut(), faer::Par::Seq, stack);
    (0..ndof).map(|i| rhs[(i, 0)]).collect()
    };
    let time_lu_s = t1.elapsed().as_secs_f64();

    // reactions at fixed dofs from the full K: r = K u - b
    let t2 = std::time::Instant::now();
    let mut reaction_sum = [0.0f64; 3];
    {
        let mut ku = vec![0.0f64; ndof];
        for (r, c, v) in &merged {
            ku[*r] += v * u[*c];
        }
        for &fd in fixed_dofs.iter() {
            let r = ku[fd] - load_vector[fd];
            reaction_sum[fd % 3] += r;
        }
    }

    // node-averaged stress (volume weighted, all 10 nodes) + volumes
    let mut acc = vec![[0.0f64; 6]; n];
    let mut wacc = vec![0.0f64; n];
    let mut volumes = Vec::with_capacity(rows.len());
    for conn in &rows {
        let xe: Vec<[f64; 3]> = conn[..nn].iter().map(|&i| mesh.nodes[i]).collect();
        let (_, vol) = element_stiffness(&xe, &d, mesh.order);
        volumes.push(vol);
        let ue: Vec<f64> = conn[..nn]
            .iter()
            .flat_map(|&i| [u[3 * i], u[3 * i + 1], u[3 * i + 2]])
            .collect();
        let sig = element_stress(&xe, &d, mesh.order, &ue);
        for k in 0..nn {
            let i = conn[k];
            for q in 0..6 {
                acc[i][q] += vol * sig[k][q];
            }
            wacc[i] += vol;
        }
    }
    let mut sigma = vec![[0.0f64; 6]; n];
    for i in 0..n {
        if wacc[i] > 0.0 {
            for q in 0..6 {
                sigma[i][q] = acc[i][q] / wacc[i];
            }
        }
    }
    // strain energy: 0.5 * u^T K u  (K diagonal fixed rows contribute 0 since u=0 there)
    let mut energy = 0.0;
    for (r, c, v) in &merged {
        energy += v * u[*r] * u[*c];
    }
    let time_stress_s = t2.elapsed().as_secs_f64();

    SolveResult {
        u,
        sigma,
        volumes,
        strain_energy: 0.5 * energy,
        reaction_sum,
        backend: backend.into(),
        time_assembly_s,
        time_lu_s,
        time_stress_s,
    }
}

/// Element-averaged Voigt stress (6,) at the element centroid/average of gauss points.
fn element_stress(xe: &[[f64; 3]], d: &[[f64; 6]; 6], order: u8, ue: &[f64]) -> [[f64; 6]; 10] {
    let _nn = if order == 2 { 10 } else { 4 };
    let mut sig = [[0.0; 6]; 10];
    if order == 2 {
        let mut s = [0.0; 6];
        for (gp, w) in GAUSS4_RST.iter().zip([0.25; 4]) {
            let dn = tet10_dn_ref(gp[0], gp[1], gp[2]);
            let mut j = [[0.0; 3]; 3];
            for k in 0..10 {
                for i in 0..3 {
                    for m in 0..3 {
                        j[i][m] += dn[k][i] * xe[k][m];
                    }
                }
            }
            let inv = inv3(&j);
            let mut dndx = [[0.0; 3]; 10];
            for k in 0..10 {
                for m in 0..3 {
                    // physical gradient = dN @ inv(J).T
                    dndx[k][m] = dn[k][0] * inv[m][0] + dn[k][1] * inv[m][1] + dn[k][2] * inv[m][2];
                }
            }
            let b = build_b(&dndx, 10);
            let eps = strain_from_b(&b, 30, ue);
            for q in 0..6 {
                let mut acc = 0.0;
                for p in 0..6 {
                    acc += d[q][p] * eps[p];
                }
                s[q] += w * acc;
            }
        }
        for k in 0..10 {
            sig[k] = s;
        }
    } else {
        let mut j = [[0.0; 3]; 3];
        for k in 0..4 {
            for i in 0..3 {
                for m in 0..3 {
                    j[i][m] += TET4_DN[k][i] * xe[k][m];
                }
            }
        }
        let inv = inv3(&j);
        let mut dndx = [[0.0; 3]; 4];
        for k in 0..4 {
            for m in 0..3 {
                dndx[k][m] = TET4_DN[k][0] * inv[m][0] + TET4_DN[k][1] * inv[m][1] + TET4_DN[k][2] * inv[m][2];
            }
        }
        let b = build_b(&dndx, 4);
        let eps = strain_from_b(&b, 12, ue);
        for q in 0..6 {
            let mut acc = 0.0;
            for p in 0..6 {
                acc += d[q][p] * eps[p];
            }
            sig[0][q] = acc;
        }
        for k in 1..4 {
            sig[k] = sig[0];
        }
    }
    sig
}

fn strain_from_b(b: &[[f64; 30]], ncols: usize, ue: &[f64]) -> [f64; 6] {
    let mut eps = [0.0; 6];
    for q in 0..6 {
        let mut acc = 0.0;
        for c in 0..ncols {
            acc += b[q][c] * ue[c];
        }
        eps[q] = acc;
    }
    eps
}

pub fn von_mises(sigma: &[[f64; 6]]) -> Vec<f64> {
    sigma
        .iter()
        .map(|s| {
            let sxx = s[0];
            let syy = s[1];
            let szz = s[2];
            let txy = s[3];
            let tyz = s[4];
            let tzx = s[5];
            (0.5 * ((sxx - syy).powi(2) + (syy - szz).powi(2) + (szz - sxx).powi(2))
                + 3.0 * (txy * txy + tyz * tyz + tzx * tzx))
            .sqrt()
        })
        .collect()
}

/// load point displacement delta along force direction and stiffness k = |F|/delta.
pub fn stiffness(u: &[f64], load_nodes: &[usize], force: &[f64; 3]) -> (f64, f64) {
    let fmag = (force[0].powi(2) + force[1].powi(2) + force[2].powi(2)).sqrt();
    if fmag <= 0.0 {
        return (0.0, f64::INFINITY);
    }
    let dir = [force[0] / fmag, force[1] / fmag, force[2] / fmag];
    let mut delta = 0.0;
    for &i in load_nodes {
        delta += u[3 * i] * dir[0] + u[3 * i + 1] * dir[1] + u[3 * i + 2] * dir[2];
    }
    delta /= load_nodes.len() as f64;
    if delta.abs() < 1e-30 {
        (delta, f64::INFINITY)
    } else {
        (delta, fmag / delta)
    }
}

pub fn face_nodes(nodes: &[[f64; 3]], face: &str) -> Result<Vec<usize>, String> {
    let axis = match face.as_bytes()[0] {
        b'x' => 0usize,
        b'y' => 1,
        b'z' => 2,
        _ => return Err(format!("bad face {face}")),
    };
    let is_min = face.ends_with("_min");
    let is_max = face.ends_with("_max");
    if !is_min && !is_max {
        return Err(format!("bad face {face}"));
    }
    let v: Vec<f64> = nodes.iter().map(|p| p[axis]).collect();
    let lo = v.iter().cloned().fold(f64::INFINITY, f64::min);
    let hi = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let tol = 1e-7 * (hi - lo).abs() + 1e-14;
    let target = if is_min { lo } else { hi };
    Ok(nodes
        .iter()
        .enumerate()
        .filter_map(|(i, p)| {
            if (p[axis] - target).abs() <= tol {
                Some(i)
            } else {
                None
            }
        })
        .collect())
}

pub fn build_load_vector(n: usize, loads: &[(String, [f64; 3])], nodes: &[[f64; 3]]) -> (Vec<f64>, Vec<usize>) {
    let ndof = 3 * n;
    let mut b = vec![0.0; ndof];
    let mut load_nodes = Vec::new();
    for (face, force) in loads {
        let idx = face_nodes(nodes, face).expect("bad face in load");
        let per = [force[0] / idx.len() as f64, force[1] / idx.len() as f64, force[2] / idx.len() as f64];
        for &i in &idx {
            for k in 0..3 {
                b[3 * i + k] += per[k];
            }
        }
        load_nodes.extend(idx);
    }
    (b, load_nodes)
}
