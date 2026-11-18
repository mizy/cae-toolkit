//! Analytical benchmarks + patch test against the native solver.
use std::collections::HashSet;

use structural_solver_native::fem::{self, face_nodes};
use structural_solver_native::mesh::TetMesh;

/// Structured T10 mesh (nx*ny*nz hexes, 6 tets each, dedup nodes) — mirrors the
/// Python reference grid generator.
fn rule_grid(nx: usize, ny: usize, nz: usize, l: f64, w: f64, h: f64) -> TetMesh {
    let sx = l / nx as f64;
    let sy = w / ny as f64;
    let sz = h / nz as f64;
    let mut nodes: Vec<[f64; 3]> = Vec::new();
    let mut nid: std::collections::HashMap<[i64; 3], usize> = Default::default();
    fn get(
        nid: &mut std::collections::HashMap<[i64; 3], usize>,
        nodes: &mut Vec<[f64; 3]>,
        sx: f64,
        sy: f64,
        sz: f64,
        x: usize,
        y: usize,
        z: usize,
    ) -> usize {
        let key = [2 * x as i64, 2 * y as i64, 2 * z as i64]; // even: corner
        if let Some(&i) = nid.get(&key) {
            return i;
        }
        let i = nodes.len();
        nodes.push([x as f64 * sx, y as f64 * sy, z as f64 * sz]);
        nid.insert(key, i);
        i
    }
    let mut elements: Vec<[usize; 10]> = Vec::new();
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let v = [
                    get(&mut nid, &mut nodes, sx, sy, sz, i, j, k),
                    get(&mut nid, &mut nodes, sx, sy, sz, i + 1, j, k),
                    get(&mut nid, &mut nodes, sx, sy, sz, i + 1, j + 1, k),
                    get(&mut nid, &mut nodes, sx, sy, sz, i, j + 1, k),
                    get(&mut nid, &mut nodes, sx, sy, sz, i, j, k + 1),
                    get(&mut nid, &mut nodes, sx, sy, sz, i + 1, j, k + 1),
                    get(&mut nid, &mut nodes, sx, sy, sz, i + 1, j + 1, k + 1),
                    get(&mut nid, &mut nodes, sx, sy, sz, i, j + 1, k + 1),
                ];
                let tets = [
                    [v[0], v[1], v[3], v[4]],
                    [v[1], v[2], v[3], v[4]],
                    [v[1], v[5], v[6], v[4]],
                    [v[1], v[2], v[6], v[4]],
                    [v[2], v[6], v[7], v[4]],
                    [v[2], v[3], v[7], v[4]],
                ];
                for t in tets {
                    let c: Vec<[f64; 3]> = t.iter().map(|&a| nodes[a]).collect();
                    let d = (c[1][0] - c[0][0]) * ((c[2][1] - c[0][1]) * (c[3][2] - c[0][2]) - (c[2][2] - c[0][2]) * (c[3][1] - c[0][1]))
                        - (c[1][1] - c[0][1]) * ((c[2][0] - c[0][0]) * (c[3][2] - c[0][2]) - (c[2][2] - c[0][2]) * (c[3][0] - c[0][0]))
                        + (c[1][2] - c[0][2]) * ((c[2][0] - c[0][0]) * (c[3][1] - c[0][1]) - (c[2][1] - c[0][1]) * (c[3][0] - c[0][0]));
                    let t = if d < 0.0 { [t[0], t[2], t[1], t[3]] } else { t };
                    let cc: Vec<[f64; 3]> = t.iter().map(|&a| nodes[a]).collect();
                    let edges = [(0usize, 1usize), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)];
                    let mut conn = [0usize; 10];
                    conn[..4].copy_from_slice(&t);
                    for (e, (a, b)) in edges.iter().enumerate() {
                        let mid = [
                            (cc[*a][0] + cc[*b][0]) / 2.0,
                            (cc[*a][1] + cc[*b][1]) / 2.0,
                            (cc[*a][2] + cc[*b][2]) / 2.0,
                        ];
                        // half-index space: corners are even, edge midpoints odd
                        let key2 = [
                            (mid[0] / sx * 2.0).round() as i64,
                            (mid[1] / sy * 2.0).round() as i64,
                            (mid[2] / sz * 2.0).round() as i64,
                        ];
                        let id = if let Some(&x) = nid.get(&key2) {
                            x
                        } else {
                            let i = nodes.len();
                            nodes.push(mid);
                            nid.insert(key2, i);
                            i
                        };
                        conn[4 + e] = id;
                    }
                    elements.push(conn);
                }
            }
        }
    }
    TetMesh { nodes, elements, order: 2 }
}

fn solve_case(mesh: &TetMesh, fix: &str, load_face: &str, force: [f64; 3]) -> fem::SolveResult {
    let n = mesh.nodes.len();
    let mut fixed = HashSet::new();
    for i in face_nodes(&mesh.nodes, fix).unwrap() {
        for k in 0..3 {
            fixed.insert(3 * i + k);
        }
    }
    let (b, _) = fem::build_load_vector(n, &[(load_face.to_string(), force)], &mesh.nodes);
    fem::solve_linear(mesh, 210e9, 0.3, &fixed, &b, "cpu")
}

#[test]
fn patch_test_t10_energy_exact() {
    let c = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let mut nodes: Vec<[f64; 3]> = c.to_vec();
    nodes.extend([
        [0.5, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.0, 0.5, 0.0],
        [0.0, 0.0, 0.5],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
    ]);
    let (ke, vol) = fem::element_stiffness_pub(&nodes, 1.0, 0.0, 2);
    let mut u = vec![0.0; 30];
    for i in 0..10 {
        u[3 * i] = nodes[i][0];
    }
    let se: f64 = (0..30)
        .map(|a| (0..30).map(move |b| (a, b)))
        .flatten()
        .map(|(a, b)| 0.5 * ke[a][b] * u[a] * u[b])
        .sum();
    assert!((se - 0.5 * 1.0 * vol).abs() < 1e-12, "SE={se} vol={vol}");
}

#[test]
fn tension_bar_t10() {
    let mesh = rule_grid(20, 2, 2, 0.1, 0.01, 0.01);
    let res = solve_case(&mesh, "x_min", "x_max", [1000.0, 0.0, 0.0]);
    let delta = 1000.0 * 0.1 / (210e9 * 0.01 * 0.01);
    // load-face mean displacement (dof 0 = ux)
    let ln = fem::face_nodes(&mesh.nodes, "x_max").unwrap();
    let d: f64 = ln.iter().map(|&i| res.u[3 * i]).sum::<f64>() / ln.len() as f64;
    let rel = (d - delta).abs() / delta;
    assert!(rel < 0.02, "delta={d} ana={delta} rel={rel}");
}

#[test]
fn cantilever_t10() {
    let mesh = rule_grid(20, 2, 2, 0.1, 0.01, 0.01);
    let res = solve_case(&mesh, "x_min", "x_max", [0.0, 0.0, -1000.0]);
    let i_zz = 0.01 * 0.01f64.powi(3) / 12.0;
    let delta = 1000.0 * 0.1f64.powi(3) / (3.0 * 210e9 * i_zz);
    let ln = fem::face_nodes(&mesh.nodes, "x_max").unwrap();
    let mut d = 0.0;
    for &i in &ln {
        d += res.u[3 * i + 2];
    }
    d /= ln.len() as f64;
    d = d.abs();
    let rel = (d - delta).abs() / delta;
    assert!(rel < 0.05, "delta={d} ana={delta} rel={rel}");
}

#[test]
fn rule_grid_volume_ok() {
    let mesh = rule_grid(20, 2, 2, 0.1, 0.01, 0.01);
    // volume via centroid detJ
    let mut vol = 0.0;
    for conn in &mesh.elements {
        let xe: Vec<[f64; 3]> = conn.iter().map(|&i| mesh.nodes[i]).collect();
        let (_, v) = fem::element_stiffness_pub(&xe, 1.0, 0.0, 2);
        vol += v;
    }
    println!("nodes={} elems={} vol={vol} expect=1e-5", mesh.nodes.len(), mesh.elements.len());
    assert!((vol - 1e-5).abs() / 1e-5 < 0.01, "vol={vol}");
}
