//! structural-solver native core: MSH 2.2 -> linear elastic solve -> binary/JSON out.
//!
//! Usage: structural_solver_native <mesh.msh> <bc.json> <out_prefix>
//!
//! bc.json: {"material":{"E":...,"nu":...}, "fixes":["x_min"], "loads":[{"face":"x_max","force":[0,0,-100]}]}
//! Outputs: <out_prefix>.u.bin (f64 LE, ndof), <out_prefix>.sigma.bin (f64 LE, n*6),
//!          <out_prefix>.summary.json
use structural_solver_native::{fem, mesh};

use std::collections::HashSet;
use std::io::Write;
use std::path::Path;

use serde::Deserialize;

#[derive(Deserialize)]
struct Bc {
    material: Material,
    #[serde(default)]
    fixes: Vec<String>,
    #[serde(default)]
    loads: Vec<Load>,
}

#[derive(Deserialize)]
struct Material {
    #[serde(rename = "E")]
    e: f64,
    nu: f64,
}

#[derive(Deserialize)]
struct Load {
    face: String,
    force: [f64; 3],
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        eprintln!("usage: {} <mesh.msh> <bc.json> <out_prefix>", args[0]);
        std::process::exit(2);
    }
    let msh_path = Path::new(&args[1]);
    let bc_path = Path::new(&args[2]);
    let out_prefix = Path::new(&args[3]);

    let t_total = std::time::Instant::now();
    let tm = match mesh::read_msh22(msh_path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("mesh error: {e}");
            std::process::exit(1);
        }
    };
    let bc: Bc = match std::fs::read_to_string(bc_path) {
        Ok(s) => serde_json::from_str(&s).unwrap_or_else(|e| {
            eprintln!("bc.json error: {e}");
            std::process::exit(1);
        }),
        Err(e) => {
            eprintln!("bc error: {e}");
            std::process::exit(1);
        }
    };

    let n = tm.nodes.len();
    let _ndof = 3 * n;
    let mut fixed_dofs = HashSet::new();
    for f in &bc.fixes {
        match fem::face_nodes(&tm.nodes, f) {
            Ok(idx) => {
                for &i in &idx {
                    for k in 0..3 {
                        fixed_dofs.insert(3 * i + k);
                    }
                }
            }
            Err(e) => {
                eprintln!("fix error: {e}");
                std::process::exit(1);
            }
        }
    }
    let loads: Vec<(String, [f64; 3])> = bc.loads.iter().map(|l| (l.face.clone(), l.force)).collect();
    let (load_vector, load_nodes) = fem::build_load_vector(n, &loads, &tm.nodes);
    let fsum = bc.loads.iter().fold([0.0f64; 3], |acc, l| {
        [acc[0] + l.force[0], acc[1] + l.force[1], acc[2] + l.force[2]]
    });
    let fmag = (fsum[0].powi(2) + fsum[1].powi(2) + fsum[2].powi(2)).sqrt();
    let _fdir = if fmag > 0.0 {
        [fsum[0] / fmag, fsum[1] / fmag, fsum[2] / fmag]
    } else {
        [0.0, 0.0, 0.0]
    };

    let res = fem::solve_linear(&tm, bc.material.e, bc.material.nu, &fixed_dofs, &load_vector);

    let (delta, stiffness) = fem::stiffness(&res.u, &load_nodes, &fsum);
    let vm = fem::von_mises(&res.sigma);
    let max_u = (0..n)
        .map(|i| {
            let ux = res.u[3 * i];
            let uy = res.u[3 * i + 1];
            let uz = res.u[3 * i + 2];
            (ux * ux + uy * uy + uz * uz).sqrt()
        })
        .fold(0.0f64, f64::max);
    let max_vm = vm.iter().cloned().fold(0.0f64, f64::max);
    let mean_vm = vm.iter().sum::<f64>() / vm.len() as f64;

    // write binary outputs
    let write_f64 = |path: &Path, data: &[f64]| -> std::io::Result<()> {
        let mut f = std::fs::File::create(path)?;
        for v in data {
            f.write_all(&v.to_le_bytes())?;
        }
        Ok(())
    };
    let upath = out_prefix.with_extension("u.bin");
    let spath = out_prefix.with_extension("sigma.bin");
    write_f64(&upath, &res.u).unwrap();
    let mut sflat = Vec::with_capacity(n * 6);
    for s in &res.sigma {
        sflat.extend_from_slice(s);
    }
    write_f64(&spath, &sflat).unwrap();

    let summary = serde_json::json!({
        "nodes": n,
        "elements": tm.elements.len(),
        "order": tm.order,
        "max_displacement_m": max_u,
        "load_point_delta_m": delta,
        "stiffness_n_per_m": stiffness,
        "max_von_mises_pa": max_vm,
        "mean_von_mises_pa": mean_vm,
        "strain_energy_j": res.strain_energy,
        "reaction_sum_n": res.reaction_sum,
        "reaction_balance": if fmag > 0.0 {
            let rx = res.reaction_sum[0] + fsum[0];
            let ry = res.reaction_sum[1] + fsum[1];
            let rz = res.reaction_sum[2] + fsum[2];
            ((rx * rx + ry * ry + rz * rz).sqrt()) / fmag
        } else {
            0.0
        },
        "time_assembly_s": res.time_assembly_s,
        "time_lu_s": res.time_lu_s,
        "time_stress_s": res.time_stress_s,
        "time_total_s": t_total.elapsed().as_secs_f64(),
    });
    let spath2 = out_prefix.with_extension("summary.json");
    std::fs::write(&spath2, serde_json::to_string_pretty(&summary).unwrap()).unwrap();
    println!("OK nodes={n} elems={} order={} delta={delta:.6e} k={stiffness:.6e}",
        tm.elements.len(), tm.order);
}
