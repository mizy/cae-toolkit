//! Shared native/WASM linear static equation solver. FEM assembly stays upstream.
mod gpu;
use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LinearSystem {
    pub row_ptr: Vec<u32>,
    pub col_idx: Vec<u32>,
    pub values: Vec<f64>,
    pub rhs: Vec<f64>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Solution {
    pub solution: Vec<f64>,
    pub iterations: usize,
    pub relative_residual: f64,
    pub converged: bool,
    pub backend: String,
    pub fallback_reason: Option<String>,
}

impl LinearSystem {
    pub fn validate(&self) -> Result<Vec<f64>, String> {
        let n = self.rhs.len();
        if n == 0
            || self.row_ptr.len() != n + 1
            || self.row_ptr[0] != 0
            || self.values.len() != self.col_idx.len()
            || self.row_ptr[n] as usize != self.values.len()
        {
            return Err("Invalid CSR dimensions".into());
        }
        let mut diagonal = vec![0.0; n];
        for i in 0..n {
            if self.row_ptr[i] > self.row_ptr[i + 1]
                || self.row_ptr[i + 1] as usize > self.values.len()
                || !self.rhs[i].is_finite()
            {
                return Err("Invalid CSR row or force".into());
            }
            let mut previous = None;
            for j in self.row_ptr[i] as usize..self.row_ptr[i + 1] as usize {
                let col = self.col_idx[j] as usize;
                if col >= n || !self.values[j].is_finite() || previous.is_some_and(|p| col <= p) {
                    return Err("CSR columns must be sorted, unique and finite".into());
                }
                previous = Some(col);
                if col == i {
                    diagonal[i] = self.values[j];
                }
            }
            if !diagonal[i].is_finite() || diagonal[i] <= 0.0 {
                return Err("Missing positive diagonal; check constraints".into());
            }
        }
        // PCG requires a symmetric positive definite system. Symmetry is checked
        // here; positive definiteness is also monitored through search curvature.
        for i in 0..n {
            for j in self.row_ptr[i] as usize..self.row_ptr[i + 1] as usize {
                let col = self.col_idx[j] as usize;
                let range = self.row_ptr[col] as usize..self.row_ptr[col + 1] as usize;
                let transposed = self.col_idx[range.clone()]
                    .binary_search(&(i as u32))
                    .map(|k| self.values[range.start + k])
                    .unwrap_or(0.0);
                if (self.values[j] - transposed).abs() > 1e-9 * (diagonal[i] * diagonal[col]).sqrt()
                {
                    return Err("PCG requires a symmetric matrix".into());
                }
            }
        }
        Ok(diagonal)
    }

    pub fn residual(&self, x: &[f64]) -> Vec<f64> {
        (0..self.rhs.len())
            .map(|i| {
                self.rhs[i]
                    - (self.row_ptr[i] as usize..self.row_ptr[i + 1] as usize)
                        .map(|j| self.values[j] * x[self.col_idx[j] as usize])
                        .sum::<f64>()
            })
            .collect()
    }
}

pub(crate) fn norm(x: &[f64]) -> f64 {
    x.iter().fold(0.0_f64, |a, &b| a.hypot(b))
}

/// @entry Solve the same CSR equations on CPU f64 or wgpu (native + browser).
pub async fn solve(
    system: &LinearSystem,
    backend: &str,
    tolerance: f64,
    max_iterations: usize,
) -> Result<Solution, String> {
    if !(1e-10..=0.1).contains(&tolerance) || max_iterations == 0 || max_iterations > 20000 {
        return Err("Invalid solver limits".into());
    }
    let diagonal = system.validate()?;
    if backend == "auto" {
        let reason = match gpu::solve(system, &diagonal, tolerance, max_iterations).await {
            Ok(result) if result.converged => return Ok(result),
            Ok(result) => format!(
                "GPU stopped after {} iterations, residual {:.3e}",
                result.iterations, result.relative_residual
            ),
            Err(error) => error,
        };
        let mut result = solve_cpu(system, &diagonal, tolerance.min(1e-8), max_iterations)?;
        result.fallback_reason = Some(reason);
        return Ok(result);
    }
    match backend {
        "gpu" => gpu::solve(system, &diagonal, tolerance, max_iterations).await,
        "cpu" => solve_cpu(system, &diagonal, tolerance.min(1e-8), max_iterations),
        _ => Err("Backend must be auto, gpu or cpu".into()),
    }
}

fn solve_cpu(
    a: &LinearSystem,
    diagonal: &[f64],
    tolerance: f64,
    max: usize,
) -> Result<Solution, String> {
    let n = a.rhs.len();
    let mut x = vec![0.0; n];
    let bnorm = norm(&a.rhs);
    if !bnorm.is_finite() {
        return Err("Force magnitude exceeds supported range".into());
    }
    if bnorm == 0.0 {
        return Ok(Solution {
            solution: x,
            iterations: 0,
            relative_residual: 0.0,
            converged: true,
            backend: "rust-cpu-f64".into(),
            fallback_reason: None,
        });
    }
    let mut r = a.rhs.clone();
    let mut p: Vec<f64> = r.iter().zip(diagonal).map(|(r, d)| r / d).collect();
    let mut rz: f64 = r.iter().zip(&p).map(|(r, p)| r * p).sum();
    let mut count = 0;
    for k in 0..max {
        let ap: Vec<f64> = (0..n)
            .map(|i| {
                (a.row_ptr[i] as usize..a.row_ptr[i + 1] as usize)
                    .map(|j| a.values[j] * p[a.col_idx[j] as usize])
                    .sum()
            })
            .collect();
        let pap: f64 = p.iter().zip(&ap).map(|(p, a)| p * a).sum();
        if !pap.is_finite() || pap <= 0.0 {
            return Err("PCG breakdown; check positive definiteness".into());
        }
        let alpha = rz / pap;
        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }
        count = k + 1;
        if norm(&r) / bnorm <= tolerance {
            r = a.residual(&x);
            if norm(&r) / bnorm <= tolerance {
                break;
            }
            for i in 0..n {
                p[i] = r[i] / diagonal[i];
            }
            rz = r.iter().zip(&p).map(|(r, p)| r * p).sum();
            continue;
        }
        let next: f64 = (0..n).map(|i| r[i] * r[i] / diagonal[i]).sum();
        for i in 0..n {
            p[i] = r[i] / diagonal[i] + (next / rz) * p[i];
        }
        rz = next;
    }
    let residual = norm(&a.residual(&x)) / bnorm;
    Ok(Solution {
        solution: x,
        iterations: count,
        relative_residual: residual,
        converged: residual <= tolerance,
        backend: "rust-cpu-f64".into(),
        fallback_reason: None,
    })
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub async fn solve_json(json: String, backend: String) -> Result<String, wasm_bindgen::JsValue> {
    let a: LinearSystem =
        serde_json::from_str(&json).map_err(|e| wasm_bindgen::JsValue::from_str(&e.to_string()))?;
    let solution = solve(&a, &backend, 1e-5, 6000)
        .await
        .map_err(|e| wasm_bindgen::JsValue::from_str(&e))?;
    serde_json::to_string(&solution).map_err(|e| wasm_bindgen::JsValue::from_str(&e.to_string()))
}
