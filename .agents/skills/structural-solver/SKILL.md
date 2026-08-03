---
name: structural-solver
description: Linear static structural analysis (stiffness k = F/delta) with deformation visualization for watertight CAD/mesh parts. Builds a tetrahedral volume mesh (T10) from parametric geometry, solves the linear elastic FE system with scipy, and emits deformed_mesh.vtp (displaced geometry + displacement/von-Mises point arrays), a JSON report, and a self-contained HTML report with embedded before/after deformation renderings, field plots, analytical benchmark validation, and pass/fail gates. Use when the user needs static stiffness, deformation, or stress for a part.
---

# Structural Solver

Linear static elasticity (small strain, isotropic material) for parametric part geometry.
The solver is self-contained: gmsh builds box meshes, the hinge uses a deterministic
numpy mesher (gmsh OCC boolean meshing of knuckle cylinders hangs, so the hinge avoids it),
rendering uses pyvista offscreen.

## Native Rust core

The FE hot path (T4/T10 element stiffness, sparse assembly, faer sparse LU solve,
stress recovery) is implemented in Rust under `native/` and is used automatically
when the release binary exists:

```bash
cd native && cargo build --release
```

`solve.py` shells out to `native/target/release/structural_solver_native` with the
mesh (MSH 2.2) and BCs (JSON), reads displacement/stress binaries back, and falls
back to the Python scipy solver when the binary is missing or fails. Results agree
with the Python reference to ~1e-10 relative (same quadrature, node ordering,
Jacobian convention). Measured: hinge 1.5 mm case (9120 T10 / ~27k dof) solves in
~0.2 s vs ~20 s in Python; end-to-end CLI ~10x faster.

## Quick Start

```bash
cd .agents/skills/structural-solver
../../../.venv/bin/python scripts/solve.py solve \
  --geometry hinge --units mm --material steel --order 2 --mesh-size 1.5 \
  --fix x_min --load "x_max: fz=-100" --convergence \
  --output-dir outputs/hinge --name hinge
```

Outputs in `outputs/hinge/`:

- `hinge_deformed.vtp` — displaced surface geometry (scaled for visibility) with
  point arrays `displacement` (true mm), `displacement_magnitude`, `von_mises_stress` (MPa)
- `hinge_mesh.vtp` — original surface geometry with the same arrays
- `hinge_volume.vtu` — full-volume tetrahedral mesh with the same arrays
- `hinge_report.json` — authoritative numeric report
- `hinge_report.html` — self-contained report: interactive 3D viewer (drag rotate /
  right-drag pan / wheel zoom; toggles for original wireframe, deformed mesh colored
  by displacement, red load arrows with force labels, red fixed-constraint markers,
  axes; pure Canvas2D renderer, no WebGL requirement, works offline in any browser),
  plus static deformation overlay, |u|
  and von-Mises plots, field table (delta, k, max sigma, strain energy), load/fix
  list, analytical benchmarks (tension bar, cantilever), and gates table
- `hinge_*.png` — rendered images

## Supported inputs

- `--geometry box:Lx,Ly,Lz` (gmsh OCC, any units)
- `--geometry hinge` (numpy rectangular-knuckle hinge, mm only; see plan.md)
- `--mesh file.msh` (existing gmsh MSH 2.2 tetrahedral mesh)

Units: `--units m` (m/N/Pa) or `--units mm` (mm/N/MPa). Internally SI.

## Conventions and assumptions

- Linear small-strain isotropic elasticity; T10 default (`--order 2`), T4 supported.
- Axis convention: blade 1 fixed at `x_min`, load applied on `x_max` face.
- `deformed.vtp` geometry is displaced by `display_scale × displacement` so the
  deformation is visible; the true displacement is in the point arrays.
- `k = |F| / delta` with delta = mean load-face displacement along the load direction.
- Hinge v1 model is a simplified one-body part (rectangular knuckle block, no pin,
  no screw holes) to keep gmsh booleans robust; see docs/plan.md.
- Static stiffness is the load-point translational stiffness only.

## Validation

Analytical benchmarks run inside every report:

- tension bar: delta = F·L/(E·A) — rel error < 1%
- cantilever: delta = F·L³/(3·E·I) — rel error < 5%

Gates: reaction balance |ΣR+F|/|F| < 1%, energy consistency, VTP read-back audit,
positive Jacobians, optional convergence (coarse/fine delta change < 15%).
