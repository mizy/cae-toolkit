---
name: structural-solver
description: Linear static structural analysis (stiffness k = F/delta) with deformation visualization for watertight CAD/mesh parts. Builds a tetrahedral volume mesh (T10) from parametric geometry, solves the linear elastic FE system with scipy, and emits deformed_mesh.vtp (displaced geometry + displacement/von-Mises point arrays), a JSON report, and a self-contained HTML report with embedded before/after deformation renderings, field plots, analytical benchmark validation, and pass/fail gates. Use when the user needs static stiffness, deformation, or stress for a part.
---

# Structural Solver

Linear static elasticity (small strain, isotropic material) for parametric part geometry.
For the experimental browser WebGPU solver and its example, see
[`web/README.md`](web/README.md). It solves an exported FEM system and is not a
replacement for the native workflow.
The solver is self-contained: gmsh builds box meshes, the hinge uses a deterministic
numpy mesher (gmsh OCC boolean meshing of knuckle cylinders hangs, so the hinge avoids it),
rendering uses pyvista offscreen.

## Shared Rust + wgpu equation core

`core/` owns the CSR equation contract, CPU f64 PCG, GPU PCG WGSL kernels and
true-residual correction. It builds as a native binary and as WASM for `web/`.
GPU vectors and reductions stay resident for batches of up to 512 iterations.
The browser page accepts model + conditions and uses the local `serve_web.py`
service for automatic initial solid T10 meshing/assembly, then solves in WASM
and WebGPU. See [`web/README.md`](web/README.md) for build/run instructions.

`native/` retains Rust FEM assembly, stress recovery and faer CPU LU, and now
calls `core/` for its GPU equation solve. `solve.py --backend auto` attempts that
GPU path for T10 when the native binary exists; explicit `gpu` fails on errors,
`cpu` uses the CPU route. Automatic mode warns before Python fallback. T4 uses
the Python path through the legacy CLI. Build both Rust packages after checkout;
binaries and generated WASM are not committed.

The supported physical model remains solid, small-strain, linear isotropic
elasticity. Shell/beam elements, midsurface extraction, arbitrary CAD input and
automatic mesh-convergence refinement are not yet implemented.

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

## Gravity and known bore loads

The local model preparation path accepts single closed STL surfaces and preserves
source facets when generating volume elements. `scripts/loads.py` integrates T4/T10
body forces from explicit density and gravity, and planar T3/T6 cylindrical surface
pressure or half-bore bearing loads. Cylinder regions require explicit center,
axis, radius and tolerance; this is not interactive or general automatic hole
recognition. The page exposes self weight; the preparation API exposes cylinder
loads. See `web/README.md` for the precise input limitations.

Use `scripts/validate_loads.py --output-dir OUTPUT` to reproduce synthetic STL
bearing/pressure and cantilever self-weight checks. It runs native CPU/GPU solves,
checks resultants, reactions, source topology/volume and analytical beam deflection,
and writes HTML/JSON evidence. No bolt pretension, friction or contact is included.
