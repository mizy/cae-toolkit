---
name: structural-solver
description: Linear static structural analysis and deformation visualization for tetrahedral solids and planar triangular thin shells. Use when the user needs static stiffness, deformation, or stress for a part. The solid and shell CLI paths return JSON and VTP fields, and the solid path also emits an HTML report.
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

The physical model uses small-strain linear isotropic elasticity. The thin-shell
case below uses the Python CPU solver. Beam elements, automatic midsurface
extraction and arbitrary CAD input are outside the supported CLI inputs.

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

## 3D HTML postprocessing

Generate a self-contained HTML report from a solved VTP surface or VTU volume:

```bash
../../../.venv/bin/python scripts/solve.py postprocess \
  --mesh outputs/hinge/hinge_mesh.vtp --mesh-units m --units mm \
  --summary outputs/hinge/hinge_report.json --output outputs/hinge/hinge_3d.html
```

Use the original geometry with its result fields. `--units` specifies the
stored field units (m/Pa or mm/MPa), and `--mesh-units` defaults to the field
length unit. The solid solver's original VTP has coordinates in meters
and displacement in the requested units. The shell solver's original VTP
uses the requested length units for both. Supply a 3-component point
array named `displacement` or `u_mm` (for mm), and optional von Mises fields
on points or cells, or open a VTP with scalar or vector fields only. The report can
also show signed displacement components and any stored
`vertical_mm`, `axial_mm`, or stress components. New solid outputs carry the
six Voigt stress components and a geometry scale on displaced outputs.
The postprocessor uses that metadata to restore the original coordinates
and derives principal stresses when the tensor is available.

Open the report in a WebGL-capable browser. It embeds its data and viewer,
starts at 1× deformation, and supports rotation, zoom, cutaway views, original
geometry overlay, animation, peak location, point/cell inspection, model-space
distance, PNG export, and full-range or P99 color scales. A VTP cutaway clips
the existing surface. Add `--volume-sections` for a tetrahedral VTU to embed
interior field samples along X, Y and Z at five positions per axis. That option
increases the report size. Stored numeric point and cell arrays can be selected
in the field menu. Switch surface, wireframe or points, isolate triangles by
threshold, or draw an isoline on a continuous point field. Point thresholds use
triangle vertex averages. Include repeatable `--conclusion` arguments to place
engineering findings at the top of the report. This command performs
postprocessing only.

## Planar thin shell

```bash
../../../.venv/bin/python scripts/solve.py shell \
  --mesh plate-midsurface.vtp --units mm --thickness 0.4 \
  --young-modulus 2000 --poisson-ratio 0.38 \
  --fix x_min --load "x_max: fz=-0.001" \
  --output-dir outputs/plate --name plate
```

Supply one planar triangular midsurface as VTP or STL, with connected mesh
vertices and uniform thickness. The plane can have any 3D orientation. Cutouts
can be represented as mesh boundaries. `--fix` clamps endpoint translations
and the midpoint normal slope on each selected boundary edge. `--load` applies a total force
in N, distributed uniformly per unit edge length in the global directions
`fx/fy/fz`. Select bounding edges with `x_min` etc., or name a VTP scalar
point array of 0/1 values. An edge is selected when both endpoints are tagged.
Both flags can be repeated. Choose a material preset with `--material` or supply
Young's modulus and Poisson's ratio as in the example.

This path assembles constant-strain triangular membrane elements and quadratic
Morley Kirchhoff–Love bending elements. Bending uses three corner displacements
and one normal slope per edge. It has no transverse shear term, so a thin
plate does not require tetrahedra through its thickness or suffer transverse
shear locking in this model.
[Morley's constant-moment plate-bending element](https://journals.sagepub.com/doi/10.1243/03093247V061020)

The JSON report records force balance, free-DOF residual and energy consistency.
The two VTP files carry translations and magnitudes as point data, with top,
bottom and maximum von Mises stress as cell data. Geometry and displacement
fields use the requested length units. The report records the normal defining
the top face. The deformed geometry defaults to 1×, using the vertex
displacements to draw the midsurface.

Use this model for small displacement and rotation in a thin, planar patch.
This CLI treats the patch as a stand-alone part. Curved shells, shell–solid
coupling, contact, transverse shear, gravity and face pressure are outside the
current shell path. A mixed-thickness assembly still uses the solid workflow.

The shell regression uses a 200 × 20 × 0.4 mm strip with E = 2000 MPa and ν = 0.
A 4 μN transverse end force gives 0.050012 mm with 720 triangles and 2,427 DOFs,
compared with 0.050000 mm from the thin cantilever solution. The membrane
regression checks axial displacement and stress on a rotated patch. Run both
via `python -m pytest tests/test_shell.py`.

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
