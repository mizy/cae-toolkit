# Browser linear-static prototype

This experiment solves an existing linear-elastic FEM system locally in the browser.
Python assembles the mesh and stiffness matrix, applies fixed boundary conditions,
and exports a double-precision SciPy reference. WebGPU performs f32 sparse matrix
vector products; PCG vectors and reductions run on the CPU. Scalar Jacobi and
3×3 nodal block Jacobi are selectable; block Jacobi requires node-major xyz DOFs.
A true residual against
the original f64 matrix is checked before accepting a solution. This is not yet a
browser mesher, general CAD importer, stress viewer, or validated engineering tool.

From the repository root, generate the built-in T10 tensile example:

```sh
.venv/bin/python .agents/skills/structural-solver/scripts/export_web_case.py --output .agents/skills/structural-solver/web/example.json
.venv/bin/python .agents/skills/structural-solver/scripts/export_web_case.py --load 'x_max: fz=-100' --output .agents/skills/structural-solver/web/example-bending.json
cd .agents/skills/structural-solver/web
python3 -m http.server 18766 --bind 127.0.0.1
```

Open `http://127.0.0.1:18766/` in a WebGPU-enabled browser, select a case and **载入示例**
and **开始求解**. To use another generated case, see
`python scripts/export_web_case.py --help` from the skill directory and choose the
resulting JSON in the page. `--fix` and `--load` can be repeated for combined
cases. Load faces touching fixed nodes are rejected. The generated examples are
ignored by Git. The downloaded coordinates and displacement vector are in meters,
with x/y/z components interleaved per mesh node; element connectivity is not exported.

The exporter shares the existing solver's axis-aligned face selection and equal
nodal-force distribution; this is not generally equivalent to uniform traction.
Geometry, element quality, conditioning, mesh convergence, stress, and physical
boundary conditions require independent assessment. Per-iteration GPU readback
makes this version a correctness prototype, not an optimized GPU implementation.
The SciPy reference solves the same assembled system, so it checks the linear solver
but does not independently validate the FEM assembly or physical model. The reported
solve time includes GPU setup and iteration but excludes export, assembly and input
validation/preconditioner preparation.
