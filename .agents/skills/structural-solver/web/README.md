# Model + conditions, shared native/browser solver

The user supplies a model and physical conditions. Python automatically creates an
initial T10 solid mesh and assembled equations. `core/` owns the CSR contract,
CPU f64 PCG (residual target at most 1e-8), symmetric diagonal scaling, GPU PCG kernels and f64 true-residual
verification. Native Rust and Rust/WASM call the same code and WGSL; JavaScript
owns the page only. The existing `native/` FEM program also calls `core/` in GPU
mode, retaining faer LU in CPU mode and its existing stress/report pipeline.

## Build and run

From the repository root:

```sh
cargo build --release --manifest-path .agents/skills/structural-solver/core/Cargo.toml
cargo build --release --manifest-path .agents/skills/structural-solver/native/Cargo.toml
wasm-pack build .agents/skills/structural-solver/core --target web --out-dir ../web/pkg --release
.venv/bin/python .agents/skills/structural-solver/scripts/serve_web.py --port 18768
```

Open `http://127.0.0.1:18768/`. Enter model dimensions/units and material,
constraint and force. The local Python service prepares the model; all GPU
solving happens in the browser. This workflow requires Python locally; an already
exported equation JSON can also be solved from a static server without Python.
Automatic mode tries GPU then selects CPU f64 on failure or non-convergence, with
the actual backend and reason shown. The advanced panel forces GPU or Rust CPU f64, and imports existing equation JSON.
Built WASM and native binaries are ignored by Git and must be built after checkout.

For native model/condition analysis, create `input.json`:

```json
{"model":{"geometry":"box:100,10,10","units":"mm"},"conditions":{"material":"steel","fixes":["x_min"],"loads":["x_max: fx=1000"]}}
```

```sh
.venv/bin/python .agents/skills/structural-solver/scripts/analyze.py input.json --output /tmp/result.json
```

`--backend cpu` selects CPU f64. For an already prepared CSR JSON:

```sh
.agents/skills/structural-solver/core/target/release/structural-equations /tmp/case.json gpu
```

The existing `scripts/solve.py solve` CLI supports `--backend auto|gpu|cpu`.
Auto attempts Rust GPU for T10, warns before retrying native CPU f64, then
follows the existing Python fallback only if native execution fails;
explicit GPU mode fails if unavailable or unconverged. CPU uses the native faer
path where available. Reports identify the actual backend.

## Algorithm and limits

GPU vectors, dot reductions and PCG updates remain in device buffers for batches
of up to 512 iterations. At batch boundaries, Rust reads the displacement
correction, accumulates it in f64, checks the original system's true residual,
and restarts with a correction when needed. GPU arithmetic is f32; the outer
verification is f64 on the CPU. Native uses Metal on supported Macs; browsers
use WebGPU. Explicit GPU failure is shown; automatic mode retries CPU and reports why.

Current model generation supports boxes, the existing simplified one-body
hinge and single closed STL inputs, with linear isotropic materials, axis-aligned fully fixed faces and equal
nodal forces. Initial box mesh size is half the smallest dimension. Generated
examples and solve outputs are ignored. Automatic mesh-convergence studies,
arbitrary CAD import, contact, shell/beam elements and midsurface extraction
remain future work; thin geometry must not silently be claimed as a shell model.
The SciPy reference uses the same FEM assembly and checks solver agreement only.
Small residuals alone do not establish mesh, physical model or peak-stress accuracy.

## STL, gravity and cylindrical loads

The page now accepts one closed, consistently oriented STL component (up to 10 MB).
Units must be stated explicitly; no hole filling or automatic repair is performed.
The volume mesher preserves the existing planar surface facets and creates linear-
geometry T10 elements. This can retain poor source surface quality; arbitrary STL
quality, intersecting surfaces and general assemblies are not validated by the
synthetic test. Fixed faces are still coordinate-extreme node selections; visual
picking and general surface-region assignment are not implemented.

For self weight, set the external force components to zero, enable **包含自重**,
and specify density in kg/m³ and acceleration in m/s². Density is an explicit
assumption, not inferred from the STL. The 7850 kg/m³ initial value is a steel
example; changing material does not determine the actual part density.

The preparation API also accepts `conditions.gravity`, `conditions.density`, and
`conditions.cylinder`. A cylinder region is explicitly specified in SI units:

```json
{"type":"bearing","center":[0.05,0.02,0.004],"axis":[0,0,1],"radius":0.006,"tolerance":0.0002,"force":[1000,0,0]}
```

`type: "pressure"` uses `pressure_pa` instead of the force. Pressure acts into the
solid on the selected bore; bearing uses a compressive half-bore cosine pattern,
normalized to the requested projected force. Incomplete cylindrical selections or
large resultant-direction errors are rejected. T6 load integration currently
requires planar facets with straight midside nodes. This is a known-region linear
load approximation, not bolt contact, friction, torque or pretension simulation.
The page currently configures gravity and fixed/nodal loads; cylinder loads are
supplied through the preparation API or a prepared equation JSON.

Reproduce the synthetic through-hole, uniform pressure and self-weight checks:

```sh
.venv/bin/python .agents/skills/structural-solver/scripts/validate_loads.py --output-dir .agents/skills/structural-solver/outputs/loads-validation
```

The output includes source STL, equation JSON files, `validation.json`, a plot and
`validation.html`. It checks both native CPU and GPU plus load resultants/reactions,
STL topology/volume preservation and the gravity cantilever analytical deflection.
`browser-check.html` runs the same cases through WebGPU when the skill directory
is served over localhost; it reports results on screen. It requires a built WASM
package. Browser file upload acceptance is separate from equation-solver validation.

## Self-contained CLI package

On macOS Apple Silicon, `scripts/package_cli.py` builds a single executable that
embeds the Python/Gmsh/SciPy preprocessing runtime plus the Rust/wgpu equation
solver. The user running this executable does not need Python, Cargo or Gmsh
installed separately. This is a bundled full workflow, rather than an all-Rust
STL mesher. Build it with the solver environment's Python after installing
`requirements-package.txt` into that environment:

```sh
.venv/bin/python .agents/skills/structural-solver/scripts/package_cli.py
```

Both the packaged executable and `scripts/analyze.py` accept:

```sh
structural-solver --model part.stl --conditions conditions.json --units mm --output result.json
structural-solver input.json --backend auto --output result.json
```

`conditions.json` is the physical-conditions object (material, fixes, loads,
optional density/gravity/cylinder). Units are mandatory for direct STL input.
The release configuration has been tested only on macOS arm64; build/validate a
separate package for other platforms. The smaller `structural-equations` binary
is pure Rust and still accepts prepared CSR equations only.
