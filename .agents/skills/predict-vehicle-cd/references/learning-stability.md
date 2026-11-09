# Performance and stability validation — 2026-10-28

## Changes

- Replaced the remaining per-triangle Python pixel loop with bounded NumPy
  batches and `minimum.at` depth reduction. No new runtime dependency or learned
  representation was introduced. Existing checkpoints retain their input meaning.
- Moved file-to-view rendering into `render.render_geometry`, consumed by CLI
  inference; dataset preparation reads each geometry once for rendering and hashing.
- Shared numerical depth/mask validation between training data and inference.
  Invalid image shapes, nonfinite/out-of-range depth, nonbinary/empty masks and
  nonzero background depth fail before the model runs.
- Checkpoint loading validates finite positive scalers, vector/model dimensions,
  parameter ranges, finite weights and strict state-dict compatibility.
- Unsupported Re/yaw and geometry ratios fail closed by default. Geometry-only
  diagnostic extrapolation requires an explicit API/CLI option. Tiny numeric drift
  on constant trained features snaps to the trained value rather than being
  amplified by the 1e-8 normalization floor. Negative passive-vehicle Cd is rejected;
  Cl retains its signed meaning.
- CLI timing now separates model loading, geometry loading/rendering, and inference.
  Inference includes validation and tensor conversion; it is not bare neural-network
  forward time. Total timing begins after argument parsing and excludes Python imports.

These checks do not recognize every possible out-of-distribution shape. Passing a
range check is not proof of physical accuracy or of vehicle identity.

## Performance evidence

The exact same Auris input and checkpoint were measured before and after rendering
optimization. Geometry loading plus five-view construction decreased from 8.793 s
to 0.683 s (about 12.9x); current total measured CLI processing was 0.727 s. Imports
and process startup are outside that timer. Output Cd remained 0.4079514.

An independent renderer sweep of the 11 known cars took 5.88 s total, median 0.455 s
per car and worst 1.53 s, excluding mesh file loading. Every output depth/mask array
matched the previous prepared input exactly. Slow reference parity tests also cover
random triangles, degenerate triangles, occlusion and forced chunk boundaries.

Repeated example inference is about 25 ms median and 29 ms p95 on this host with
two CPU threads, including checks/conversion. These are bounded local measurements;
other machines and larger meshes may differ. Chunking bounds candidate-pixel
batches, while mesh storage and per-face setup still scale with input triangle count.

## Stability and real examples

The consolidated feature suite passes 58 tests; Ruff lint/format, Python
compilation and diff whitespace checks pass. VTK emits dependency deprecation
warnings. Ten of twelve real cases pass default geometric range checks; Fortuner
and Porsche are rejected by default and evaluated only as explicit diagnostics.


The reusable harness exercises 11 cached vehicles (7 train, 2 validation, 2 test),
a documented canonical Porsche 911 mesh not used in training, and synthetic sphere
and rectangular-prism inputs. Each accepted/diagnostic result is repeated 50 times.

- Repeated coefficients have zero spread in the recorded sweep.
- A fresh Ford render matches its cached prediction exactly.
- Equivalent meter/mm conversion, 90-degree axis rotation and translation preserve
  every mask. Maximum depth drift is 4.89e-6 car lengths due to float32 source
  coordinates, below the declared 1e-5 tolerance; coefficient drift is zero.
- A 300-inference memory probe after 20 warmups had zero output drift. RSS increased
  from 329.0 MB to 334.0 MB over the bounded probe (about 5 MB); this is not an
  indefinite leak guarantee.
- A changed Re formerly yielded Cd of order millions because constant-condition
  normalization amplified the input. Current inference rejects it, including when
  geometry diagnostics are enabled. A 1e-7 condition representation drift stays
  invariant rather than being amplified.

| Example | Role | Cd / outcome | Interpretation |
| --- | --- | ---: | --- |
| Ford F-150 Raptor | Training | 0.5813 vs source 0.560 | Training fit only |
| Cybertruck | Training | 0.3631 vs source 0.384 | Training fit only |
| BYD Seal | Held-out test | 0.2598 vs source 0.219 | Label conditions remain mixed |
| Toyota Auris Wagon | Held-out test | 0.4080 vs source 0.280 | Large real error retained |
| Toyota Fortuner | Validation | Default rejects height ratio; diagnostic 0.3615 | Geometry outside trained range |
| Porsche 911 | Unseen vehicle | Default rejects ratios; diagnostic 0.0366 | Diagnostic is not credible aerodynamic prediction |
| Synthetic sphere | Nonvehicle stress | Rejected negative Cd | No coefficient is returned as usable output |
| Vehicle-proportioned prism | Nonvehicle stress | Numeric diagnostic 0.3034 | Ratios in range do not establish vehicle semantics |

The new guards prevent known invalid usage; they do not improve the trained
coefficient model's generalization. Weights are unchanged. The initial two-car
held-out Cd MAE remains about 0.0844 for five views, and no real Cl accuracy claim
is supported. More consistent labeled geometry/condition cases are required for
engineering accuracy acceptance.

## Reproduction and artifacts

With the repository Python environment and skill scripts on PYTHONPATH:

```bash
python -m pytest -q .agents/skills/predict-vehicle-cd/tests
python -m vehicle_cd_learning.validate_examples \
  outputs/predict-vehicle-cd/reference-assumed/prepared \
  outputs/predict-vehicle-cd/reference-assumed/five/model.pt \
  outputs/predict-vehicle-cd/reference-audit/cd_training_manifest.json \
  --extra-vehicle /path/to/documented-canonical-unseen-car.vtp \
  --output-dir outputs/vehicle-learning/example-checks --repeats 50
```

Use `predict --allow-geometry-extrapolation` only for explicit numerical diagnostics.
Fixed-condition range errors and invalid inputs remain errors. Evaluation itself
explicitly opts into geometric extrapolation to measure held-out shapes and
records which cases use it; it does not imply default-mode acceptance.

Generated local evidence is under ignored directories:

- `outputs/predict-vehicle-cd/reference-assumed/examples/example-validation.json`:
  per-case roles, source labels, default rejection/diagnostic status, repetitions,
  image checks, coordinate invariance, and separated timing.
- `.../examples/*-preview.png`: inspected real and synthetic five-view contact sheets.
- `.../examples/memory-stability.json`: bounded repeated-process RSS samples.
- `.../examples/cli-policy-check.json`: default Porsche rejection, explicit
  diagnostic opt-in and Re rejection with the override enabled.
- `.../query-prediction-optimized.json`: same-input end-to-end performance comparison.

`learning.md` specifies the data and checkpoint contract; `learning-validation.md`
preserves the original training and accuracy report. All personal mesh paths and
weights stay out of version control.
