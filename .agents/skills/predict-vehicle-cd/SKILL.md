---
name: predict-vehicle-cd
description: Estimate a vehicle drag coefficient (Cd) from an arbitrary 3D car mesh by automatically normalizing vehicle axes and units, rendering a standardized side-view depth map with an alpha silhouette mask, comparing those artifacts against a packaged reference vehicle library, and similarity-weighting reference Cd values. Use when Codex needs a fast geometry-similarity Cd estimate for VTP, VTK, STL, OBJ, GLB, or GLTF vehicle meshes, especially before running CFD or a learned model.
---

# Predict Vehicle Cd

For the verified vehicle/aircraft training membership and downloaded-only sources,
read [the training data inventory](../cad-surface-mesher/docs/aerodynamic-training-datasets.md).

Estimate Cd from a 3D vehicle mesh with the packaged side-view reference library. The predictor accepts raw vehicle meshes and first normalizes them to the project vehicle convention: length on `X`, width on `Y`, height on `Z`, top at `+Z`, front target `-X`, meters-scale dimensions, and ground at `z = 0`. Treat the result as a fast shape-similarity prior for triage, candidate ranking, or sanity checks before higher-fidelity CFD/model inference.

## Quick Start

From the repository root, run the standalone predictor from this skill directory:

```bash
cd .agents/skills/predict-vehicle-cd
python scripts/vehicle_cd_similarity.py predict /path/to/car.vtp \
  --library assets/reference-library/reference_cars.json \
  --category pickup \
  --output-dir /tmp/cd_estimate
```

Supported input mesh formats are `vtp`, `vtk`, `stl`, `obj`, `glb`, and `gltf`.

The script prints JSON with:

- `cd`: similarity-weighted Cd estimate
- `confidence`: qualitative confidence from the top side-view match
- `query_category`: optional category passed through `--category`; when supplied, same-category references receive a small ranking bonus
- `orientation`: which normalized front/rear candidate won (`front_as_normalized` or `front_rotated_180_z`)
- `normalization`: raw bounds, axis order, detected unit/scale, final bounds, and any axis warning
- `top_k`: nearest reference cars, scores, weights, and Cd values
- `artifacts`: generated query side-view `depth.png` with alpha mask and debug `normal.png`
- `normalized_mesh`: generated `query_normalized.vtp` when `--output-dir` is set

## Workflow

1. Use `scripts/vehicle_cd_similarity.py predict` for one-off estimates.
2. Inspect `normalization` and `query_normalized.vtp` when a source model has unknown units or orientation.
3. Inspect `top_k` before trusting the scalar Cd.
4. Inspect `cd_source` because the packaged Cd labels come from mixed public wind-tunnel, paper, OEM, and secondary-publication conditions.
5. For reference cars with known real dimensions, pass `--target-dimensions L,W,H` or set `target_dimensions_m` in the manifest so the rendered artifacts use real length, width, and height.
6. Rebuild the packaged library after changing reference cars:

```bash
python scripts/vehicle_cd_similarity.py build-library \
  assets/reference-library/seed_manifest.json \
  --output-dir assets/reference-library
```

## Method

The predictor is intentionally self-contained:

- Reads `vtp`, `vtk`, `stl`, `obj`, `glb`, and `gltf` meshes through PyVista, with a `trimesh` first pass for `glb/gltf` scene files and an optional fallback when PyVista cannot load a file, then triangulates and extracts the surface.
- Normalizes vehicle geometry before rendering:
  - longest axis -> length `X`, middle axis -> width `Y`, shortest axis -> height `Z`
  - uses horizontal PCA to resolve length/width when raw horizontal spans are close
  - detects meter, millimeter, centimeter, and unit-scaled meshes from the length span
  - uniformly scales to meters, keeping realistic 2.5-7.0 m lengths when already plausible
  - can force real exterior dimensions with `--target-dimensions L,W,H` or per-car `target_dimensions_m`
  - flips `Z` only when the top/bottom point distribution strongly indicates the mesh is upside down
  - centers `X/Y` and aligns the lowest point to `z = 0`
  - compares both the normalized pose and a 180 degree rotation around `Z` for front/rear ambiguity
- Renders only the current side view by CPU rasterizing triangles into:
  - normalized side depth PNG whose alpha channel is the binary silhouette mask
  - outward normal RGB map for diagnostics
- Scores each reference with a weighted blend:
  - mask IoU
  - depth similarity on the shared mask
  - normal cosine similarity on the shared mask, currently weighted at `0.0` because leave-one-out validation showed it was not useful for this library
- When `--category` is provided, adds a small same-category ranking bonus before top-k weighting.
- Computes Cd with inverse-distance softmax weighting over top-k references.

## Cached whole-car multiview similarity

Use `scripts/vehicle_cd_multiview.py build-library` and `predict` for the optional
float-preserving five-view matcher. Read `references/multiview.md` for the cache,
query format, condition assumptions, self-reference exclusion and same-fold
validation results. The first 26-car comparison has slightly worse average error
than the legacy side matcher, so keep the legacy method available as the baseline.

## Supervised five-view Cd/Cl learning

For training and deploying the depth-image CNN predictor, read
`references/learning.md`. It covers five-view preparation, explicit geometry
conventions, Cd-only or Cd/Cl regression, family-disjoint validation, and the
`parameters` / `four` / `five` ablations. The existing similarity predictor remains
a separate baseline; its mixed-condition reference labels are exploratory data,
not matched-condition CFD truth. Install optional `requirements-learning.txt`.

For public geometry/coefficient collection, read `references/public-datasets.md`.
For selective DrivAerRim acquisition and complete-car reconstruction, read
[the public-data expansion record](../cad-surface-mesher/docs/aerodynamic-data-expansion-20261109.md).
For the early physics-only conditional flow experiments, read
`references/physics-flow-prototype.md`. These are two-dimensional analytical
verification cases, not vehicle-flow models or substitutes for CFD.
For the real-vehicle 3D weak-supervision experiment using the frozen dataset,
read `references/vehicle-flow-weak-supervision.md`. It records the proxy boundary
and closure assumptions, integration checks, training and fixed-weight inference.
For full-view detail pooling, optional condition-dependent view attention and the
frozen-split comparison command, read `references/learning-cnn-design.md`.
For the paired 32/128 per-view compression experiment with a shared Cd/Cl head,
read `references/learning-feature-width.md`.
For the current absolute-Cd/Cl-only readout continuation, compatible candidate
checkpoint and per-source regressions, read `references/learning-absolute-readout.md`.
For the separate fixed-encoder head-architecture and RBF correction comparison,
read `references/learning-head-variants.md`; previous readout arms kept the same head.
For the fixed-baseline pre-compression comparison, the scope of the earlier27%
Rim result and the next accuracy-validation protocol, read
`references/learning-kernel-readout.md`.
For the completed three-seed full-network continuation, geometry-use diagnosis,
multiscale readout and same-reference absolute-Cd comparison with Radiant, read
`references/learning-end-to-end-radiant.md`. Keep the overall validation winner
separate from the candidate with the best observed Rim detail result.
For the subsequent source-gradient/parameter-averaging repair attempts and the
unified coefficient-correction entry point, read
`references/learning-accuracy-repair-20261117.md`; its small Cl improvement does
not resolve the Rim Cd or Windsor generalization issues.
For the later single-run256/512 ×8/16 paired-delta comparison, native512 inputs,
direction/amplitude evaluation and the sampling-sensitivity counterexample,
read `references/learning-delta-factorial.md`. Its B candidate predicts deltas,
and keeps the existing absolute-coefficient entry points separate.
For the completed sampling-consistency continuations and canonical geometry
inference entry point, read `references/learning-sampling-consistency.md`.
All three learned continuations failed the joint preservation/stability gates;
keep B. The new entry re-renders source geometry into a shared canonical frame;
it does not establish invariance of the bare network to arbitrary shifted images.
For the later single-seed online image-jitter comparison on all4,121 vehicle and
fixed-wing training records, read `references/learning-online-jitter-full-pool.md`.
Both six-epoch arms completed. The auxiliary coefficient readout shares B's CNN;
Rim deltas still use the original pair head. Raw-render sampling improvements are
mixed, including worse Cl sensitivity. Per the user's later decision, retain
online epoch5 as the next experiment's starting point through
`outputs/aerodynamic-online-jitter-20261117/retained-online.pt` and `retention.json`.
Keep original B as the comparison baseline and retain all measured regressions.
For the retained-epoch5 Cl investigation, read
`references/learning-cl-sampling-diagnosis.md`. Validation interventions locate
direction-dependent bottom-view sensitivity and amplification in the pair adapter,
hidden readout and Cl row. Two row-only repairs fail joint per-phase preservation;
the continuation checkpoint stays `retained-online.pt`.
For stability-first geometry input and the completed broader phase experiment,
read [the sampling stability validation](references/learning-stability-first.md).
`vehicle_cd_learning.registered_prediction` runs the retained checkpoint on a
shared physical sampling grid. Zero drift on that grid measures renderer
repeatability. The retained weights and the tested continuations still fail
on independently sampled raw views of the same wheel geometry.
For the auditable complete-car wheel-assembly change demo, paired continuation,
and prediction-versus-source-CFD delta checks, read
`references/learning-detail-contribution.md`; changes in prediction or input pixels
do not certify physical part-force contributions.
Keep matched-condition CFD pools separate from the assumed-condition reference
pool. A geometry-disjoint split within one base family must be reported as such.
Read `references/learning-full-attention.md` for full-source collection, exact
Windsor alias removal, and the retained channel/spatial/view attention workflow.
Read `references/learning-builtin-inclusive.md` for the conditional attention
model that actually trains all 26 built-in Cd cases together with the CFD pools.
Its declared source profiles preserve assumed conditions, mask missing Cl, and
limit the built-in profile's CLI output to Cd.

## Reference Library

The packaged library lives in `assets/reference-library/reference_cars.json` with side-view artifacts beside it. The Cd values include source provenance in each `cd_source`; align conditions before using the scalar estimate for engineering decisions.

Read `references/method.md` when changing the scoring formula or adding more views. Read `references/reference-library.md` when editing the reference-car schema.
