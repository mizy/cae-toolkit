# Five-view aerodynamic coefficient learning

The [independent-sampling evaluation](learning-stability-first.md) finds that the retained model can reverse a real wheel change when the reference and query use independently sampled raw depth images. The report includes a same-car counterexample, a comparison with earlier end-to-end CNN training, and the original vehicle, aircraft and mirror replays. Zero drift on registered rays measures renderer repeatability.

The [absolute coefficients and design-change audit](learning-absolute-and-design-change.md) measures the retained fifth epoch on whole-vehicle Cd/Cl, matched wheel replacements, a five-level mirror deformation and independently shifted views. It includes same-ID absolute baselines, a calibrated fixed-reference Rim estimate, and the resolution of small signed drag changes against CFD.

The [retained-epoch5 Cl diagnosis](learning-cl-sampling-diagnosis.md) adds validation-only component/view interventions and train-only gradient checks. Positive horizontal sampling errors are amplified by the paired readout despite lower average CNN feature variation. Weak-ridge and bounded Cl-row repairs improve some phases but fail joint per-phase protection, so epoch5 remains the starting point. The wider readout experiment proposed there is covered by the sampling stability validation.

The [full-pool online image-jitter comparison](learning-online-jitter-full-pool.md) completed two six-epoch continuations with all4,121 vehicle/aircraft train records and seed42. An auxiliary real-label coefficient readout updates B's shared CNN while matched Rim pairs retain delta supervision. Online jitter gives mixed raw-render phase results, worsens Cl sensitivity at the main phase and weakens amplitude versus the matched control; no candidate passes all gates. The user subsequently chose to retain online epoch5 as the starting point for further experiments (`retained-online.pt`, `retention.json`), with original B kept for comparison. Auxiliary absolute outputs are evaluated separately from B's delta capability.

The [sampling-consistency repair](learning-sampling-consistency.md) adds a canonical source-geometry entry point and records three completed six-epoch continuations. None passed the joint sampling-stability and real-response gates, so B is retained. Re-rendering into a common frame preserves its original predictions and removes upstream phase differences at that input boundary; bare-network phase robustness and nominal aerodynamic error remain unresolved.

The [single-run delta factorial](learning-delta-factorial.md) shifts the objective to same-condition ΔCd/ΔCl: genuine256/512 inputs crossed with native8/16 spatial pooling, matched initialization and executed pair exposure. B(512/8) wins validation and the fixed-reference mean errors, with stronger Cd response than the tested Radiant SDF package. Unseen-to-unseen Cl and camera-sampling stability remain weak; this does not promote a generic production predictor.

The [2026-11-17 repair comparison](learning-accuracy-repair-20261117.md) tests paired CNN source-gradient projection, parameter averaging and a unified multiscale residual. Six continuations completed; projection/averaging did not pass the joint validation gates. The retained neural base plus a Cl-only residual slightly lowers aggregate Cl MAE, leaves Cd unchanged and still worsens Windsor Cl. It does not replace the previously stronger standalone Rim Cd candidate or establish a generalization fix.

The [completed three-seed end-to-end and multiscale comparison](learning-end-to-end-radiant.md) adds train-side geometry-permutation evidence and current source-gradient measurements. The validation-selected neural ensemble improves aggregate Cd/Cl point estimates; a separate multiscale readout gives lower anchored Cd MAE than the tested Radiant SDF package on all three fixed references and exceeds a zero-geometry-response control. These are distinct candidates, with source regressions and known-family evaluation limits recorded. Original defaults remain available.

The [fixed-baseline pre-compression readout validation](learning-kernel-readout.md) clarifies the scope of the earlier27% Rim-only result and tests the corresponding5120-feature route across the vehicle cohort. Both historical-test MAE point estimates improve slightly, but paired uncertainty includes zero and some sources regress. Keep the previously retained absolute-MAE candidate as the working baseline; no default promotion occurred. The document defines the next paired end-to-end validation and marks it as not yet run.

For the current vehicle A candidate, use the [dimensionless input entry point](learning-dimensionless-input.md): Re is supplied directly, absolute speed/length features are removed, and preparation/training/inference share the 18-value plus18-mask contract. The legacy5/9-value contracts described below remain available with their original checkpoints.

The [absolute Cd/Cl readout experiment](learning-absolute-readout.md) follows the user's current absolute-error-only priority. A simple MAE-continuation checkpoint is available through the original dimensionless loader as an exploratory candidate; per-source regressions are reported and the old default is preserved. Feature normalization/balancing did not establish stable Cl improvement.

The [shared coefficient-head comparison](learning-head-variants.md) separately tests original-head continuation, linear/MLP/normalized corrections and RBF on the same frozen fused features. RBF wins validation within this experiment but does not surpass the retained prior continuation on both historical-test MAEs; default weights remain unchanged.

The [2026-11-16 completion record](learning-label-completion.md) documents separate vehicle and joint datasets: 930 vehicle plus 1497 aircraft Re estimates, with 26 audit-only Cl pseudo-labels. Observed supervision and frozen checkpoints remain unchanged.

The [paired-detail demonstration](learning-detail-contribution.md) includes matched A1/A2 continuation, fixed-reference and all-query pair metrics, full source-geometry replay, and an offline interactive CFD comparison. The current delta-loss candidate is not promoted: its response remains weak and results depend on the reference.

The [subsequent detail diagnosis and external-model comparison](learning-radiant-detail-diagnosis.md) adds frozen-stage information probes, train-only gradient measurements and actual cached Radiant SDF predictions on the same92 queries. This is a raw-source model-level comparison, with unverified external training exposure/live identity; it does not rank architectures under equal training data.

## Scope and architecture

Predict mean Cd and Cl directly from orthographic numerical depth/mask views and
conditions. No CFD volume mesh or flow-field decoder is required. Surface meshes
are used for rendering. This is a conditional supervised surrogate; deployment
accuracy must be established on labeled cases from the intended domain.

Four views (front, rear, right, top) share a CNN. A separate CNN encodes a
higher-resolution bottom view. Each outputs 128 features; fixed-order
concatenation preserves view identity. A parameter MLP outputs 32 features.
The five-view regression head is 672 -> 256 -> 64 -> target count. The model
learns feature contributions; there are no manually chosen view weights.
Available ablations are `parameters`, `four`, and `five`. Cd-only training is
supported when Cl labels are unavailable; missing Cl is never replaced by zero.

## Geometry convention

Declare units as `unit_scale_to_m` and raw `front_axis`/`up_axis` explicitly.
Canonical front is -X, right is +Y, up is +Z. X/Y are centered and the lowest
point is placed at z=0. Use complete vehicles including tires if ground height
is to have its usual physical meaning. Detached geometry and omitted tires can
invalidate the implied ground relationship. All dimensions are divided by car
length, preserving proportions. No inferred front/rear choice or anisotropic
scaling occurs in this learning path.

All cameras use a fixed window (default 1.3 car lengths), depth origin, and
length-based scale. Models exceeding it are rejected. Default body views are
256x256 and bottom is 512x512. Inputs are lossless float32 NPZ arrays, each
`(2,H,W)` depth plus binary mask; PNG contact sheets are inspection only.
Subpixel features and occluded geometry may be lost. A high-resolution image
cannot recover an invisible internal channel.

Parameters are width/length, height/length, log(Re), sin(yaw), cos(yaw).
Re must use the same reference length in every case. Scale is represented through
Re for this incompressible model. Ground, wheel, turbulence, solver, averaging,
reference-area and force-axis conventions are fixed per dataset/checkpoint.
Geometry must capture ride height changes; the code does not guess body clearance.
For bodies whose source mesh omits tires/supports, declare
`geometry.ground_clearance_m` as the measured distance from the lowest mesh point
to the ground. The default is zero for a complete vehicle with ground-contacting
tires. This offset survives rendering; it does not change the body dimensions or
the geometry hash used to prevent the same body leaking across splits.

## Source manifest

The paths below are examples; supply your own local geometry. Source manifests
are JSON, with one record per geometry-condition pair:

```json
{
  "data_kind": "cfd",
  "provenance": "Dataset name, version, license and simulation settings",
  "target_names": ["cd", "cl"],
  "conditions": {
    "ground": "moving at freestream speed",
    "wheels": "rotating at matching speed",
    "turbulence": "SST; document inlet turbulence here",
    "solver": "document solver/version and convergence settings",
    "averaging": "document steady solution or averaging interval",
    "reference_area": "zero-yaw frontal projected area, per case",
    "force_axes": "drag along freestream; lift +Z; yaw positive toward +Y"
  },
  "cases": [
    {
      "case_id": "sedan-a-condition-0",
      "geometry_id": "sedan-a",
      "family_id": "sedan-family-a",
      "split": "train",
      "geometry": {
        "path": "meshes/sedan-a.vtp",
        "unit_scale_to_m": 1.0,
        "front_axis": "-x",
        "up_axis": "+z"
      },
      "reynolds": 3000000,
      "yaw_deg": 0,
      "cd": 0.28,
      "cl": 0.03
    }
  ]
}
```

A training dataset needs distinct train and val families; acceptance needs test
families too. The single record above illustrates schema only. All derivatives
of one base vehicle must share `family_id` and split. A geometry or geometry hash
cannot cross splits. Optional `pair_id` groups design variants under exactly the
same conditions for delta and direction metrics. Target standardization uses
training cases only; a target without variation is rejected.

Paired evaluation requires identical profile and yaw, and compares Re with a
relative tolerance of `1e-7` to accommodate float32 STL coordinate rounding.
This is a numerical tolerance, not permission to merge different flow settings;
`pair_tolerance` separately controls the minimum target change for direction
accuracy. A real change in reference length can change Re even at the same
speed and viscosity, so verify the source conditions before assigning `pair_id`.

The strict default `split_policy=family_disjoint` evaluates new vehicle families.
An explicit `split_policy=geometry_disjoint` supports published datasets made
from morphs of one base vehicle: keep their common `family_id`, and record the
source partitions under `split_provenance`. Geometry IDs and hashes still cannot
cross splits. This evaluates held-out shapes within an existing family, with
seen/unseen-family counts in evaluation reports; it does not establish accuracy
on a different vehicle family.

Use `data_kind=synthetic` for software fixtures. `reference_assumed` identifies
mixed public/reference Cd labels assigned assumed conditions with user consent.
Such labels can exercise the learner, but cannot demonstrate matched-condition
CFD accuracy. Keep these runs separate from coherent CFD/experimental datasets;
combine them only with an explicit source/condition policy and new validation.
Do not promote the old similarity predictor's outputs to ground truth.

## Commands

From the repository root, activate its Python environment and expose skill scripts:

```bash
source .venv/bin/activate
uv pip install -r .agents/skills/predict-vehicle-cd/requirements-learning.txt
export PYTHONPATH="$PWD/.agents/skills/predict-vehicle-cd/scripts"

python -m vehicle_cd_learning.prepare /path/to/manifest.json \
  --output-dir outputs/vehicle-learning/prepared
python -m vehicle_cd_learning.train outputs/vehicle-learning/prepared \
  --output-dir outputs/vehicle-learning/five --variant five --epochs 100
python -m vehicle_cd_learning.evaluate outputs/vehicle-learning/five/model.pt \
  outputs/vehicle-learning/prepared --output-dir outputs/vehicle-learning/evaluation \
  --pair-tolerance 0.001
```

Use a fresh output directory for preparation and training. Repeat training with
`--variant parameters` and `--variant four` in separate output directories. The
training CLI defaults to CPU and supports `--device` and `--seed`.

For single-case inference, create a JSON with `geometry`, `reynolds`, `yaw_deg`,
and `conditions` using the same conventions as the source manifest:

```bash
python -m vehicle_cd_learning.predict outputs/vehicle-learning/five/model.pt \
  /path/to/query.json --output outputs/vehicle-learning/prediction.json
```

The checkpoint owns model shape, target names, render configuration, normalization,
training parameter ranges, fixed conditions, provenance and frozen dataset hash.
Inference rejects fixed-condition mismatch, unsupported Re/yaw ranges, and geometry
ratios outside the trained range by default. `--allow-geometry-extrapolation` enables
explicit geometric diagnostics; it never bypasses condition checks or invalid
inputs. It also rejects nonfinite or negative passive-vehicle drag predictions.
Range membership is not calibrated uncertainty or proof that the input is a vehicle.

## Verification and result interpretation

Evaluation reads only the requested split, verifies dataset/checkpoint identity
and checksummed numerical views, and reports MAE, RMSE, p95 absolute error, family
metrics, paired delta errors and direction accuracy above a chosen noise tolerance.
It also writes a scatter plot, predictions, and train-mean baseline. Evaluation
explicitly permits geometric extrapolation to measure held-out cases and marks
those cases/counts; its numbers are not default-mode acceptance. Timed inference
includes input validation, tensor conversion, and the model, after warmup. The
single-case CLI reports loading, rendering, inference, and total time separately;
its first inference is not warmed up.
A declared zero pair tolerance includes every nonzero difference; engineering
validation should set it from CFD/experimental repeatability.

Software verification:

```bash
python -m pytest -q .agents/skills/predict-vehicle-cd/tests
python .agents/skills/predict-vehicle-cd/tests/make_learning_fixture.py \
  outputs/vehicle-learning/synthetic-source
python -m vehicle_cd_learning.prepare \
  outputs/vehicle-learning/synthetic-source/manifest.json \
  --output-dir outputs/vehicle-learning/synthetic-prepared \
  --ordinary-size 64 --bottom-size 128
```

The analytic fixture provides 48 cases in 12 family groups, visible roof/underbody
changes and two conditions. Its labels are deliberately non-aerodynamic. Learning
those labels demonstrates software behavior only. Compare all variants and seeds
on real independent families before claiming engineering accuracy. Acceptance must
specify an error tolerance based on the smallest design change you need to detect;
there is no built-in universal acceptable Cd/Cl percentage.

## Existing reference library

```bash
python -m vehicle_cd_learning.import_references \
  .agents/skills/predict-vehicle-cd/assets/reference-library/seed_manifest.json \
  --output-dir outputs/vehicle-learning/reference-audit
```

This inventories source labels and available models. It preserves missing fields
and label provenance. Eligible explicitly normalized meshes can be rendered and
used for exploratory Cd training with declared assumed conditions; this does not
supply missing Cl or recover original CFD boundary conditions. Raw models without
confirmed orientation need annotation first. Outputs and weights belong outside
Git; do not commit personal meshes or absolute-path local manifests.

## Example and stability checks

Run the repeatability and real-mesh sweep against a frozen prepared dataset and
its source manifest. The source manifest must carry the explicit mesh paths and
axis/unit declarations used to make the prepared images:

```bash
python -m vehicle_cd_learning.validate_examples \
  outputs/vehicle-learning/prepared \
  outputs/vehicle-learning/five/model.pt \
  /path/to/source-manifest.json \
  --output-dir outputs/vehicle-learning/examples \
  --extra-vehicle /path/to/canonical-unseen-car.vtp \
  --repeats 50
```

The report checks cached cases across train/validation/test groups, repeats
inference, compares a fresh geometry render with its cached prediction, tests
metre/millimetre and axis/translation invariance, and runs synthetic nonvehicle
stress inputs. Supply an unseen vehicle only when its canonical front/up axes and
unit scale are documented. Synthetic shapes exercise input handling. Negative drag is rejected; valid
numerical outputs on nonvehicles have no physical prediction claim. Reference Cd comparisons retain their source and condition
caveats. The report separates render time, checkpoint loading and repeated
inference time. Keep reports and previews under the ignored `outputs/` directory.

Current measured performance, supported-domain checks and example outcomes are
recorded in `learning-stability.md`. The original `learning-validation.md` preserves
the first implementation results and the limitations of its labels.

For missing reference conditions, the reference importer uses a *hypothesized*
low-drag straight-ahead condition: 120 km/h, zero yaw, moving ground and wheels,
20 °C air approximated by density 1.204 kg/m³ and viscosity 1.81e-5 Pa·s.
Re is derived using the recorded vehicle length. The speed, equipment settings,
and source-label conditions are assumptions, not recovered measurements; the
reported Cd is not proved to be the minimum. It is not justified to infer a
speed response from a single assumed label per vehicle. Existing checkpoints
and prepared datasets retain the conditions they were created with and must be
retrained separately for a new manifest.

## Fit every built-in labeled vehicle

The actual built-in `reference_cars.json` defines membership (26 labels), not the
seed manifest's pending entries. `prepare_builtin` resolves all its models using
an explicit local resolution JSON, then aligns side silhouettes against the built-in
images without using Cd to choose orientation. Visual review remains required:
the MPV, RAV4 and Fortuner legacy references were found upside down. Resolution
entries may use `{"path": "/absolute/model.vtp", "invert_z_after_matching": true}`
to record that visual correction. All numerical views are regenerated afterwards.
Automatic axis/unit normalization for older meshes is an assumption; inspect the
orientation audit and contact sheets before training. Original personal meshes are
not edited.

`train --fit-all` uses every prepared case, including any pre-existing split labels,
and saves the specified final epoch. It records zero independent validation cases,
no held-out IDs, and **training_fit** metrics. These metrics are not a generalization
claim. Choose a fixed training duration before inspecting independent results.

```bash
python -m vehicle_cd_learning.train /path/to/all26/prepared \
  --output-dir outputs/all26/final --fit-all --epochs 60 --batch-size 4
python -m vehicle_cd_learning.cross_validate /path/to/all26/prepared \
  --output-dir outputs/all26/cross-validation --folds 3 --epochs 60
```

Cross-validation trains fixed-epoch models on disjoint family groups and evaluates
each held-out family exactly once. It does not use a held-out fold for checkpoint
selection. Fold reports retain default-policy rejection reasons while numerical
out-of-fold errors use the raw model output, including negative or extrapolated
predictions, to avoid hiding failed cases. The final all-sample model is a separate
artifact. Every known label can therefore contribute to the final model without
misrepresenting its training error as an independent test.

## Representation jitter experiment

`train --augment` adds training-only sampling jitter and depth noise after batch
loading. A shared random XYZ offset projects to each view using the renderer's
`VIEW_AXES` convention; magnitude is at most one ordinary-view pixel per axis.
Depth and mask use the same sampling grid. Interpolated depth is normalized by
mask coverage; masks remain binary and background depth stays zero. Independent
Gaussian depth noise has standard deviation 1e-4 of the normalized depth window;
values are clipped to [0,1]. Geometry files, conditions and labels are unchanged.
This is a measurement/representation-robustness hypothesis, not new CFD labels.

Validation, inference and cached images remain unaugmented. Augmentation has its
own seeded generator, so it does not alter model initialization or batch ordering.
`--model-size small` reduces the five-view model from 1,089,569 parameters to 44,681
for Cd-only output. Both options are saved in checkpoint/report metadata. Old
checkpoints without a size field retain the standard architecture.

```bash
python -m vehicle_cd_learning.cross_validate /path/to/all26/prepared \
  --output-dir outputs/augmentation/small-jitter --folds 3 --epochs 60 \
  --model-size small --augment
python -m vehicle_cd_learning.train /path/to/all26/prepared \
  --output-dir outputs/augmentation/final-small-jitter --fit-all --epochs 60 \
  --batch-size 4 --model-size small --augment
```

Use the same family split and epoch count for comparisons. All augmented versions
of a vehicle stay inside its training fold. Read `learning-augmentation.md` for the
measured four-group comparison; increased image variation is not an increase in
independent aerodynamic examples.

## Full-view multiscale attention (no part crops)

Per-part mirror crops were explicitly rejected and their experiment was stopped.
They are not part of the current model, input contract or training CLI. Only the
five complete depth/mask images are used; no part boxes, location annotations or
per-part encoders are required.

`--encoder pyramid` builds 1/2, 1/4 and 1/8 resolution feature maps for each full
view, projects them to eight channels, and fuses them at 1/2 resolution before
spatial pooling. `--encoder pyramid_attention` adds lightweight learned channel
and spatial gates to that same full-view feature map. The block is CBAM-style,
not quadratic all-pairs Transformer attention. It covers every image position
using the same convolutional processing. Attention scores are not physical drag
contributions or proof of recognizing a named part.

```bash
python -m vehicle_cd_learning.prepare /path/to/all26/source.json \
  --output-dir outputs/full-view/prepared --ordinary-size 1024 --bottom-size 1024
python -m vehicle_cd_learning.cross_validate outputs/full-view/prepared \
  --output-dir outputs/full-view/pyramid-attention --folds 3 --epochs 60 \
  --model-size small --encoder pyramid_attention --device mps
```

Compare `plain`, `pyramid`, and `pyramid_attention` at the same input resolution,
folds, epoch count and training device. CPU is still the default. The current
1024 full view corresponds to about 5.76 mm/pixel for a 4.5 m car in the fixed
1.3-length window. This improves small-part representation over 256 inputs but
does not guarantee robust detection of every 5 mm modification. Higher global
resolution is needed for stricter geometric sensitivity targets. Models cannot
recover parts absent from the input surface, and scalar Cd labels alone do not
establish correct responses to local physical modifications.
