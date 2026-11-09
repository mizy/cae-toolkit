# Whole-view CNN design and public CFD experiments

## Retained architecture

Every sample still supplies five complete depth/mask views. The existing
`PyramidEncoder` owns the half-, quarter- and eighth-resolution feature maps.
`--encoder pyramid_detail` changes its final pooling to foreground-masked mean
and maximum in 8x8 spatial bins. Mean pooling retains distributed shape evidence;
maximum pooling provides a path for a small strong activation. Empty bins are
zero. No vehicle part labels, crops or manually selected regions are introduced.
The additional maximum features increase model capacity, so an accuracy change
cannot be attributed solely to detail preservation.

`--fusion attention` uses `ViewAttention` in the existing `CoefficientModel`.
It scores each complete-view feature from the feature, encoded conditions and
view identity. The resulting softmax weights multiply each view feature by
`number_of_views * weight` before the same ordered concatenation head. Initial
scores are zero, giving unit scales. This permits per-case view selection while
retaining view identity. Weights are not drag contribution percentages, and the
current fixed-yaw CFD pool cannot validate a learned response to changing yaw.

Legacy encoders and checkpoints default to `fusion=concat`. Neither a new model
nor the old predictor is replaced automatically.

## Reproducible comparison

Use a fresh output folder and a prepared dataset with train/val/test splits:

```bash
export PYTHONPATH="$PWD/.agents/skills/predict-vehicle-cd/scripts"
.venv/bin/python -m vehicle_cd_learning.compare_cnn /path/to/prepared \
  --output-dir outputs/predict-vehicle-cd/cnn-comparison \
  --seeds 17 42 73 --epochs 60 --batch-size 4 --device mps
```

The runner freezes the dataset fingerprint, seeds, hyperparameters and arms in
`protocol.json` before training. The parameter-only baseline, ordinary pyramid,
detail pyramid and detail pyramid with view attention use the same source data
and split. Checkpoints are chosen only by validation MSE; test evaluation begins
after all model training has completed. `comparison.json` reports each seed and
mean/std across seeds. These seeds share one test set and are not independent
datasets. MSE gives equal weight to standardized Cd and Cl when both are labeled.
The current runner uses explicit raw diagnostic evaluation and retains strict
inference rejection reasons for every held-out case. Production prediction keeps
its existing condition and geometry checks. No rejected cases are silently
removed to improve the aggregate score.

Keep the existing 26-car family-disjoint experiment separate from public CFD
experiments. The old labels have mixed provenance and assumed conditions. A new
small error on morphs of one CFD base car cannot be compared numerically with
the 26-car MAE as an improvement on unseen production vehicles.

## Data and geometry boundaries

Re uses each complete body's length, with source speed and kinematic viscosity.
Source force reference area/length are retained separately; a source wheelbase
Re must not silently become a body-length Re. Cd and Cl use a documented reference
area and force axes. Missing density remains missing when dimensionless labels
and kinematic viscosity suffice; it is not manufactured to fill the manifest.

Public source importers retain immutable revisions and file hashes. Keep original
labels and meshes under ignored outputs and never train on the old predictor's
estimated coefficients. Use the explicit `geometry_disjoint` policy for morphs
of one base family and preserve their common family ID. Default family-disjoint
validation and unconditional geometry-ID/hash leakage checks remain in place.

For a mesh without ground-contacting tires, an explicit `ground_clearance_m`
keeps the body above the ground in numerical views. It is measured in metres,
independent of source coordinate units. This is necessary for simplified bodies
that omit support legs. The caller must supply the actual clearance.

The full-view representation can still miss occluded or subpixel surfaces.
Improved pooling or attention cannot establish accurate mirror/underbody delta-Cd
without paired CFD labels and a sensitivity validation target.

## Ownership and evidence

Preparation reuses `prepare` and `render_five_views`; training and evaluation reuse
`train_model`, `load_model` and `evaluate`. `compare_cnn.compare` owns the fixed
ablation protocol and consumes these existing entry points. `ViewAttention` owns
only its learned view scores and is consumed by `CoefficientModel.forward`.

Current run artifacts live under `outputs/predict-vehicle-cd/cnn-design-20261030/`.

## Results from 2026-10-30

The collected CFD pool contains 108 geometry/Cd/Cl records across three base
families: 60 DrivAerML, 24 AhmedML and 24 WindsorML. They are separate prepared
datasets and models because the CFD boundary conditions and source force-axis
definitions differ. The earlier 26 mixed-source reference vehicles remain a
separate exploratory pool. Aircraft coefficient metadata is inventoried but no
aircraft geometry is used in the car CNN.

### DrivAerML primary comparison

Five complete 512x512 views, 40 train / 10 validation / 10 test, seeds 17/42/73,
60 epochs, batch 4, small model, MPS, no augmentation. All training completed
before test evaluation. Checkpoint selection used validation MSE only.

| Model | Cd MAE mean ± seed std | Cl MAE mean ± seed std | Cd P95 error, mean across seeds |
| --- | ---: | ---: | ---: |
| Parameter MLP | 0.012630 ± 0.000191 | 0.049847 ± 0.000615 | 0.037770 |
| Existing pyramid | **0.011032 ± 0.001144** | 0.039353 ± 0.001715 | 0.026716 |
| Detail pooling | 0.012759 ± 0.001372 | 0.036998 ± 0.005654 | 0.024091 |
| Detail pooling + view attention | 0.012387 ± 0.000601 | **0.032992 ± 0.001125** | **0.021291** |

The train-mean baseline has Cd MAE 0.012962 and Cl MAE 0.050733. Detail pooling
does not improve Cd mean error in this pilot. Attention improves Cl and the Cd
error tail while worsening Cd MAE relative to the original pyramid. Its Cl MAE
is about 16.2% lower and mean Cd P95 about 20.3% lower than the original pyramid;
these describe this small fixed test set, not universal accuracy gains.

This run used the pre-existing guarded evaluator with explicit geometric
extrapolation and all ten cases included; all test Re/yaw values are within
training ranges. A strict post-run coverage audit rejects three of ten cases
because their width/height ratios exceed the training range; those cases remain
in every reported error aggregate. The current runner's diagnostic mode records
default rejection coverage, as used for the secondary pools below. Runtime/code
hashes and the precise evaluation revision are recorded in each protocol.

The lowest validation MSE among the declared runs selects
`drivaerml-comparison/seed-42/pyramid/model/model.pt`. A fresh CLI query for held-out
run 26 returned Cd=0.274159 and Cl=0.018931; its CFD labels are 0.274448 and
0.051665. This is a runtime smoke example, not the aggregate accuracy evidence.
It took about 1.4 s excluding interpreter/import startup. Copy conditions from
the frozen checkpoint/prepared data when constructing queries; a later source
manifest version may change its condition JSON schema.

### Existing 26-car grouped comparison

Same 1024 complete views, three family-disjoint folds, seed 42, 60 epochs and
batch size 4. All 26 predictions are included, including seven cases rejected
by the default deployment range checks in each CNN arm.

| Model | Cd MAE | RMSE | P95 absolute error |
| --- | ---: | ---: | ---: |
| Legacy side similarity | **0.038976** | **0.054410** | 0.104457 |
| Existing pyramid | 0.048568 | 0.065915 | 0.156573 |
| Detail pooling | 0.048601 | 0.066890 | 0.119778 |
| Detail + view attention | 0.049457 | 0.064356 | **0.102997** |

The new pooling has essentially unchanged MAE in this single-seed run, with a
smaller tail. View attention improves RMSE/P95 but worsens MAE. The old similarity
method remains the accuracy baseline. These Cd-only cross-family numbers are
not directly comparable to the two-target, within-family DrivAerML numbers.

### Secondary CFD smoke experiments

Each uses 16/4/4 source partitions, seed 42, the same four arms and 60 epochs.
Errors include all four held-out samples; no test-based model tuning followed.

| Pool / model | Cd MAE | Cl MAE |
| --- | ---: | ---: |
| Ahmed / parameters | 0.043169 | 0.203943 |
| Ahmed / pyramid | 0.051839 | 0.169496 |
| Ahmed / detail | 0.044485 | 0.194484 |
| Ahmed / detail + attention | 0.043123 | 0.217154 |
| Windsor / parameters | 0.009812 | 0.301871 |
| Windsor / pyramid | 0.018670 | 0.300604 |
| Windsor / detail | 0.011597 | 0.396497 |
| Windsor / detail + attention | 0.021805 | 0.402353 |

Three of four Ahmed test samples are outside the training body-length Reynolds
range and are explicitly marked as rejected by default inference. These are raw
extrapolation diagnostics. Windsor has no default range rejections, but its Cl
errors are large and attention worsens both mean targets. Four test shapes and
one seed support software/data-path checks, not a stable architecture ranking.

### Geometry and data corrections

- DrivAerML STLs/force/reference files: all 180 files from 60 cases matched the
  pinned source object hashes. The 48 secondary STLs also matched pinned hashes.
- Windsor includes four support pins reaching the floor. Its variable body
  clearance is already present in the STL; additional renderer clearance is zero.
  Earlier generated views with a repeated clearance offset are marked rejected
  and were never used for training.
- Windsor's published body rotation is -2.5 degrees about raw +Y. Mapping the
  inverse body-frame freestream through canonical `(x,-z,y)` gives learner yaw
  +2.5 degrees. Raw Cd/Fx and Cl/Fy labels retain the source yawed-model axes;
  they are not relabeled as freestream-projected drag at nonzero yaw.
- Ahmed omits support pins and uses the documented 50 mm lowest-body clearance.
  Its paper and current README disagree on the averaging duration; the source
  manifest records the conflict. Missing operating details remain explicit.

No default predictor or all-vehicle deployment model was replaced. Whole-view
attention remains an optional experiment. Accurate microgeometry delta-Cd and
unseen production-vehicle accuracy still need dedicated CFD validation.

Final software verification: 113 focused tests passed, Ruff lint/format checks,
Python compilation and diff whitespace checks passed. Existing VTK/NumPy
deprecation warnings remain. A fresh metadata-only collector smoke passed for
both secondary sources, and a fresh geometry-to-coefficient CLI smoke passed for
the validation-selected DrivAerML model. Generated evidence includes
`data-and-results.json`, `drivaerml-comparison.png`, each pool's `comparison.json`,
`builtin26/view-attention.json`, `drivaerml-default-coverage.json`, and
`example-{query,prediction,comparison}.json` under the current run folder.
