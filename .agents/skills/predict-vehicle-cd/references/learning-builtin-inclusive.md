# Attention training including every built-in reference car

This run implements the explicit requirement that every packaged labeled vehicle
participates in the new attention model's training. It builds one conditional
model from all three complete CFD pools plus all 26 built-in vehicle meshes and
their Cd values. The authoritative membership/labels remain
`assets/reference-library/reference_cars.json`; the source builder checks every
ID and coefficient against that file before preparation.

## Data and label handling

| Profile | Train | Validation | Test | Observed outputs |
| --- | ---: | ---: | ---: | --- |
| drivaerml | 400 | 34 | 50 | Cd, Cl |
| ahmedml | 399 | 50 | 50 | Cd, Cl |
| windsorml | 274 | 35 | 35 | Cd, Cl |
| builtin_assumed | **26** | 0 | 0 | **Cd only** |
| Total | **1,099** | **119** | **135** | 1,353 geometry/condition records |

All 1,353 geometry hashes are unique. Every original CFD case ID, split and
geometry hash is preserved. The 26 built-ins are all train cases, so any reported
built-in fit is explicitly an in-sample fit, not independent vehicle accuracy.
Original CFD and reference datasets/checkpoints remain unchanged.

Built-in Cl values are JSON null and runtime NaN. Masked loss selects finite
observed target entries before subtraction; unknown Cl contributes no loss or
direct output gradient. Training normalization, metrics and scatter plots also
exclude unavailable labels. Cd normalization sees 1,099 training labels; Cl sees
only the 1,073 real CFD training labels. Actual NaN/Inf in source labels is
rejected; only explicit null denotes missing Cl.

Each source has an explicit `condition_profile` containing its original fixed
conditions, source provenance, data kind and labeled targets. Input parameters
are the existing five geometry/Re/yaw values plus four declared profile one-hot
features. The existing parameter encoder and view-attention block consume these
nine inputs. This separates the assumed moving-ground/rotating-wheel reference
labels from the fixed CFD settings rather than treating them as one condition.
It supports the four declared profiles, not arbitrary unseen physics settings.

The checkpoint owns profile-specific training parameter ranges. Inference checks
Re/yaw against the selected profile even if another profile spans the requested
value. Unknown/mixed one-hot profiles and mismatched condition dictionaries fail.
For `builtin_assumed`, the CLI returns Cd only and declares Cl unsupported;
it does not present the unvalidated latent Cl output as a prediction.

## Architecture and training

The model retains five complete 512x512 depth/mask views, multiscale foreground
mean/max pooling, channel/spatial attention, and condition-dependent view gates.
There are no local part crops or manually assigned part weights. The source
DrivAerML attention checkpoint initializes only the compatible image and bottom
encoders. The nine-input condition encoder, view gates and output head start
fresh under the merged training normalization; this is feature transfer, not an
optimizer/head-state resume.

Protocol: seed 42, small model, 80 epochs, batch 8, AdamW 0.001, MPS, no
augmentation. The entire training set is traversed each epoch without dropping
the final batch. Only the original 119 CFD validation cases select the checkpoint;
the 135 CFD test cases remain untouched until final evaluation. Geometry, source
and missing-label checks are part of the real CLI path.

Prepared data, protocol, weights and inclusion evidence are under
`outputs/predict-vehicle-cd/builtin-inclusive-20261030/`:

```bash
export PYTHONPATH="$PWD/.agents/skills/predict-vehicle-cd/scripts"
.venv/bin/python -m vehicle_cd_learning.combine_builtin \
  --output-dir outputs/predict-vehicle-cd/builtin-inclusive-20261030
.venv/bin/python -m vehicle_cd_learning.train \
  outputs/predict-vehicle-cd/builtin-inclusive-20261030/prepared \
  --output-dir outputs/predict-vehicle-cd/builtin-inclusive-20261030/model \
  --encoder pyramid_detail_attention --fusion attention --model-size small \
  --epochs 80 --batch-size 8 --seed 42 --device mps --preload \
  --encoder-checkpoint outputs/predict-vehicle-cd/full-attention-20261030/run-v3/drivaerml/model/model.pt
```

Use fresh output folders to rerun those commands. `inclusion-audit.json` proves
the prepared membership and unchanged splits. `model/model.pt` stores all actual
training IDs and per-profile observed Cd/Cl counts. The inclusion-gradient check
uses a real built-in case to verify nonzero encoder gradients from its Cd label
and zero direct Cl supervision. The final proof is recorded after training.

CFD holdout evaluation and built-in training fit use the existing evaluator:

```bash
.venv/bin/python -m vehicle_cd_learning.evaluate \
  outputs/predict-vehicle-cd/builtin-inclusive-20261030/model/model.pt \
  outputs/predict-vehicle-cd/builtin-inclusive-20261030/prepared \
  --diagnostic --output-dir outputs/predict-vehicle-cd/builtin-inclusive-20261030/test
.venv/bin/python -m vehicle_cd_learning.evaluate \
  outputs/predict-vehicle-cd/builtin-inclusive-20261030/model/model.pt \
  outputs/predict-vehicle-cd/builtin-inclusive-20261030/prepared \
  --split train --condition-profile builtin_assumed --diagnostic \
  --output-dir outputs/predict-vehicle-cd/builtin-inclusive-20261030/builtin-training-fit
```

Metrics report observed-label counts and JSON null for unobserved targets. Raw
diagnostic evaluation retains rejected-range cases and records the reasons;
production inference guards remain strict. Built-in fit is never called a
held-out accuracy result.

Ownership: `combine_builtin` reuses `prepare` for the 26 complete-view renders and
`load_dataset` for source integrity. The existing `train_model` owns masked loss,
training-only normalization and checkpoint persistence. `CoefficientModel` reuses
its parameter encoder and `ViewAttention` for profile features; `predict` owns
profile/range validation and supported-output filtering. `evaluate` owns partial
label metrics and profile selection. No new model/context/cache framework is added.

## Completed results

The 80-epoch run completed and validation selected epoch 47. The final checkpoint
contains all 26 packaged IDs in `train_ids`, and `train_source_counts` records
`builtin_assumed: {cases:26, cd:26, cl:0}`. Global observed training labels are
Cd=1,099 and Cl=1,073. `final-inclusion-proof.json` checks membership against the
actual reference library and performs a real native-sample backward pass: the
Cd loss gives nonzero encoder gradients, while the unlabeled Cl output gradient
is exactly zero. This proves effective supervision, beyond listing the data.

The built-in profile's **training fit** over all 26 vehicles is Cd MAE 0.007422,
RMSE 0.008967 and P95 absolute error 0.016764. Cl has observed count zero and all
Cl metrics are null. These 26 cases were fitted; these values do not measure
generalization to another production vehicle.

The unchanged 135-case CFD test set, including all three default-rejected range
diagnostics, has Cd MAE 0.013381 and Cl MAE 0.063522. A same-data parameter-only
MLP gives Cd MAE 0.020998 and Cl MAE 0.136127.

| CFD profile | Test cases | Previous specialist Cd/Cl MAE | New joint Cd/Cl MAE |
| --- | ---: | ---: | ---: |
| DrivAerML | 50 | 0.007713 / 0.029596 | 0.009572 / 0.032397 |
| AhmedML | 50 | 0.014916 / 0.072812 | 0.016185 / 0.066779 |
| WindsorML | 35 | 0.019721 / 0.108076 | 0.014816 / 0.103334 |

The test-count-weighted previous-specialist average is Cd=0.013494 and
Cl=0.065949, versus the joint model's 0.013381 / 0.063522. Overall mean error is
similar, with profile-specific tradeoffs: DrivAerML Cd/Cl and AhmedML Cd are worse
than their specialists. Original specialist weights remain available. This is a
comparison of one joint conditional model with three specialized models, not an
isolated experiment attributing every change to the 26 added records.

Four fresh geometry CLI smokes passed, one per declared profile. The Model 3
reference query returned Cd=0.221239 against its training label 0.23. Its output
contains `unsupported_targets:["cl"]` and no Cl coefficient. Example queries and
outputs are under `examples/<profile>/`. For example:

```bash
.venv/bin/python -m vehicle_cd_learning.predict \
  outputs/predict-vehicle-cd/builtin-inclusive-20261030/model/model.pt \
  outputs/predict-vehicle-cd/builtin-inclusive-20261030/examples/builtin_assumed/query.json
```

`summary.json` contains final metrics, sample counts and comparisons;
`final-inclusion-proof.json` contains all 26 IDs and observed-label evidence.
The earlier `active-inclusion-proof.json` is an explicitly early epoch-8 check,
not the final checkpoint proof. The primary final artifact is `model/model.pt`.

Software validation: 162 tests passed, with Ruff lint/format and compilation
checks. Known VTK/NumPy deprecation warnings remain. Real-car labels retain their
assumed-condition status; missing Cl was neither recovered nor pseudo-labeled.
The fixed profile one-hots identify four observed source/operating regimes, not
arbitrary new wall, wheel, turbulence or velocity settings.
