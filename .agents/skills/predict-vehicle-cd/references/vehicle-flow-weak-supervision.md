# Existing vehicle data for experimental 3D field learning

This implementation reuses the frozen five-view vehicle dataset and its trained
CNN. It has a real vehicle geometry path, three-dimensional coordinate decoder,
total surface-force integration, and physics/boundary losses. Interior velocity,
pressure and shear labels are not supplied or fabricated.

All 1,099 original training records supervise the coefficient head, including
all 26 built-in Cd labels. Unknown built-in Cl remains masked. The initial
experiments froze the encoder; the retained A partially fine-tunes it as
described below. The field decoder conditions on the
168 encoded features plus the head's two standardized coefficient predictions;
the true coefficient labels are never inference inputs. Thus the supervised
head is shared with field conditioning. Only the DrivAer pilot geometries
currently supervise field forces and physics; the remaining records do not
silently acquire field labels. The original 119-case validation split remains
separate and no test predictions are used.

The decoder uses smooth multi-frequency coordinate features. It does not yet
implement the planned coupled coarse/fine 3D latent grids. It predicts u/U,
v/U, w/U and Cp. Its spatial derivatives include the whole coordinate encoding.
Surface force sums pressure and viscous traction; a closed-surface constant
stress control variate removes force caused by finite-sample normal imbalance.

## Physics assumptions

The first model uses a prescribed constant dimensionless eddy viscosity added
to 1/Re. This closes a regularized momentum model; it does not reproduce the
source hybrid RANS/LES closure. The proxy domain in car-length coordinates is
[-3,6] × [-2,2] × [0,3], with constant flow on outer faces and a stationary
floor. These dimensions and the full-floor condition are not verified original
CFD boundary settings. Every report identifies this mismatch. The source
labels are weak supervision under that approximation, not new flow truth.

Source topology, winding, signed volume and constant-pressure force are checked
before experimental quadrature. Self-intersections and component orientation
are not independently established. Solid masking uses the resulting oriented
surface under that stated source-geometry assumption. These checks do not
constitute a general CAD repair or watertightness certification.

Loss values or coefficient agreement alone do not establish an accurate field.
The initial sparse fixed surface quadrature produced noticeably different
forces from a denser independent quadrature. The next experiment uses a larger
surface pool, redraws integration samples each step, and checks independent
quadrature. Proxy boundary/closure errors remain even if sampling stabilizes.

## Commands

From the repository root:

```bash
export PYTHONPATH="$PWD/.agents/skills/predict-vehicle-cd/scripts"
.venv/bin/python -m vehicle_flow_learning.train_vehicle \
  outputs/predict-vehicle-cd/vehicle-flow-data-20261104/vehicle-data.json \
  outputs/predict-vehicle-cd/builtin-inclusive-20261030/model/model.pt \
  --output-dir outputs/predict-vehicle-cd/my-vehicle-flow-run \
  --epochs 30 --field-train-cases 4 --field-val-cases 2 \
  --surface-count 65536 --force-count 4096 --collocation-count 128 \
  --boundary-weight 10 --allow-proxy-physics
```

Each epoch traverses every original training record for coefficient supervision.
The field pilot cycles over only the requested training geometries. Final field
coefficients are also reported for held-out validation geometries. An atomic
epoch checkpoint stores optimizer and Torch RNG state. Resume with the same
command, output directory and settings, add `--resume`, and increase `--epochs`.
The encoder/data identities and training settings are checked on resume.

`--field-cl-weight` sets the lift term's weight in the normalized field-force
loss and defaults to 1. It does not modify the original coefficient head's
missing-label mask. The fourfold experiment is saved separately under
`outputs/predict-vehicle-cd/vehicle-flow-optimization-20261105/cl-weight-4/`.
It improves Cd/Cl and Cl/Cd mean absolute errors on the 34 development cases,
while mean proxy PDE and boundary errors increase; its flow field has no direct
CFD-field validation. The original CNN and field weights are preserved.

## Paired feedback and latent-only comparison

The A/B experiment uses `--finetune-encoder` for both variants. It caches
pre-projection full-view pyramid features, then trains the existing image and
bottom projection layers, parameter encoder, view attention, coefficient head
and field decoder. Earlier convolution/pooling stages remain frozen. This lets
B's auxiliary coefficient labels influence its shared latent without feeding
the predicted coefficients into the field decoder.

- A: `--conditioning feedback`, field condition is latent plus predicted Cd/Cl.
- B: `--conditioning latent`, field condition is latent only.

Both share all common field initialization tensors and the training random
stream. A's two extra input columns add 48 field parameters. `--seed` selects a
paired training seed. `--reuse-prepared` permits a fresh directory containing
cloned `surfaces` and compatible encoded files; identities are verified and an
existing model cannot be overwritten without explicit resume. Partial-fine-tuned
feature exports are bound to the exact resulting checkpoint SHA.

`evaluate_ab` checks both completed runs, the shared inputs/initialization and
random state before evaluating the same validation samples:

```bash
.venv/bin/python -m vehicle_flow_learning.evaluate_ab \
  outputs/predict-vehicle-cd/ab-comparison-20261105/A-42 \
  outputs/predict-vehicle-cd/ab-comparison-20261105/B-42 \
  --audit outputs/predict-vehicle-cd/vehicle-flow-data-20261104/vehicle-data.json \
  --output /tmp/vehicle-ab-comparison.json
```

Three paired seeds (42/43/44), each 31 epochs, used all 1,099 coefficient training
records and 32 field-training geometries. On the same 34 development-validation
geometries, mean Cd/Cl/L-D MAE is A=0.009744/0.041041/0.149161 and
B=0.009823/0.041771/0.151786. A is slightly better in all three paired seeds, but
case-bootstrap intervals include zero difference. Both remain worse than the
original scalar CNN's coefficient/ratio metrics on this set. Two geometries
where B was better also retain that ordering with independent surface sampling.
The experiment does not establish a universal architecture winner or accurate
CFD fields. Detailed results are in `ab-comparison-20261105/summary.json`.

The user retained A as the development direction on 2026-11-05. All three A
checkpoints remain intact. A subsequent boundary/force audit found substantial
wall penetration and outer-velocity error, and a single-seed continuation to
62 epochs did not qualify to replace the 31-epoch checkpoint. See
[the A diagnostics report](vehicle-flow-a-diagnostics.md) for fixed-validation
measurements, evidence limits and the next experiment's acceptance criteria.

Raw geometry inference accepts the existing query JSON convention with an
explicit DrivAer profile and conditions. Add the actual source
`reference_area_m2` to request field-integrated forces. Query coordinates are an
N×3 NPY in car lengths in the same canonical frame as the vehicle.

```bash
.venv/bin/python -m vehicle_flow_learning.predict_vehicle \
  outputs/predict-vehicle-cd/my-vehicle-flow-run/model.pt \
  /path/to/case.json /path/to/queries.npy --output /path/to/field.npy
```

The weights stay fixed during inference. Geometry is loaded once; the validated
coefficient call captures the same encoded features used by the field decoder.
Prediction is batched, and solid/domain-exterior points become NaN. The metadata
reports each timing stage, the actual field-force estimate, and the separate
coefficient-head baseline. Neither estimate is mislabeled as the other.

The raw-geometry command accepts `--sampling systematic` as an optional
random-offset area-CDF surface sampler; `iid` remains the default and preserves
the earlier behavior. On two development geometries, eight seeds and 65,536
points, systematic sampling reduced average differences from two high-density
reference samples. Those references also varied, so this does not establish the
requested integration tolerance or real aerodynamic accuracy. Sampling method
is recorded in the result's surface metadata.

`run-001`, `run-002`, `run-003` and the larger-pool `run-004` are independent
experiments under `outputs/predict-vehicle-cd/vehicle-flow-implementation-20261104/`.
They are development artifacts, not validated replacements for the existing
coefficient predictor. Keep model-specific feature files with their encoder
hash, geometry surface and length. The model currently depends on its original
encoder checkpoint; it is not yet a portable standalone release package.

## Existing owners and new consumers

`CoefficientModel.encode` exposes its existing complete-view encoding to the
coefficient head and `VehicleField`. `CaseDataset` and `load_dataset` provide the
frozen views and splits. `masked_mse_loss` retains the old missing-label contract.
`surface.prepare` binds actual source geometry to oriented samples, weights and
source Aref. `train_vehicle` owns the joint experiment and its checkpoint.
`predict_vehicle` reuses the old input guards and rendering, then calls the same
field decoder and force operator. No separate runtime or registry is introduced.
`diagnose_vehicle.audit_run` reuses `evaluate_ab.validate_run/read_sample` and
`vehicle.field_derivatives/momentum_residual`; its only consumer is the diagnostic
CLI. It adds boundary-region measurements and pressure/viscous force
decomposition without modifying model weights or the online inference path.
