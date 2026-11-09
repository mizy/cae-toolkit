# First implementation validation — 2026-10-28

## Delivered behavior

Five-view numerical depth/mask rendering, explicit units and axes, Cd-only or
Cd/Cl training, three architecture variants, checkpoint loading, single-case
inference and family-disjoint evaluation are implemented. Default views are four
256x256 images and one 512x512 bottom image. No CFD volume mesh is generated.

The existing `mesh_io.read_mesh` and `artifacts.triangle_faces` are reused by the
new renderer. The old automatic normalization and view-specific depth scaling are
not used in the learning path: supervised data declares orientation and units and
uses fixed length-based cameras. `ViewEncoder` is consumed by the shared body and
separate bottom branches. `CaseDataset` serves train/validation loaders; the
checkpoint-owned statistics are also used by inference. No registry or runtime
framework was added.

## Software checks

- 21 focused tests passed using the repository Python environment. Tests cover
  signed orientation and front/rear identity, meters/millimeters, masks, occlusion,
  bottom-view bumps, raster parity, finite model gradients, shared CNN weights,
  NPZ integrity, family/geometry leakage including unit-scaled renamed duplicates,
  model save/load, Cd-only labels and held-out evaluation.
- Ruff lint and formatting checks passed for the new modules and tests.
- Python compilation and `git diff --check` passed.
- VTK/NumPy emit deprecation warnings; there were no test failures.
- Raster optimization matches a brute-force implementation in all five views.
  It also produces exactly equal input arrays for the two real vehicles rendered
  by both the original and optimized implementation.

## Analytic software fixture

48 cases / 24 geometries / 12 families, with 8 train, 2 validation and 2 test
families. Roof and underbody changes are visible, and the two conditions vary Re
and yaw. Labels are analytic test signals with no aerodynamic meaning. Training
used 80 epochs, batch size 8 and seed 42 at 64/128 image resolution.

| Variant | Test Cd MAE | Test Cl MAE |
| --- | ---: | ---: |
| Train-mean baseline | 0.002531 | 0.010000 |
| Parameters | 0.000893 | 0.001115 |
| Four views | 0.000495 | 0.000565 |
| Five views | 0.000848 | 0.000521 |

Additional five-view seeds 7 and 19 yielded Cd MAE 0.000916 / 0.001243 and Cl MAE
0.001425 / 0.001212. This demonstrates seed variability and does not establish
bottom-view superiority. A separate two-case overfit smoke checks that the full
network can reduce standardized training loss below 0.001.

## User reference-data exploratory run

The seed manifest contains 30 records including pending entries; its already
built reference library contains 26 cars. The inventory found 21 mesh paths. Only
11 source Cd records had existing normalized VTP meshes with documented canonical
orientation and dimensions, so only these were eligible for automated preparation.
No prediction JSON was promoted into a source label. Missing raw-model orientation
and missing Cl were left unresolved rather than guessed.

With user authorization, exploratory conditions were assigned Re=3,000,000, yaw=0,
moving ground, rotating wheels and SST. They are assumptions, not recovered
wind-tunnel or CFD settings. Labels retain their mixed public/user provenance.
The data/checkpoints identify this as `reference_assumed` and inference warns about
its applicability. Fixed Re/yaw data do not validate changed-condition behavior.

The deterministic family split is 7 train / 2 validation / 2 test. Test vehicles
are Toyota Auris Wagon and BYD Seal. Training used 60 epochs, batch size 4, seed 42,
and the full default 256/512 images. Checkpoints were chosen on validation only.

| Variant | Train Cd MAE | Test Cd MAE | Test Cd RMSE |
| --- | ---: | ---: | ---: |
| Train-mean baseline | — | 0.097500 | 0.102159 |
| Parameters | 0.056246 | 0.038453 | 0.040086 |
| Four views | 0.044631 | 0.053751 | 0.057717 |
| Five views | 0.013979 | 0.084371 | 0.094962 |

The five-view model fits training data better but performs worse on these two
held-out references than the parameter-only baseline. This is consistent with
small-data overfitting; this run does not prove improved aerodynamic prediction.
The source conditions are mixed and the test count is only two. No engineering
accuracy acceptance claim is made, and Cl accuracy has not been validated on real
labels. The next accuracy gate requires a larger, consistently labeled dataset
and a tolerance tied to the intended design differences.

A fresh single-case geometry render for Toyota Auris Wagon produced Cd=0.4079514,
matching its cached held-out prediction within 1e-7 (source Cd=0.28). The mismatch
to the source label is retained as evidence, not hidden by parameter tuning.
On this host, five-view prediction API time was approximately 0.027 seconds; geometry
rendering for that query took 8.8 seconds. These are local timings, not hardware
independent performance promises.

## Local artifacts and reproduction

All generated data and weights are ignored by Git:

- `outputs/predict-vehicle-cd/reference-audit/`: inventory and assumed-condition source manifest.
- `outputs/predict-vehicle-cd/reference-assumed/prepared/`: frozen family split,
  checksummed float-depth inputs and preview sheets.
- `outputs/predict-vehicle-cd/reference-assumed/{parameters,four,five}/`: models,
  training history, train/test reports and plots.
- `outputs/predict-vehicle-cd/reference-assumed/query-prediction.json`: fresh-render inference.
- `outputs/predict-vehicle-cd/validation-synthetic/`: analytic runs, extra seeds and overfit smoke.

See `learning.md` for the source schema and exact CLI commands. To repeat the
reference run, import the seed manifest into a new output directory, prepare its
`cd_training_manifest.json`, then train each variant with `--epochs 60
--batch-size 4 --seed 42`. Keep the same frozen split for the comparison.
