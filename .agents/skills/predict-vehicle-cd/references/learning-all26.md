# All 26 built-in vehicles trained — 2026-10-28

The built-in reference_cars.json, not pending seed entries, defines membership.
All 26 IDs and Cd labels were matched exactly to the prepared dataset and the
final checkpoint's train_ids. The five broken mesh paths were resolved to local
DrivAer sedan, Model 3, AeroSUV, Zeekr 009 and Hyundai pickup models. The other 21
meshes were read from their recorded locations. No labeled built-in vehicle was
excluded. Source meshes and the original label library were not overwritten.

Five numerical views are prepared at 256/512 resolution. Older raw meshes use the
existing axis/unit normalization, with orientation matched to the built-in side
silhouette without consulting Cd. Visual review found upside-down MPV, RAV4 and
Fortuner references; those three were corrected before training. The interrupted
uncorrected run is marked REJECTED; use run26-upright. Orientation resolution and
view previews are saved beside the source manifest. These older mesh scales and
source conditions remain assumptions, not calibrated CFD geometry evidence.

## Final model

- Five-view CNN, Cd only; no invented Cl labels.
- All 26 vehicles, seed 42, 60 epochs, batch size 4.
- Fixed final epoch, no validation-based checkpoint selection.
- Source conditions follow the user-authorized low-drag reference assumption:
  zero yaw, 120 km/h, moving ground and rotating wheels. Reynolds number is
  derived per vehicle length using rho=1.204 kg/m3 and mu=1.81e-5 Pa s.
- Training-fit Cd MAE: 0.001279; RMSE: 0.001765.
- Checkpoint validation metric is null, validation count is zero and heldout_ids
  is empty. Training fit is not reported as new-vehicle accuracy.

## Independent evaluation

Three family-disjoint folds, fixed 60 epochs and seed 42. Every car is predicted
once by a fold model that did not train on it. No fold's labels are used for
checkpoint selection. ID and canonical-geometry hash separation was verified.
Out-of-fold errors include raw diagnostic predictions even when public inference
would reject a parameter range, preventing failure-case exclusion from metrics.
Seven fold cases were rejected by default range policy, but remain in the error
aggregate. The final all-26 model and fold models are separate artifacts.

| Method | Out-of-fold Cd MAE | RMSE |
| --- | ---: | ---: |
| Five-view CNN | 0.056376 | 0.073201 |
| Training-fold mean | 0.055883 | 0.077580 |
| Existing side-view similarity | 0.038976 | 0.054410 |

The similarity baseline uses only the same fold's training cars, the original
stored side images, top-k=3, temperature=.08, epsilon=1e-6 and no category bonus.
Each held-out car and its label is removed from the candidate reference pool.
This compares each method's own representation; original side-view orientation
and rasterization differ from the corrected five-view preparation.

The CNN has learned the training labels, but independent performance does not
improve on the existing method or even the mean baseline in MAE. All labels still
come from mixed public/user sources, with assumed conditions. This experiment
establishes complete training coverage, not verified aerodynamic accuracy.

A fresh Porsche query at the new assumed speed was rejected because its derived
Re fell outside the full-training range. No trusted coefficient was fabricated.

## Artifacts

Local artifacts are under `outputs/predict-vehicle-cd/all-builtin-training/`:

- `source-inventory.json`: all 26 memberships and completed-training pointers.
- `mesh-resolutions.json`: repaired paths and visual upright corrections.
- `run26-upright/source.json`: complete source manifest and assumptions.
- `run26-upright/prepared/`: five-view inputs and checked previews.
- `run26-upright/final-five/model.pt`: final all-sample weights.
- `run26-upright/final-five/training-fit/evaluation.json`: per-car training fit.
- `run26-upright/cv-three-fold/out-of-fold.json`: independent predictions.
- `run26-upright/comparison-out-of-fold.json`: same-fold similarity comparison.

The CLI now supports `train --fit-all`; a regression test verifies every case is
included and that no validation accuracy claim is emitted. Reused entry points are
read_mesh, normalize_vehicle_mesh, render_five_views and prepare. The new
prepare_builtin entry point resolves membership and orientation; cross_validate
runs the existing trainer on disjoint family subsets. Neither changes label truth.
