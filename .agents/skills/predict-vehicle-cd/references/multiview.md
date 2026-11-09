# Cached whole-view similarity predictor

`vehicle_cd_multiview.py` provides an optional whole-car five-view reference matcher.
The established single-side predictor remains available and retains its default
behavior. This first comparison does not justify replacing it on accuracy.

## Representation and scoring

Use complete front/rear/right/top/bottom depth+mask images with the existing fixed
coordinate/length convention. No local part crops or manual region annotations
are used. Input views must be equal square resolutions divisible by 256; the
current 26-car cache was built from full 1024x1024 numerical views.

For each view, area pooling produces float32 occupancy and mask-weighted mean
depth at 32, 128 and 256 resolution. Fractional coverage is retained, and the
reference cache introduces no 8-bit depth quantization. All views and scales
receive equal weight. Each scale uses .6 soft mask IoU + .4 overlap-weighted depth
similarity (depth MAE normalized by .30), followed by the existing top-3 inverse
distance softmax (.08 temperature, 1e-6 epsilon). No scoring weights were tuned
using the evaluation outcomes. Gradients/normals and learned calibration are not
included in this first version.

An exact or near-exact self reference can dominate the result. A geometry score
is not calibrated confidence, and a reference-weight change is not a physical
prediction of the aerodynamic effect of a small modification. Outputs remain a
convex combination of reference Cd values.

## Build once, query repeatedly

The cache stores descriptors for every supplied prepared case, labels, IDs,
family/geometry provenance, conditions and checksums. It is not a learned model.
`load_library()` returns metadata and arrays that can be retained across repeated
`predict_descriptors()` calls in a long-lived process. The CLI loads the cache
once per invocation. No cache singleton or background service is introduced.

```bash
source .venv/bin/activate
export PYTHONPATH="$PWD/.agents/skills/predict-vehicle-cd/scripts"
python .agents/skills/predict-vehicle-cd/scripts/vehicle_cd_multiview.py build-library \
  outputs/predict-vehicle-cd/full-view-attention/prepared \
  --output-dir outputs/vehicle-multiview/library
python .agents/skills/predict-vehicle-cd/scripts/vehicle_cd_multiview.py predict \
  /path/to/query.json --library outputs/vehicle-multiview/library/library.json \
  --output outputs/vehicle-multiview/prediction.json
```

A query JSON declares the geometry explicitly:

```json
{
  "geometry": {
    "path": "/path/to/car.vtp",
    "unit_scale_to_m": 1.0,
    "front_axis": "-x",
    "up_axis": "+z"
  }
}
```

The bank's fixed operating profile applies when conditions are omitted. If
`conditions`, `reynolds` or `yaw_deg` are supplied, they must match that profile.
A fixed-speed bank derives Re from the query's measured length and declared air
properties; otherwise all bank records must share one Re. A single bank supports
one yaw. It does not learn a speed/yaw response from mixed labels. The current
bank uses the user's low-drag-reference assumption, not recovered source tests.

Use `--exclude-id model3` to exclude that exact reference when testing Model 3.
Unknown IDs are errors. For proper generalization assessment exclude the entire
held-out family, including any geometry aliases; the validation runner does so.

## Validation results

The current bank contains all 26 built-in labeled vehicles. The exact same
seed-42 family folds (9/9/8) and reference label values were used for both methods;
all held-out IDs/families and geometry hashes were excluded. All 26 outputs are
reported. The old baseline uses native stored side images and no category bonus.

| Method | Cd MAE | RMSE | P95 absolute error |
| --- | ---: | ---: | ---: |
| Old single-side similarity | 0.038976 | 0.054410 | 0.104457 |
| New whole-view similarity | 0.041852 | 0.058577 | 0.102051 |

Average error is slightly worse; the P95 improvement is small and not evidence of
uniformly better predictions. Labels retain mixed public/user provenance and
assumed conditions. This is a representation comparison, not CFD validation.

The all-26 bank occupies 86.3 MB in memory and about 82 MiB on disk. Its initial
build took 2.11 s on this host. Across 15 warmed runs with the same 26 references:

| Scoring stage | Median |
| --- | ---: |
| New multiview bank | 54.1 ms |
| Old method, preloaded reference artifacts | 26.3 ms |
| Old normal scoring path, including reference PNG reads | 59.2 ms |

These timings exclude query rendering and feature extraction. The new scoring
stage is not faster than fully preloaded single-side scoring.

One fresh Model 3 query from the same canonical mesh took 15.77 s through the
legacy native entry point, versus 0.824 s for the new entry point (0.095 s library
load, 0.534 s render, 0.114 s describe, 0.081 s score). Both exclude interpreter
startup/imports. The old path renders two orientation candidates; the new path
uses explicit axes and five batched-rendered views. The end-to-end improvement
comes mostly from renderer/cache implementation and is not inherent to using
more views; the legacy renderer could also be optimized.

## Mirror sensitivity

The previously audited Model 3 deformation changes both outer mirror heights,
leaving body points, all XY coordinates and whole-car bounds exactly unchanged.
Both predictors use their normal whole-car inputs. No local training crop is used.

| Outer mirror height | Old Cd, self excluded | New Cd, self excluded |
| --- | ---: | ---: |
| Original | 0.219720019 | 0.245500756 |
| -25% | 0.219721371 | 0.245495016 |
| -50% | 0.219720595 | 0.245530934 |

With the self reference retained, both remain effectively at Cd=.23. The new
original-to-50%-flattened geometry similarity drops to .998895, with most change
in the front/rear views; the old side-only comparison was .999867. The new input
comparison is more responsive, but its coefficient change is nonmonotonic and has
no CFD ground truth. It does not establish useful design-optimization sensitivity.

## Reproduction and ownership

```bash
python .agents/skills/predict-vehicle-cd/scripts/validate_multiview.py \
  outputs/vehicle-multiview/library/library.json \
  outputs/predict-vehicle-cd/full-view-attention/prepared \
  .agents/skills/predict-vehicle-cd/assets/reference-library/reference_cars.json \
  --output-dir outputs/vehicle-multiview/validation
```

Generated evidence for this run is in
`outputs/predict-vehicle-cd/multiview-similarity/`: `library/`,
`validation/comparison.json`, `model3-prediction.json`, `fresh-query-timing.json`,
and `mirror-sensitivity.json`. All generated arrays and personal meshes stay out
of Git.

The CLI reuses `load_dataset` for prepared-data integrity and `render_geometry`
for whole-view rendering. `describe_views` owns descriptor construction;
`score_descriptors` and `predict_descriptors` own matching. The private shared
scoring function avoids repeated validation within a prediction. Cache persistence
is owned by `build_library`/`load_library`; validation consumes those same APIs.
