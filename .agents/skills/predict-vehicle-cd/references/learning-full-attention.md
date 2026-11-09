# Full-data attention training

## Frozen model protocol

The continuation requested on 2026-10-30 retains attention as the main CNN path.
Each sample supplies five complete 512x512 float depth/mask images. The encoder
is `pyramid_detail_attention`: the existing multiscale pyramid, foreground mean
and max pooling, and channel/spatial attention. Fusion remains the existing
condition-dependent view attention. There are no part crops or fixed part weights.

The new combined encoder starts its two sigmoid gates at 0.5 and applies a factor
of two, preserving initial feature amplitude. View gates start uniformly, with
unit feature scales. Moving view-gate initialization after the regression head
keeps shared encoder/head initialization paired with concat for the same seed.
Existing stored model configurations and weights retain their numerical behavior.
An actual earlier view-attention checkpoint reproduced its saved prediction with
maximum absolute difference zero after the code change.

The full runs use seed 42, 80 epochs, batch size 8, small model, MPS, AdamW at
0.001, no augmentation, and the minimum validation standardized Cd/Cl MSE
checkpoint. Cd and Cl have equal weight after training-only standardization.
The test split is never fitted. Numerical tensors are loaded once into RAM by
`train --preload`; one 450-case train+val pool occupies approximately 4.7 GB at
512 resolution. This direct list of prepared samples avoids per-epoch NPZ
decompression; no cross-request cache or background inference service is added.

## Full source collection and duplicate handling

Only geometry, coefficient/reference tables and source documentation are fetched;
full volume/surface CFD fields are unnecessary for scalar training. Downloads
retain the previously pinned revisions and validate source size and hashes.
Pilot manifests and prior results remain immutable beside the full manifests.

| Pool | Raw paired cases | Training | Validation | Test |
| --- | ---: | ---: | ---: | ---: |
| AhmedML | 499 | 399 | 50 | 50 |
| WindsorML, raw | 350 | 280 | 35 | 35 |
| WindsorML, exact-mesh deduplicated | 344 | 274 | 35 | 35 |
| DrivAerML, official public cases | 484 | 400 | 34 | 50 |

Ahmed run 500 is excluded for missing variable-area labels and geometry parameter
rows. Windsor runs 350-353 (train) and 354 (test) have no published STL. Six other
Windsor pairs are exact mesh aliases: 30/31, 133/134, 230/231, 241/242, 270/271,
304/305. Their binary STL facet records match exactly; only generator timestamp
bytes in their 80-byte headers differ. Recorded geometry CSV parameters differ,
so they are preserved as source metadata rather than substituted for mesh inputs.

The strict geometry-hash split guard caught three alias pairs across held-out
partitions before any Windsor model trained. Deduplication retains the original
test representative first, then validation, then a stable training ID. Labels are
unchanged and all raw source assets remain in provenance. The retained IDs are
30, 133, 230, 242, 270 and 305. Earlier 350-case prepared files are explicitly
marked rejected and kept for audit. The standalone `deduplicate_windsor` command
verifies exact facet equality, Re/yaw and geometry declarations before exclusion;
rounded hash collisions and different conditions cannot be silently combined.

The completed total after these verified exclusions is 1,327 paired records across
three source families, including 1,073 training cases, 119 validation and 135 test.
Pools are trained separately because their physical conditions and coefficient
axis definitions differ. These are morphs of three baseline bodies, not 1,327
independent production vehicle families. The earlier 26 assumed-condition
reference vehicles remain a separate exploratory comparison.

## Entry points and current artifacts

Full geometry collection uses the existing source owners:

```bash
export PYTHONPATH="$PWD/.agents/skills/predict-vehicle-cd/scripts"
.venv/bin/python -m vehicle_cd_learning.import_drivaerml --all --workers 8
.venv/bin/python -m vehicle_cd_learning.collect_bluff_bodies --full --workers 6
```

The Windsor derived source is
`outputs/predict-vehicle-cd/public-data/windsorml/full-source-deduplicated.json`.
To reproduce its selection from the original complete-view audit artifacts:

```bash
.venv/bin/python -m vehicle_cd_learning.deduplicate_windsor \
  outputs/predict-vehicle-cd/public-data/windsorml/full-source.json \
  outputs/predict-vehicle-cd/full-attention-20261030/run/windsorml/prepared/dataset.json \
  --output /path/to/fresh-full-source-deduplicated.json
```

`train_full_attention` waits for the verified full source manifests, prepares
numerical views with the existing `prepare`, calls `train`, and evaluates all
held-out cases with the existing diagnostic evaluator. It freezes configuration,
source checksums and code hashes. Completed checkpoints are reused on restart;
partial model/preparation folders are preserved as interrupted evidence.

```bash
.venv/bin/python -m vehicle_cd_learning.train_full_attention \
  --output-dir outputs/predict-vehicle-cd/full-attention-20261030/run-v3 \
  --epochs 80 --batch-size 8 --seed 42 --device mps
```

Current results/protocols live in that run-v3 directory. Its Ahmed folder points
to the already completed, unchanged first run. Windsor was restarted only after
the source aliases and a constant-input normalization inconsistency were resolved.
`CaseDataset` now uses the same zeroing rule as inference for effectively fixed
inputs: standard deviation at most 1e-8 and deviation from the mean at most 1e-6.
This prevents serialization noise in fixed Windsor width/Re from becoming a
training signal; unsupported condition extrapolation remains visible. The earlier
v2 model and its pre-correction dataset code are retained for audit. All fixed
Ahmed inputs were exactly zero already, so its completed checkpoint is reused.
`inspect_attention` exports the five complete
input views, fixed-scale sigmoid maps, and view gates; gates describe the learned
model, not physical drag-contribution percentages or validated local delta-Cd.

## Completed results

AhmedML full attention, official 50-case test set:

| Model | Cd MAE | Cl MAE |
| --- | ---: | ---: |
| Dimensions/Re/yaw parameter MLP | 0.025930 | 0.162239 |
| Full-view channel/spatial/view attention | **0.014916** | **0.072812** |

Attention-model Cd RMSE is 0.029462 and P95 absolute error 0.055112; Cl RMSE is
0.125787 and P95 0.212747. Two cases fail default range checks but remain in the
raw diagnostic aggregate. On the same four cases from the earlier small pilot,
Cd MAE changes from 0.043123 to 0.019630 and Cl from 0.217154 to 0.125711. Both
training population and attention architecture changed, so this does not isolate
an attention-only causal effect. One seed and a single source family cannot
establish unseen-family accuracy; large individual errors remain.

WindsorML, 344 unique shapes after exact alias removal, 35-case held-out test:

| Model | Cd MAE | Cl MAE |
| --- | ---: | ---: |
| Dimensions/Re/yaw parameter MLP | 0.024606 | 0.228752 |
| Full-view channel/spatial/view attention | **0.019721** | **0.108076** |

Cd RMSE is 0.026588 and P95 0.051609; Cl RMSE is 0.170417 and P95 0.314646.
There are zero default range rejections. Same-condition pairwise geometry-change
direction accuracy is 72.2% for Cd (576 differences above 0.001) and 82.1% for Cl
(592 differences above 0.001). These pairs share 35 test shapes and are correlated;
they are not hundreds of independent tests or mirror-specific validation.
On the four prior pilot test IDs, Cd MAE is 0.018654 and Cl 0.074872, compared
with 0.021805 and 0.402353 previously. Expanded data, the combined attention
architecture and alias removal changed together; individual large errors remain.

DrivAerML, official 400/34/50 source partitions, 484 unique geometry hashes:

| Model | Cd MAE | Cl MAE |
| --- | ---: | ---: |
| Dimensions/Re/yaw parameter MLP | 0.011773 | 0.037608 |
| Full-view channel/spatial/view attention | **0.007713** | **0.029596** |

Cd RMSE is 0.010092 and P95 0.022131; Cl RMSE is 0.035323 and P95 0.059310.
One held-out case exceeds the training height-ratio range and remains in the raw
diagnostic aggregate. On the same ten earlier pilot test IDs, Cd MAE is 0.008612
versus 0.011710 and Cl MAE 0.022904 versus 0.033937. The full models use one
seed; these are dataset-specific, within-family results, not a global car claim.

All three final attention checkpoints completed 80 epochs. Selected best epochs
are DrivAerML 46, AhmedML 45 and WindsorML 50. Together they fit 1,073 training
records and retain 119 validation plus 135 test records. Three default-rejected
range-extrapolation cases remain in the combined diagnostic population. The
separate pools are not represented as one unified cross-condition network.

`run-v3/full-summary.json` contains final counts, model paths, metrics, source
scope and example CLI results. Each pool contains `model/model.pt`,
`model/training_report.json`, `test/evaluation.json`, a parameter-only control,
and `example-query.json` / `example-prediction.json`. Fresh geometry-to-CPU CLI
smokes passed for all three; observed times excluding interpreter/import startup
were about 0.35 s (Ahmed), 0.21 s (Windsor) and 1.42 s (DrivAer). These are local
examples, not latency guarantees or selectively reported accuracy results.

Final software gate: 138 tests passed, Ruff lint/format and Python compilation
passed, and legacy checkpoint prediction delta remained zero. Existing VTK/NumPy
deprecation warnings remain. No default legacy predictor or original 26-car
reference library was replaced. Millimetre-level feature sensitivity, actual
part drag contributions and unseen vehicle families still require validation.
