# Per-view feature compression experiment

Status: stopped at the user's explicit request on 2026-11-09 to start aircraft
training. The 32-feature arm completed 40 epochs; the 128-feature arm was
interrupted and retains only intermediate weights. This is not a completed
paired experiment and supports no final width comparison. See the experiment's
`stopped.json`. Existing source checkpoints were preserved.

2026-11-09. User-selected experiment 1 compares 32 versus 128 features per
complete view. Cd and Cl continue to share the same predictor. The experiment
does not change the source data, five 512-pixel views, convolution channels,
foreground pooling, condition encoder, attention hidden width, or the head's
64/16 hidden widths. Its input-dependent projection, attention input and head
input matrices necessarily change shape. Parameter counts are 89,677 and
318,733.

The normal `small`/`standard` switch changes several components together and
cannot isolate this question. `--view-feature-size` exposes just this width for
pyramid models. Omit it to preserve legacy initialization/loading behavior.

Both arms transfer the same pre-projection pyramid from the existing DrivAer
encoder and initialize their projection layers afresh. Otherwise the old
32-feature projection would have a supervised initialization unavailable to the
128-feature arm. Equal-shape parameters are initialized from an identical
reference model; `common_initial_sha256` binds those initial tensors after source
transfer. The view-attention hidden size remains 16 for both small-model arms.

The fixed protocol uses seed 42, 40 epochs per arm, batch 8, AdamW 0.001 and
sampling_jitter_v1 on MPS. The loader and augmentation use their own same-seed
generators in both arms. All 1,099 training records and 26 built-in Cd-only cars
remain present. The 119 reused development-validation records select each arm's
checkpoint; the 135 historical test records are not used. No unseen-family
accuracy claim follows from this comparison. A single paired seed is an initial
ablation, not a stable architecture ranking.

From the repository root, use a fresh output directory for each arm:

```bash
export PYTHONPATH="$PWD/.agents/skills/predict-vehicle-cd/scripts"
.venv/bin/python -m vehicle_cd_learning.train \
  outputs/predict-vehicle-cd/builtin-inclusive-20261030/prepared \
  --output-dir /tmp/vehicle-width-128-new \
  --encoder pyramid_detail_attention --fusion attention --model-size small \
  --view-feature-size 128 --epochs 40 --batch-size 8 --seed 42 \
  --device mps --preload --augment \
  --encoder-checkpoint outputs/predict-vehicle-cd/full-attention-20261030/run-v3/drivaerml/model/model.pt
```

Use `--view-feature-size 32` for the paired baseline, not a previously trained
checkpoint with transferred projections. The experiment protocol, fixed input
hashes, logs and results live under
`outputs/predict-vehicle-cd/feature-width-20261109/`. After both runs complete:

```bash
.venv/bin/python -m vehicle_cd_learning.compare_feature_width \
  outputs/predict-vehicle-cd/feature-width-20261109
```

The comparison verifies completion, dataset, common initialization, train IDs,
normalization and source-label counts. It evaluates clean train/validation,
per-source Cd/Cl, signed Cl/Cd, sign errors, and built-in training fit. Missing
built-in Cl remains null and excluded. The existing raw-mesh predictor can load
either checkpoint through its stored `model_config`.

Existing owners: `CoefficientModel` owns the width-dependent model dimensions;
`train_model` owns paired initialization, transfer and checkpoint metadata;
`predict.load_model` reads the stored dimensions. The diagnostic CLI
`compare_feature_width` is the concrete consumer of the new metadata and reuses
`CaseDataset`, `load_model` and `evaluate.metrics`. No new runtime or registry is
introduced.
