# Per-view feature compression experiment

Latest result, 2026-11-12: the user requested a **single-seed128-width A-model**
trial and a controlled common vehicle-detail edit. This new trial completed40
epochs (seed42) using the scene experiment's common 4,121 train /614 validation
/819 historical test records and256-pixel complete views. The existing32-width
A-seed42 run was reused only after reconstructing its random initialization and
matching its recorded initial SHA; the128 trial verified shared pre-projection
initialization and identical actual sampling counts/RNG. This is separate from
the stopped2026-11-09 vehicle-only experiment below.

| Width per view | Parameters | Vehicle Cd MAE | Vehicle Cl MAE | Vehicle Cl/Cd MAE | Aircraft Cd MAE | Aircraft Cl MAE | Aircraft Cl/Cd MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
|32|90,173|0.011823|0.051925|0.168135|0.008595|0.069948|1.299640|
|128|319,229|0.010345|0.061547|0.207541|0.007645|0.068859|1.235506|

These are equal-source macro MAEs for one matched seed, not the prior
three-seed means.128 improves mean vehicle/aircraft Cd by12.5%/11.0%, while
vehicle Cl/Cl÷Cd worsen18.5%/23.4%; aircraft Cl/Cl÷Cd improve1.6%/4.9%.
There is no overall replacement decision; the existing32-width A baseline and
default predictors remain unchanged. Training epoch time was about26.84minutes.

Later on2026-11-12, the user confirmed nominal zero-yaw operating assumptions
for all26 built-in cars. A separate prepared dataset now supplies those assumed
conditions; it has not been retrained. Both width arms above retain the frozen
scene dataset, where those operating inputs were missing. See
[the training inventory](../../cad-surface-mesher/docs/aerodynamic-training-datasets.md).

The detail probe preselected a held-out DrivAerML run11 and shortened both
exterior mirror outboard extensions15%/30%, keeping roots, original camera,
physical conditions and reference area fixed. All five views remain complete.
At30% (about50.86mm inward each side),32 predicts ΔCd=-0.003911 and128 predicts
ΔCd=+0.002467. With all numeric geometry inputs fixed, image-only changes still
produce -0.003701/+0.002610: both networks sense the visual change, but disagree
on its aerodynamic direction. Modified cases have no paired CFD labels.
The128 response to the smaller15% edit changes sign across deterministic
±0.1-pixel camera offsets; sensitivity is not proof of accurate physical detail
understanding. The zero-change controls produce exactly zero prediction change.

The original source has3 exact self-intersection pairs; all design copies inherit
those same3 pairs and add none. No source repair or CFD-ready acceptance is
claimed. Full-resolution VTP files contain753,250 triangles, while the verified
interactive viewer explicitly previews200,000 per model. Raw source re-rendering
matches the frozen numerical cache exactly.

Artifacts: `outputs/aerodynamic-width-128-20261112/` has the protocol, checkpoint,
full comparison, README and final-verification; the primary interactive report
and whole-car mesh copies are under
`outputs/aerodynamic-width-128-20261112-geometry/probe/report.html`.
`aircraft_learning.width_experiment` reuses `train_model`, `SceneCases`,
`sampling_weights` and `compare_scenes.evaluate_run`. The new
`vehicle_cd_learning.detail_sensitivity` owns the geometry/rendering/input-path
probe, and `detail_report` calls the existing `html_report.write_html_report` for
the single-file3D report.225 learning/geometry tests and Ruff checks passed.

## Archived vehicle-only experiment — 2026-11-09

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
