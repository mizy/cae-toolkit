# Full-view multiscale attention experiment — 2026-10-28

The user rejected per-part crops because they do not scale to all vehicle parts.
That experiment was stopped and its crop/render/data/model additions were removed.
Partial artifacts under mirror-experiment are marked STOPPED and are not included
in the results below. The retained implementation uses exactly five full images.

## Architecture and protocol

- Input: front, rear, right, top, bottom, each a complete 1024x1024 float depth/mask
  tensor. No part names, bounding boxes, local images or location annotations.
- Baseline: the existing small CNN at this same full-view resolution.
- Pyramid: 1/2, 1/4 and 1/8 maps from each complete view; 1x1 lateral projections to
  eight channels; bilinear upsampling and fusion at 1/2 resolution, then spatial
  pooling and feature projection. All positions share the same processing.
- Pyramid attention: lightweight CBAM-style channel and spatial gates on the fused
  full-view map. There is no all-pairs Transformer attention or manual part weighting.
- Shared body-view encoder and separate bottom encoder are owned by CoefficientModel.
  PyramidEncoder is selected through the existing model/training entry points.
- Same 26 source labels, 3 family folds, seed 42, fixed 60 epochs, batch size 4,
  AdamW defaults, no augmentation. All three arms trained on MPS, scored on CPU.
- Ordinary and bottom input resolution are both 1024. Labels, geometry hashes,
  condition vectors and family memberships match the earlier all-26 dataset.

At a 4.5 m length and 1.3-length camera window, pixel spacing is about 5.76 mm.
This improves over the previous ordinary 256 image's 23.6 mm spacing, but does not
guarantee robust sensing of every 5 mm modification. Attention cannot recover
unrecorded subpixel surfaces or parts absent from source meshes.

## Same-protocol results

| Encoder | Parameters | Cd MAE | RMSE | P95 absolute error |
| --- | ---: | ---: | ---: | ---: |
| Plain small CNN | 44,681 | 0.052555 | 0.070789 | 0.121113 |
| Full-view pyramid | 55,705 | 0.048568 | 0.065915 | 0.156573 |
| Full-view pyramid + attention | 55,987 | 0.050235 | 0.064958 | 0.137469 |
| Legacy side similarity, same families | — | 0.038976 | 0.054410 | 0.104457 |

Pyramid features improve mean error over the matched-resolution baseline, while
large-error tails worsen. Attention improves RMSE slightly over the plain pyramid
but worsens MAE, with two of three folds worse and one better. Neither beats the
legacy reference method. These are single-seed exploratory comparisons against
mixed-source labels under assumed conditions, not engineering acceptance.

All 26 out-of-fold predictions are included, and geometry-hash/ID separation was
verified for every fold. Default inference rejects seven out-of-range fold cases
in each arm; raw diagnostic errors remain in the aggregate to avoid selection bias.
The displayed legacy comparison uses the same fold membership and its original
stored side-view representation; it does not use 1024 CNN inputs.

CPU inference API medians on one supported case after warmup (10 repetitions,
including numerical input checks/conversion) were 0.111 s, 0.117 s and 0.149 s for
the three encoders respectively. These are local observations, not universal
performance promises. Training uses MPS for all arms to avoid backend confounding.

## Attention evidence and verification

The saved heatmap is from DrivAer Sedan held out of attention fold 0. It displays
all five complete views with a fixed absolute 0..1 color scale, not independent
per-image contrast stretching. Spatial gates largely distinguish body foreground
from background; this does not establish part-specific aerodynamic importance or
correct sensitivity to mirror/underbody changes. Attention values are not drag
contribution percentages.

64 focused tests pass, including full-frame gradient coverage, full-view spatial
attention-map extent, finite attention gradients, checkpoint roundtrip and legacy
model compatibility. Ruff lint/format, compilation and diff checks pass. Only
existing VTK/NumPy dependency deprecation warnings remain.

No new all-26 production model is substituted automatically. The fold checkpoints
are evaluation artifacts; the earlier full-training weights remain available.

## Local artifacts and commands

Ignored artifacts are under `outputs/predict-vehicle-cd/full-view-attention/`:

- `protocol.json`, `comparison.json`: frozen setup and measured comparisons.
- `prepared/`: only the five complete depth/mask images per car.
- `{plain,pyramid,pyramid_attention}/out-of-fold.json`: all predictions.
- `full-attention-example.png`, `attention-example.json`: complete-view gates.

```bash
python -m vehicle_cd_learning.prepare /path/to/all26/source.json \
  --output-dir outputs/full-view/prepared --ordinary-size 1024 --bottom-size 1024
python -m vehicle_cd_learning.cross_validate outputs/full-view/prepared \
  --output-dir outputs/full-view/pyramid-attention --folds 3 --epochs 60 \
  --model-size small --encoder pyramid_attention --device mps
```

Use identical settings with `--encoder plain` and `--encoder pyramid` to compare.
The global rendering, CaseDataset, train_model and cross_validate call chains are
reused. The only retained new encoder abstraction is PyramidEncoder, consumed by
the shared complete body-view branch and the complete bottom-view branch.
