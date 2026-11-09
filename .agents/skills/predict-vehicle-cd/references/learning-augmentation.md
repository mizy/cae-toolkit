# Depth sampling jitter experiment — 2026-10-28

The experiment uses the same 26 built-in labeled vehicles, corrected five-view
inputs, three family folds, seed 42, 60 fixed epochs, batch size 4, AdamW learning
rate .001 and weight decay .01 as the earlier all-vehicle comparison. No held-out
case or augmented sibling enters its fold's training set or selects its checkpoint.
All 26 out-of-fold predictions are retained, including default-policy rejections.

The standard baseline is reused from the preceding experiment. Restoring its three
checkpoints under the refactored standard architecture reproduces every previous
out-of-fold prediction exactly. Source dataset hashes and case/fold membership were
checked. Augmentation changes only training inputs; evaluation uses clean views.

## Intervention

- Standard network: 1,089,569 parameters. Small network: 44,681 parameters.
- Per-sample XYZ sampling offset shared across cameras, at most one ordinary-view
  pixel per axis. The renderer's common camera-axis definitions own projection signs.
- Depth and mask sampling are synchronized. Valid depth interpolates with mask
  coverage normalization. Masks remain binary; background remains exactly zero.
- Small depth noise: normalized standard deviation 1e-4, clipped to the valid window.
- Labels and condition parameters are unchanged. This models representation noise;
  it does not claim new physical configurations with measured Cd.
- Separate augmentation RNG preserves the baseline initialization and shuffle stream.

## Independent results

| Configuration | Cd MAE | Cd RMSE | P95 absolute error |
| --- | ---: | ---: | ---: |
| Standard CNN | 0.056376 | 0.073201 | 0.144964 |
| Standard CNN + jitter | 0.060544 | 0.079011 | 0.160744 |
| Small CNN | 0.056315 | 0.073761 | 0.132272 |
| Small CNN + jitter | 0.052524 | 0.071500 | 0.159106 |
| Legacy similarity, same folds | 0.038976 | 0.054410 | 0.104457 |

Small+jitter improves mean absolute error by 6.83% relative to the standard CNN,
but its P95 error worsens. Its fold MAEs are 0.05361 / 0.06365 / 0.03878, versus
0.03757 / 0.06814 / 0.06429 for the standard model. Gains are uneven and do not
beat the legacy method. Adding jitter to the large network alone worsens MAE.

These are one-seed, small-data comparative results, not a guaranteed improvement
or an independent final acceptance after candidate selection. Labels remain mixed
public/user references with assumed low-drag conditions. Augmentation creates no
new independent CFD evidence and does not correct label/geometry mismatch.

## Reproducibility and artifacts

All artifacts below are local, ignored outputs under
`outputs/predict-vehicle-cd/augmentation-experiment/`:

- `protocol.json`: frozen inputs, settings, model sizes and baseline parity.
- `comparison.json`: all metrics, per-fold errors and source report paths.
- `standard-jitter/`, `small/`, `small-jitter/`: fold models and raw predictions.
- `jitter-preview.png`: real-car clean/jittered inputs and magnified differences.
- `final-small-jitter/`: experimental all-26 candidate, selected using the comparison;
  its training-fit score is separate from the out-of-fold results.

Focused tests verify reproducibility, nonmutation of input images, binary masks,
zero background, opposite-view depth consistency, the small network's gradient
flow, and augmentation call counts confined to optimizer batches. The original
checkpoint inference remains compatible. Shared `CaseDataset` and `train_model`
entry points are reused; `jitter_views` is called only by training, and `ViewEncoder`
serves the existing shared-view and bottom-view branches at the selected size.
