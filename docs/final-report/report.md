# Final report — Traffic accident severity grid classifier

The result of the analysis-driven improvement campaign (rounds 1–3), evaluated as a
single clean, seeded, reproducible run. This is the configuration carried out of the
[experiment log](../README.md): ResNet18 transfer head (`tp=1`), frequency-derived
effective-number class weights, and the ordinal-neighbour soft-label classification
loss. No new variable is introduced here — this run **certifies the kept config** for
the report.

## Run provenance

| Field | Value |
|---|---|
| Date | 2026-06-22 |
| Seed | 42 (training RNG; val/test split has its own fixed seed) |
| Hardware | GeForce MX450 (2 GB), CUDA |
| Backbone | ResNet18 ImageNet transfer, last two layers stripped (`tp=1`) |
| Batch size | 64 |
| Optimizer | Adam, lr 1e-4, StepLR×Cosine schedule |
| Class weights | effective-number (β 0.999, max-ratio 10, No-accident cap 1.0) |
| Class loss | ordinal-neighbour soft labels, `neighbor_smoothing=0.2` |
| Test set | 50/50 carve from `val/` (fixed split), N = 311 occupied cells @ conf 0.4 |
| Stopping | early-stopped at epoch 66/100; best val-total epoch 47 |
| Train time | ~1669 s (~28 min) |
| Artifacts | [`loss_curve.png`](loss_curve.png), [`confusion_matrix.png`](confusion_matrix.png), [`training-run.log`](training-run.log) |

Effective-number class weights applied (from real occupied-cell frequencies):

| Class | Count | Weight |
|---|---|---|
| No accident | 1707 | 0.466 |
| Minor | 570 | 0.877 |
| Moderate | 454 | 1.044 |
| Severe | 314 | 1.414 |
| Totaled Vehicle | 382 | 1.200 |

## Headline result

**Overall classification accuracy: 77.17%** (240 / 311 correctly classified occupied
cells).

This sits squarely on the established noise floor for this config —
**77.2% ± 1.4** across three seeds on the fixed test set
([variance-check.md](../variance-check.md)). So 77.17% is the *expected* central
result, not an outlier high or low: it is the number to report.

### Test-set loss terms

| Term | Value |
|---|---|
| Total | 2.89433 |
| Objectness | 0.80900 |
| Classification | 1.02763 |
| Regression (bbox) | 0.02004 |

> Note: the classification loss is **not** comparable to pre-round-3 numbers — the
> ordinal soft targets raise the minimum achievable cross-entropy. Judge classification
> quality by accuracy / recall, not by this term.

## Per-class performance

| Accident type | Precision | Recall | Support |
|---|---|---|---|
| No accident | 94.97% | 94.97% | 159 |
| Minor | 68.09% | 64.00% | 50 |
| Moderate | 41.18% | 35.90% | 39 |
| Severe | 30.30% | 50.00% | 20 |
| Totaled Vehicle | 86.84% | 76.74% | 43 |

The two extremes of the severity scale are solved (No accident ~95%, Totaled ~87%
precision). The aggregate accuracy is genuine but is **carried by these easy classes
plus the dominant No-accident support**. The mid-scale severities — Moderate and
Severe — remain the bottleneck, exactly as every round since the class-weight round
predicted.

## Confusion matrix (rows = true, cols = predicted)

conf_threshold = 0.4, N = 311

```
          NoAcc  Minor   Mod   Sev   Tot   support
true NoAcc  151      2     4     1     1     159
true Minor    4     32    10     3     1      50
true Mod      3      8    14    13     1      39
true Sev      0      3     5    10     2      20
true Tot      1      2     1     6    33      43
```

See [`confusion_matrix.png`](confusion_matrix.png) for the rendered version.

## Error structure — the errors are ordinal

| Metric | Value |
|---|---|
| Ordinal MAE (all cells) | 0.322 |
| Total misclassifications | 71 |
| Off-by-one (neighbour severity) | 50 (70% of errors) |
| Off-by-two-or-more (far) | 21 (30% of errors) |
| MAE \| true = Moderate | 0.744 (n = 39) |
| MAE \| true = Severe | 0.650 (n = 20) |

**70% of all errors are confusions with an immediately adjacent severity level.** The
model has correctly learned the *ordering* of severity; what it lacks is fine
discrimination at the Moderate↔Severe boundary (true Moderate is predicted Severe 13
times — its single largest confusion). This is the signature the ordinal soft-label
loss was designed to produce, and it confirms the residual weakness is a
*separability* problem at the middle of the scale, not a model that confuses
unrelated classes.

## Training dynamics

From [`training-run.log`](training-run.log) and the history pickle:

- Early stopping at epoch 66; best validation-total at epoch 47 (val-total 2.823),
  best validation-classification loss 0.997 at epoch 61.
- Classification train/val gap: start −0.229 → last-10-epoch mean **+0.431** → final
  +0.456. The smaller gap vs. earlier rounds (+0.65) is partly mechanical — soft
  targets raise the train-CE floor — so it is not read as "overfit fixed," but the
  curve is stable and the run stopped on a clean plateau (see
  [`loss_curve.png`](loss_curve.png)).
- Final train cls 0.571 / val cls 1.028. That the train fit never collapses to ~0
  corroborates the round-3 finding: the val gap is a **class-confusion ceiling**, not
  classic overfit, and is why L2/weight-decay was rejected.

## Verdict for the report

This is the certified final state of the classifier. **77.2% overall accuracy** on a
held-out, severity-imbalanced test set, with the two scale extremes solved and the
residual error concentrated, *by design*, in off-by-one confusions at the
Moderate/Severe boundary (70% of all errors are neighbour-level; ordinal MAE 0.32).

The Moderate/Severe ceiling is **data-limited** — test support is only 39 and 20
cells respectively, so per-class recall swings ~9–10 points from the training seed
alone. Further loss surgery faces diminishing returns; the next real lever is more
labelled mid-severity data, not architecture or regularization.
