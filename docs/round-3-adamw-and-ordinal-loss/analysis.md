# Class-Weight Results Analysis (Round 2 outcome → Round 3 plan)

**Date:** 2026-06-21
**Scope:** outcome of implementing recommendation **#1** from [`../round-2-class-weights/analysis.md`](../round-2-class-weights/analysis.md) — replacing the hand-picked `class_w` with weights derived from real label frequencies.
**Change under test (one variable):** `cnn/losses.py` `class_w` `[4.0, 0.6, 1.0, 6.5, 0.6]` → frequency-derived effective-number weights, computed at startup in `models/conv.py` and read via new `parameters.yaml` keys. Everything else (batch_size=64, LR, schedule, optimizer) held at the round-1 baseline.

> **Data source:** all numbers below come from [`../round-2-class-weights/results.txt`](../round-2-class-weights/results.txt) — the test report, confusion matrix, and history signals from the 2026-06-21 run on the real Kaggle dataset (`marslanarshad/car-accidents-and-deformation-datasetannotated`), GPU GeForce MX450.

---

## TL;DR

The frequency-derived weights are a **confirmed win**: overall classification accuracy **71.55% → 75.84% (+4.3 pts)**, and the classes moved exactly as the round-2 analysis predicted. The remaining weakness is now **isolated and characterized**: the middle severity classes (Moderate, Severe) confuse with their **ordinal neighbors**, not with "No accident." Two follow-ups are now backed by direct evidence that was unavailable during round 2:

1. **Ordinal-neighbor confusion** (confusion matrix) → recommendation **#3: ordinal-aware / soft-label loss**.
2. **Overfitting on the classification term** (history train/val gap widened −0.16 → +0.65) → recommendation **#4: AdamW + weight_decay**.

---

## The weights that were applied

Computed from occupied-cell frequencies across `dataset/labels/{train,val}` (β=0.999, max-ratio cap 10×, `no_accident_weight_cap=1.0`):

| Class | occupied count | share | old `class_w` | new `class_w` | direction |
|---|---|---|---|---|---|
| No accident | 1707 | 49.8% | 4.0 | **0.466** | slashed (was over-weighted) |
| Minor | 570 | 16.6% | 0.6 | **0.877** | raised (was wrongly down-weighted) |
| Moderate | 454 | 13.2% | 1.0 | 1.044 | ≈ unchanged |
| Severe | 314 | 9.2% | 6.5 | **1.414** | slashed |
| Totaled | 382 | 11.1% | 0.6 | **1.200** | raised |

The cap on class 0 never bound — effective-number already placed it at 0.466.

---

## Verified findings

### 1. Accuracy and per-class metrics vs. the round-1 baseline

| Class | weight (old→new) | recall: base → new | precision: base → new |
|---|---|---|---|
| No accident | 4.0 → 0.47 | 93.7% → 92.9% (flat) | 84% → **93%** ↑ |
| **Minor** | 0.6 → 0.88 | 41.2% → **64.6% (+23)** | 75% → 70% |
| Moderate | 1.0 → 1.04 | 36.4% → **27.0% (−9)** | 46% → 38% |
| Severe | 6.5 → 1.41 | 61.4% → 47.4% (−14) | 39% → 31% |
| **Totaled** | 0.6 → 1.20 | 52.6% → **82.5% (+30)** | 87% → 73% |

**Overall: 71.55% → 75.84%.**

- **Minor** and **Totaled** (both up-weighted) are the headline recall wins (+23, +30) — direct confirmation of the round-2 thesis that they were wrongly down-weighted.
- **No accident** (cut 4.0→0.47) kept its recall *and* gained 9 pts of precision: cutting the over-weighted easy class stopped it over-predicting. This is recommendation **#2** working as intended.
- **Severe** (cut 6.5→1.41) lost recall, as expected — 6.5 was inflating it. An accepted trade.

### 2. The loss terms are NOT comparable across rounds

Test classification loss "rose" 1.052 → 1.177, but the cross-entropy is **weighted by `class_w`, which changed this round** — so the magnitude is not comparable. Accuracy is the fair metric, and it improved. The two weight-independent terms both improved slightly: objectness 0.859 → 0.825, regression 0.0221 → 0.0175.

### 3. Confusion matrix → the remaining error is **ordinal-neighbor confusion**

Test matrix (rows = true, cols = predicted), `conf_threshold=0.4`:

```
            NoAcc Minor  Mod  Sev  Tot   support
true NoAcc   143    4    3    1    3      154
true Minor     4   31    7    3    3       48
true Mod       6    6   10   12    3       37   <- only 10 correct
true Sev       0    2    5    9    3       19
true Tot       1    1    1    4   33       40
```

Where the errors land:

- **Moderate** errors: 18 of 27 go to adjacent severities (Minor 6 + Severe 12) = **67%**; only 6 leak to "No accident."
- **Severe** errors: 8 of 10 go to neighbors (Moderate 5 + Totaled 3) = **80%**; **zero** to "No accident."

This resolves the open sub-question from `../round-2-class-weights/analysis.md` §Diagnosis: the middle classes confuse **with each other along the severity axis**, *not* with "No accident." The objectness gate is healthy ("No accident" precision 93%). → indicates **ordinal/soft-label loss (#3)**, not the gating fix (cutting `class_w[0]` / threshold sweep).

### 4. History signals (newly measurable — the pickle now exists)

- **Overfitting confirmed (#4):** classification train/val gap widened from **−0.164** (start) to **+0.653** (last-10-epoch mean). This is the curve evidence that was missing in round 2; AdamW + weight_decay is now justified rather than conditional.
- **Schedule (#5), mild:** ran 78/100 epochs (early-stopped), best val at **epoch 57** — inside the long 45→105 cosine down-swing, with the next warm restart (epoch 105) never reached before the cap. Real but mild: val kept improving to epoch 57, so the schedule is not catastrophically stranding training. Lower priority than #4.

---

## Caveats

- **Small test set.** Per-class support: No accident 154, Minor 48, Moderate 37, **Severe 19**, Totaled 40 (298 occupied cells total). Severe's −14 recall is a swing of ~3 samples; the Moderate/Severe percentages are noisy. The *pattern* (errors concentrated on ordinal neighbors) is robust; the exact deltas are not.
- **One run, no seed sweep.** Effects are attributed to the single `class_w` change because nothing else moved, but a repeat run would quantify variance.
- **Augmented dataset.** The Kaggle dataset is pre-augmented (filenames like `*_flip70`), so geometric augmentation in-pipeline (the deferred item) is partly redundant.

---

## Ranked recommendations (Round 3, one variable at a time)

| # | Change | File | Now backed by | Metric to watch |
|---|---|---|---|---|
| 1 | **AdamW + `weight_decay`** (the un-done Priority 3b) | `cnn/model.py:164` + yaml key | history gap −0.16→+0.65 (overfitting, finding 4) | train↔val classification gap; Moderate recall |
| 2 | **Ordinal-aware / soft-label classification loss** | `cnn/losses.py` (`self.ce`) | confusion matrix: 67–80% of mid-class errors are ordinal neighbors (finding 3) | Moderate/Severe recall vs. neighbor off-diagonals |
| 3 | **Cap/align the cosine schedule** so early-stop can't strand the 45→105 down-swing | `parameters.yaml` / `cnn/model.py` | best-val@57 mid-decay, restart@105 never reached (finding 4) | stop-epoch vs. cosine minima |
| — | *Deferred:* geometric augmentation | `cnn/dataset.py` | dataset already pre-augmented | val classification loss |

### Recommended first move

**#1 — AdamW + `weight_decay`.** Lowest-risk single change, directly targets the now-confirmed overfitting gap, and is the only remaining item from the original `../round-1-mechanical-fixes/analysis.md` Priority 3b. Then **#2 (ordinal/soft loss)** to attack the Moderate/Severe ceiling that the matrix isolates.

---

## Constraints carried forward (from `CLAUDE.md`)

- Hyperparameters (incl. `weight_decay`, any ordinal-loss params) belong in `parameters.yaml`, read by key — never hardcoded in `cnn/`.
- `S` and `num_classes` stay consistent across dataset, network output channels, and loss.
- **Change one variable at a time**, retrain with `python -m models.conv train`, evaluate with `python -m models.conv run`, and attribute each effect before stacking the next.
