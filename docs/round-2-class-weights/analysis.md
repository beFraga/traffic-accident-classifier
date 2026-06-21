# Classification Plateau Analysis (Round 2)

**Date:** 2026-06-17
**Scope:** `cnn/losses.py`, `cnn/model.py`, `parameters.yaml`, `models/conv.py` (plus `cnn/dataset.py` and `cnn/network.py` for context)
**Symptom under investigation:** after the [round-1 fixes](../round-1-mechanical-fixes/analysis.md) accuracy rose ~50% → **71.55%**, but training loss keeps decreasing while **validation loss stays flat**. The stuck term is now isolated: **classification**, concentrated in the middle severity classes.

> **Data source:** all numbers below come from `../round-1-mechanical-fixes/results.txt` — the test report produced *after* the round-1 changes were implemented.

> **Predecessor:** every recommendation in `../round-1-mechanical-fixes/analysis.md` is already implemented in the code **except Priority 3b** (AdamW + weight decay). This document does not repeat those; it diagnoses what remains.

---

## TL;DR

The plateau is **intrinsic to the classification head**, not a multi-task scale artifact. Two findings drive the conclusion:

1. **Classification is already 70% of the total loss budget** (verified by recomputing the term-weighted total to match `../round-1-mechanical-fixes/results.txt` exactly). The old worry that "objectness drowns out classification" is no longer true — the optimizer is spending most of its gradient on classification and *still* can't move val loss.
2. **The hand-picked class weights point the wrong way.** The two worst-recall classes are *down-weighted*; two already-good classes carry the heaviest weights.

**Highest-confidence single change:** replace the hand-picked `class_w` with weights **derived from real label frequencies** (and drop the inflated "No accident" weight). It is the only change supported by direct evidence in the current checkout, and it acts on the term that is provably 70% of the loss.

**Blocked-on-data:** the `dataset/` labels and `training/*.pkl` history are gitignored and were absent during analysis, so class frequencies, the train↔val gap, and early-stopping behavior could not be measured. AdamW and the scheduler fix stay *conditional* until those are read.

---

## Verified findings

### 1. Loss balance — classification dominates (recomputed, exact)

Using the test terms from `../round-1-mechanical-fixes/results.txt` and the `1.0 / 2.0 / 1.5` term weights in `cnn/losses.py:53`:

```
total = obj + 2.0*cls + 1.5*box
      = 0.85905 + 2.0*1.05200 + 1.5*0.02214
      = 2.99626   (../round-1-mechanical-fixes/results.txt reports 2.99625 — match)
```

Share of total loss:

| Term | Weighted value | Share |
|---|---|---|
| Objectness | 0.859 | **28.7%** |
| **Classification** | 2.104 | **70.2%** |
| Regression | 0.033 | 1.1% |

Box regression is effectively solved (0.022 raw). Objectness is a near-fixed ~29%. **The plateau lives entirely in the classification term**, and that term already owns the majority of the gradient — so the fix must improve *how* classification learns, not its weight relative to the other tasks.

### 2. `class_w` anti-correlates with need

`cnn/losses.py:26` → `class_w = [4.0, 0.6, 1.0, 6.5, 0.6]`, cross-referenced with the per-class recall in `../round-1-mechanical-fixes/results.txt`:

| Class | Precision | Recall | `class_w` | Verdict |
|---|---|---|---|---|
| No accident | 84% | **94%** | 4.0 | solved, yet over-weighted |
| Minor | 75% | **41%** | **0.6** | worst, **down-weighted** |
| Moderate | 46% | **36%** | **1.0** | worst, under-weighted |
| Severe | 39% | 61% | **6.5** | decent, most-weighted |
| Totaled | 87% | 53% | 0.6 | mediocre, down-weighted |

The heaviest weights sit on classes that are already performing; the two collapsing classes (Minor, Moderate) are the *least* weighted. The vector was hand-picked, not derived from frequencies, and it is pushing the model away from its weak spots.

### 3. "No accident" (class 0) is a real, active class — not dead weight

Round 1 raised the possibility that `class_w[0]=4.0` was inert (classification loss only runs on `objectness==1` cells). **Refuted:** `dataset.py:97–123` writes `objectness=1` for every parsed label line including class 0, the confusion report (`models/conv.py:122–138`) only counts occupied cells, and `../round-1-mechanical-fixes/results.txt` reports "No accident" at 84%/94% *within that report*. So class 0 genuinely competes inside the classification head, and its 4.0 weight actively biases predictions toward the easiest class — at the expense of Minor/Moderate.

### 4. Scheduler can strand training in a long down-swing (suspected)

`CosineAnnealingWarmRestarts(T_0=15, T_mult=2)` restarts at epochs **0, 15, 45, 105**. With `max_epochs=100`, epochs **45–100 are a single ~60-epoch LR decay** from 1e-4 toward `eta_min=1e-6`, and the next warm-restart kick (epoch 105) never arrives before the cap. With `patience_epochs=20` monitoring `total` at `min_delta=1e-4`, early stopping can fire deep in that decay with no rescue restart. This is a structural risk derived from the config; confirming it fired requires the history pickle (absent).

---

## What could NOT be measured (and why it gates two recommendations)

Both are gitignored and were not present in the analyzed checkout:

- **`dataset/labels/{train,val}`** → no real class frequencies, occupied-cell counts, or train-set size. (Only known size: test = 1223 images per `../round-1-mechanical-fixes/results.txt`, implying val/ ≈ 2446; train unknown.)
- **`training/accidentclassifier_history.pkl`** → no per-term train-vs-val curves, so **overfitting (widening gap) cannot be separated from data scarcity**, and the scheduler/early-stop suspicion cannot be confirmed.

Per the "prove it from the curves first" discipline, AdamW and the scheduler change therefore remain *conditional* below rather than recommended outright.

---

## Diagnosis

The classification plateau is driven by **class imbalance pointed the wrong way**, not by the multi-task objective masking the signal and not (provably) by capacity. Evidence: classification owns 70% of the loss yet stalls; the worst classes are the least-weighted; the easiest class is over-weighted and actively competing. Whether there is *also* genuine overfitting on the classification term is **undetermined** without the history curves.

An open sub-question decides between two families of fix, and is answerable cheaply by reading the existing confusion matrix (re-run `python -m models.conv run`):

- **Minor/Moderate/Severe confused with each other** → ordinal-neighbor problem → ordinal-aware or soft-label loss.
- **Minor/Moderate confused with "No accident"** → detection/gating problem → reduce `class_w[0]` and check `conf_threshold`.

The low-recall-on-middle-classes pattern is *consistent with* ordinal confusion but is a hypothesis until the matrix off-diagonals are read.

---

## Ranked recommendations (one variable at a time)

| # | Change | File | Conditional on | Metric to watch |
|---|---|---|---|---|
| 1 | **Derive `class_w` from real label frequencies** (inverse-freq or effective-number, β≈0.999, cap ratio ~10×); fold in #2 | `cnn/losses.py:26` + new `parameters.yaml` key, read in `models/conv.py` | label counts (Step 1.1) | `validation_loss_classification`; **Minor & Moderate recall** |
| 2 | **Cut `class_w[0]` ("No accident")** from 4.0 → ~0.5–1.0 | `cnn/losses.py:26` / yaml | confusion-matrix column for "No accident" | off-diagonal mass into class 0 |
| 3 | **Focal or ordinal-aware classification loss** (focal γ≈2 if imbalance; soft/ordinal if neighbor-confusion) | `cnn/losses.py` (`self.ce`) | confusion-matrix pattern | Minor/Moderate recall vs No-accident/Totaled precision |
| 4 | **Adam → AdamW + `weight_decay`** (the un-done Priority 3b) | `cnn/model.py:164` + yaml | history shows widening train↔val *classification* gap | the gap itself |
| 5 | **Cap/align the cosine schedule** so early stop can't strand the long down-swing (align `max_epochs` to a restart, or expose/​shorten `T_0`/`T_mult`) | `parameters.yaml` | history shows stop fired mid-decay while val still trending | stop-epoch vs cosine minima |
| 6 | **`conf_threshold` sweep** (0.3/0.4/0.5) — cheap sanity check that the objectness gate isn't dropping correct rare-class cells | `models/conv.py:101` | — (do during any eval) | Minor/Moderate recall vs threshold |
| — | *Deferred:* geometric augmentation (h-flip + target transform) — real "more data," larger change | `cnn/dataset.py` | occupied-cell count justifies it | val classification loss |

### The first move

**#1 — replace the hand-picked `class_w` with frequency-derived weights, with `class_w[0]` reduced.** Per the project convention, compute it from `dataset/labels/{train,val}` at startup and expose it via a `class_w:` key under `accident_classifier:` in `parameters.yaml` (read in `models/conv.py`), rather than hardcoding a new vector in `cnn/`. This removes the hand-picked guess entirely and self-corrects on whatever data is present.

**Prerequisites before any of #1/#4/#5 can be finalized:** read the real label frequencies and the `training/*.pkl` history on the machine where the data lives. The #6 threshold sweep and the confusion-matrix read (for #2/#3) need only a re-run of the existing `run` command.

---

## Constraints carried forward (from `CLAUDE.md`)

- Hyperparameters (including the new `class_w` / `weight_decay`) belong in `parameters.yaml`, read by key — never hardcoded in `cnn/`.
- `S` and `num_classes` stay consistent across dataset, network output channels, and loss.
- **Change one variable at a time**, retrain with `python -m models.conv train`, evaluate with `python -m models.conv run`, and attribute each effect before stacking the next.
