# Prompt: Diagnose and improve the validation plateau (round 2)

> Paste everything below the line into a fresh Claude Code session at the repo root.

---

You are working on a YOLO-style **grid classifier** for traffic-accident severity
(`cnn/`, orchestrated by `models/conv.py`, hyperparameters in `parameters.yaml`).
Read `CLAUDE.md` first for the architecture and conventions, then read
`docs/training-plateau-analysis.md` — that was a prior analysis and **its
recommendations are already implemented in the code** (single cosine scheduler,
`break` removed from the label loop, LR raised to `1e-4`, color/blur augmentation,
frozen early ResNet layers, `pos_weight=40`, sigmoid on bbox outputs, mixup
disabled). **The one item NOT yet done is Priority 3b** (AdamW + weight decay — the
optimizer in `cnn/model.py` is still plain `Adam`).

## Current state (after those fixes)

Latest run (`docs/output-after-round1.txt`):
- Overall classification accuracy rose from ~50% to **71.55%**.
- Test loss breakdown: objectness 0.859, **classification 1.052** (dominant),
  regression 0.022.
- Per-class precision / recall:
  - No accident: 84% / 94% — essentially solved
  - Minor: 75% / **41%**
  - Moderate: **46% / 36%**
  - Severe: **39%** / 61%
  - Totaled: 87% / 53%
- **Symptom:** training loss keeps decreasing while **validation loss is flat** —
  the classic generalization gap. The stuck term is **classification**, and the
  damage is concentrated in the middle severity classes.

## Your job

Figure out what will actually move validation performance, and justify each
proposal with **evidence from this repo's data and training history — not from
priors**. Do not start editing code until you have measured the things below.
The previous round already exhausted the obvious mechanical fixes; the remaining
ceiling is almost certainly **class imbalance + limited data + the multi-task
objective**, so prove or disprove that before touching the network.

### Step 1 — Measure before theorizing

Write a small throwaway script (or inline analysis) to establish:

1. **Class frequency** across `dataset/labels/{train,val}` — count how often each of
   the 5 severity indices actually appears. Quantify the imbalance and compare it to
   the hand-picked `class_w = [4.0, 0.6, 1.0, 6.5, 0.6]` in `cnn/losses.py`. Are the
   weights even pointing at the right classes? (Note the worst classes — Moderate,
   Severe — and check whether the weights reflect their scarcity.)
2. **Does class 0 ("No accident") ever appear in an `objectness==1` cell?**
   Classification loss only runs on occupied cells, so if class 0 never co-occurs
   with objectness, `class_w[0]=4.0` is dead weight — confirm or refute.
3. **Dataset sizes** — how many train / val / test images, and how many *occupied
   grid cells* total? This decides whether "more regularization" or "more/synthetic
   data / resampling" is the right lever. The test set is carved 50/50 from `val/`
   (seeded), so val and test share a distribution — note any train↔val skew.
4. **Per-term train-vs-val curves.** Load the history pickle in `training/`
   (`accidentclassifier_history.pkl`) and look at `train_loss_*` vs
   `validation_loss_*` over epochs for each term (objectness, classification,
   regression). Confirm *which* term's gap is widening and *when* it plateaus. Check
   whether early stopping (`patience_epochs: 20`) is firing during a cosine
   down-swing and cutting training short.

### Step 2 — Diagnose

From the measurements, classify the bottleneck and say why:
- Is the classification plateau driven by **imbalance** (model defaults to easy
  classes), **too little data** for the rare classes, **label/encoding issues**, or
  **genuine overfitting** (train↔val gap widening on the classification term)?
- Is the **multi-task loss** masking the problem — i.e. is total loss dropping
  because objectness/box improve while classification stalls? Recompute the
  effective gradient balance now that `pos_weight=40` and the `2.0/1.5` term weights
  are in play.

### Step 3 — Propose targeted changes, ranked, one variable at a time

For each proposal give: the file, the specific change, the evidence it addresses,
and the metric you'd watch to confirm it helped. Candidates worth evaluating
(include or discard each based on Step 1–2 evidence, don't apply blindly):

- **Re-derive `class_w` from the measured frequencies** (e.g. inverse-frequency or
  effective-number weighting) instead of the hand-picked vector.
- **AdamW + `weight_decay`** (the unimplemented Priority 3b) — only if the
  classification train↔val gap is genuinely widening.
- **Address the middle-class confusion directly** — inspect the confusion matrix
  from `plot_traffic_confusion_matrix`: are Minor/Moderate/Severe being confused with
  each other (ordinal neighbors) or with "No accident"? That distinguishes a
  *labeling/ordinal* problem from a *detection* problem and changes the fix.
- **More aggressive augmentation, including geometric** (horizontal flip + the bbox/
  grid-target transform it requires) — the biggest untapped "more data" lever, but
  it's a larger change because targets must be transformed too.
- **Rebalancing the loss term weights or the classification sampling** (e.g. focal
  loss for classification) if imbalance is the proven cause.
- **Capacity / `conf_threshold`** sanity checks: confirm the accuracy metric's
  objectness gate isn't silently dropping correct predictions for the rare classes.

### Constraints (from `CLAUDE.md`)

- Hyperparameters go in `parameters.yaml`, read by key — do not hardcode in `cnn/`.
- `S` and `num_classes` must stay consistent across dataset, network, and loss.
- **Change one variable at a time** and re-train so each effect is attributable;
  retrain with `python -m models.conv train` and evaluate with
  `python -m models.conv run`.
- Don't re-apply anything already implemented (see the list above). Verify the
  current code rather than trusting this summary.

Deliverable: a short written diagnosis backed by the Step 1 numbers, then a ranked
change list. Implement only the single highest-confidence change first and report
the before/after on validation classification loss and per-class recall.
