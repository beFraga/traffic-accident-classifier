# Training Plateau Analysis & Recommendations

**Date:** 2026-06-17
**Scope:** `cnn/network.py`, `cnn/losses.py`, `parameters.yaml` (plus `cnn/model.py` and `cnn/dataset.py`, which interact with them)
**Symptom under investigation:** training loss keeps improving while **validation loss plateaus** — the model fits the training data but does not generalize.

---

## TL;DR

The plateau is most likely **not** a simple "network too big / too small" problem. There is a learning-rate configuration bug that would stall training on its own, and a data-labeling bug that caps achievable performance. Fix those first, *then* diagnose overfit vs. underfit from the train/val gap.

**Recommended order of changes (one at a time):**

1. **Fix the double LR scheduler** in `cnn/model.py` (LR collapses to ~0 → looks like a plateau).
2. **Fix the `break` in `cnn/dataset.py`** (only the first annotation per image is being used).
3. Raise the learning rate (~`1e-4`) and re-read the train/val gap to classify overfit vs. underfit.
4. If overfitting: add data augmentation + AdamW weight decay + freeze early ResNet layers.
5. Rebalance the loss (objectness scale, `pos_weight`, bbox activation) and test removing mixup.

---

## How to diagnose overfit vs. underfit

The "too big vs. too small" question is answered by the **gap between the curves**, not the validation curve alone:

| Signature | Diagnosis | Cure |
|---|---|---|
| Train loss keeps dropping, val loss flattens/rises, **gap widens** | **Overfitting** (too much capacity / too little data/regularization) | More regularization, more data/augmentation, less capacity |
| **Both** train and val stuck high and close together | **Underfitting** (too little capacity / LR too low / too few epochs) | More capacity, higher LR, train longer |
| Train and val both flatten suddenly and early | **Optimizer stopped moving** (LR collapsed) | Fix the LR schedule |

The reported behavior (train improving, val flat) points to **overfitting** — *but* the third row is a strong candidate here because of the scheduler bug below, and it mimics a plateau. Resolve the mechanical issues first so the diagnosis is trustworthy.

---

## Priority 1 — Learning-rate schedule (most likely cause of the plateau)

**File:** `cnn/model.py` (configures the optimizer/schedulers for `parameters.yaml`).

Two schedulers are attached to the **same optimizer** and both stepped every epoch:

```python
lr_scheduler = StepLR(self.optimizer, lr_decay_every_n_epoch, gamma=0.5)        # halves LR every 15 epochs
lr_cosine    = CosineAnnealingWarmRestarts(self.optimizer, T_0=15, T_mult=2, eta_min=1e-6)
self.schedulers = [lr_scheduler, lr_cosine]
```

**Why it breaks:** both schedulers write the learning rate of the *same* parameter group every epoch, so their effects **compound**. StepLR keeps multiplying the LR (×0.5 @ epoch 15, ×0.25 @ 30, ×0.125 @ 45, …) while Cosine independently overwrites/oscillates it. The net LR collapses toward `eta_min = 1e-6` very quickly. Once the LR is near zero, weights stop updating and **both curves flatten** — indistinguishable from a "plateau."

**Fix — use exactly one scheduler:**

```python
# Option A (recommended): cosine warm restarts only — periodic LR "kicks" help escape plateaus
self.schedulers = [lr_cosine]

# Option B: step decay only
self.schedulers = [lr_scheduler]
```

**Addresses:** the plateau directly (a frozen optimizer, not overfit/underfit). **Highest-value single change.**

---

## Priority 2 — Hyperparameters (`parameters.yaml`)

```yaml
accident_classifier:
  batch_size: 64
  learning_rate: 0.00005
  max_epochs: 100
  patience_epochs: 20
  lr_decay_every_n_epoch: 15
  lr_decay_rate: 0.5
  num_classes: 5
  S: 7
```

- **`learning_rate: 0.00005`** is low. It is acceptable for *fine-tuning* the ResNet, but combined with the collapsing schedule above the model barely moves. After fixing the scheduler, try **`1e-4` to `3e-4`**. *(Underfitting lever.)*
- **`batch_size: 64`** — fine, no change needed.
- **`patience_epochs: 20`** — after the scheduler fix, confirm early stopping isn't firing during a cosine down-swing (which would halt a model that's about to recover on the next restart).
- **`S` and `num_classes`** must stay consistent across dataset, network output channels, and loss — do not change one in isolation.

---

## Priority 3 — Regularization (the real fix *if* it is overfitting)

Once the LR is healthy, if the train/val gap still widens, attack overfitting.

### 3a. Data augmentation (biggest missing lever)

**File:** `cnn/dataset.py` — the transform currently only resizes + normalizes:

```python
self.transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.RandomApply([transforms.GaussianBlur(3)], p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

Augmentation is effectively "more data" — the most reliable generalization fix. **Note:** start with the color/blur augmentations above; geometric augmentations (flips, crops) also require transforming the bbox/grid targets, so they are a larger change. *(Addresses overfitting.)*

### 3b. Weight decay (Adam → AdamW)

**File:** `cnn/model.py`:

```python
self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.learning_rate, weight_decay=1e-4)
```

Penalizes large weights so the model relies on general patterns. *(Addresses overfitting.)*

### 3c. `TransferResnet` capacity (`cnn/network.py`)

Currently the **entire** ResNet18 backbone (~11M params) is trainable. On a small dataset that is a lot of capacity to memorize with. Freeze early/generic layers, or do two-phase training (freeze → train head → unfreeze):

```python
for p in self.backbone[:6].parameters():   # freeze early/generic layers
    p.requires_grad = False
```

*(Addresses overfitting — fewer free parameters, preserves pretrained general features.)*

### Architecture notes (`cnn/network.py`)

- The from-scratch `AccidentClassifierNet` downsamples 224 → 112 → 56 → 28 → 14 → 7, landing exactly at the `S = 7` grid. Spatial design is sound.
- Both nets already use BatchNorm + LeakyReLU + Dropout2d (0.2 in the scratch head, 0.3 in the transfer head). Capacity is not obviously too small, so **underfitting via "network too short" is unlikely** — focus on the LR and regularization first.

---

## Priority 4 — Loss function (`cnn/losses.py`)

```python
self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([7.0]))
self.ce  = nn.CrossEntropyLoss(label_smoothing=0.1)
self.mse = nn.MSELoss()
self.class_w = torch.tensor([4.0, 0.6, 1.0, 6.5, 0.6])
...
total_loss = loss_obj + (2.0 * loss_cls) + (1.5 * loss_box)
```

1. **Objectness dominates by scale.** `loss_obj` is BCE averaged over **all 49 grid cells**; `loss_cls`/`loss_box` are averaged over only the **few occupied cells**. Even with the `2.0`/`1.5` weights, the *gradient* is dominated by objectness — the model can drive total loss down while barely learning classification, which reads as a "val plateau" on the metric that matters. **Action:** plot the per-term validation losses separately (they are already returned — watch `validation_loss_classification`) to see *which* term is stuck.

2. **`pos_weight = 7.0` is a hard-coded guess.** The appropriate value ≈ (empty cells / occupied cells). On a 7×7 grid with few accidents per image, that ratio is closer to ~40–48, not 7. Too low makes the model lazy about detecting accidents. *(Underfitting on the detection task.)*

3. **Bounding-box outputs have no activation.** The network emits **unbounded** raw values for `x, y, w, h`, but targets lie in `[0, 1]`. MSE between an unbounded prediction and a `[0,1]` target produces large, noisy gradients. Apply a sigmoid to the 4 coordinate outputs before the MSE. *(Stability / generalization.)*

4. **Possible dead weight: `class_w[0] = 4.0` ("No accident").** Classification loss only runs on cells where `objectness == 1` (an accident *is* present), so class 0 may never appear there. If so, weighting it does nothing. Verify the label encoding and re-derive class weights from **actual class frequencies** rather than hand-picked values.

---

## Priority 5 — Mixup augmentation

**File:** `cnn/model.py`, `train_one_epoch` — 50% of batches blend two images **and their targets**:

```python
images  = lam * images  + (1 - lam) * images[index, :]
targets = lam * targets + (1 - lam) * targets[index, :]
```

For plain classification mixup is great, but here the target includes **objectness, one-hot class, and bbox coordinates**. Linearly averaging two *different* bounding boxes yields a coordinate pointing at neither object; averaging objectness yields fractional "half-objects." This injects noise into exactly the signals to be learned. **Action:** disable mixup and compare — if val improves, it was hurting.

---

## Critical adjacent bug (outside the three files, but it caps everything)

**File:** `cnn/dataset.py`. The label-parsing loop's `break` is at the *loop* level, not inside the `if`:

```python
for line in f:
    ...
    if target[grid_y, grid_x, 0] == 0:
        target[...] = ...
    break   # <-- exits after the FIRST line, always
```

The comment says "one object per cell," but as written it keeps **only the first annotation in the entire file** and discards every other accident in the image. If images contain multiple accidents, the model is trained on mostly-wrong targets — which caps validation performance regardless of architecture. **Action:** remove the `break` and let the per-cell `if ... == 0` guard enforce "one object per cell."

---

## Summary table

| # | Change | File | Fixes |
|---|---|---|---|
| 1 | Use a single LR scheduler | `cnn/model.py` | Plateau (LR collapse) |
| 2 | Remove `break` in label loop | `cnn/dataset.py` | Wrong targets (caps performance) |
| 3 | Raise LR to ~`1e-4`, re-read curve gap | `parameters.yaml` | Underfitting / diagnosis |
| 4 | Add color/blur augmentation | `cnn/dataset.py` | Overfitting |
| 5 | Adam → AdamW (`weight_decay=1e-4`) | `cnn/model.py` | Overfitting |
| 6 | Freeze early ResNet layers | `cnn/network.py` | Overfitting |
| 7 | Recompute `pos_weight`; sigmoid bbox outputs; verify class weights | `cnn/losses.py`, `cnn/network.py` | Detection quality / stability |
| 8 | Test disabling mixup | `cnn/model.py` | Label noise |

**Change one thing at a time** so each improvement can be attributed.
