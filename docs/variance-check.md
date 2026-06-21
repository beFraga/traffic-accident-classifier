# Variance Check — how much of our signal is noise?

**Date:** 2026-06-21
**Scope:** the kept round-3 config (plain Adam, effective-number class weights,
ordinal-neighbour soft labels `neighbor_smoothing=0.2`), retrained under 3 different
**training-process seeds** (1, 2, 3). Only the RNG seed changed; the held-out test set
is fixed (`TrafficDataManager` splits val/test with its own `manual_seed(42)`), so this
isolates training stochasticity (head init, train-loader shuffle, photometric aug).

Wired via `set_seed()` / `resolve_seed()` in `models/conv.py` (`$TAC_SEED` env override,
`seed:` key in `parameters.yaml` as fallback). Raw logs: `training/var_seed{1,2,3}.log`.

---

## TL;DR

Per-class recall on the **rare/mid classes swings ~9–10 points from the seed alone.**
That is the same magnitude as the per-class "improvements" we have been attributing to
loss changes. Overall accuracy is much more stable (~±1.4). Concretely:

- **Overall accuracy** is a trustworthy metric: noise spread 2.8 pts (sd 1.4).
- **Moderate / Severe recall** are NOT trustworthy at single-run resolution: spread
  9.1 / 9.5 pts (sd 4.6 / 5.1).
- The round-3 ordinal-loss headline ("Moderate recall +11") sits **at the noise floor**
  and is therefore **not established** by a single run.

---

## Measurements (same config, seeds 1/2/3, fixed test set)

| Metric | seed1 | seed2 | seed3 | mean | min..max (spread) | sd |
|---|---|---|---|---|---|---|
| Overall accuracy | 78.5 | 77.3 | 75.7 | **77.2** | 75.7..78.5 (2.8) | 1.4 |
| No accident recall | 97.4 | 95.8 | 92.3 | 95.2 | 92.3..97.4 (5.1) | 2.6 |
| Minor recall | 65.3 | 62.8 | 63.3 | 63.8 | 62.8..65.3 (2.6) | 1.4 |
| Moderate recall | 27.5 | 36.6 | 32.5 | 32.2 | 27.5..36.6 (**9.1**) | 4.6 |
| Severe recall | 55.0 | 45.5 | 47.4 | 49.3 | 45.5..55.0 (**9.5**) | 5.1 |
| Totaled recall | 85.0 | 79.1 | 82.9 | 82.3 | 79.1..85.0 (5.9) | 3.0 |

Noise tracks support, as expected: the smallest-support classes (Moderate 42, Severe 22
in the test set) have the widest bands; the common classes (Minor, overall) are tight.

---

## Re-reading the prior rounds against the noise floor

| Prior claim (single run) | Delta | Noise band | Verdict |
|---|---|---|---|
| round1→round2 overall acc | +4.3 | ±2.8 spread | **real** (exceeds band) |
| round2 Minor recall | +23 | ~3 (common class) | **real** |
| round2 Totaled recall | +30 | ~6 | **real** |
| round2→round3 overall acc | −0.5 ("flat") | ±2.8 | within noise (consistent) |
| **round3 Moderate recall** | **+11** | **±9.1** | **NOT established** — at the floor |

The kept round-3 run reported Moderate=38.1% and overall=75.31%. Re-running the *same*
config gives Moderate {27.5, 36.6, 32.5} (mean 32.2) and overall {78.5, 77.3, 75.7}
(mean 77.2). So the 38.1% was a high draw, and the config's true overall mean (~77%) is
actually a bit **higher** than the single number we recorded — but still inside the band
relative to round-2.

---

## Implications

1. **Report overall accuracy as mean ± sd across seeds (~77.2% ± 1.4), not a single
   run.** Single numbers over-/under-sell by up to ~1.5 pts.
2. **Single-run per-class deltas below ~10 pts on Moderate/Severe are not interpretable.**
   We were over-reading them. Minor/Totaled round-2 wins were large enough to survive.
3. **The `neighbor_smoothing` sweep is not worth running as single runs** — the effect it
   targets (mid-class recall) is smaller than its measurement noise. Resolving it would
   need 3+ seeds *per setting* (≥6 runs, ~3.5 h), which is a poor trade given the
   data-limited ceiling.
4. **The ordinal loss stays kept** — not because "+11" is proven, but because it does not
   hurt overall accuracy (config mean ~77%) and it improves error *structure* (errors stay
   ordinal-neighbour / lower severity-MAE), which is a more stable property than per-class
   recall on 22–42 samples.

## Recommended stance

Treat loss-level micro-tuning as **done**. The remaining Moderate/Severe ceiling is
data-limited and below our noise floor; only **data-level** changes (more labelled data
for the rare severities, label-quality review, or geometric augmentation *with* target
transforms) have a plausible chance of clearing the ~3 pt overall / ~10 pt per-class bar.
Any future model change should be judged by **multi-seed mean ± sd**, not a single run.
