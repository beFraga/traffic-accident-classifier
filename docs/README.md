# Experiment log

A chronological record of the work to lift the traffic-accident grid classifier off
its validation plateau. The discipline throughout: **change one variable at a time**,
retrain (`python -m models.conv train`), evaluate (`python -m models.conv run`), and
attribute each effect before stacking the next.

## How this is organised

One directory per **round**. Each round holds two files:

| File | What it is |
|---|---|
| `analysis.md` | The diagnosis + ranked plan that *motivated* that round's change. It analyses the **previous** round's `results.txt` and proposes what to try next. |
| `results.txt` | The raw test report from running that round's change (accuracy, per-class precision/recall, loss terms, confusion matrix, history signals) plus an inline verdict. |

So the chain reads: `round-N/analysis.md` consumes `round-(N-1)/results.txt` and produces `round-N/results.txt`. Round 1's `analysis.md` diagnoses the original plateau.

`prompts/` holds reusable session prompts (not part of the round narrative).

## Scoreboard

Overall classification accuracy on the held-out test set (carved 50/50 from `val/`),
GeForce MX450, ResNet18 transfer (`tp=1`), batch_size 64.

| Round | Change (one variable) | Accuracy | Headline effect | Verdict |
|---|---|---|---|---|
| — | baseline | ~50% | — | — |
| [1](round-1-mechanical-fixes/) | Mechanical fixes: single LR scheduler, removed label-loop `break`, LR→1e-4, color/blur aug, frozen early ResNet, `pos_weight=40`, sigmoid bbox, mixup off | **71.55%** | broke the plateau | kept |
| [2](round-2-class-weights/) | Hand-picked `class_w` → frequency-derived effective-number weights | **75.84%** | Minor +23, Totaled +30 recall | kept |
| [3](round-3-adamw-and-ordinal-loss/) #1 | Adam → AdamW + `weight_decay` (1e-4, 1e-2) | 74.39–75.42% | gap frozen ~+0.63, acc flat→down | **rejected** (overfit is a class-confusion ceiling, not L2-fixable) |
| [3](round-3-adamw-and-ordinal-loss/) #2 | Uniform label smoothing → ordinal-neighbour soft labels (`neighbor_smoothing=0.2`) | 75.31% | **Moderate recall 27%→38% (+11)**, ordinal MAE 0.36 | kept |

## Current state of the code

- Optimizer: plain **Adam** (AdamW tried and reverted in round 3).
- Classification loss: **ordinal-neighbour soft labels**, `neighbor_smoothing=0.2`.
- Class weights: **effective-number**, derived from label frequencies at startup.
- Remaining ceiling on Moderate/Severe is largely **data-limited** (test support 42/22),
  so expect diminishing returns from further loss surgery.

## Open candidates (next rounds)

- Sweep `neighbor_smoothing` (0.1 / 0.3) for the recall vs. ordinal-MAE sweet spot.
- Round-3 #3: cosine-schedule realignment so early-stop can't strand the long
  45→105 down-swing (documented as mild / low priority).

## Conventions (from `CLAUDE.md`)

- Hyperparameters live in `parameters.yaml`, read by key — never hardcoded in `cnn/`.
- `S` and `num_classes` stay consistent across dataset, network, and loss.
