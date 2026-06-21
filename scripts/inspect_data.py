"""
Phase-0 evidence dump for docs/classification-plateau-analysis.md.

The round-2 analysis gates most recommendations on three measurements that live
with the data (not in the repo). This script prints all three in one shot so the
conditional decisions (#3 loss change, #4 AdamW, #5 schedule) can be made:

  1. Class frequencies counted as the loss sees them (occupied grid cells),
     plus the effective-number weights they imply (feeds #1/#2).
  2. Per-term train-vs-val curves from training/accidentclassifier_history.pkl:
     best-val epoch, classification train/val gap, and whether early-stop fired
     during a cosine down-swing (feeds #4 and #5).
  3. A reminder to read the confusion-matrix off-diagonals from
     `python -m models.conv run` (feeds #3).

Run from the repo root:
    python -m scripts.inspect_data
"""
import pickle
from pathlib import Path

import numpy as np
import yaml

try:
    import matplotlib
    matplotlib.use("Agg")  # headless-safe; we save a PNG instead of showing it
    import matplotlib.pyplot as plt
except ImportError:  # plotting is optional; the text report is the point
    plt = None

from cnn.dataset import count_occupied_cell_classes, effective_number_weights

WORKDIR = Path().absolute()
DATASET_PATH = WORKDIR / "dataset"
SAVE_DIR = WORKDIR / "training"
HISTORY_FILE = SAVE_DIR / "accidentclassifier_history.pkl"

CLASS_NAMES = ["No accident", "Minor", "Moderate", "Severe", "Totaled Vehicle"]


def _raw_line_counts(base_path, num_classes, splits=("train", "val")):
    """Count raw label lines (before one-object-per-cell dedup), for comparison."""
    counts = np.zeros(num_classes, dtype=np.int64)
    for split in splits:
        label_dir = Path(base_path) / "labels" / split
        if not label_dir.is_dir():
            continue
        for label_path in label_dir.glob("*.txt"):
            for line in label_path.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls = int(parts[0])
                if 0 <= cls < num_classes:
                    counts[cls] += 1
    return counts


def report_class_frequencies(params):
    S, num_classes = params["S"], params["num_classes"]
    print("=" * 70)
    print(" 1. CLASS FREQUENCIES  (feeds recommendations #1 / #2)")
    print("=" * 70)

    if not (DATASET_PATH / "labels").is_dir():
        print(f"  No labels found under {DATASET_PATH/'labels'} — restore dataset/ and re-run.")
        return

    occ = count_occupied_cell_classes(DATASET_PATH, S=S, num_classes=num_classes)
    raw = _raw_line_counts(DATASET_PATH, num_classes)

    if occ.sum() == 0:
        print("  Label dirs present but empty.")
        return

    weights = effective_number_weights(
        occ,
        beta=params.get("class_weight_beta", 0.999),
        max_ratio=params.get("class_weight_max_ratio", 10.0),
        class0_cap=params.get("no_accident_weight_cap"),
    )
    hand_picked = [4.0, 0.6, 1.0, 6.5, 0.6]  # the original vector the analysis flagged

    print(f"  {'Class':<16} {'raw':>7} {'occupied':>9} {'share':>7} {'eff-num w':>10} {'old w':>7}")
    print("  " + "-" * 62)
    total = occ.sum()
    for i, name in enumerate(CLASS_NAMES[:num_classes]):
        share = occ[i] / total
        print(f"  {name:<16} {raw[i]:>7} {occ[i]:>9} {share:>6.1%} {weights[i]:>10.3f} {hand_picked[i]:>7.1f}")
    print("\n  effective_number weights (paste-ready):")
    print(f"    class_w: {[round(w, 3) for w in weights]}")
    print("\n  Watch: classes whose new weight moves OPPOSITE the old hand-picked one")
    print("  are exactly the ones the analysis says were mis-weighted.")


def report_history():
    print("\n" + "=" * 70)
    print(" 2. TRAINING HISTORY  (feeds recommendations #4 AdamW / #5 schedule)")
    print("=" * 70)

    if not HISTORY_FILE.exists():
        print(f"  No history at {HISTORY_FILE} — train once, then re-run this script.")
        return

    with open(HISTORY_FILE, "rb") as fp:
        h = pickle.load(fp)

    val_total = h.get("validation_loss_total", [])
    if not val_total:
        print("  History present but has no validation_loss_total.")
        return

    n_epochs = len(val_total)
    best_epoch = int(np.argmin(val_total))

    # Cosine warm restarts (T_0=15, T_mult=2) restart at the start of epochs 15, 45, 105.
    restarts = []
    boundary, step = 15, 15
    while boundary < n_epochs + 1:
        restarts.append(boundary)
        step *= 2
        boundary += step
    last_restart = max([0] + [r for r in restarts if r <= n_epochs])
    in_downswing = best_epoch >= last_restart and (best_epoch == n_epochs - 1)

    print(f"  epochs run:            {n_epochs}  (max_epochs cap = 100)")
    print(f"  best val-total epoch:  {best_epoch}  (val_total = {val_total[best_epoch]:.5f})")
    print(f"  cosine restarts at:    {restarts or '[none reached]'}")
    print(f"  last restart <= end:   epoch {last_restart}")

    early_stopped = n_epochs < 100
    print(f"  early-stopped:         {early_stopped}")
    if early_stopped and best_epoch == n_epochs - 1:
        print("  ⚠ #5 SIGNAL: stop fired while val was still its own best (down-swing) —")
        print("    the schedule may be stranding training before the next warm restart.")

    # #4 signal: per-term classification train vs val gap over the last 10 epochs
    tr_cls = h.get("train_loss_classification", [])
    va_cls = h.get("validation_loss_classification", [])
    if tr_cls and va_cls:
        w = min(10, len(tr_cls))
        gap_now = np.mean(va_cls[-w:]) - np.mean(tr_cls[-w:])
        gap_start = va_cls[0] - tr_cls[0]
        print(f"  classification gap:    start={gap_start:+.4f}  last{w}={gap_now:+.4f}")
        if gap_now > gap_start + 1e-3:
            print("  ⚠ #4 SIGNAL: classification train/val gap WIDENED → overfitting →")
            print("    AdamW + weight_decay is justified.")
        else:
            print("  → classification gap not clearly widening; AdamW less urgent than data scarcity.")

    _plot_curves(h)


def _plot_curves(h):
    if plt is None:
        print("\n  (matplotlib not installed — skipping per-term curve plot)")
        return
    terms = ["total", "classification", "objectness", "regression"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, term in zip(axes.ravel(), terms):
        tr = h.get(f"train_loss_{term}", [])
        va = h.get(f"validation_loss_{term}", [])
        if tr:
            ax.plot(tr, label="train")
        if va:
            ax.plot(va, label="val")
        ax.set_title(term)
        ax.set_xlabel("epoch")
        ax.legend(loc="upper right")
    fig.suptitle("Per-term train vs. validation loss")
    fig.tight_layout()
    out = SAVE_DIR / "history_per_term.png"
    fig.savefig(out, dpi=110)
    print(f"\n  Saved per-term curves → {out}")


def main():
    with open("./parameters.yaml", "r") as yf:
        params = yaml.load(yf, Loader=yaml.SafeLoader)["accident_classifier"]

    report_class_frequencies(params)
    report_history()

    print("\n" + "=" * 70)
    print(" 3. CONFUSION MATRIX  (feeds recommendation #3)")
    print("=" * 70)
    print("  Run `python -m models.conv run` and read the off-diagonals:")
    print("    Minor/Moderate/Severe confused WITH EACH OTHER → ordinal/soft loss.")
    print("    Minor/Moderate confused WITH 'No accident'      → gating: cut class_w[0],")
    print("                                                       sweep conf_threshold.")


if __name__ == "__main__":
    main()
