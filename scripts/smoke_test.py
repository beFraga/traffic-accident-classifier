"""
End-to-end smoke test for the recent training fixes.

It does NOT need the real dataset: it generates a tiny synthetic one in the
expected `images/{train,val}` + `labels/{train,val}` layout, then exercises the
real dataset/network/loss/model classes and asserts each fix behaves at runtime.

Run from the repo root:
    python -m scripts.smoke_test
"""
import os
import tempfile
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from cnn.dataset import (
    TrafficGridDataset,
    TrafficDataManager,
    count_occupied_cell_classes,
    effective_number_weights,
)
from cnn.network import TransferResnet
from cnn.losses import AccidentDetectionLoss
from cnn.model import AccidentClassifier

S, NUM_CLASSES = 7, 5


def make_fake_dataset(base, n_train=8, n_val=6):
    """Write random images + YOLO label files. Some labels have 2 objects in
    DIFFERENT grid cells to verify the `break` fix keeps more than one object."""
    for split, n in (("train", n_train), ("val", n_val)):
        img_dir = os.path.join(base, "images", split)
        lbl_dir = os.path.join(base, "labels", split)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)
        for i in range(n):
            arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            Image.fromarray(arr).save(os.path.join(img_dir, f"img{i}.png"))
            with open(os.path.join(lbl_dir, f"img{i}.txt"), "w") as f:
                # one object near top-left (cell 0,0)
                f.write(f"{i % NUM_CLASSES} 0.1 0.1 0.2 0.2\n")
                # a second object near bottom-right (cell 6,6) -> different cell
                f.write(f"{(i + 1) % NUM_CLASSES} 0.9 0.9 0.2 0.2\n")


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    with tempfile.TemporaryDirectory() as tmp:
        base = os.path.join(tmp, "dataset")
        make_fake_dataset(base)

        # --- Fix: removed `break` -> multiple objects per image are kept ---
        train_ds = TrafficGridDataset(base, split="train", S=S, num_classes=NUM_CLASSES)
        _, target = train_ds[0]
        n_objects = int(target[..., 0].sum().item())
        assert n_objects == 2, f"expected 2 objects kept, got {n_objects} (break not removed?)"
        print(f"[ok] label loop keeps multiple objects per image (objectness cells = {n_objects})")

        # --- Fix: augmentation on train only, not val ---
        def has(transform, cls):
            return any(isinstance(t, cls) for t in transform.transforms)
        val_ds = TrafficGridDataset(base, split="val", S=S, num_classes=NUM_CLASSES)
        assert has(train_ds.transform, T.ColorJitter), "train split missing augmentation"
        assert not has(val_ds.transform, T.ColorJitter), "val split should NOT be augmented"
        print("[ok] photometric augmentation applied to train split only")

        # --- Fix: early ResNet layers frozen ---
        net = TransferResnet(num_classes=NUM_CLASSES, S=S)
        frozen = sum(p.numel() for p in net.backbone.parameters() if not p.requires_grad)
        trainable = sum(p.numel() for p in net.backbone.parameters() if p.requires_grad)
        assert frozen > 0, "no frozen backbone params"
        assert trainable > 0, "entire backbone frozen (later layers should train)"
        print(f"[ok] ResNet backbone partially frozen (frozen={frozen:,}, trainable={trainable:,})")

        # --- Fix: bbox sigmoid + higher pos_weight -> loss is finite ---
        loss_fn = AccidentDetectionLoss(num_classes=NUM_CLASSES)
        assert float(loss_fn.bce.pos_weight.item()) == 40.0, "pos_weight not updated"
        x = torch.randn(2, 3, 224, 224)
        pred = net(x)
        true = torch.zeros(2, S, S, 1 + NUM_CLASSES + 4)
        true[:, 0, 0, 0] = 1.0
        true[:, 0, 0, 1] = 1.0
        true[:, 0, 0, 6:10] = torch.tensor([0.1, 0.1, 0.2, 0.2])
        losses = loss_fn(pred, true)
        for k, v in losses.items():
            assert torch.isfinite(v), f"loss term {k} is not finite"
        print(f"[ok] forward + loss finite (total={losses['total'].item():.4f})")

        # --- Phase 1: frequency-derived class weights match the loss's occupied-cell view ---
        counts = count_occupied_cell_classes(base, S=S, num_classes=NUM_CLASSES)
        # 8 train + 6 val images, each with 2 objects in distinct cells -> 28 occupied cells
        assert int(counts.sum()) == 28, f"expected 28 occupied cells, counted {counts.sum()}"
        weights = effective_number_weights(counts, beta=0.999, max_ratio=10.0, class0_cap=1.0)
        assert len(weights) == NUM_CLASSES, "weight vector wrong length"
        assert all(np.isfinite(weights)) and all(w > 0 for w in weights), "weights must be finite & positive"
        assert weights[0] <= 1.0 + 1e-9, f"class0 cap not applied (got {weights[0]})"
        assert max(weights) <= min(weights) * 10.0 + 1e-6, "max_ratio cap not enforced"
        # weights flow into the loss verbatim
        wloss = AccidentDetectionLoss(num_classes=NUM_CLASSES, class_w=weights)
        assert torch.allclose(wloss.class_w, torch.tensor(weights, dtype=torch.float32)), "loss didn't take class_w"
        print(f"[ok] class weights derived from occupied-cell counts (counts={counts.tolist()}, weights={[round(w,3) for w in weights]})")

        # --- Fix: single LR scheduler, no LR collapse over a short run ---
        params = {
            "learning_rate": 1e-4, "batch_size": 4, "max_epochs": 2,
            "patience_epochs": 5, "lr_decay_every_n_epoch": 15, "lr_decay_rate": 0.5,
            "num_classes": NUM_CLASSES, "S": S, "class_w": weights,
        }
        from pathlib import Path
        dm = TrafficDataManager(base, batch_size=4, S=S, num_classes=NUM_CLASSES)
        model = AccidentClassifier(Path(tmp) / "training", dm, params,
                                   device=torch.device("cpu"), early_stopping=True, tp=1)
        assert len(model.schedulers) == 1, f"expected 1 scheduler, got {len(model.schedulers)}"
        assert torch.allclose(model.loss.class_w, torch.tensor(weights, dtype=torch.float32)), \
            "params['class_w'] did not reach the model's loss"
        lr_before = model.optimizer.param_groups[0]["lr"]
        model.train()
        lr_after = model.optimizer.param_groups[0]["lr"]
        assert lr_after > 1e-6, f"LR collapsed to {lr_after}"
        print(f"[ok] single scheduler, no LR collapse (lr {lr_before:.2e} -> {lr_after:.2e})")
        assert len(model.history.get("train_loss_total", [])) == 2, "training did not run 2 epochs"
        print(f"[ok] trained 2 epochs (train_loss_total = {model.history['train_loss_total']})")

    print("\nAll smoke-test checks passed.")


if __name__ == "__main__":
    main()
