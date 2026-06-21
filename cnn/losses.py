import torch
import torch.nn.functional as F
import torch.nn as nn

class BaseLoss(object):
    key_names = None
    def __init__(self):
        if self.key_names == None:
            raise NotImplementedError("Losses subclasses must implement `key_names` attribute")

        if 'total' not in self.key_names:
            raise NotImplementedError("The key `total` must be present for backdrop")


class AccidentDetectionLoss:
    def __init__(self, num_classes=5, class_w=None, neighbor_smoothing=0.0):
        self.num_classes = num_classes
        self.key_names = ['total', 'objectness', 'classification', 'regression']
        # pos_weight should approximate (empty cells / occupied cells). On a 7x7=49
        # grid with few accidents per image that ratio is ~40, not 7; too low makes
        # the model lazy about detecting accidents. Tune to your dataset's density.
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([40.0]))
        self.mse = nn.MSELoss()

        # Classification uses ordinal-neighbor soft labels instead of plain CE
        # (docs/round-3-adamw-and-ordinal-loss/results.txt #2). The severity classes are ordered, and
        # 67-80% of mid-class errors fall on adjacent severities, so we replace the
        # old uniform label_smoothing=0.1 with smoothing that places (1 - s) on the
        # true severity and splits s across the immediate ordinal neighbours.
        self.neighbor_smoothing = float(neighbor_smoothing)

        # Per-class CE weights. These are NOT hand-picked here: they are derived from
        # real label frequencies and passed in by models/conv.py (see
        # docs/round-2-class-weights/analysis.md #1). Falls back to uniform when
        # nothing is supplied so the loss is still usable in isolation.
        if class_w is None:
            class_w = [1.0] * num_classes
        self.class_w = torch.tensor(class_w, dtype=torch.float32)

    def _ordinal_soft_ce(self, logits, target):
        """Class-weighted cross-entropy against ordinal-neighbour soft labels.

        Builds a soft target with (1 - s) mass on the true class and s split as
        s/2 to each existing immediate neighbour (an edge class keeps the half it
        cannot give away). Weighting is by the true class and normalised by the
        weight sum, matching nn.CrossEntropyLoss(weight=..., reduction='mean') so
        the magnitude stays comparable to the round-2 weighted CE.
        """
        M, C = logits.shape
        s = self.neighbor_smoothing
        device = logits.device

        soft = torch.zeros(M, C, device=device)
        rows = torch.arange(M, device=device)
        soft[rows, target] = 1.0
        if s > 0:
            half = s / 2.0
            lmask = target > 0
            soft[rows[lmask], target[lmask] - 1] += half
            soft[rows[lmask], target[lmask]] -= half
            rmask = target < (C - 1)
            soft[rows[rmask], target[rmask] + 1] += half
            soft[rows[rmask], target[rmask]] -= half

        log_probs = F.log_softmax(logits, dim=-1)
        loss_per = -(soft * log_probs).sum(dim=-1)
        w = self.class_w.to(device)[target]
        return (w * loss_per).sum() / w.sum().clamp_min(1e-8)

    def __call__(self, pred, true):
        # 1. Objectness Localization Loss
        self.bce.pos_weight = self.bce.pos_weight.to(pred.device)
        loss_obj = self.bce(pred[..., 0], true[..., 0])
        
        # Create mask filtering grid locations where accidents actually occur
        mask = (true[..., 0] == 1.0)
        loss_cls = torch.tensor(0.0, device=pred.device)
        loss_box = torch.tensor(0.0, device=pred.device)
        
        if mask.sum() > 0:
            # 2. Categorical Classification Loss (ordinal-neighbour soft labels)
            pred_classes = pred[mask][..., 1:1 + self.num_classes]
            true_classes = torch.argmax(true[mask][..., 1:1 + self.num_classes], dim=-1)
            loss_cls = self._ordinal_soft_ce(pred_classes, true_classes)
            
            # 3. Bounding Box Coordinate Bounding Loss
            # Squash raw box logits to [0, 1] (targets are normalized coords); without
            # this the unbounded predictions produce large, noisy MSE gradients.
            pred_boxes = torch.sigmoid(pred[mask][..., 1 + self.num_classes:])
            true_boxes = true[mask][..., 1 + self.num_classes:]
            loss_box = self.mse(pred_boxes, true_boxes)
            
        # Total scaled multi-task loss equation step
        total_loss = loss_obj + (2.0 * loss_cls) + (1.5 * loss_box)
        
        return {
            'total': total_loss, 
            'objectness': loss_obj, 
            'classification': loss_cls, 
            'regression': loss_box
        }