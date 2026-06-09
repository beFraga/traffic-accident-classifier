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
    def __init__(self, num_classes=5):
        self.num_classes = num_classes
        self.key_names = ['total', 'objectness', 'classification', 'regression']
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([7.0]))
        self.ce = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.mse = nn.MSELoss()

        self.class_w = torch.tensor([4.0, 0.6, 1.0, 6.5, 0.6])

    def __call__(self, pred, true):
        # 1. Objectness Localization Loss
        self.bce.pos_weight = self.bce.pos_weight.to(pred.device)
        loss_obj = self.bce(pred[..., 0], true[..., 0])
        
        # Create mask filtering grid locations where accidents actually occur
        mask = (true[..., 0] == 1.0)
        loss_cls = torch.tensor(0.0, device=pred.device)
        loss_box = torch.tensor(0.0, device=pred.device)
        
        if mask.sum() > 0:
            # 2. Categorical Classification Loss
            pred_classes = pred[mask][..., 1:1 + self.num_classes]
            true_classes = torch.argmax(true[mask][..., 1:1 + self.num_classes], dim=-1)
            self.ce.weight = self.class_w.to(pred.device)
            loss_cls = self.ce(pred_classes, true_classes)
            
            # 3. Bounding Box Coordinate Bounding Loss
            pred_boxes = pred[mask][..., 1 + self.num_classes:]
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