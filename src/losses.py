"""Loss functions for SGL training.

The official model is trained with plain binary cross-entropy on the sigmoid
segmentation output (``base_bce_loss`` in loss/bceloss.py). The SGL final model
sums two BCE terms -- one against the human label, one against the teacher
(pseudo) label:

    loss = BCE(pred, vessel_gt) + BCE(pred, teacher)
"""
import torch
import torch.nn as nn

EPS = 1e-14


class BCELoss(nn.Module):
    """Mean binary cross-entropy over all pixels (matches base_bce_loss)."""

    def forward(self, pred, target):
        bce = -target * torch.log(pred + EPS) - (1 - target) * torch.log(1 - pred + EPS)
        return bce.mean()


class SGLoss(nn.Module):
    """Study-Group dual-target loss: BCE against GT + BCE against teacher."""

    def __init__(self):
        super().__init__()
        self.bce = BCELoss()

    def forward(self, pred, vessel_gt, teacher):
        return self.bce(pred, vessel_gt) + self.bce(pred, teacher)
