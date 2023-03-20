from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass
class LossCurve:
    unreduced_loss: torch.Tensor
    indices: torch.Tensor
    y: torch.Tensor
    y_hat: torch.Tensor
    softmax_confidence: torch.Tensor

    @staticmethod
    def create(
        loss: torch.Tensor, indices: torch.Tensor, y: torch.Tensor, logits: torch.Tensor
    ):
        y_hat = logits.argmax(dim=1)
        softmax_confidence = F.softmax(logits, dim=1)[range(len(y)), y]

        return LossCurve(loss, indices, y, y_hat, softmax_confidence)

    def to_dict(self):
        return {
            "unreduced_loss": self.unreduced_loss,
            "loss": self.unreduced_loss.mean(),
            "indices": self.indices,
            "y": self.y,
            "y_hat": self.y_hat,
            "softmax_confidence": self.softmax_confidence,
        }

    def __getitem__(self, item):
        return self.to_dict()[item]
