import torch
from torch.nn import functional as F


class LossCurve:
    @staticmethod
    def create(
        loss: torch.Tensor, indices: torch.Tensor, y: torch.Tensor, logits: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        y_hat = logits.argmax(dim=1)
        softmax_confidence = F.softmax(logits, dim=1)[range(len(y)), y]

        return {
            "loss": loss.mean(),
            "unreduced_loss": loss,
            "indices": indices,
            "y": y,
            "y_hat": y_hat,
            "softmax_confidence": softmax_confidence,
        }
