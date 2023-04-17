from typing import Dict

import torch


class LossCurve:
    @staticmethod
    def create(
        loss: torch.Tensor, indices: torch.Tensor, y: torch.Tensor, logits: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        y_hat = logits.argmax(dim=1)

        return {
            "loss": loss.mean(),
            "unreduced_loss": loss,
            "indices": indices,
            "y": y,
            "y_hat": y_hat,
        }
