from typing import Dict

import torch


class ProxyOutput:
    @staticmethod
    def create(
        loss: torch.Tensor, indices: torch.Tensor, y: torch.Tensor, logits: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        return {"loss": loss, "indices": indices, "y": y, "logits": logits}
