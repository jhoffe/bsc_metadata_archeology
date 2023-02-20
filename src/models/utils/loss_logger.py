import torch
import os

class LossCurveLogger:
    losses: dict[int, torch.Tensor]
    save_path: str

    def __init__(self, save_path: str):
        self.losses = {}
        self.save_path = save_path

    def log(self, idx: int, loss: torch.Tensor) -> "LossCurveLogger":
        self.losses[idx] = loss.detach()

        return self

    def flush(self) -> "LossCurveLogger":
        self.losses = {}

        return self

    def save(self) -> "LossCurveLogger":
        losses = []

        if os.path.exists(self.save_path):
            losses = torch.load(self.save_path)

        losses.append(self.losses)

        torch.save(losses, self.save_path)

        return self