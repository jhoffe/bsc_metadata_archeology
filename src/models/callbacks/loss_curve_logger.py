import os
from time import gmtime, strftime

import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers.wandb import WandbLogger


class LossCurveLogger(Callback):
    def __init__(self, dir: str, wandb_suffix: str) -> None:
        super().__init__()
        self.dir = dir
        self.wandb_suffix = wandb_suffix
        self.loss_curves = []

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        unreduced_losses = outputs["unreduced_loss"]

        if pl_module.global_rank == 0:
            all = [
                torch.zeros(unreduced_losses.shape, device=pl_module.device)
            ] * trainer.num_devices
            torch.distributed.gather(unreduced_losses, all)

            all_losses = []

            for losses in all:
                all_losses.append((batch_idx, losses.detach()))

            self.loss_curves += all_losses
        else:
            torch.distributed.gather(unreduced_losses)

    def get_path(self, version: int) -> str:
        return os.path.join(self.dir, f"losses_v{version}.pt")

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if isinstance(pl_module.logger, WandbLogger) and pl_module.global_rank == 0:
            epoch_losses = {}

            if pl_module.current_epoch > 0:
                epoch_losses = torch.load(self.get_path(pl_module.current_epoch - 1))

            os.makedirs(self.dir, exist_ok=True)
            path = self.get_path(pl_module.current_epoch)

            epoch_losses[pl_module.current_epoch] = [(batch_idx, lc.tolist()) for batch_idx, lc in self.loss_curves]
            torch.save(epoch_losses, path)
            artifact = wandb.Artifact(
                f"losses-{self.wandb_suffix}",
                type="loss_curves",
                metadata={"epoch": pl_module.current_epoch},
            )
            artifact.add_file(path)

            pl_module.logger.experiment.log_artifact(artifact, "loss_curves")
