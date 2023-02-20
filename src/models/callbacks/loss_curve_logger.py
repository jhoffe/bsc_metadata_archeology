import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers.wandb import WandbLogger


class LossCurveLogger(Callback):
    def setup(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str
    ) -> None:
        pl_module.loss_curves = []

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

            pl_module.loss_curves += all_losses
        else:
            torch.distributed.gather(unreduced_losses)

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if isinstance(pl_module.logger, WandbLogger) and pl_module.global_rank == 0:
            path = f"models/losses_v{pl_module.current_epoch}.pt"
            torch.save(pl_module.loss_curves, path)
            artifact = wandb.Artifact(
                "losses",
                type="loss_curves",
                metadata={"epoch": pl_module.current_epoch},
            )
            artifact.add_file(path)

            pl_module.logger.experiment.log_artifact(artifact, "losses")
