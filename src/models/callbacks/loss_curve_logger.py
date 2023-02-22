import os

import pyarrow as pa
import pyarrow.parquet as pq
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
        filenames = outputs["filenames"]

        if pl_module.global_rank == 0:
            all = [
                torch.zeros(unreduced_losses.shape, device=pl_module.device)
            ] * trainer.num_devices
            torch.distributed.gather(unreduced_losses, all)

            all_filenames = [None] * trainer.num_devices
            torch.distributed.gather_object(filenames, all_filenames)

            all_losses = []

            for losses, fns in zip(all, all_filenames):
                all_losses.append((batch_idx, losses.detach(), fns))

            self.loss_curves += all_losses
        else:
            torch.distributed.gather(unreduced_losses)
            torch.distributed.gather_object(filenames)

    def get_path(self, version: int) -> str:
        return os.path.join(self.dir, f"losses_v{version}.pt")

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if pl_module.global_rank == 0:
            os.makedirs(self.dir, exist_ok=True)
            path = self.get_path(pl_module.current_epoch)

            epoch_losses = [
                (batch_idx, lc.tolist(), filenames)
                for batch_idx, lc, filenames in self.loss_curves
            ]

            indices = []
            losses = []
            filenames = []
            for batch_idx, batch_loss, batch_filenames in self.loss_curves:
                indices.extend([batch_idx]*len(batch_filenames))
                losses.extend(batch_loss.tolist())
                filenames.extend(batch_filenames)

            pa_indices = pa.array(indices, type=pa.uint32())
            pa_losses = pa.array(losses, type=pa.float16())
            pa_filenames = pa.array(filenames)
            pa_epochs = pa.array([pl_module.current_epoch]*len(losses), type=pa.uint8())

            pa_table = pa.table(
                [pa_indices, pa_losses, pa_filenames, pa_epochs],
                names=["batch_idx", "loss", "filename", "epoch"]
            )
            pq.write_to_dataset(
                pa_table,
                self.dir,
                partition_cols=["epoch"],
                use_legacy_dataset=False
            )

            self.loss_curves = []

            if isinstance(pl_module.logger, WandbLogger):
                self.log_artifact(pl_module.logger.experiment, pl_module.current_epoch)

    def log_artifact(self, experiment, epoch: int) -> None:
        artifact = wandb.Artifact(
                f"loss_curves-epoch-{epoch}", type="loss_curves"
        )
        artifact.add_dir(self.dir)

        experiment.log_artifact(artifact, "loss_curves")
