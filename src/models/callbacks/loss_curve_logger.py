import os

import lightning as L
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from lightning.pytorch.strategies import SingleDeviceStrategy


class LossCurveLogger(L.Callback):
    def __init__(self, dir: str, wandb_suffix: str) -> None:
        super().__init__()
        self.dir = dir
        self.wandb_suffix = wandb_suffix
        self.loss_curves = []

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        unreduced_losses = outputs["unreduced_loss"]
        indices = outputs["indices"]
        y = outputs["y"]
        y_hat = outputs["y_hat"]

        self.loss_curves.append(
            (
                batch_idx,
                unreduced_losses.detach(),
                indices.detach(),
                y.detach(),
                y_hat.detach(),
            )
        )

    def get_path(self, version: int) -> str:
        return os.path.join(self.dir, f"losses_v{version}.pt")

    def on_train_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        batch_ids = []
        indices = torch.cat(
            [indices for _, _, indices, _, _ in self.loss_curves], 0
        ).to(pl_module.dtype)
        y = torch.cat([y for _, _, _, y, _ in self.loss_curves], 0).to(pl_module.dtype)
        y_hat = torch.cat([y_hat for _, _, _, _, y_hat in self.loss_curves], 0).to(
            pl_module.dtype
        )

        for batch_idx, _, fns, _, _ in self.loss_curves:
            batch_ids.extend([batch_idx] * len(fns))

        losses = torch.cat([lc[1] for lc in self.loss_curves], 0)

        if not isinstance(trainer.strategy, SingleDeviceStrategy):
            if pl_module.global_rank == 0:
                device_batch_ids = [None] * trainer.num_devices
                device_losses = [
                    torch.zeros(losses.shape, device=pl_module.device)
                ] * trainer.num_devices
                device_indices = [
                    torch.zeros(
                        losses.shape, device=pl_module.device, dtype=pl_module.dtype
                    )
                ] * trainer.num_devices
                device_y = [
                    torch.zeros(
                        losses.shape, device=pl_module.device, dtype=pl_module.dtype
                    )
                ] * trainer.num_devices
                device_y_hat = [
                    torch.zeros(
                        losses.shape, device=pl_module.device, dtype=pl_module.dtype
                    )
                ] * trainer.num_devices

                torch.distributed.gather_object(batch_ids, device_batch_ids)
                torch.distributed.gather(losses, device_losses)
                torch.distributed.gather(indices, device_indices)
                torch.distributed.gather(y, device_y)
                torch.distributed.gather(y_hat, device_y_hat)

                batch_ids = []
                indices = torch.cat(device_indices, 0)
                losses = torch.cat(device_losses, 0)
                y = torch.cat(device_y, 0)
                y_hat = torch.cat(device_y_hat, 0)

                for i in range(trainer.num_devices):
                    batch_ids.extend(device_batch_ids[i])
            else:
                torch.distributed.gather_object(batch_ids)
                torch.distributed.gather(losses)
                torch.distributed.gather(indices)
                torch.distributed.gather(y)
                torch.distributed.gather(y_hat)

                self.loss_curves = []
                return

        os.makedirs(self.dir, exist_ok=True)

        pa_batch_indices = pa.array(batch_ids, type=pa.uint16())
        pa_losses = pa.array(losses.cpu().numpy())
        pa_sample_indices = pa.array(indices.cpu().numpy(), type=pa.uint32())
        pa_y = pa.array(y.cpu().numpy(), type=pa.uint16())
        pa_y_hat = pa.array(y_hat.cpu().numpy(), type=pa.uint16())
        pa_epochs = pa.array([pl_module.current_epoch] * len(losses), type=pa.uint16())

        pa_table = pa.table(
            [pa_batch_indices, pa_losses, pa_sample_indices, pa_y, pa_y_hat, pa_epochs],
            names=["batch_idx", "loss", "sample_index", "y", "y_hat", "epoch"],
        )
        pq.write_to_dataset(
            pa_table, self.dir, partition_cols=["epoch"], use_legacy_dataset=False
        )

        self.loss_curves = []
