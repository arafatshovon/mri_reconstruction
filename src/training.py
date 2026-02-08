from typing import Any, Dict, Iterable, Tuple

import fastmri
from fastmri.data import transforms
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.datasets.fastmri_dataset import Mri_Data
from src.metrics.psnr import psnr
from src.metrics.ssim import ssim
from src.models.varnet import VarNet


class VarNetWCNNDataModule(pl.LightningDataModule):
    def __init__(self, train_file, val_file, batch_size, transform, challenge):
        super().__init__()
        self.batch_size = batch_size
        self.train_file = train_file
        self.val_file = val_file
        self.transform = transform
        self.challenge = challenge

    def setup(self, stage=None):
        if stage == "fit":
            self.train_dataset = Mri_Data(
                self.train_file, self.transform, self.challenge
            )
            self.val_dataset = Mri_Data(self.val_file, self.transform, self.challenge)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2
        )


class VarNetWCNNLightningModule(pl.LightningModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.model = VarNet(
            num_cascades=config["num_cascades"],
            sens_chans=config["sens_chans"],
            sens_pools=config["sens_pools"],
            chans=config["chans"],
            pools=config["pools"],
        )
        self.loss = fastmri.SSIMLoss()
        self.config = config
        self.automatic_optimization = False
        self.acum_step = config["acum_step"]

        self.table = None
        if config.get("use_wandb", False):
            try:
                import wandb

                self.table = wandb.Table(
                    columns=["Ground_truth", "Prediction", "PSNR", "SSIM", "Run"]
                )
            except ModuleNotFoundError:
                self.table = None

    def forward(self, masked_kspace, mask):
        return self.model(masked_kspace, mask)

    def training_step(self, batch, batch_idx):
        masked_kspace, mask, target, *_rest = batch
        max_value = _rest[-2]
        if masked_kspace.dim() == 6:
            masked_kspace = masked_kspace.squeeze(0)
            mask = mask.squeeze(0)
            target = target.squeeze(0)
            max_value = max_value.squeeze(0)

        total_loss = 0
        optim = self.optimizers()
        for x, y, z, w in self.generator(
            masked_kspace, mask, target, max_value, self.acum_step
        ):
            output = self(x, y)
            z, output = transforms.center_crop_to_smallest(z, output)
            l1 = F.l1_loss(output.unsqueeze(1), z.unsqueeze(1))
            l2 = self.loss(output.unsqueeze(1), z.unsqueeze(1), data_range=w)
            total_loss += l1.item() + l2.item()
            self.manual_backward(l1 + l2)
            optim.step()
            optim.zero_grad()
        self.log(
            "train_loss",
            total_loss,
            on_step=True,
            on_epoch=True,
            batch_size=self.config["batch_size"],
        )

    def validation_step(self, batch, batch_idx):
        masked_kspace, mask, target, *_rest = batch
        max_value = _rest[-2]
        if masked_kspace.dim() == 6:
            masked_kspace = masked_kspace.squeeze(0)
            mask = mask.squeeze(0)
            target = target.squeeze(0)
            max_value = max_value.squeeze(0)

        psnr_total, ssim_total, val_loss, step = 0.0, 0.0, 0.0, 0
        for x, y, z, w in self.generator(
            masked_kspace, mask, target, max_value, target.shape[0]
        ):
            output = self(x, y)
            z, output = transforms.center_crop_to_smallest(z, output)
            l1 = F.l1_loss(output.unsqueeze(1), z.unsqueeze(1)).item()
            l2 = self.loss(output.unsqueeze(1), z.unsqueeze(1), data_range=w).item()
            val_loss += l1 + l2
            z, output = z.detach().cpu().numpy(), output.detach().cpu().numpy()
            psnr_total += psnr(z, output)
            ssim_total += ssim(z, output)
            step += 1

        metrics = {
            "val loss": val_loss,
            "psnr": psnr_total / step,
            "ssim": ssim_total / step,
        }
        self.log_dict(
            metrics,
            on_step=True,
            on_epoch=True,
            batch_size=self.config["batch_size"],
        )

    def on_train_epoch_end(self):
        lr_scheduler = self.lr_schedulers()
        lr_scheduler.step()
        avg_train_loss = self.trainer.callback_metrics["train_loss"]
        print(f"Train Loss: {avg_train_loss}")

    def on_validation_epoch_end(self):
        avg_val_loss = self.trainer.callback_metrics["val loss"]
        avg_psnr = self.trainer.callback_metrics["psnr"]
        avg_ssim = self.trainer.callback_metrics["ssim"]
        print(f"\nepoch: {self.current_epoch + 1}/{self.config['epochs']}")
        print(f"Val Loss: {avg_val_loss}")
        print(f"PSNR: {avg_psnr}")
        print(f"SSIM: {avg_ssim}")
        optimizer = self.optimizers()
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Current learning rate: {current_lr}")

    def configure_optimizers(self):
        optim = torch.optim.Adam(
            self.model.parameters(), lr=self.config["lr"], weight_decay=0.0
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optim, self.config["lr_step_size"], self.config["lr_gamma"]
        )
        return {"optimizer": optim, "lr_scheduler": scheduler}

    def update_table(self, target, output):
        if self.table is None:
            return
        import wandb
        from skimage.metrics import peak_signal_noise_ratio, structural_similarity

        for index in range(target.shape[0]):
            norm_target = self.normalize_image(target[index])
            norm_output = self.normalize_image(output[index])
            snr = peak_signal_noise_ratio(
                target[index],
                output[index],
                data_range=target[index].max() - target[index].min(),
            )
            sim = structural_similarity(
                target[index],
                output[index],
                data_range=target[index].max() - target[index].min(),
            )
            self.table.add_data(
                wandb.Image(norm_target),
                wandb.Image(norm_output),
                snr,
                sim,
                self.config.get("run", 0),
            )

    def generator(self, x, y, z, w, acum_step):
        length = x.shape[0]
        start = 0
        while start < length:
            if start + acum_step <= length:
                yield (
                    x[start : start + acum_step],
                    y[start : start + acum_step],
                    z[start : start + acum_step],
                    w[start : start + acum_step],
                )
            else:
                yield x[start:], y[start:], z[start:], w[start:]
                break
            start += acum_step

    def normalize_image(self, img):
        img = img.astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min())
        return img
