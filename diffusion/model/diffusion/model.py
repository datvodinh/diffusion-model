import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
import diffusion
import wandb
from torchvision.utils import make_grid
from torch.optim.lr_scheduler import OneCycleLR


class DiffusionModel(pl.LightningModule):
    def __init__(
        self,
        lr: float = 1e-4,
        max_timesteps: int = 1000,
        beta_1: float = 0.0001,
        beta_2: float = 0.02,
        in_channels: int = 3,
        dim: int = 32,
        num_classes: int | None = 10,
        sample_per_epochs: int = 50
    ):
        super().__init__()
        self.model = diffusion.ConditionalUNet(
            c_in=in_channels,
            c_out=in_channels,
            num_classes=num_classes
        )
        self.lr = lr
        self.max_timesteps = max_timesteps
        self.in_channels = in_channels
        self.dim = dim
        self.num_classes = num_classes

        self.scheduler = diffusion.LinearScheduler(
            max_timesteps, beta_1, beta_2
        )

        self.criterion = nn.MSELoss()

        self.spe = sample_per_epochs
        self.epoch_count = 0
        self.train_loss = []
        self.val_loss = []

    def _batch_index_select(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        device: torch.device
    ):
        # x.shape = [T,]
        # t.shape = [B,]
        if x.device != device:
            x = x.to(device)
        if t.device != device:
            t = t.to(device)
        x_select = x.gather(dim=-1, index=t)
        return x_select[:, None, None, None]  # [B,1]

    def noising(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor
    ):
        noise = torch.randn_like(x_0, device=x_0.device)
        new_x = self.scheduler.get('sqrt_alpha_hat', t) * x_0
        new_noise = self.scheduler.get('sqrt_one_minus_alpha_hat', t) * noise
        return new_x + new_noise, noise

    def sampling(self, labels=None, n_samples: int = 16, demo: bool = False):
        return diffusion.ddpm_sampling(
            model=self.model,
            scheduler=self.scheduler,
            n_samples=n_samples,
            max_timesteps=self.max_timesteps,
            in_channels=self.in_channels,
            dim=self.dim,
            demo=demo,
            labels=labels
        )

    def forward(self, x_0, labels):
        t = torch.randint(
            low=0, high=self.max_timesteps, size=(x_0.shape[0],), device=x_0.device
        )
        x_noise, noise = self.noising(x_0, t)
        noise_pred = self.model(x_noise, t, labels)
        return noise, noise_pred

    def training_step(self, batch, idx):
        if isinstance(batch, torch.Tensor):
            x_0 = batch
            labels = None
        else:
            x_0, labels = batch
        if np.random.random() < 0.1:
            labels = None
        noise, noise_pred = self(x_0, labels)
        loss = self.criterion(noise, noise_pred)
        self.train_loss.append(loss)
        return loss

    def validation_step(self, batch, idx):
        if isinstance(batch, torch.Tensor):
            x_0 = batch
            labels = None
        else:
            x_0, labels = batch
        noise, noise_pred = self(x_0, labels)
        loss = self.criterion(noise, noise_pred)
        self.val_loss.append(loss)
        return loss

    def on_train_epoch_end(self) -> None:
        self.log_dict(
            {
                "train_loss": sum(self.train_loss) / len(self.train_loss)
            },
            sync_dist=True
        )
        self.train_loss.clear()

        if self.epoch_count % self.spe == 0:
            wandblog = self.logger.experiment
            x_t = self.sampling(n_samples=16, demo=False)
            img_array = [x_t[i] for i in range(x_t.shape[0])]

            wandblog.log(
                {
                    "sampling": wandb.Image(
                        make_grid(img_array, nrow=4).permute(1, 2, 0).cpu().numpy(),
                        caption="Sampled Image!"
                    )
                }
            )

        self.epoch_count += 1

    def on_validation_epoch_end(self):
        self.log_dict(
            {
                "val_loss": sum(self.val_loss) / len(self.val_loss)
            },
            sync_dist=True
        )
        self.val_loss.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            params=self.parameters(),
            lr=self.lr,
            weight_decay=0.001,
            betas=(0.9, 0.999)
        )
        scheduler = OneCycleLR(
            optimizer=optimizer,
            max_lr=self.lr,
            total_steps=self.trainer.estimated_stepping_batches,

        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler
        }


if __name__ == "__main__":
    a = torch.randn(32, 3, 32, 32)
    model = DiffusionModel(max_timesteps=10)
    n, n_pred = model(a)
    print(n.shape, n_pred.shape)
    print(torch.mean((n-n_pred)**2))
    print(model.sampling(1))
