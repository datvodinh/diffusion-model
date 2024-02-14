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
        self.beta = self._linear_scheduler(max_timesteps, beta_1, beta_2)
        self.sqrt_beta = torch.sqrt(self.beta)
        self.alpha = 1 - self.beta
        self.sqrt_alpha = torch.sqrt(self.alpha)
        self.alpha_hat = torch.cumprod(1 - self.beta, dim=0)
        self.sqrt_alpha_hat = torch.sqrt(self.alpha_hat)
        self.sqrt_1_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat)

        self.criterion = nn.MSELoss()

        self.spe = sample_per_epochs
        self.epoch_count = 0
        self.train_loss = []
        self.val_loss = []

    def _linear_scheduler(
        self,
        max_timesteps: int,
        beta_1: float = 0.0001,
        beta_2: float = 0.02
    ):
        return torch.linspace(beta_1, beta_2, max_timesteps)

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
        new_x = self._batch_index_select(self.sqrt_alpha_hat, t, device=x_0.device) * x_0
        new_noise = self._batch_index_select(self.sqrt_1_minus_alpha_hat, t, device=x_0.device) * noise
        return new_x + new_noise, noise

    @torch.no_grad()
    def sampling(
        self,
        n: int,
        labels: torch.Tensor,
        cfg_scale: int = 3
    ):
        x_t = torch.randn(
            n, self.in_channels, self.dim, self.dim, device=self.model.device
        )
        self.model.eval()
        for t in range(self.max_timesteps-1, -1, -1):
            time = torch.full((n,), fill_value=t, device=self.model.device)
            pred_noise = self.model(x_t, time, labels)
            if cfg_scale > 0:
                uncond_pred_noise = self.model(x_t, time)
                pred_noise = torch.lerp(uncond_pred_noise, pred_noise, cfg_scale)
            sqrt_alpha = self._batch_index_select(
                self.sqrt_alpha, time, device=self.model.device
            )
            sqrt_one_minus_alpha_hat = self._batch_index_select(
                self.sqrt_1_minus_alpha_hat, time, device=self.model.device)
            sqrt_beta = self._batch_index_select(
                self.sqrt_beta, time, device=self.model.device
            )
            noise = torch.randn_like(x_t, device=self.model.device) if (
                t > 0
            ) else torch.zeros_like(x_t, device=self.model.device)

            x_t = 1 / sqrt_alpha * (
                x_t - (1-sqrt_alpha) / sqrt_one_minus_alpha_hat * pred_noise
            ) + sqrt_beta * noise

        x_t = (x_t.clamp(-1, 1) + 1) / 2 * 255.  # range [0,255]
        self.model.train()
        return x_t.type(torch.uint8)

    def forward(self, x_0, labels):
        t = torch.randint(
            low=0, high=self.max_timesteps, size=(x_0.shape[0],), device=x_0.device
        )
        x_noise, noise = self.noising(x_0, t)
        noise_pred = self.model(x_noise, t, labels)
        return noise, noise_pred

    def training_step(self, batch, idx):
        x_0, labels = batch
        if np.random.random() < 0.1:
            labels = None
        noise, noise_pred = self(x_0, labels)
        loss = self.criterion(noise, noise_pred)
        self.train_loss.append(loss)
        return loss

    def validation_step(self, batch, idx):
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
        self.epoch_count += 1

        if self.epoch_count % self.spe == 0:
            n = 32
            labels = torch.randint(
                low=0, high=self.num_classes, size=(n,), device=self.model.device
            )
            x_t = self.sampling(n, labels).cpu()

            img_array = [x_t[i] for i in range(x_t.shape[0])]
            try:
                wandblog = self.logger.experiment
                wandblog.log(
                    {
                        "sampling": wandb.Image(
                            make_grid(img_array).permute(1, 2, 0).numpy(),
                            caption="Sampled Image!"
                        )
                    }
                )
            except:
                pass

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
            weight_decay=0.1,
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
