import torch
import pytorch_lightning as pl
import diffusion
import matplotlib.pyplot as plt


class DiffusionModel(pl.LightningModule):
    def __init__(
        self,
        lr: float = 1e-4,
        max_timesteps: int = 1000,
        beta_1: float = 0.0001,
        beta_2: float = 0.02,
        in_channels: int = 3,
        dim: int = 32
    ):
        super().__init__()
        self.model = diffusion.UNet(
            dim=dim, in_channels=in_channels
        )
        self.lr = lr
        self.max_timesteps = max_timesteps
        self.in_channels = in_channels
        self.dim = dim
        self.beta = self._linear_scheduler(max_timesteps, beta_1, beta_2)
        self.sqrt_beta = torch.sqrt(self.beta)
        self.alpha = 1 - self.beta
        self.sqrt_alpha = torch.sqrt(self.alpha)
        self.alpha_hat = torch.cumprod(1 - self.beta, dim=0)
        self.sqrt_alpha_hat = torch.sqrt(self.alpha_hat)
        self.sqrt_1_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat)

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
        if self.device is None:
            self.device = x_0.device
        x_0 = x_0 * 2 - 1  # range [-1,1]
        noise = torch.randn_like(x_0, device=x_0.device)
        mean = self._batch_index_select(self.sqrt_alpha_hat, t, device=x_0.device) * x_0
        std = self._batch_index_select(self.sqrt_1_minus_alpha_hat, t, device=x_0.device) * noise
        return torch.clamp(mean + std, min=-1, max=1), noise

    @torch.no_grad()
    def sampling(self, n: int):
        print(f"Sampling {n} images!")
        x_t = torch.randn(n, self.in_channels, self.dim, self.dim, device=self.model.device)
        self.model.eval()
        for t in range(self.max_timesteps-1, -1, -1):
            time = torch.full((n,), fill_value=t, device=self.model.device)
            pred_noise = self.model(x_t, time)
            sqrt_alpha = self._batch_index_select(self.sqrt_alpha, time, device=self.model.device)
            sqrt_one_minus_alpha_hat = self._batch_index_select(
                self.sqrt_1_minus_alpha_hat, time, device=self.model.device)
            sqrt_beta = self._batch_index_select(self.sqrt_beta, time, device=self.model.device)
            noise = torch.rand_like(x_t, device=self.model.device) if (
                t > 0
            ) else torch.zeros_like(x_t, device=self.model.device)

            x_t = 1 / sqrt_alpha * (
                x_t - (1-sqrt_alpha) / sqrt_one_minus_alpha_hat * pred_noise
            ) + sqrt_beta * noise

        x_t = (x_t.clamp(-1, 1) + 1) / 2  # range [0,1]
        self.model.train()
        return x_t

    def on_train_epoch_end(self) -> None:
        x_t = self.sampling(16).cpu()
        plt.figure(figsize=(12, 12))
        for i in range(16):
            plt.subplot(4, 4, i+1)
            plt.imshow(x_t[i].permute(1, 2, 0))
            plt.axis('off')

        plt.show()

    def forward(self, x_0):
        t = torch.randint(
            low=0, high=self.max_timesteps, size=(x_0.shape[0],), device=x_0.device
        )
        x_noise, noise = self.noising(x_0, t)
        noise_pred = self.model(x_noise, t)
        return noise, noise_pred

    def training_step(self, batch, idx):
        x_0, _ = batch
        noise, noise_pred = self(x_0)
        loss = torch.mean((noise - noise_pred)**2)
        self.log_dict(
            {
                "train_loss": loss
            },
            sync_dist=True
        )
        return loss

    def validation_step(self, batch, idx):
        x_0, _ = batch
        noise, noise_pred = self(x_0)
        loss = torch.mean((noise - noise_pred)**2)
        self.log_dict(
            {
                "val_loss": loss
            },
            sync_dist=True
        )
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            params=self.parameters(),
            lr=self.lr,
            betas=(0.9, 0.999)
        )
        return {
            'optimizer': optimizer
        }


if __name__ == "__main__":
    a = torch.randn(32, 3, 32, 32)
    model = DiffusionModel(max_timesteps=10)
    n, n_pred = model(a)
    print(n.shape, n_pred.shape)
    print(torch.mean((n-n_pred)**2))
    print(model.sampling(1))
