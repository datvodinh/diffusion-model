import torch
import pytorch_lightning as pl
import diffusion


class DiffusionModel(pl.LightningModule):
    def __init__(
        self,
        lr: float = 1e-4,
        max_timesteps: int = 1000,
        in_channels: int = 3,
        dim: int = 64
    ):
        super().__init__()
        self.forward_diffusion = diffusion.ForwardDiffusion(max_timesteps=max_timesteps)
        self.reverse_diffusion = diffusion.UNet(dim=dim, in_channels=in_channels)
        self.max_timesteps = max_timesteps
        self.lr = lr

    def forward(self, x_0):
        t = torch.randint(
            low=0, high=self.max_timesteps, size=(x_0.shape[0],), device=x_0.device
        )
        x_noise, noise = self.forward_diffusion(x_0, t)
        noise_pred = self.reverse_diffusion(x_noise, t)
        return noise, noise_pred

    def training_step(self, batch, idx):
        x_0, _ = batch
        noise, noise_pred = self(x_0)
        loss = torch.mean((noise - noise_pred)**2)
        self.log_dict(
            {
                "train_loss": loss
            }
        )
        return loss

    def sampling(self, x_t):
        pass

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
    model = DiffusionModel(max_timesteps=1000)
    n, n_pred = model(a)
    print(n.shape, n_pred.shape)
    print(torch.mean((n-n_pred)**2))
