import torch
import torch.nn as nn


class ForwardDiffusion(nn.Module):
    def __init__(
        self,
        max_timesteps: int,
        beta_1: float = 0.0001,
        beta_2: float = 0.02,
    ):
        super().__init__()
        self._beta_scheduler = self._linear_scheduler(max_timesteps, beta_1, beta_2)
        alpha_cumprod = torch.cumprod(1 - self._beta_scheduler, dim=0)
        self._sqrt_alpha = torch.sqrt(alpha_cumprod)
        self._sqrt_1_minus_alpha = torch.sqrt(1 - alpha_cumprod)

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
        t: torch.Tensor
    ):
        # x.shape = [T,]
        # t.shape = [B,]
        x_select = x.gather(dim=-1, index=t)
        return x_select.unsqueeze(-1)  # [B,1]

    def forward(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor
    ):
        noise = torch.randn_like(x_0, device=x_0.device)
        mean = self._batch_index_select(self._sqrt_alpha.to(x_0.device), t) * x_0
        std = self._batch_index_select(self._sqrt_1_minus_alpha.to(x_0.device), t) * noise
        return torch.clamp(mean + std, min=-1, max=1), noise


if __name__ == "__main__":
    a = torch.rand(3, 3)
    t = torch.tensor([1, 0, 2])
    model = ForwardDiffusion(max_timesteps=5)
    print(a)
    print(model(a, t))
