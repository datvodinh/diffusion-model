import torch


class LinearScheduler:
    def __init__(
        self,
        max_timesteps: int = 1000,
        beta_1: int = 0.0001,
        beta_2: int = 0.02
    ) -> None:
        self.beta = torch.linspace(beta_1, beta_2, max_timesteps)
        self.sqrt_beta = torch.sqrt(self.beta)[:, None, None, None]
        self.alpha = (1 - self.beta)[:, None, None, None]
        self.sqrt_alpha = torch.sqrt(self.alpha)
        self.alpha_hat = torch.cumprod(1 - self.beta, dim=0)[:, None, None, None]
        self.sqrt_alpha_hat = torch.sqrt(self.alpha_hat)
        self.sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat)

    def get(self, key: str, t: torch.Tensor):
        return self.__dict__[key].to(t.device)[t]
