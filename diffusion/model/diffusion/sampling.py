import torch


@torch.no_grad()
def ddpm_sampling(
    model,
    scheduler,
    n_samples: int = 16,
    max_timesteps: int = 1000,
    in_channels: int = 3,
    dim: int = 32,
    num_classes: int = 10,
    cfg_scale: int = 3,
):
    labels = torch.randint(
        low=0, high=num_classes, size=(n_samples,), device=model.device
    )
    x_t = torch.randn(
        n_samples, in_channels, dim, dim, device=model.device
    )
    model.eval()
    for t in range(max_timesteps-1, 0, -1):
        time = torch.full((n_samples,), fill_value=t, device=model.device)
        pred_noise = model(x_t, time, labels)
        if cfg_scale > 0:
            uncond_pred_noise = model(x_t, time)
            pred_noise = torch.lerp(uncond_pred_noise, pred_noise, cfg_scale)
        sqrt_alpha = scheduler.get('sqrt_alpha', time)
        somah = scheduler.get('sqrt_one_minus_alpha_hat', time)
        sqrt_beta = scheduler.get('sqrt_beta', time)
        noise = torch.randn_like(x_t, device=model.device) if (
            t > 1
        ) else torch.zeros_like(x_t, device=model.device)

        x_t = 1 / sqrt_alpha * (
            x_t - (1-sqrt_alpha) / somah * pred_noise
        ) + sqrt_beta * noise
        x_t = x_t.clamp(-1, 1)

    x_t = (x_t + 1) / 2 * 255.  # range [0,255]
    model.train()
    return x_t.type(torch.uint8)

