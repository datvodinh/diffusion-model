import torch


def ddpm_sampling_timestep(
    x_t,
    model,
    scheduler,
    labels,
    t,
    n_samples: int = 16,
    cfg_scale: int = 3,
):
    time = torch.full((n_samples,), fill_value=t, device=model.device)
    pred_noise = model(x_t, time, labels)
    if cfg_scale > 0:
        uncond_pred_noise = model(x_t, time, None)
        pred_noise = torch.lerp(uncond_pred_noise, pred_noise, cfg_scale)
    alpha = scheduler.get('alpha', time)
    sqrt_alpha = scheduler.get('sqrt_alpha', time)
    somah = scheduler.get('sqrt_one_minus_alpha_hat', time)
    sqrt_beta = scheduler.get('sqrt_beta', time)
    if t > 0:
        noise = torch.randn_like(x_t, device=model.device)
    else:
        noise = torch.zeros_like(x_t, device=model.device)

    x_t_new = 1 / sqrt_alpha * (x_t - (1-alpha) / somah * pred_noise) + sqrt_beta * noise
    return x_t_new.clamp(-1, 1)


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
    demo: bool = True,
    labels=None
):
    if labels is None:
        labels = torch.randint(
            low=0, high=num_classes, size=(n_samples,), device=model.device
        )
    else:
        n_samples = labels.shape[0]

    x_t = torch.randn(
        n_samples, in_channels, dim, dim, device=model.device
    )
    model.eval()
    for t in range(max_timesteps-1, -1, -1):
        x_t = ddpm_sampling_timestep(x_t=x_t, model=model, scheduler=scheduler,
                                     labels=labels, t=t, n_samples=n_samples,
                                     cfg_scale=cfg_scale)
        if demo:
            yield ((x_t + 1) / 2 * 255).type(torch.uint8)
    model.train()
    if not demo:
        x_t = (x_t + 1) / 2 * 255.  # range [0,255]
        return x_t.type(torch.uint8)
