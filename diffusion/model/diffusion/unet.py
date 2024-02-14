
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from einops import rearrange, repeat


class SelfAttention(nn.Module):
    def __init__(
            self,
            channels: int
    ):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return rearrange(attention_value, 'b (h w) c -> b c h w', h=H, w=W).contiguous()


class DoubleConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int | None = None,
        residual: bool = False
    ):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return (x + self.double_conv(x)) / 1.414
        else:
            return self.double_conv(x)


class DownSample(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        emb_dim: int = 256
    ):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        _, _, H, W = x.shape
        emb = repeat(self.emb_layer(t), 'b d -> b d h w', h=H, w=W).contiguous()
        return x + emb


class UpSample(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        emb_dim: int = 256
    ):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels)
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        _, _, H, W = x.shape
        emb = repeat(self.emb_layer(t), 'b d -> b d h w', h=H, w=W).contiguous()
        return x + emb


class UNet(pl.LightningModule):
    def __init__(
        self,
        c_in: int = 3,
        c_out: int = 3,
        time_dim: int = 256
    ):
        super().__init__()
        self.time_dim = time_dim

        self.time_embed = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.inc = DoubleConv(in_channels=c_in, out_channels=64)
        self.down1 = DownSample(in_channels=64, out_channels=128)
        self.sa1 = SelfAttention(channels=128)
        self.down2 = DownSample(in_channels=128, out_channels=256)
        self.sa2 = SelfAttention(channels=256)
        self.down3 = DownSample(in_channels=256, out_channels=256)
        self.sa3 = SelfAttention(channels=256)

        self.mid1 = DoubleConv(in_channels=256, out_channels=512)
        self.mid2 = DoubleConv(in_channels=512, out_channels=512)

        self.up1 = UpSample(in_channels=512, out_channels=256)
        self.sa4 = SelfAttention(channels=256)
        self.up2 = UpSample(in_channels=256, out_channels=128)
        self.sa5 = SelfAttention(channels=128)
        self.up3 = UpSample(in_channels=128, out_channels=64)
        self.sa6 = SelfAttention(channels=64)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2).float().to(t.device) / channels)
        ) * t.repeat(1, channels // 2)

        pos_enc = torch.zeros((t.shape[0], channels), device=t.device)
        pos_enc[:, 0::2] = torch.sin(inv_freq)
        pos_enc[:, 1::2] = torch.cos(inv_freq)
        return pos_enc

    def forward_unet(self, x, t):
        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.mid1(x4)
        x4 = self.mid2(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.outc(x)
        return output

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor
    ):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)
        t = self.time_embed(t)
        return self.forward_unet(x, t)


class ConditionalUNet(UNet):
    def __init__(
        self,
        c_in: int = 3,
        c_out: int = 3,
        time_dim: int = 256,
        num_classes: int | None = None,
    ):
        super().__init__(c_in, c_out, time_dim)
        if num_classes is not None:
            self.cls_embed = nn.Embedding(num_classes, time_dim)

    def forward(
            self,
            x: torch.Tensor,
            t: torch.Tensor,
            label: torch.Tensor | None = None
    ):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)
        t = self.time_embed(t)
        if label is not None:
            t += self.cls_embed(label)
        return self.forward_unet(x, t)


if __name__ == '__main__':
    net = ConditionalUNet()
    print(sum([p.numel() for p in net.parameters()]))
    x = torch.randn(2, 3, 32, 32)
    t = x.new_tensor([500] * x.shape[0]).long()
    print(t)
    print(net(x, t).shape)
