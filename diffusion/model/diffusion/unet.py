import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from einops.layers.torch import Rearrange
from einops import rearrange, repeat


def upsample_block(dim_in: int, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2),
        nn.Conv2d(
            in_channels=dim_in,
            out_channels=dim_out or dim_in,
            kernel_size=3,
            padding=1
        )
    )


def downsample_block(dim_in: int, dim_out=None):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1=2, p2=2),
        nn.Conv2d(
            in_channels=dim_in*4,
            out_channels=dim_out or dim_in,
            kernel_size=1
        )
    )


class RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x: torch.Tensor):
        return F.normalize(x, dim=1) * self.g * (x.shape[1] ** 0.5)


class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, embed_dim: int, theta: int = 10000):
        super().__init__()
        self.dim = embed_dim
        self.theta = theta

    def forward(self, x: torch.Tensor):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Block(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        groups: int = 8
    ):
        super().__init__()
        self.proj = nn.Conv2d(dim_in, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(
        self,
        x: torch.Tensor,
        scale_shift: tuple[torch.Tensor] | None = None
    ):
        x = self.proj(x)
        x = self.norm(x)
        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        time_emb_dim: int = None,
        groups: int = 8
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if time_emb_dim else None

        self.block1 = Block(dim_in, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim_in, dim_out, 1) if dim_in != dim_out else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        time_emb: torch.Tensor | None = None
    ):

        scale_shift = None
        if (self.mlp is not None) and (time_emb is not None):
            time_emb = self.mlp(time_emb).unsqueeze(-1).unsqueeze(-1)  # [B,C,1,1]
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)


class LinearAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        head_dim: int = 32,
        num_mem_kv: int = 4
    ):
        super().__init__()
        self.scale = head_dim ** -0.5
        self.num_heads = num_heads
        hidden_dim = head_dim * num_heads

        self.norm = RMSNorm(dim)

        self.mem_kv = nn.Parameter(torch.randn(2, num_heads, head_dim, num_mem_kv))
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            RMSNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.num_heads), qkv)
        mem_k, mem_v = map(lambda t: repeat(t, 'h c n -> b h c n', b=b), self.mem_kv)
        k = torch.cat([mem_k, k], dim=-1)
        v = torch.cat([mem_v, v], dim=-1)
        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)
        q = q * self.scale
        context = torch.einsum('bhdn, bhen -> bhde', [k, v])
        out = torch.einsum('bhde, bhdn -> bhen', [context, q])
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h=self.num_heads, x=h, y=w)
        return self.to_out(out)


class FullAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        head_dim: int = 32,
        num_mem_kv=4,
    ):
        super().__init__()
        self.heads = num_heads
        hidden_dim = head_dim * num_heads

        self.norm = RMSNorm(dim)
        self.mem_kv = nn.Parameter(torch.randn(2, num_heads, num_mem_kv, head_dim))
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x: torch.Tensor):
        b, c, h, w = x.shape
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h (x y) c', h=self.heads), qkv)
        mem_k, mem_v = map(lambda t: repeat(t, 'h c n -> b h c n', b=b), self.mem_kv)
        k = torch.cat([mem_k, k], dim=-2)
        v = torch.cat([mem_v, v], dim=-2)
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=0.
        )
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        return self.to_out(out)


class UNet(pl.LightningModule):
    def __init__(
        self,
        dim: int,
        out_dim: int | None = None,
        dim_mults: tuple[int] = (1, 2, 4, 8),
        in_channels: int = 3,
        groups: int = 8,
        learned_variance: bool = False,
        attn_head_dim: int = 32,
        attn_heads: int = 4,
        full_attn=None,    # defaults to full attention only for inner most layer
    ):
        super().__init__()
        self.init_conv = nn.Conv2d(in_channels, dim, 7, padding=3)

        dims = [dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # time embeddings
        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbedding(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # attention
        if not full_attn:
            full_attn = (*((False,) * (len(dim_mults) - 1)), True)

        num_stages = len(dim_mults)
        full_attn = ((full_attn,) * num_stages)
        attn_heads = ((attn_heads,) * num_stages)
        attn_head_dim = ((attn_head_dim,) * num_stages)

        assert len(full_attn) == len(dim_mults)

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, ((dim_in, dim_out), _, heads, head_dim) in \
                enumerate(zip(in_out, full_attn, attn_heads, attn_head_dim)):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ResnetBlock(dim_in, dim_in, time_dim, groups),
                ResnetBlock(dim_in, dim_in, time_dim, groups),
                LinearAttention(dim_in, head_dim=head_dim, num_heads=heads),
                downsample_block(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_dim, groups)
        self.mid_attn = FullAttention(mid_dim, num_heads=attn_heads[-1], head_dim=attn_head_dim[-1])
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_dim, groups)

        for ind, ((dim_in, dim_out), _, heads, head_dim) in \
                enumerate(zip(*map(reversed, (in_out, full_attn, attn_heads, attn_head_dim)))):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                ResnetBlock(dim_out + dim_in, dim_out, time_dim, groups),
                ResnetBlock(dim_out + dim_in, dim_out, time_dim, groups),
                LinearAttention(dim_out, head_dim=head_dim, num_heads=heads),
                upsample_block(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding=1)
            ]))

        default_out_dim = in_channels * (1 if not learned_variance else 2)
        self.out_dim = out_dim or default_out_dim

        self.final_res_block = ResnetBlock(dim * 2, dim, time_dim, groups)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

        self.downsample_factor = 2 ** (len(self.downs) - 1)

    def forward(
        self,
        x: torch.Tensor,
        time: torch.Tensor
    ):
        assert all([d % self.downsample_factor == 0 for d in x.shape[-2:]]
                   ), f'your input dimensions {x.shape[-2:]} need to be divisible by {self.downsample_factor}'

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)
            x = block2(x, t)
            x = attn(x) + x
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x) + x
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x) + x
            x = upsample(x)

        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)


if __name__ == "__main__":
    model = UNet(dim=32)
    x = torch.randn(4, 3, 32, 32)
    time = torch.tensor([0, 1, 2, 3])
    print(model(x, time).shape)
