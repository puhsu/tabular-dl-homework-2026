"""Kumo Medium tensor core. One unpadded table.

Checkpoint names mirror NVIDIA's SDM; the computation uses only PyTorch.
"""

import torch
from torch import nn


def groups(columns, device=None):
    raise NotImplementedError("TODO: groups")


class Cells(nn.Module):
    def __init__(self, d=128):
        super().__init__()
        self.num_freq = nn.Parameter(torch.randn(3, 32))
        self.cat_freq = nn.Parameter(torch.randn(3, 32))
        self.num_lin, self.cat_lin = nn.Linear(64, d), nn.Linear(64, d)
        self.nan_lin = nn.Linear(3, d, bias=False)

    def forward(self, x, n, cat, mean=None):
        raise NotImplementedError("TODO: forward")


class Rotary(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.register_buffer("inv_freq", 100000 ** (-torch.arange(0, width, 2) / width))

    def forward(self, x):
        raise NotImplementedError("TODO: forward")


class Block(nn.Module):
    def __init__(self, d, heads, scale, rotary=False):
        super().__init__()
        self.h = heads
        self.query_norm, self.key_value_norm = nn.RMSNorm(d), nn.RMSNorm(d)
        a = self.attn = nn.Module()
        a.qkv_lin, a.out_lin = nn.Linear(d, 3 * d), nn.Linear(d, d)
        for name in ("query_transform", "key_transform"):
            ops = [Rotary(d // heads)] if rotary else []
            setattr(
                a,
                name,
                nn.Sequential(
                    *ops, nn.RMSNorm(d // heads, eps=1e-6, elementwise_affine=False)
                ),
            )
        a.sdpa = nn.Module()
        if scale:
            s = a.sdpa.query_scaling = nn.Module()
            s.head_scale = nn.Parameter(torch.rand(heads))
            if rotary:
                s.gate = nn.Sequential(
                    nn.Linear(d // heads, 64), nn.GELU(), nn.Linear(64, d // heads)
                )
        self.mlp = nn.Sequential(
            nn.RMSNorm(d), nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d)
        )

    def forward(self, x, memory, kv_heads=None):
        raise NotImplementedError("TODO: forward")


class Induced(nn.Module):
    def __init__(self, d=128):
        super().__init__()
        self.inducing_points = nn.Parameter(torch.randn(d, d) * 0.02)
        self.inducing_block = Block(d, heads=4, scale=True)
        self.output_block = Block(d, heads=4, scale=False)

    def forward(self, columns, n):
        raise NotImplementedError("TODO: forward")


class KumoMedium(nn.Module):
    def __init__(self, classification=False):
        super().__init__()
        r = self.row_embedding = nn.Module()
        i = self.icl_block = nn.Module()
        width, depth = 256, 6
        for mod, d in ((r, width), (i, 512)):
            if classification:
                mod.y_emb = nn.Embedding(10, d)
            else:
                mod.y_lin = nn.Linear(1, d, bias=False)
            mod.norm = nn.RMSNorm(d)

        # this is encoder
        r.cell_embedding = Cells(width)
        r.readout_token = nn.Parameter(torch.randn(4, width) * 0.02)
        r.col_blocks = nn.ModuleList(Induced(width) for _ in range(depth))
        r.row_blocks = nn.ModuleList(
            Block(width, heads=4, scale=True, rotary=True) for _ in range(depth)
        )

        # this is the ICL transformer
        self.row_project = nn.Linear(1024, 512)
        i.layers = nn.ModuleList(Block(512, 8, True) for _ in range(24))
        i.head = nn.Sequential(
            nn.Linear(512, 1024),
            nn.GELU(),
            nn.Linear(1024, 10 if classification else 999),
        )

    def encoder(self, x, y, categorical):
        raise NotImplementedError("TODO: encoder")

    def icl(self, rows, y):
        raise NotImplementedError("TODO: icl")

    def forward(self, x, y, categorical):
        return self.icl(self.encoder(x, y, categorical), y)

    @classmethod
    def load(cls, path, classification=False, device="cpu"):
        with torch.device("meta"):
            model = cls(classification)
        state = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(state, strict=True, assign=True)
        return model.eval()
