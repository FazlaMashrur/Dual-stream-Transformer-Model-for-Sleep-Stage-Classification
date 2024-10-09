from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from layers.attention import MultiHeadAttention, MacaronBlock
from layers.patchEncoder import LinearPatchEncoder, LinearPatchEncoder2
from layers.norm import PreNorm


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        c_in = 1
        c_out = args.c_out
        d_model = args.d_model
        n_heads = args.n_heads
        seq_len = args.seq_len
        dropout = args.dropout
        e_layers = args.e_layers
        patch_len = args.patch_len
        emb_dropout = args.emb_dropout
        norm_type = args.norm_type
        activation = args.activation

        d_head = d_model // n_heads
        inner_dim = n_heads * d_head
        mult_ff = args.d_ff // d_model
        n_traces = 2 if args.features == "ALL" else 1

        assert (seq_len % patch_len) == 0
        n_patches = seq_len // patch_len

        self.pos_embedding = nn.Parameter(torch.randn(1, n_patches + 1, inner_dim))

        self.cls_eeg = nn.Parameter(torch.randn(1, 1, inner_dim))
        self.cls_emg = nn.Parameter(torch.randn(1, 1, inner_dim))

        self.patch_encs = nn.ModuleList(
            [
                LinearPatchEncoder2(trace_idx, patch_len, c_in, inner_dim)
                for trace_idx in range(n_traces)
            ]
        )

        self.macaron_transformer = nn.ModuleList(
            [
                MacaronBlock(
                    inner_dim,
                    n_heads,
                    d_head,
                    dropout=dropout,
                    activation=activation,
                    norm=norm_type,
                    mult=mult_ff,
                )
                for _ in range(e_layers)
            ]
        )

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(inner_dim * n_traces), nn.Linear(inner_dim * n_traces, c_out)
        )

    def forward(self, x, label):
        # note: if no context is given, cross-attention defaults to self-attention
        # x --> [batch, trace, channel, inner_dim]
        eeg, emg = [patch_enc(x) for patch_enc in self.patch_encs]
        b, n, d = eeg.shape
        eeg = eeg + self.pos_embedding[:, :n]
        emg = emg + self.pos_embedding[:, :n]
        cls_eeg = repeat(self.cls_eeg, "() n d -> b n d", b=b)
        cls_emg = repeat(self.cls_emg, "() n d -> b n d", b=b)
        src_eeg = torch.cat([eeg, cls_eeg], dim=-2)
        src_emg = torch.cat([emg, cls_emg], dim=-2)

        for block in self.macaron_transformer:
            src_eeg, src_emg = block(src_eeg, src_emg)

        cls_eeg, cls_emg = src_eeg[:, -1], src_emg[:, -1]
        # x_our --> [b, n, 2d]
        emb = torch.cat([cls_eeg, cls_emg], dim=-1)
        out = self.mlp_head(emb)
        return out, None, None, emb, label
