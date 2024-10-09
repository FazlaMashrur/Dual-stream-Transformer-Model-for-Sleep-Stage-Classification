from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from layers.attention import MultiHeadAttention
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
        self.mix_type = args.mix_type
        assert (seq_len % patch_len) == 0
        n_patches = seq_len // patch_len

        self.pos_embedding = nn.Parameter(torch.randn(1, n_patches + 1, inner_dim))
        self.trace_emdedding = nn.Parameter(torch.randn(1, n_traces, inner_dim))

        self.cls_token = nn.Parameter(torch.randn(1, 1, inner_dim))

        self.patch_encs = nn.ModuleList(
            [
                LinearPatchEncoder2(trace_idx, patch_len, c_in, inner_dim)
                for trace_idx in range(n_traces)
            ]
        )

        self.transformer = nn.ModuleList(
            [
                MultiHeadAttention(
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
            nn.LayerNorm(inner_dim), nn.Linear(inner_dim, c_out)
        )

    def forward(self, x, label):
        # note: if no context is given, cross-attention defaults to self-attention
        # x --> [batch, trace, channel, inner_dim]
        eeg, emg = [patch_enc(x) for patch_enc in self.patch_encs]
        b, n, d = eeg.shape
        eeg = eeg + self.pos_embedding[:, :n]
        emg = emg + self.pos_embedding[:, :n]

        if self.mix_type == 2:
            eeg = eeg + self.trace_emdedding[:, :1]
            emg = emg + self.trace_emdedding[:, -1:]
        cls_token = repeat(self.cls_token, "() n d -> b n d", b=b)
        trace = torch.cat([eeg, emg, cls_token], dim=-2)

        for block in self.transformer:
            trace = block(trace, context=None)

        cls_emb = trace[:, -1]
        # cls --> [b, 1, d]
        out = self.mlp_head(cls_emb)
        return out, None, None, cls_emb, label


class Mono_Model(nn.Module):
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
        self.features = args.features
        assert (seq_len % patch_len) == 0
        n_patches = seq_len // patch_len

        self.pos_embedding = nn.Parameter(torch.randn(1, n_patches + 1, inner_dim))

        self.cls_token = nn.Parameter(torch.randn(1, 1, inner_dim))

        self.patch_enc = LinearPatchEncoder2(0, patch_len, c_in, inner_dim)

        self.transformer = nn.ModuleList(
            [
                MultiHeadAttention(
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
        x = x[:, :1] if self.features == "EEG" else x[:, -1:]
        x = self.patch_enc(x)
        b, n, d = x.shape
        x = x + self.pos_embedding[:, :n]
        cls_tokens = repeat(self.cls_token, "() n d -> b n d", b=b)
        src_x = torch.cat([x, cls_tokens], dim=-2)

        for block in self.transformer:
            src_x = block(src_x, context=None)

        # emb --> [b, n, d]
        emb = src_x[:, -1]

        out = self.mlp_head(emb)
        return out, None, None, emb, label