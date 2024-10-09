from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from layers.attention import MultiHeadAttention
from layers.Freqtransform import STFT
from layers.patchEncoder import LinearPatchEncoder, LinearPatchEncoder2
from layers.transformer import Transformer
from layers.norm import PreNorm
import librosa
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        c_in = 1
        c_out = args.c_out
        d_model = args.d_model
        n_heads = args.n_heads
        seq_len = args.seq_len
        dropout = args.dropout
        path_drop = args.path_drop
        e_layers = args.e_layers
        patch_len = args.patch_len
        norm_type = args.norm_type
        activation = args.activation
        output_attentions = args.output_attentions
        d_head = d_model // n_heads
        inner_dim = n_heads * d_head
        mult_ff = args.d_ff // d_model
        n_traces = 2 if args.features == "ALL" else 1

        assert (seq_len % patch_len) == 0
        n_patches = seq_len // patch_len

        self.stft_transform = STFT(
            win_length=patch_len, n_fft=256, hop_length=patch_len, normalized_stft=False
        )

        self.eeg_stft_transformer = Transformer(
            patch_len,
            n_patches + 1,
            e_layers,
            c_in,
            inner_dim,
            n_heads=n_heads,
            d_head=d_head,
            dropout=dropout,
            path_drop=path_drop,
            activation=activation,
            norm=norm_type,
            mult=mult_ff,
            mix_type=args.mix_type,
            cls=True,
            flag="epoch",
            domain="freq",
            output_attentions=output_attentions,
        )
        self.emg_stft_transformer = Transformer(
            patch_len,
            n_patches + 1,
            e_layers,
            c_in,
            inner_dim,
            n_heads=n_heads,
            d_head=d_head,
            dropout=dropout,
            path_drop=path_drop,
            activation=activation,
            norm=norm_type,
            mult=mult_ff,
            mix_type=args.mix_type,
            cls=True,
            flag="epoch",
            domain="freq",
            output_attentions=output_attentions,
        )

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(inner_dim * n_traces), nn.Linear(inner_dim * n_traces, c_out)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, label):
        # note: if no context is given, cross-attention defaults to self-attention
        # x --> [batch, trace, channel, inner_dim]
        eeg, emg = x[:, :1], x[:, -1:]

        eeg_freq = self.stft_transform(eeg)[:, 0]

        emg_freq = self.stft_transform(emg)[:, 0]

        eeg, eeg_attn = self.emg_stft_transformer(eeg_freq)
        emg, emg_attn = self.emg_stft_transformer(emg_freq)

        cls_eeg, cls_emg = eeg[:, -1], emg[:, -1]
        # x_our --> [b, n, 2d]
        emb = torch.cat([cls_eeg, cls_emg], dim=-1)
        out = self.mlp_head(emb)

        out_dict = {
            "out": out,
            "eeg_attn": eeg_attn,
            "emg_attn": emg_attn,
            "emb": emb,
            "label": label,
        }
        return out_dict