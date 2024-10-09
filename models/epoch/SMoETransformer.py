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
from layers.transformer import Transformer, CrossAttnTransformer, MoETransformer
from layers.norm import PreNorm
from layers.head import Pooler
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
        ca_layers = args.ca_layers
        patch_len = args.patch_len
        norm_type = args.norm_type
        activation = args.activation

        d_head = d_model // n_heads
        inner_dim = n_heads * d_head
        mult_ff = args.d_ff // d_model
        n_traces = 2 if args.features == "ALL" else 1
        useRaw = args.useRaw
        assert (seq_len % patch_len) == 0
        n_patches = seq_len // patch_len
        mixffn_start_layer_index = e_layers - ca_layers
        # self.stft_transform = STFT(win_length=patch_len,n_fft=256,hop_length=patch_len)
        self.moe_transformer = MoETransformer(
            patch_len,
            n_patches,
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
            domain="time",
            mixffn_start_layer_index=mixffn_start_layer_index,
            output_attentions=False,
        )

        self.mlp_head = nn.Sequential(
            nn.Linear(inner_dim, inner_dim * 2),
            nn.LayerNorm(inner_dim * 2),
            nn.GELU(),
            nn.Linear(inner_dim * 2, c_out),
        )
        self.pooler = Pooler(inner_dim, useRaw=useRaw)

        # self.mlp_head.apply(self._init_weights)
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
        eeg, emg = x[:, 0], x[:, 1]

        x, eeg_feats, emg_feats = self.moe_transformer(eeg, emg)

        cls_feat = self.pooler(x)
        cls_raw = x[:, -1]
        cls_eeg, cls_emg = eeg_feats[:, -1], emg_feats[:, -1]
        # x_our --> [b, n, 2d]
        out = self.mlp_head(cls_feat)
        out_dict = {"out": out, "cls_raw": cls_raw, "emb": cls_feat, "label": label}
        return out_dict
