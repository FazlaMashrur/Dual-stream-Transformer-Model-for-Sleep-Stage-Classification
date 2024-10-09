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
from layers.epochEncoder import MLP_EpochEncoder, LSTM_EpochEncoder, LSTM_Encoder
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
        seq_layers = args.seq_layers
        patch_len = args.patch_len
        norm_type = args.norm_type
        activation = args.activation
        n_sequences = args.n_sequences
        self.output_attentions = args.output_attentions
        d_head = d_model // n_heads
        inner_dim = n_heads * d_head
        mult_ff = args.d_ff // d_model
        n_traces = 2 if args.features == "ALL" else 1

        assert (seq_len % patch_len) == 0
        n_patches = seq_len // patch_len

        # self.stft_transform = STFT(win_length=patch_len,n_fft=256,hop_length=patch_len)
        self.eeg_ep_encoder = LSTM_EpochEncoder(
            patch_len, inner_dim, depth=e_layers, dropout=dropout, activation=activation
        )

        self.emg_ep_encoder = LSTM_EpochEncoder(
            patch_len, inner_dim, depth=e_layers, dropout=dropout, activation=activation
        )

        self.seq_lstm = LSTM_Encoder(
            inner_dim, depth=seq_layers, dropout=dropout, activation=activation
        )

        self.proj = self._head = nn.Sequential(
            nn.LayerNorm(inner_dim * n_traces),
            nn.Linear(inner_dim * n_traces, inner_dim),
        )
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(inner_dim), nn.Linear(inner_dim, c_out)
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
        eeg, emg = x[:, :, 0], x[:, :, 1]

        eeg = self.eeg_ep_encoder(eeg)
        emg = self.emg_ep_encoder(emg)

        # x_our --> [b, n, 2d]
        emb = torch.cat([eeg, emg], dim=-1)
        emb = self.proj(emb)
        emb = self.seq_lstm(emb)

        out = self.mlp_head(emb)
        out = rearrange(out, "b e d -> (b e) d")
        emb = rearrange(emb, "b e d -> (b e) d")
        label = rearrange(label, "b e d -> (b e) d")

        out_dict = {"out": out, "emb": emb, "label": label}
        return out_dict
