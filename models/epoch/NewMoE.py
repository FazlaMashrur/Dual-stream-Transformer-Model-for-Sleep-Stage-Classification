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
from layers.transformer import (
    Transformer,
    CrossAttnTransformer,
    MoETransformer,
    NewMoETransformer,
)
from layers.norm import PreNorm
from layers.head import Pooler, cls_head
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
        self.moe_transformer = NewMoETransformer(
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

        self.cls_head = cls_head(inner_dim, c_out)
        self.cls_head_eeg = cls_head(inner_dim, c_out)
        self.cls_head_emg = cls_head(inner_dim, c_out)

    def forward(self, x, label):
        # note: if no context is given, cross-attention defaults to self-attention
        # x --> [batch, trace, channel, inner_dim]
        eeg, emg = x[:, 0], x[:, 1]

        infer = self.moe_transformer.infer(eeg, emg)
        logits = self.cls_head(infer["cls_feats"])
        infer_eeg = self.moe_transformer.infer_eeg(eeg)
        infer_emg = self.moe_transformer.infer_emg(emg)
        logits_eeg = self.cls_head_eeg(infer_eeg["cls_feats"])
        logits_emg = self.cls_head_emg(infer_emg["cls_feats"])

        out_dict = {
            "out": logits,
            "out_eeg": logits_eeg,
            "out_emg": logits_emg,
            "cls_feats": infer["cls_feats"],
            "cls_feats_eeg": infer_eeg["cls_feats"],
            "cls_feats_emg": infer_emg["cls_feats"],
            "raw_cls_feats": infer["raw_cls_feats"],
            "label": label,
        }
        return out_dict
