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
    MoETransformer,
    SeqNewMoETransformer,
    SeqNewMoETransformer2,
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
        seq_layers = args.seq_layers
        patch_len = args.patch_len
        norm_type = args.norm_type
        activation = args.activation
        self.n_sequences = args.n_sequences
        self.output_attentions = args.output_attentions
        d_head = d_model // n_heads
        inner_dim = n_heads * d_head
        mult_ff = args.d_ff // d_model
        n_traces = 2 if args.features == "ALL" else 1
        self
        assert (seq_len % patch_len) == 0
        n_patches = seq_len // patch_len
        mixffn_start_layer_index = seq_layers - ca_layers
        # self.stft_transform = STFT(win_length=patch_len,n_fft=256,hop_length=patch_len)
        self.ep_moe_transformer = NewMoETransformer(
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

        self.moe_transformer = SeqNewMoETransformer2(
            patch_len,
            self.n_sequences,
            seq_layers,
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
            cls=False,
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
        eeg, emg = x[:, :, 0], x[:, :, 1]
        b, e, c, t = eeg.shape
        eeg = rearrange(eeg, "b e c t -> (b e) c t")
        emg = rearrange(emg, "b e c t -> (b e) c t")

        ep_infer_eeg = self.ep_moe_transformer.infer_eeg(eeg)
        ep_infer_emg = self.ep_moe_transformer.infer_emg(emg)

        cls_eeg = ep_infer_eeg["cls_feats"]
        cls_emg = ep_infer_emg["cls_feats"]

        cls_eeg = rearrange(cls_eeg, "(b e) d -> b e d", e=e)
        cls_emg = rearrange(cls_emg, "(b e) d -> b e d", e=e)

        # x_our --> [b, n, 2d]

        infer = self.moe_transformer.infer(cls_eeg, cls_emg)
        logits = self.cls_head(infer["cls_feats"])
        infer_eeg = self.moe_transformer.infer_eeg(cls_eeg)
        infer_emg = self.moe_transformer.infer_emg(cls_emg)

        logits_eeg = self.cls_head_eeg(infer_eeg["cls_feats"])
        logits_emg = self.cls_head_emg(infer_emg["cls_feats"])

        logits = rearrange(logits, "b e d -> (b e) d")
        logits_eeg = rearrange(logits_eeg, "b e d -> (b e) d")
        logits_emg = rearrange(logits_emg, "b e d -> (b e) d")

        label = rearrange(label, "b e d -> (b e) d")

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
