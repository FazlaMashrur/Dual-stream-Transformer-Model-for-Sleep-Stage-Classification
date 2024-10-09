from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from layers.attention import MacaronBlock, MultiHeadAttention, MultiPathBlock
from layers.patchEncoder import LinearPatchEncoder, LinearPatchEncoder2
from layers.head import Pooler


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
        n_sequences = args.n_sequences
        norm_type = args.norm_type
        activation = args.activation

        d_head = d_model // n_heads
        inner_dim = n_heads * d_head
        mult_ff = args.d_ff // d_model
        n_traces = 2 if args.features == "ALL" else 1

        assert (seq_len % patch_len) == 0
        n_patches = seq_len // patch_len

        self.pos_embedding = nn.Parameter(torch.randn(1, 1, n_patches + 1, inner_dim))
        self.ep_embedding = nn.Parameter(torch.randn(1, n_sequences + 1, inner_dim))

        self.cls_token = nn.Parameter(torch.randn(1, 1, 1, inner_dim))

        self.patch_encs = nn.ModuleList(
            [
                LinearPatchEncoder2(trace_idx, patch_len, c_in, inner_dim)
                for trace_idx in range(n_traces)
            ]
        )

        self.mono_blocks = nn.ModuleList(
            [
                MultiPathBlock(
                    inner_dim,
                    n_heads,
                    d_head,
                    dropout=dropout,
                    activation=activation,
                    norm=norm_type,
                    mult=mult_ff,
                    with_emffn=False,
                )
                for _ in range(e_layers)
            ]
        )

        self.cross_blocks = nn.ModuleList(
            [
                MultiPathBlock(
                    inner_dim,
                    n_heads,
                    d_head,
                    dropout=dropout,
                    activation=activation,
                    norm=norm_type,
                    mult=mult_ff,
                    with_emffn=True,
                    n_patch=n_patches + 1,
                )
                for _ in range(e_layers)
            ]
        )

        self.cross_epoch_tranfomer = nn.ModuleList(
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

        self.pooler = Pooler(inner_dim)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(inner_dim), nn.Linear(inner_dim, c_out)
        )

    def forward(self, x, label):
        # note: if no context is given, cross-attention defaults to self-attention
        # x --> [batch, epoch, trace, channel, seq_len]
        eeg, emg = [patch_enc(x) for patch_enc in self.patch_encs]
        # rearrange the epoch dimmension to batch dimmension
        b, e, n, d = eeg.shape
        # pos_embedding--> 1, 1, n+1, d
        eeg = eeg + self.pos_embedding[:, :, :n]
        emg = emg + self.pos_embedding[:, :, :n]

        for block in self.mono_blocks:
            eeg = block(eeg, modality_type="eeg")
            emg = block(emg, modality_type="emg")

        cls_tokens = repeat(self.cls_token, "() () n d -> b e n d", b=b, e=e)
        eeg = torch.cat([cls_tokens, eeg], dim=-2)
        co_emb = torch.cat([eeg, emg], dim=-2)

        for block in self.cross_blocks:
            co_emb = block(co_emb, modality_type="mix")

        # cls_emb = co_emb[:,0]
        emb = self.pooler(co_emb)
        # emb --> [batch, epoch, inner_dim*2]
        emb = emb + self.ep_embedding[:, :e]

        for block in self.cross_epoch_tranfomer:
            emb = block(emb, context=None)

        out = self.mlp_head(emb)
        out = rearrange(out, "b e d -> (b e) d", b=b)
        label = rearrange(label, "b e d -> (b e) d", b=b)
        return out, None, None, emb, label


# class Model(nn.Module):

#     def __init__(self, args):
#         super().__init__()
#         c_in = 1
#         c_out = args.c_out
#         d_model = args.d_model
#         n_heads = args.n_heads
#         seq_len = args.seq_len
#         dropout = args.dropout
#         e_layers = args.e_layers
#         patch_len = args.patch_len
#         emb_dropout = args.emb_dropout
#         n_sequences = args.n_sequences

#         d_head = d_model // n_heads
#         inner_dim = n_heads * d_head
#         n_traces = 2 if args.features == "ALL" else 1

#         assert (seq_len % patch_len) == 0
#         n_patches = seq_len // patch_len

#         self.pos_embedding = nn.Parameter(torch.randn(1, 1, n_patches + 1, inner_dim))
#         self.ep_embedding = nn.Parameter(torch.randn(1, n_sequences + 1, inner_dim))

#         self.cls_token = nn.Parameter(torch.randn(1,1,1, inner_dim))

#         self.patch_encs = nn.ModuleList(
#             [LinearPatchEncoder2(trace_idx, patch_len, c_in, inner_dim)
#                 for trace_idx in range(n_traces)])

#         self.eeg_transformer = nn.ModuleList(
#             [MultiHeadAttention(inner_dim, n_heads, d_head, dropout=dropout)
#                 for _ in range(e_layers)]
#         )
#         self.emg_transformer = nn.ModuleList(
#             [MultiHeadAttention(inner_dim, n_heads, d_head, dropout=dropout)
#                 for _ in range(e_layers)]
#         )
#         self.cross_epoch_tranfomer = nn.ModuleList(
#             [MultiHeadAttention(inner_dim, n_heads, d_head, dropout=dropout)
#                 for _ in range(e_layers)])

#         self.proj = nn.Sequential(
#             nn.LayerNorm(inner_dim * n_traces),
#             nn.Linear(inner_dim * n_traces, inner_dim),
#             )

#         self.mlp_head = nn.Sequential(
#             nn.LayerNorm(inner_dim),
#             nn.Linear(inner_dim, c_out)
#             )

#     def forward(self, x, label):
#         # note: if no context is given, cross-attention defaults to self-attention
#         # x --> [batch, epoch, trace, channel, seq_len]
#         eeg, emg = [patch_enc(x) for patch_enc in self.patch_encs]
#         # rearrange the epoch dimmension to batch dimmension
#         b, e, n, d = eeg.shape
#         # pos_embedding--> 1, 1, n+1, d
#         eeg = eeg + self.pos_embedding[:, :, :n]
#         emg = emg + self.pos_embedding[:, :, :n]

#         cls_tokens = repeat(self.cls_token, '() () n d -> b e n d', b=b, e=e)
#         src_eeg = torch.cat([eeg, cls_tokens], dim=-2)
#         src_emg = torch.cat([emg, cls_tokens], dim=-2)

#         for block in self.eeg_transformer:
#             src_eeg = block(src_eeg, context=None)

#         for block in self.emg_transformer:
#             src_emg = block(src_emg, context=None)

#         # cls_x1 --> [batch, epoch, inner_dim]
#         cls_eeg, cls_emg = src_eeg[:,:,-1], src_emg[:,:,-1]

#         # emb --> [batch, epoch, inner_dim*2]
#         emb = torch.cat([cls_eeg, cls_emg], dim=-1)
#         # emb --> [batch, epoch, inner_dim]
#         emb = self.proj(emb)
#         emb = emb + self.ep_embedding[:,:e]

#         for block in self.cross_epoch_tranfomer:
#             emb = block(emb, context=None)

#         out = self.mlp_head(emb)
#         out = rearrange(out, 'b e d -> (b e) d', b=b)
#         label = rearrange(label, 'b e d -> (b e) d', b=b)
#         return out, None, None, emb, label
