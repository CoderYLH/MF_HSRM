#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
# Modified for the MF-HSRM architecture-only release: selected definitions,
# self-contained imports, and removal of experiment/profiling utilities.

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from typing import Optional
from .base import BaseLayer
from .conv_layer import ConvLayer
from .dropout import Dropout


class LinearSelfAttention_improve(BaseLayer):
    """Linear attention with depthwise context enhancement, as selected in MF-HSRM.

    Input/output: [B, C, patch_area, number_of_patches]."""

    def __init__(
        self,
        opts,
        embed_dim: int,
        attn_dropout: Optional[float] = 0.0,
        bias: Optional[bool] = True,
        *args,
        **kwargs
    ) -> None:
        super().__init__()
        self.qkv_proj = ConvLayer(
            opts=opts,
            in_channels=embed_dim,
            out_channels=1 + 2 * embed_dim,
            bias=bias,
            kernel_size=1,
            use_norm=False,
            use_act=False,
        )
        self.context_conv = nn.Conv2d(
            embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim
        )
        self.attn_dropout = Dropout(p=attn_dropout)
        self.out_proj = ConvLayer(
            opts=opts,
            in_channels=embed_dim,
            out_channels=embed_dim,
            bias=bias,
            kernel_size=1,
            use_norm=False,
            use_act=False,
        )
        self.activation = nn.GELU()
        self.embed_dim = embed_dim

    def __repr__(self):
        return "{}(embed_dim={}, attn_dropout={})".format(
            self.__class__.__name__, self.embed_dim, self.attn_dropout.p
        )

    def _forward_self_attn(self, x: Tensor, *args, **kwargs) -> Tensor:
        qkv = self.qkv_proj(x)
        query, key, value = torch.split(
            qkv, split_size_or_sections=[1, self.embed_dim, self.embed_dim], dim=1
        )
        key = self.context_conv(key)
        value = self.context_conv(value)
        context_scores = F.softmax(query, dim=-1)
        context_scores = self.attn_dropout(context_scores)
        context_vector = key * context_scores
        context_vector = torch.sum(context_vector, dim=-1, keepdim=True)
        out = self.activation(value) * context_vector.expand_as(value)
        out = self.out_proj(out)
        return out

    def _forward_cross_attn(
        self, x: Tensor, x_prev: Optional[Tensor] = None, *args, **kwargs
    ) -> Tensor:
        batch_size, in_dim, kv_patch_area, kv_num_patches = x.shape
        q_patch_area, q_num_patches = x.shape[-2:]
        assert (
            kv_patch_area == q_patch_area
        ), "The number of pixels in a patch for query and key_value should be the same"
        qk = F.conv2d(
            x_prev,
            weight=self.qkv_proj.block.conv.weight[: self.embed_dim + 1, ...],
            bias=self.qkv_proj.block.conv.bias[: self.embed_dim + 1, ...],
        )
        query, key = torch.split(qk, split_size_or_sections=[1, self.embed_dim], dim=1)
        value = F.conv2d(
            x,
            weight=self.qkv_proj.block.conv.weight[self.embed_dim + 1 :, ...],
            bias=self.qkv_proj.block.conv.bias[self.embed_dim + 1 :, ...],
        )
        key = self.context_conv(key)
        value = self.context_conv(value)
        context_scores = F.softmax(query, dim=-1)
        context_scores = self.attn_dropout(context_scores)
        context_vector = key * context_scores
        context_vector = torch.sum(context_vector, dim=-1, keepdim=True)
        out = self.activation(value) * context_vector.expand_as(value)
        out = self.out_proj(out)
        return out

    def forward(
        self, x: Tensor, x_prev: Optional[Tensor] = None, *args, **kwargs
    ) -> Tensor:
        if x_prev is None:
            return self._forward_self_attn(x, *args, **kwargs)
        else:
            return self._forward_cross_attn(x, *args, x_prev=x_prev, **kwargs)
