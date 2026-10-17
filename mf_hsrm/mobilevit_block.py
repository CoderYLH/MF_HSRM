# For licensing see accompanying LICENSE file.
# Modified for the MF-HSRM architecture-only release: selected definitions,
# self-contained imports, and removal of experiment/profiling utilities.

import math
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from typing import Optional, Union, Sequence, Tuple
from .base import BaseModule
from .conv_layer import ConvLayer
from .normalization import get_normalization_layer
from .linear_attention import LinearSelfAttention_improve


class MobileViTBlockv3_improve(BaseModule):
    """MobileViTv3 block selected by the complete MF-HSRM configuration.

    The image path applies HFS weighting. The supplied video path retains its
    original direct local/global concatenation. Computation is preserved from
    the source implementation."""

    def __init__(
        self,
        opts,
        in_channels: int,
        attn_unit_dim: int,
        ffn_multiplier: Optional[Union[Sequence[Union[int, float]], int, float]] = 2.0,
        n_attn_blocks: Optional[int] = 2,
        attn_dropout: Optional[float] = 0.0,
        dropout: Optional[float] = 0.0,
        ffn_dropout: Optional[float] = 0.0,
        patch_h: Optional[int] = 8,
        patch_w: Optional[int] = 8,
        conv_ksize: Optional[int] = 3,
        dilation: Optional[int] = 1,
        attn_norm_layer: Optional[str] = "layer_norm_2d",
        *args,
        **kwargs
    ) -> None:
        super(MobileViTBlockv3_improve, self).__init__()
        cnn_out_dim = attn_unit_dim
        conv_3x3_in = ConvLayer(
            opts=opts,
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=conv_ksize,
            stride=1,
            use_norm=True,
            use_act=True,
            dilation=dilation,
            groups=in_channels,
        )
        conv_1x1_in = ConvLayer(
            opts=opts,
            in_channels=in_channels,
            out_channels=cnn_out_dim,
            kernel_size=1,
            stride=1,
            use_norm=False,
            use_act=False,
        )
        self.local_rep = nn.Sequential(conv_3x3_in, conv_1x1_in)
        self.global_rep, attn_unit_dim = self._build_attn_layer(
            opts=opts,
            d_model=attn_unit_dim,
            ffn_mult=ffn_multiplier,
            n_layers=n_attn_blocks,
            attn_dropout=attn_dropout,
            dropout=dropout,
            ffn_dropout=ffn_dropout,
            attn_norm_layer=attn_norm_layer,
        )
        self.fc = nn.Linear(2 * cnn_out_dim, 2)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_proj = ConvLayer(
            opts=opts,
            in_channels=2 * cnn_out_dim,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            use_norm=True,
            use_act=False,
        )
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.patch_area = self.patch_w * self.patch_h
        self.cnn_in_dim = in_channels
        self.cnn_out_dim = cnn_out_dim
        self.transformer_in_dim = attn_unit_dim
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.ffn_dropout = ffn_dropout
        self.n_blocks = n_attn_blocks
        self.conv_ksize = conv_ksize

    def _build_attn_layer(
        self,
        opts,
        d_model,
        ffn_mult,
        n_layers,
        attn_dropout,
        dropout,
        ffn_dropout,
        attn_norm_layer,
        *args,
        **kwargs
    ):
        global_rep = [
            LinearSelfAttention_improve(
                opts=opts, embed_dim=d_model, attn_dropout=attn_dropout
            )
            for _ in range(n_layers)
        ]
        global_rep.append(
            get_normalization_layer(
                opts=opts, norm_type=attn_norm_layer, num_features=d_model
            )
        )
        return (nn.Sequential(*global_rep), d_model)

    def __repr__(self) -> str:
        repr_str = "{}(".format(self.__class__.__name__)
        repr_str += "\n\t Local representations"
        if isinstance(self.local_rep, nn.Sequential):
            for m in self.local_rep:
                repr_str += "\n\t\t {}".format(m)
        else:
            repr_str += "\n\t\t {}".format(self.local_rep)
        repr_str += "\n\t Global representations with patch size of {}x{}".format(
            self.patch_h, self.patch_w
        )
        if isinstance(self.global_rep, nn.Sequential):
            for m in self.global_rep:
                repr_str += "\n\t\t {}".format(m)
        else:
            repr_str += "\n\t\t {}".format(self.global_rep)
        if isinstance(self.conv_proj, nn.Sequential):
            for m in self.conv_proj:
                repr_str += "\n\t\t {}".format(m)
        else:
            repr_str += "\n\t\t {}".format(self.conv_proj)
        repr_str += "\n)"
        return repr_str

    def unfolding_pytorch(self, feature_map: Tensor) -> Tuple[Tensor, Tuple[int, int]]:
        batch_size, in_channels, img_h, img_w = feature_map.shape
        patches = F.unfold(
            feature_map,
            kernel_size=(self.patch_h, self.patch_w),
            stride=(self.patch_h, self.patch_w),
        )
        patches = patches.reshape(
            batch_size, in_channels, self.patch_h * self.patch_w, -1
        )
        return (patches, (img_h, img_w))

    def folding_pytorch(self, patches: Tensor, output_size: Tuple[int, int]) -> Tensor:
        batch_size, in_dim, patch_size, n_patches = patches.shape
        patches = patches.reshape(batch_size, in_dim * patch_size, n_patches)
        feature_map = F.fold(
            patches,
            output_size=output_size,
            kernel_size=(self.patch_h, self.patch_w),
            stride=(self.patch_h, self.patch_w),
        )
        return feature_map

    def resize_input_if_needed(self, x):
        batch_size, in_channels, orig_h, orig_w = x.shape
        if orig_h % self.patch_h != 0 or orig_w % self.patch_w != 0:
            new_h = int(math.ceil(orig_h / self.patch_h) * self.patch_h)
            new_w = int(math.ceil(orig_w / self.patch_w) * self.patch_w)
            x = F.interpolate(
                x, size=(new_h, new_w), mode="bilinear", align_corners=True
            )
        return x

    def forward_spatial(self, x: Tensor, *args, **kwargs) -> Tensor:
        x = self.resize_input_if_needed(x)
        fm_conv = self.local_rep(x)
        patches, output_size = self.unfolding_pytorch(fm_conv)
        patches = self.global_rep(patches)
        fm = self.folding_pytorch(patches=patches, output_size=output_size)
        pooled_local_feat = self.global_avg_pool(fm_conv).view(x.size(0), -1)
        pooled_global_feat = self.global_avg_pool(fm).view(x.size(0), -1)
        combined_pooled_feat = torch.cat([pooled_local_feat, pooled_global_feat], dim=1)
        weights = F.softmax(self.fc(combined_pooled_feat), dim=1)
        w_local, w_global = (
            weights[:, 0].view(-1, 1, 1, 1),
            weights[:, 1].view(-1, 1, 1, 1),
        )
        fused_feat = torch.cat((w_local * fm_conv, w_global * fm), dim=1)
        fm = self.conv_proj(fused_feat) + x
        return fm

    def forward_temporal(
        self, x: Tensor, x_prev: Tensor, *args, **kwargs
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Retain the source video path; x_prev is unused by the selected attention type."""
        x = self.resize_input_if_needed(x)
        fm_conv = self.local_rep(x)
        patches, output_size = self.unfolding_pytorch(fm_conv)
        for global_layer in self.global_rep:
            patches = global_layer(patches)
        fm = self.folding_pytorch(patches=patches, output_size=output_size)
        fm = self.conv_proj(torch.cat((fm, fm_conv), dim=1))
        fm = fm + x
        return (fm, patches)

    def forward(
        self, x: Union[Tensor, Tuple[Tensor]], *args, **kwargs
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        if isinstance(x, Tuple) and len(x) == 2:
            return self.forward_temporal(x=x[0], x_prev=x[1])
        elif isinstance(x, Tensor):
            return self.forward_spatial(x)
        else:
            raise NotImplementedError
