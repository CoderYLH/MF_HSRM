#
# For licensing see accompanying LICENSE file.
# Modified for the MF-HSRM architecture-only release: selected definitions,
# self-contained imports, and removal of experiment/profiling utilities.

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from .base import BaseVideoEncoder
from .mobilevit_v3 import MobileViTv3
from .mobilevit_st import SpatioTemporalMobileViT
from .identity import Identity


class ChannelWiseGate(nn.Module):
    """Channel-wise gating (CWG) for feature tensors of shape [B, C, T, H, W]."""

    def __init__(self, dim: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(dim // reduction, dim, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        B, C, _, _, _ = x.shape
        y = self.avg_pool(x).view(B, C)
        y = self.fc(y).view(B, C, 1, 1, 1)
        return x * y


class DualDimensionalCrossInteraction(nn.Module):
    """DSTCMA: spatial RGB-query/IR-key-value attention followed by temporal convolution."""

    def __init__(self, dim: int, num_heads: int = 8, temporal_kernel: int = 3):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** (-0.5)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj_ir = nn.Linear(dim, dim, bias=False)
        self.v_proj_ir = nn.Linear(dim, dim, bias=False)
        self.out_proj_spatial = nn.Linear(dim, dim)
        padding = (temporal_kernel - 1) // 2
        self.temporal_conv = nn.Sequential(
            nn.Conv3d(
                dim,
                dim,
                kernel_size=(temporal_kernel, 1, 1),
                padding=(padding, 0, 0),
                groups=dim,
                bias=False,
            ),
            nn.GELU(),
            nn.Conv3d(dim, dim, kernel_size=1, bias=False),
        )

    def forward(self, video_seq: Tensor, ir_feat: Tensor) -> Tensor:
        """Fuse video features [B, C, T, H, W] with IR features [B, C, H_ir, W_ir]."""
        B, C, T, H, W = video_seq.shape
        if ir_feat.shape[-2:] != (H, W):
            ir_grid = F.interpolate(
                ir_feat, size=(H, W), mode="bilinear", align_corners=False
            )
        else:
            ir_grid = ir_feat
        video_tokens = video_seq.permute(0, 2, 3, 4, 1).reshape(B, T * H * W, C)
        ir_tokens = ir_grid.permute(0, 2, 3, 1).reshape(B, H * W, C)
        Q = self.q_proj(video_tokens)
        K = self.k_proj_ir(ir_tokens)
        V = self.v_proj_ir(ir_tokens)
        Q = Q.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        attn = Q @ K.transpose(-2, -1) * self.scale
        attn = attn.softmax(dim=-1)
        out_spatial = (attn @ V).transpose(1, 2).contiguous().view(B, T * H * W, C)
        out_spatial = self.out_proj_spatial(out_spatial)
        out_spatial = out_spatial.view(B, T, H, W, C).permute(0, 4, 1, 2, 3)
        temporal_fused = self.temporal_conv(out_spatial)
        return temporal_fused


class RGBIRFusionModelMiddle(BaseVideoEncoder):
    """MF-HSRM with the complete model's DSTCMA and CWG modules enabled.

    Inputs: RGB video [B, T, 3, H, W] and IR image [B, 3, H, W].
    Output: unnormalized class scores [B, num_classes]."""

    def __init__(self, opts, *args, **kwargs) -> None:
        super().__init__(opts=opts)
        self.opts = opts
        self.video_model = SpatioTemporalMobileViT(opts=opts)
        self.image_model = MobileViTv3(opts=opts)
        conf_dict = self.image_model.model_conf_dict
        dim_l3 = conf_dict["layer3"]["out"]
        dim_l4 = conf_dict["layer4"]["out"]
        dim_l5 = conf_dict["layer5"]["out"]
        self.cross_fusion_l3 = DualDimensionalCrossInteraction(dim=dim_l3)
        self.cross_fusion_l4 = DualDimensionalCrossInteraction(dim=dim_l4)
        self.cross_fusion_l5 = DualDimensionalCrossInteraction(dim=dim_l5)
        self.gate_l3 = ChannelWiseGate(dim=dim_l3)
        self.gate_l4 = ChannelWiseGate(dim=dim_l4)
        self.gate_l5 = ChannelWiseGate(dim=dim_l5)
        self.os_keys = self.video_model.os_keys
        self.temporal_conv = self.video_model.temporal_conv
        self.global_pool = self.video_model.global_pool
        out_channels = conf_dict["exp_before_cls"]["out"]
        fusion_dim = 256
        self.adaptive_video = nn.Linear(out_channels, fusion_dim)
        num_classes = getattr(opts, "model.video_classification.n_classes", 3)
        self.classifier = nn.Sequential(
            nn.ReLU(inplace=True), nn.Linear(fusion_dim, num_classes, bias=True)
        )
        self.reset_parameters(opts=opts)

    def forward(self, x_video: Tensor, x_image: Tensor) -> Tensor:
        if x_video.dim() == 6:
            x_video = x_video.squeeze(1)
        assert x_video.dim() == 5, f"Expected 5D video input, got {x_video.dim()}D"
        B, T, C, H, W = x_video.shape
        ir = self.image_model.conv_1(x_image)
        ir = self.image_model.layer_1(ir)
        ir = self.image_model.layer_2(ir)
        frame_inputs = []
        mobilevit_img = self.video_model.mobilevit_model
        for t in range(T):
            frame = x_video[:, t]
            frame = mobilevit_img.conv_1(frame)
            frame = mobilevit_img.layer_1(frame)
            frame = mobilevit_img.layer_2(frame)
            frame_inputs.append(frame)
        spatial_cache = self.video_model._init_cache()
        cross_fusions = [
            self.cross_fusion_l3,
            self.cross_fusion_l4,
            self.cross_fusion_l5,
        ]
        gates = [self.gate_l3, self.gate_l4, self.gate_l5]
        for stage_idx in range(3):
            stage = stage_idx + 3
            os_key = self.os_keys[stage_idx]
            ir = getattr(self.image_model, f"layer_{stage}")(ir)
            video_layer = getattr(mobilevit_img, f"layer_{stage}")
            has_inverted = len(video_layer) > 1
            inverted = video_layer[0] if has_inverted else Identity()
            mit_block = video_layer[-1]
            stage_outputs = []
            for t in range(T):
                frame = frame_inputs[t]
                frame = inverted(frame)
                cached_fm = (
                    spatial_cache[os_key] if spatial_cache[os_key] is not None else None
                )
                frame, new_cache = mit_block((frame, cached_fm))
                stage_outputs.append(frame)
                spatial_cache[os_key] = new_cache
            video_seq = torch.stack(stage_outputs, dim=2)
            fused = cross_fusions[stage_idx](video_seq, ir)
            fused = gates[stage_idx](fused)
            video_seq = video_seq + fused
            frame_inputs = [video_seq[:, :, t, :, :] for t in range(T)]
        outputs = video_seq
        outputs = self.temporal_conv(outputs)
        outputs = self.global_pool(outputs)
        outputs = self.adaptive_video(outputs)
        logits = self.classifier(outputs)
        return logits
