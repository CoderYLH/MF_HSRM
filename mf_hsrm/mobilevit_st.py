#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
# Modified for the MF-HSRM architecture-only release: selected definitions,
# self-contained imports, and removal of experiment/profiling utilities.

from torch import nn, Tensor
from typing import Dict, Optional
from .base import BaseVideoEncoder
from .linear_layer import LinearLayer
from .global_pool import GlobalPool
from .dropout import Dropout
from .conv_layer import ConvLayer3d
from .mobilevit_v3 import MobileViTv3


class SpatioTemporalMobileViT(BaseVideoEncoder):
    """RGB branch components used stage by stage by RGBIRFusionModelMiddle.

    This container retains the source submodules and parameter names. The full
    MF-HSRM forward method performs feature extraction and temporal aggregation."""

    os_keys = ["os_8", "os_16", "os_32"]

    def __init__(self, opts, *args, **kwargs) -> None:
        mobilevit_img_model = MobileViTv3(opts=opts)
        mobilevit_img_model.conv_1x1_exp = None
        mobilevit_img_model.classifier = None
        num_classes = getattr(opts, "model.video_classification.n_classes", 400)
        classifier_dropout = getattr(
            opts, "model.video_classification.classifier_dropout", 0.2
        )
        pool_type = getattr(opts, "model.layer.global_pool", "mean")
        super(SpatioTemporalMobileViT, self).__init__(opts=opts)
        self.mobilevit_model = mobilevit_img_model
        in_channels = mobilevit_img_model.model_conf_dict["exp_before_cls"]["in"]
        out_channels = mobilevit_img_model.model_conf_dict["exp_before_cls"]["out"]
        self.temporal_conv = nn.Sequential(
            ConvLayer3d(
                opts=opts,
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                use_norm=True,
                use_act=False,
                groups=in_channels,
            ),
            ConvLayer3d(
                opts=opts,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=1,
                use_norm=True,
                use_act=True,
            ),
        )
        self.global_pool = GlobalPool(pool_type=pool_type, keep_dim=False)
        self.classifier = nn.Sequential()
        if 0.0 < classifier_dropout < 1.0:
            self.classifier.add_module(
                name="dropout", module=Dropout(p=classifier_dropout, inplace=True)
            )
        self.classifier.add_module(
            name="fc",
            module=LinearLayer(
                in_features=out_channels, out_features=num_classes, bias=True
            ),
        )
        self.reset_module_parameters(opts=opts, module=self.temporal_conv)
        self.reset_module_parameters(opts=opts, module=self.classifier)
        self.num_classes = num_classes

    def _init_cache(self) -> Dict[str, Optional[Tensor]]:
        """Initialize a cache to store intermediate results at time step t so that they can be used at step t+1"""
        return dict(zip(self.os_keys, [None] * len(self.os_keys)))
