#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
# Modified for the MF-HSRM architecture-only release: selected definitions,
# self-contained imports, and removal of experiment/profiling utilities.

from torch import nn, Tensor
from typing import Any


class BaseLayer(nn.Module):
    """
    Base class for neural network layers
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def __repr__(self):
        return "{}".format(self.__class__.__name__)


class BaseModule(nn.Module):
    """Base class for all modules"""

    def __init__(self, *args, **kwargs):
        super(BaseModule, self).__init__()

    def __repr__(self):
        return "{}".format(self.__class__.__name__)


class BaseEncoder(nn.Module):
    """
    Base class for different classification models
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.conv_1 = None
        self.layer_1 = None
        self.layer_2 = None
        self.layer_3 = None
        self.layer_4 = None
        self.layer_5 = None
        self.conv_1x1_exp = None
        self.classifier = None
        self.round_nearest = 8
        self.dilation = 1
        output_stride = kwargs.get("output_stride", None)
        self.dilate_l4 = False
        self.dilate_l5 = False
        if output_stride == 8:
            self.dilate_l4 = True
            self.dilate_l5 = True
        elif output_stride == 16:
            self.dilate_l5 = True
        self.model_conf_dict = dict()

    def check_model(self):
        assert (
            self.model_conf_dict
        ), "Model configuration dictionary should not be empty"
        assert self.conv_1 is not None, "Please implement self.conv_1"
        assert self.layer_1 is not None, "Please implement self.layer_1"
        assert self.layer_2 is not None, "Please implement self.layer_2"
        assert self.layer_3 is not None, "Please implement self.layer_3"
        assert self.layer_4 is not None, "Please implement self.layer_4"
        assert self.layer_5 is not None, "Please implement self.layer_5"
        assert self.conv_1x1_exp is not None, "Please implement self.conv_1x1_exp"
        assert self.classifier is not None, "Please implement self.classifier"

    def reset_parameters(self, opts):
        """Initialize model weights"""
        from .init_utils import initialize_weights

        initialize_weights(opts=opts, modules=self.modules())

    def extract_features(self, x: Tensor, *args, **kwargs) -> Tensor:
        x = self.conv_1(x)
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)
        x = self.conv_1x1_exp(x)
        return x

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        x = self.extract_features(x)
        x = self.classifier(x)
        return x


class BaseVideoEncoder(nn.Module):
    """
    Base class for the video backbones
    """

    def __init__(self, opts, *args, **kwargs) -> None:
        super().__init__()
        self.round_nearest = 8
        self.model_conf_dict = dict()
        self.inference_mode = getattr(
            opts, "model.video_classification.inference_mode", False
        )

    def reset_parameters(self, opts) -> None:
        """Reset parameters for all modules in the network"""
        from .init_utils import initialize_weights

        initialize_weights(opts=opts, modules=self.modules())

    @staticmethod
    def reset_module_parameters(opts, module) -> None:
        """Reset parameters for a specific module in the network"""
        from .init_utils import initialize_weights

        initialize_weights(opts=opts, modules=module)
