#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
# Modified for the MF-HSRM architecture-only release: selected definitions,
# self-contained imports, and removal of experiment/profiling utilities.

from torch import nn, Tensor
from typing import Optional, Union, Tuple
from .base import BaseLayer
from .normalization import get_normalization_layer
from .activation import get_activation_fn


class Conv2d(nn.Conv2d):
    """
    Applies a 2D convolution over an input

    Args:
        in_channels (int): :math:`C_{in}` from an expected input of size :math:`(N, C_{in}, H_{in}, W_{in})`
        out_channels (int): :math:`C_{out}` from an expected output of size :math:`(N, C_{out}, H_{out}, W_{out})`
        kernel_size (Union[int, Tuple[int, int]]): Kernel size for convolution.
        stride (Union[int, Tuple[int, int]]): Stride for convolution. Defaults to 1
        padding (Union[int, Tuple[int, int]]): Padding for convolution. Defaults to 0
        dilation (Union[int, Tuple[int, int]]): Dilation rate for convolution. Default: 1
        groups (Optional[int]): Number of groups in convolution. Default: 1
        bias (bool): Use bias. Default: ``False``
        padding_mode (Optional[str]): Padding mode. Default: ``zeros``

        use_norm (Optional[bool]): Use normalization layer after convolution. Default: ``True``
        use_act (Optional[bool]): Use activation layer after convolution (or convolution and normalization).
                                Default: ``True``
        act_name (Optional[str]): Use specific activation function. Overrides the one specified in command line args.

    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})`
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Optional[Union[int, Tuple[int, int]]] = 1,
        padding: Optional[Union[int, Tuple[int, int]]] = 0,
        dilation: Optional[Union[int, Tuple[int, int]]] = 1,
        groups: Optional[int] = 1,
        bias: Optional[bool] = False,
        padding_mode: Optional[str] = "zeros",
        *args,
        **kwargs
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )


class ConvLayer(BaseLayer):
    """
    Applies a 2D convolution over an input

    Args:
        opts: command line arguments
        in_channels (int): :math:`C_{in}` from an expected input of size :math:`(N, C_{in}, H_{in}, W_{in})`
        out_channels (int): :math:`C_{out}` from an expected output of size :math:`(N, C_{out}, H_{out}, W_{out})`
        kernel_size (Union[int, Tuple[int, int]]): Kernel size for convolution.
        stride (Union[int, Tuple[int, int]]): Stride for convolution. Default: 1
        dilation (Union[int, Tuple[int, int]]): Dilation rate for convolution. Default: 1
        groups (Optional[int]): Number of groups in convolution. Default: 1
        bias (Optional[bool]): Use bias. Default: ``False``
        padding_mode (Optional[str]): Padding mode. Default: ``zeros``
        use_norm (Optional[bool]): Use normalization layer after convolution. Default: ``True``
        use_act (Optional[bool]): Use activation layer after convolution (or convolution and normalization).
                                Default: ``True``
        act_name (Optional[str]): Use specific activation function. Overrides the one specified in command line args.

    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})`

    .. note::
        For depth-wise convolution, `groups=C_{in}=C_{out}`.
    """

    def __init__(
        self,
        opts,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Optional[Union[int, Tuple[int, int]]] = 1,
        dilation: Optional[Union[int, Tuple[int, int]]] = 1,
        groups: Optional[int] = 1,
        bias: Optional[bool] = False,
        padding_mode: Optional[str] = "zeros",
        use_norm: Optional[bool] = True,
        use_act: Optional[bool] = True,
        act_name: Optional[str] = None,
        *args,
        **kwargs
    ) -> None:
        super().__init__()
        if use_norm:
            norm_type = getattr(opts, "model.normalization.name", "batch_norm")
            if norm_type is not None and norm_type.find("batch") > -1:
                assert not bias, "Do not use bias when using normalization layers."
            elif norm_type is not None and norm_type.find("layer") > -1:
                bias = True
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)
        assert isinstance(kernel_size, Tuple)
        assert isinstance(stride, Tuple)
        assert isinstance(dilation, Tuple)
        padding = (
            int((kernel_size[0] - 1) / 2) * dilation[0],
            int((kernel_size[1] - 1) / 2) * dilation[1],
        )
        if in_channels % groups != 0:
            raise ValueError(
                "Input channels are not divisible by groups. {}%{} != 0 ".format(
                    in_channels, groups
                )
            )
        if out_channels % groups != 0:
            raise ValueError(
                "Output channels are not divisible by groups. {}%{} != 0 ".format(
                    out_channels, groups
                )
            )
        block = nn.Sequential()
        conv_layer = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )
        block.add_module(name="conv", module=conv_layer)
        self.norm_name = None
        if use_norm:
            norm_layer = get_normalization_layer(opts=opts, num_features=out_channels)
            block.add_module(name="norm", module=norm_layer)
            self.norm_name = norm_layer.__class__.__name__
        self.act_name = None
        act_type = (
            getattr(opts, "model.activation.name", "prelu")
            if act_name is None
            else act_name
        )
        if act_type is not None and use_act:
            neg_slope = getattr(opts, "model.activation.neg_slope", 0.1)
            inplace = getattr(opts, "model.activation.inplace", False)
            act_layer = get_activation_fn(
                act_type=act_type,
                inplace=inplace,
                negative_slope=neg_slope,
                num_parameters=out_channels,
            )
            block.add_module(name="act", module=act_layer)
            self.act_name = act_layer.__class__.__name__
        self.block = block
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups
        self.kernel_size = conv_layer.kernel_size
        self.bias = bias
        self.dilation = dilation

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)

    def __repr__(self):
        repr_str = self.block[0].__repr__()
        repr_str = repr_str[:-1]
        if self.norm_name is not None:
            repr_str += ", normalization={}".format(self.norm_name)
        if self.act_name is not None:
            repr_str += ", activation={}".format(self.act_name)
        repr_str += ", bias={}".format(self.bias)
        repr_str += ")"
        return repr_str


class ConvLayer3d(BaseLayer):
    """
    Applies a 3D convolution over an input

    Args:
        opts: command line arguments
        in_channels (int): :math:`C_{in}` from an expected input of size :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})`
        out_channels (int): :math:`C_{out}` from an expected output of size :math:`(N, C_{out}, D_{out}, H_{out}, W_{out})`
        kernel_size (Union[int, Tuple[int, int]]): Kernel size for convolution.
        stride (Union[int, Tuple[int, int]]): Stride for convolution. Default: 1
        dilation (Union[int, Tuple[int, int]]): Dilation rate for convolution. Default: 1
        groups (Optional[int]): Number of groups in convolution. Default: 1
        bias (Optional[bool]): Use bias. Default: ``False``
        padding_mode (Optional[str]): Padding mode. Default: ``zeros``
        use_norm (Optional[bool]): Use normalization layer after convolution. Default: ``True``
        use_act (Optional[bool]): Use activation layer after convolution (or convolution and normalization).
                                Default: ``True``

    Shape:
        - Input: :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, D_{out}, H_{out}, W_{out})`

    .. note::
        For depth-wise convolution, `groups=C_{in}=C_{out}`.
    """

    def __init__(
        self,
        opts,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple],
        stride: Optional[Union[int, Tuple]] = 1,
        dilation: Optional[Union[int, Tuple]] = 1,
        groups: Optional[int] = 1,
        bias: Optional[bool] = False,
        padding_mode: Optional[str] = "zeros",
        use_norm: Optional[bool] = True,
        use_act: Optional[bool] = True,
        *args,
        **kwargs
    ) -> None:
        super().__init__()
        if use_norm:
            assert not bias, "Do not use bias when using normalization layers."
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        if isinstance(dilation, int):
            dilation = (dilation, dilation, dilation)
        assert isinstance(kernel_size, (tuple, list))
        assert isinstance(stride, (tuple, list))
        assert isinstance(dilation, (tuple, list))
        padding = tuple([int((kernel_size[i] - 1) / 2) * dilation[i] for i in range(3)])
        if in_channels % groups != 0:
            raise ValueError(
                "Input channels are not divisible by groups. {}%{} != 0 ".format(
                    in_channels, groups
                )
            )
        if out_channels % groups != 0:
            raise ValueError(
                "Output channels are not divisible by groups. {}%{} != 0 ".format(
                    out_channels, groups
                )
            )
        block = nn.Sequential()
        conv_layer = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )
        block.add_module(name="conv", module=conv_layer)
        self.norm_name = None
        norm_type = getattr(opts, "model.normalization.name", "batch_norm")
        if use_norm and norm_type is not None:
            if norm_type.find("batch") > -1:
                norm_type = "batch_norm_3d"
            norm_layer = get_normalization_layer(
                opts=opts, num_features=out_channels, norm_type=norm_type
            )
            block.add_module(name="norm", module=norm_layer)
            self.norm_name = norm_layer.__class__.__name__
        self.act_name = None
        act_type = getattr(opts, "model.activation.name", "prelu")
        if act_type is not None and use_act:
            neg_slope = getattr(opts, "model.activation.neg_slope", 0.1)
            inplace = getattr(opts, "model.activation.inplace", False)
            act_layer = get_activation_fn(
                act_type=act_type,
                inplace=inplace,
                negative_slope=neg_slope,
                num_parameters=out_channels,
            )
            block.add_module(name="act", module=act_layer)
            self.act_name = act_layer.__class__.__name__
        self.block = block
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups
        self.kernel_size = conv_layer.kernel_size
        self.bias = bias
        self.dilation = dilation

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)

    def __repr__(self):
        repr_str = self.block[0].__repr__()
        repr_str = repr_str[:-1]
        if self.norm_name is not None:
            repr_str += ", normalization={}".format(self.norm_name)
        if self.act_name is not None:
            repr_str += ", activation={}".format(self.act_name)
        repr_str += ")"
        return repr_str
