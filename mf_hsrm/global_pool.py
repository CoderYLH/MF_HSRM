#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
# Modified for the MF-HSRM architecture-only release: selected definitions,
# self-contained imports, and removal of experiment/profiling utilities.

import torch
from torch import Tensor
from typing import List, Optional
from .base import BaseLayer


class GlobalPool(BaseLayer):
    """
    This layers applies global pooling over a 4D or 5D input tensor

    Args:
        pool_type (Optional[str]): Pooling type. It can be mean, rms, or abs. Default: `mean`
        keep_dim (Optional[bool]): Do not squeeze the dimensions of a tensor. Default: `False`

    Shape:
        - Input: :math:`(N, C, H, W)` or :math:`(N, C, D, H, W)`
        - Output: :math:`(N, C, 1, 1)` or :math:`(N, C, 1, 1, 1)` if keep_dim else :math:`(N, C)`
    """

    pool_types = ["mean", "rms", "abs"]

    def __init__(
        self,
        pool_type: Optional[str] = "mean",
        keep_dim: Optional[bool] = False,
        *args,
        **kwargs
    ) -> None:
        super().__init__()
        if pool_type not in self.pool_types:
            raise ValueError(
                "Supported pool types are: {}. Got {}".format(
                    self.pool_types, pool_type
                )
            )
        self.pool_type = pool_type
        self.keep_dim = keep_dim

    def _global_pool(self, x: Tensor, dims: List):
        if self.pool_type == "rms":
            x = x**2
            x = torch.mean(x, dim=dims, keepdim=self.keep_dim)
            x = x ** (-0.5)
        elif self.pool_type == "abs":
            x = torch.mean(torch.abs(x), dim=dims, keepdim=self.keep_dim)
        else:
            x = torch.mean(x, dim=dims, keepdim=self.keep_dim)
        return x

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() == 4:
            dims = [-2, -1]
        elif x.dim() == 5:
            dims = [-3, -2, -1]
        else:
            raise NotImplementedError("Currently 2D and 3D global pooling supported")
        return self._global_pool(x, dims=dims)

    def __repr__(self):
        return "{}(type={})".format(self.__class__.__name__, self.pool_type)
