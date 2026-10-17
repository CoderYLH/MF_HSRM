#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
# Modified for the MF-HSRM architecture-only release: selected definitions,
# self-contained imports, and removal of experiment/profiling utilities.

from torch import nn
from typing import Optional


class Dropout(nn.Dropout):
    """
    This layer, during training, randomly zeroes some of the elements of the input tensor with probability `p`
    using samples from a Bernoulli distribution.

    Args:
        p: probability of an element to be zeroed. Default: 0.5
        inplace: If set to ``True``, will do this operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(N, *)` where :math:`N` is the batch size
        - Output: same as the input

    """

    def __init__(
        self, p: Optional[float] = 0.5, inplace: Optional[bool] = False, *args, **kwargs
    ) -> None:
        super().__init__(p=p, inplace=inplace)
