#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
# Modified for the MF-HSRM architecture-only release: selected definitions,
# self-contained imports, and removal of experiment/profiling utilities.

from torch import Tensor
from .base import BaseLayer


class Identity(BaseLayer):
    """
    This is a place-holder and returns the same tensor.
    """

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x
