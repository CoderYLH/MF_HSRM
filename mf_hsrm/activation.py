#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
# Modified for the MF-HSRM architecture-only release: selected definitions,
# self-contained imports, and removal of experiment/profiling utilities.

from torch import nn
from typing import Optional


class Swish(nn.SiLU):
    """
    Applies the `Swish (also known as SiLU) <https://arxiv.org/abs/1702.03118>`_ function.
    """

    def __init__(self, inplace: Optional[bool] = False) -> None:
        super().__init__(inplace=inplace)


def get_activation_fn(act_type="swish", **kwargs):
    """The activation used by both MobileViTv3 branches."""
    if act_type == "swish":
        return Swish()
    raise ValueError(
        f"Activation {act_type!r} is not part of this architecture release"
    )
