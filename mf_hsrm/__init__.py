"""Architecture-only MF-HSRM package."""

from .rgb_ir_middle_fusion_model import (
    RGBIRFusionModelMiddle,
    DualDimensionalCrossInteraction,
    ChannelWiseGate,
)
from .config import build_model

MFHSRM = RGBIRFusionModelMiddle
DSTCMA = DualDimensionalCrossInteraction
CWG = ChannelWiseGate
__all__ = ["build_model", "MFHSRM", "DSTCMA", "CWG"]
