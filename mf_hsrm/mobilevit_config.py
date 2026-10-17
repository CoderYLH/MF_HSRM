# For licensing see accompanying LICENSE file.
# Modified for the MF-HSRM architecture-only release: selected definitions,
# self-contained imports, and removal of experiment/profiling utilities.

from typing import Dict
from .math_utils import make_divisible, bound_fn


def get_configuration(opts) -> Dict:
    width_multiplier = getattr(opts, "model.classification.mitv3.width_multiplier", 1.0)
    ffn_multiplier = 2
    mv2_exp_mult = 2
    layer_0_dim = bound_fn(min_val=16, max_val=64, value=32 * width_multiplier)
    layer_0_dim = int(make_divisible(layer_0_dim, divisor=8, min_value=16))
    config = {
        "layer0": {"img_channels": 3, "out_channels": layer_0_dim},
        "layer1": {
            "out_channels": int(make_divisible(64 * width_multiplier, divisor=16)),
            "expand_ratio": mv2_exp_mult,
            "num_blocks": 1,
            "stride": 1,
            "block_type": "mv2",
        },
        "layer2": {
            "out_channels": int(make_divisible(128 * width_multiplier, divisor=8)),
            "expand_ratio": mv2_exp_mult,
            "num_blocks": 2,
            "stride": 2,
            "block_type": "mv2",
        },
        "layer3": {
            "out_channels": int(make_divisible(256 * width_multiplier, divisor=8)),
            "attn_unit_dim": int(make_divisible(128 * width_multiplier, divisor=8)),
            "ffn_multiplier": ffn_multiplier,
            "attn_blocks": 2,
            "patch_h": 2,
            "patch_w": 2,
            "stride": 2,
            "mv_expand_ratio": mv2_exp_mult,
            "block_type": "mobilevit",
        },
        "layer4": {
            "out_channels": int(make_divisible(384 * width_multiplier, divisor=8)),
            "attn_unit_dim": int(make_divisible(192 * width_multiplier, divisor=8)),
            "ffn_multiplier": ffn_multiplier,
            "attn_blocks": 4,
            "patch_h": 2,
            "patch_w": 2,
            "stride": 2,
            "mv_expand_ratio": mv2_exp_mult,
            "block_type": "mobilevit",
        },
        "layer5": {
            "out_channels": int(make_divisible(512 * width_multiplier, divisor=8)),
            "attn_unit_dim": int(make_divisible(256 * width_multiplier, divisor=8)),
            "ffn_multiplier": ffn_multiplier,
            "attn_blocks": 3,
            "patch_h": 2,
            "patch_w": 2,
            "stride": 2,
            "mv_expand_ratio": mv2_exp_mult,
            "block_type": "mobilevit",
        },
        "last_layer_exp_factor": 4,
    }
    return config
