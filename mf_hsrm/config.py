"""Architecture settings extracted from the supplied complete-model configuration."""

from types import SimpleNamespace


def build_model(num_classes=3):
    """Construct MF-HSRM with random weights, without loading external files."""
    from .rgb_ir_middle_fusion_model import RGBIRFusionModelMiddle

    opts = SimpleNamespace(
        **{
            # This inherited IR-branch head is retained for checkpoint compatibility.
            # MF-HSRM uses its separate num_classes-output classifier in forward().
            "model.classification.n_classes": 2,
            "model.classification.mitv3.width_multiplier": 1.0,
            "model.classification.mitv3.attn_norm_layer": "layer_norm_2d",
            "model.classification.mitv3.dropout": 0.0,
            "model.classification.mitv3.ffn_dropout": 0.0,
            "model.classification.mitv3.attn_dropout": 0.0,
            "model.video_classification.n_classes": num_classes,
            "model.video_classification.classifier_dropout": 0.1,
            "model.normalization.name": "batch_norm",
            "model.normalization.momentum": 0.1,
            "model.activation.name": "swish",
            "model.activation.inplace": False,
            "model.layer.global_pool": "mean",
            "model.layer.conv_init": "kaiming_normal",
            "model.layer.conv_init_std_dev": 0.02,
            "model.layer.linear_init": "trunc_normal",
            "model.layer.linear_init_std_dev": 0.02,
        }
    )
    return RGBIRFusionModelMiddle(opts=opts)
