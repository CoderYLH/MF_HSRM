# MF-HSRM architecture

PyTorch implementation of the Multimodal Fusion Heat Stress Recognition Model
(MF-HSRM) for cage-free broiler chickens. The model combines RGB video features
with features from a paired IR temperature image.

## Release scope

This release contains the model architecture and the layers required to construct
it.

## Construction

Install PyTorch in your Python environment. With this repository on the Python
import path, construct the model as follows:

```python
from mf_hsrm import build_model

model = build_model(num_classes=3)
```

Construction uses random weights and does not download or load any files. The
complete-model architecture is selected by default: width multiplier 1.0, the
supplied improved MobileViTv3 blocks, spatial-then-temporal cross-modal fusion,
and channel-wise gating. Ablation switches and alternative models are omitted.

The code was checked with Python 3.12 and PyTorch 2.14.1 on CPU.

## Tensor interface

`model(x_video, x_image)` accepts prepared floating-point tensors:

| Tensor | Shape | Meaning |
| --- | --- | --- |
| `x_video` | `[B, T, 3, H, W]` | RGB video of a chicken head |
| `x_image` | `[B, 3, H, W]` | Corresponding three-channel IR image representation |
| Output | `[B, 3]` | Unnormalized class scores with the default configuration |

The intended input size is 224 × 224. A 32-frame RGB sequence is supported.
The original optional singleton clip dimension, `[B, 1, T, 3, H, W]`, is also
accepted. Data preparation and the association between output indices and class
labels belong to the separate dataset pipeline and are not included here.

## Architecture files

| File in `mf_hsrm/` | Role |
| --- | --- |
| `rgb_ir_middle_fusion_model.py` | Complete dual-stream model, DSTCMA and CWG |
| `mobilevit_st.py` | RGB branch components and temporal aggregation layers |
| `mobilevit_v3.py` | MobileViTv3 backbone construction |
| `mobilevit_block.py` | Selected MobileViTv3 block and HFS image-path weighting |
| `linear_attention.py` | Linear attention with depthwise context enhancement |
| `mobilevit_config.py` | Backbone stage dimensions and block counts |
| `config.py` | Architecture settings and `build_model()` |
| Other Python files | Required convolution, normalization, activation, pooling, initialization and base layers |

The original implementation names are retained. `RGBIRFusionModelMiddle`
corresponds to MF-HSRM, `DualDimensionalCrossInteraction` to DSTCMA, and
`ChannelWiseGate` to CWG. The package also exports `MFHSRM`, `DSTCMA`, and `CWG`
as aliases.

Cross-modal fusion is applied at stages 3, 4, and 5, with 256, 384, and 512
channels, respectively. RGB features query the IR feature grid; the result passes
through temporal convolution and channel-wise gating before residual addition
to the RGB features. Temporal aggregation, global pooling, and a classifier
produce the final class scores.

## Implementation fidelity

This extraction preserves the supplied implementation's computation and parameter
names. The HFS image path weights local and global features before concatenation.
The video path retains the source's direct concatenation and does not apply these
HFS weights. With the selected linear-attention implementation, the video-block
cache is passed through the interface but is not consumed by its attention layers;
temporal convolution still mixes video features.

Inherited branch classifiers remain registered for compatibility with the source
parameter structure, although MF-HSRM's forward path uses its final classifier.
In particular, the retained IR branch classifier has two outputs; the complete
MF-HSRM classifier has three. The default model contains 9,107,118 parameters.

## Attribution

This implementation is extracted and adapted from the supplied RGBIR_model
project, which incorporates Apple CVNets/MobileViT components. Original copyright
headers, `LICENSE`, and `ACKNOWLEDGEMENTS` are retained. The applicable upstream
notices are provided without assigning a new license to the authors' additions.
