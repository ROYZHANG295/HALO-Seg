# Copyright (c) OpenMMLab. All rights reserved.
# This file is an extension, not a modification of the original.
from typing import List, Tuple

from mmseg.registry import MODELS
from .bisenetv2 import BiSeNetV2  # Directly import the original BiSeNetV2
import torch


@MODELS.register_module()
class ExposedBiSeNetV2(BiSeNetV2):
    """
    ===========================================================================
    【插件式扩展 - HALO 专用】Exposed BiSeNetV2
    ===========================================================================
    This class inherits directly from the official BiSeNetV2 backbone.
    Its SOLE purpose is to override the `forward` method to expose the
    `x_detail` feature map, which is discarded in the original implementation.

    This allows our HALO/Dynamic head to receive the detail feature for
    asymmetric Laplacian supervision, without modifying the original source code.
    This ensures 100% backward compatibility with baseline configurations.
    """

    def __init__(self, **kwargs):
        # We don't need to write anything here.
        # super().__init__() will automatically call the original BiSeNetV2's
        # __init__ method and build the network exactly as before.
        super().__init__(**kwargs)

    def forward(self, x: Tuple[List[torch.Tensor]]) -> Tuple[List[torch.Tensor]]:
        """
        Overrides the original forward pass to expose `x_detail`.
        """
        # The following code is a direct copy of the original BiSeNetV2 forward pass.
        x_detail = self.detail(x)
        x_semantic_lst = self.semantic(x)
        x_head = self.bga(x_detail, x_semantic_lst[-1])

        # --- CRITICAL MODIFICATION ---
        # The original code is:
        #   outs = [x_head] + x_semantic_lst[:-1]
        # This discards `x_detail`.
        #
        # Our modified version appends `x_detail` to the end of the list,
        # making it accessible via `out_indices`.
        #
        # Index Mapping:
        #   - outs[0]: `x_head` (BGA Fused Output)
        #   - outs[1-3]: `x_semantic_lst` intermediate stages
        #   - outs[4]: `x_semantic_lst` final stage before BGA
        #   - outs[5]: `x_detail` (Our exposed Detail Branch Output)
        # -----------------------------
        outs = [x_head] + x_semantic_lst + [x_detail]

        # The rest of the logic remains identical, filtering by out_indices.
        if self.out_indices is not None:
            outs = [outs[i] for i in self.out_indices]

        return tuple(outs)

