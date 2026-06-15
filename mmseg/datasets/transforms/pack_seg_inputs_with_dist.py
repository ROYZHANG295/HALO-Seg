# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from mmcv.transforms import to_tensor
from mmengine.structures import PixelData

from mmseg.datasets.transforms.formatting import PackSegInputs
from mmseg.registry import TRANSFORMS


@TRANSFORMS.register_module()
class PackSegInputsWithDist(PackSegInputs):
    """Pack segmentation inputs with gt_dist_map.

    This transform extends the official PackSegInputs and additionally packs
    gt_dist_map into data_sample.gt_dist_map.
    """

    def transform(self, results):
        packed_results = super().transform(results)

        if 'gt_dist_map' in results:
            dist_map = results['gt_dist_map']

            if isinstance(dist_map, np.ndarray):
                if dist_map.ndim == 2:
                    dist_map = dist_map[None, ...]
                dist_map = to_tensor(dist_map).float()

            data_sample = packed_results['data_samples']
            data_sample.set_data({
                'gt_dist_map': PixelData(data=dist_map)
            })

        return packed_results
