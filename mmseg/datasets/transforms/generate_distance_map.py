# Copyright (c) OpenMMLab. All rights reserved.
import cv2
import numpy as np
from mmcv.transforms import BaseTransform

from mmseg.registry import TRANSFORMS


@TRANSFORMS.register_module()
class GenerateDistanceMap(BaseTransform):
    """Generate truncated inverse distance map from semantic boundary.

    This transform generates a soft boundary-distance heatmap from the
    semantic segmentation map or from the existing gt_edge_map.

    The generated gt_dist_map has:
        boundary pixels: close to 1
        pixels far from boundary: close to 0

    Args:
        radius (int): Truncation radius. Recommended values: 3, 5, 7.
        ignore_index (int): Ignore label index. Default: 255.
    """

    def __init__(self, radius=5, ignore_index=255):
        self.radius = radius
        self.ignore_index = ignore_index

    def transform(self, results):
        assert 'gt_seg_map' in results, \
            '`GenerateDistanceMap` requires gt_seg_map in results.'

        gt_seg = results['gt_seg_map']

        if 'gt_edge_map' in results:
            edge = results['gt_edge_map']
            if edge.ndim == 3:
                edge = edge.squeeze()
            edge = (edge > 0).astype(np.uint8)
        else:
            edge = self._generate_edge_from_seg(gt_seg)

        # If there is no valid boundary in this crop, return zero map.
        if edge.sum() == 0:
            dist_map = np.zeros_like(gt_seg, dtype=np.float32)
            results['gt_dist_map'] = dist_map
            return results

        # cv2.distanceTransform computes distance to zero pixels.
        # Therefore:
        #   edge pixels     -> 0
        #   non-edge pixels -> 1
        non_edge = (1 - edge).astype(np.uint8)

        dist = cv2.distanceTransform(non_edge, cv2.DIST_L2, 5)

        # Truncated inverse distance:
        #   boundary: 1
        #   farther than radius: 0
        dist_map = np.maximum(0.0, 1.0 - dist / float(self.radius))
        dist_map = dist_map.astype(np.float32)

        # Ignore regions should not contribute to distance loss.
        ignore_mask = gt_seg == self.ignore_index
        dist_map[ignore_mask] = 0.0

        results['gt_dist_map'] = dist_map

        return results

    def _generate_edge_from_seg(self, gt_seg):
        """Generate semantic boundary from segmentation map."""
        h, w = gt_seg.shape
        edge = np.zeros((h, w), dtype=np.uint8)

        valid = gt_seg != self.ignore_index

        # Horizontal difference
        diff_h = gt_seg[:, 1:] != gt_seg[:, :-1]
        valid_h = valid[:, 1:] & valid[:, :-1]
        edge[:, 1:][diff_h & valid_h] = 1
        edge[:, :-1][diff_h & valid_h] = 1

        # Vertical difference
        diff_v = gt_seg[1:, :] != gt_seg[:-1, :]
        valid_v = valid[1:, :] & valid[:-1, :]
        edge[1:, :][diff_v & valid_v] = 1
        edge[:-1, :][diff_v & valid_v] = 1

        return edge
