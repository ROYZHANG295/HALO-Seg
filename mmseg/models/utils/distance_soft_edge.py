# import numpy as np
# from scipy.ndimage import distance_transform_edt
# from mmcv.transforms import BaseTransform
# from mmseg.registry import TRANSFORMS

# @TRANSFORMS.register_module()
# class DistanceTransformEdge(BaseTransform):
#     """
#     将二值硬边界转化为高斯平滑的软边界 (Distance Transform Soft Edge)
#     """
#     def __init__(self, sigma=3.0):
#         # sigma 控制边界的平滑程度。sigma 越大，软边界越宽。
#         self.sigma = sigma

#     def transform(self, results):
#         # 1. 拦截上一步 GenerateEdge 生成的二值边界 (0 是背景，1 是边界)
#         edge_map = results['gt_edge_map']
        
#         # 2. scipy 的 edt 函数计算的是到 0 的距离。
#         # 所以我们要计算每个像素到边界 (值为 1) 的距离，需要把边界设为 0，背景设为 1
#         inverted_edge = (edge_map == 0).astype(np.uint8)
        
#         # 3. 计算距离场 (Distance Map)
#         # dist_map 记录了每个像素到最近边界的欧氏距离
#         dist_map = distance_transform_edt(inverted_edge)
        
#         # 4. 高斯衰减，将距离转化为 [0, 1] 的连续概率分布
#         soft_edge = np.exp(-(dist_map ** 2) / (2 * self.sigma ** 2))
        
#         # 5. 覆盖原来的硬边界 (转换为 float32 以支持软标签 Loss)
#         results['gt_edge_map'] = soft_edge.astype(np.float32)
        
#         return results


import cv2
import numpy as np
from mmcv.transforms import BaseTransform
from mmseg.registry import TRANSFORMS

@TRANSFORMS.register_module()
class DistanceTransformEdge(BaseTransform):
    """
    使用 OpenCV 工业级加速的距离变换软边界
    """
    def __init__(self, sigma=3.0):
        self.sigma = sigma

    def transform(self, results):
        edge_map = results['gt_edge_map']
        
        # 1. OpenCV 的 distanceTransform 计算的是非零像素到零像素的距离。
        # 原来边界是 1，背景是 0。
        # 我们要把边界变成 0，背景变成非零（比如 255），这样才能算背景到边界的距离。
        inverted_edge = np.where(edge_map == 1, 0, 255).astype(np.uint8)
        
        # 2. 核心提速点：调用 OpenCV 的距离变换
        # cv2.DIST_L2 表示欧氏距离
        # 5 表示掩码大小 (3x3 较快但略粗糙，5x5 精度极高且速度依然极快)
        dist_map = cv2.distanceTransform(inverted_edge, cv2.DIST_L2, 5)
        
        # 3. 高斯衰减，生成软标签
        soft_edge = np.exp(-(dist_map ** 2) / (2 * self.sigma ** 2))
        
        # 4. 覆盖原来的硬边界
        results['gt_edge_map'] = soft_edge.astype(np.float32)
        
        return results
    

