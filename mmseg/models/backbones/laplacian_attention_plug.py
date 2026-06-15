import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.registry import MODELS


class LaplacianChannelAttention(nn.Module):
    """
    LCRA: Laplacian Channel Residual Algorithm (已修复维度 Bug)
    """
    def __init__(self, channels: int, gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        # 1. 提取通道特征：使用全局最大值 (Global Max Pooling)
        channel_feat = x.amax(dim=(2, 3))  # [B, C]
        
        # 2. 修复：计算通道间的欧氏距离矩阵 (利用广播机制)
        # 对于标量，欧氏距离就是绝对值差
        # c1: [B, C, 1], c2: [B, 1, C]
        c1 = channel_feat.unsqueeze(2)
        c2 = channel_feat.unsqueeze(1)
        dist = torch.abs(c1 - c2)  # [B, C, C]
        
        # 3. 高斯核构建相似性矩阵
        sim = torch.exp(-dist ** 2 / (2 * self.gamma ** 2))
        
        # 4. 构建拉普拉斯矩阵 L = D - A
        degree = sim.sum(dim=-1, keepdim=True)  # 度矩阵 D
        laplacian = degree - sim  # 拉普拉斯矩阵 L
        
        # 5. 计算注意力并加权
        attn = torch.softmax(-laplacian, dim=-1)
        x_reshaped = x.view(B, C, -1)  # [B, C, N]
        out = torch.bmm(attn, x_reshaped).view(B, C, H, W)
        
        return out + x  # 残差连接


class LightweightSpatialAttention(nn.Module):
    """
    轻量级空间注意力替代方案
    """
    def __init__(self, channels: int, kernel_size: int = 7):
        super().__init__()
        self.dw_conv = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=channels,
            bias=False
        )
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dw_conv(x)
        x = self.bn(x)
        return x + residual


@MODELS.register_module()
class LaplacianAttention(nn.Module):
    """
    最终版：即插即用的 Laplacian Attention
    """
    def __init__(self, channels: int, gamma_c: float = 1.0, spatial_kernel: int = 7):
        super().__init__()
        self.channel_attn = LaplacianChannelAttention(channels, gamma_c)
        self.spatial_attn = LightweightSpatialAttention(channels, spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x