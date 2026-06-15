# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, build_norm_layer
from mmengine.model import BaseModule
import torch.nn.functional as F  # 🔥 [修改点 0]: 引入 functional 库

from mmseg.models.utils import DAPPM, BasicBlock, Bottleneck, resize
from mmseg.registry import MODELS
from mmseg.utils import OptConfigType

#########################################################################
# ######## [修改点 1] 新增 LaplacianAttention 类 (放在 PIDNet 之前) ########
#########################################################################
class LaplacianAttention(nn.Module):
    """
    拉普拉斯边缘注意力模块
    作用：提取边缘 -> 生成权重图 -> 用于增强特征
    """
    def __init__(self, out_channels):
        super().__init__()
        # 1. 定义拉普拉斯卷积核 (固定不可学习)
        # 这是一个经典的边缘检测算子
        kernel = torch.tensor([[0., 1., 0.],
                               [1., -4., 1.],
                               [0., 1., 0.]], dtype=torch.float32)
        # register_buffer 保证它会自动转到 GPU，且随模型保存，但不会被梯度更新
        self.register_buffer('kernel', kernel.view(1, 1, 3, 3))
        
        # 2. 通道调整层 (1x1 Conv)
        # 把 1通道的边缘图 -> 映射成 和特征图一样的通道数 (out_channels)
        self.conv_adjust = nn.Conv2d(1, out_channels, kernel_size=1)

    def init_zero(self):
        # 1. 权重依然设为 0
        nn.init.constant_(self.conv_adjust.weight, 0)
        
        # 2. 🔥【核心修改】偏置设为 -6.0
        # 这样 Sigmoid(-6.0) ≈ 0.0024
        # 初始时刻: Feature * (1 + 0.0024) ≈ Feature (几乎无感)
        if self.conv_adjust.bias is not None:
            nn.init.constant_(self.conv_adjust.bias, -6.0)

    def forward(self, x_raw, target_shape):
        """
        x_raw: 原始 RGB 输入 (B, 3, H, W)
        target_shape: 目标特征图的大小 (H_out, W_out)
        """
        # 1. RGB 转 灰度 (B, 1, H, W)
        # 维度变化: (B, 3, H, W) -> (B, 1, H, W)
        x_gray = 0.299 * x_raw[:, 0:1] + 0.587 * x_raw[:, 1:2] + 0.114 * x_raw[:, 2:3]
        
        # 2. 提取边缘 (B, 1, H, W)
        # 经过拉普拉斯卷积，边缘处数值大，平坦处数值接近 0
        edge = F.conv2d(x_gray, self.kernel, padding=1)
        edge = torch.abs(edge) # 取绝对值
        
        # 3. 下采样到目标尺寸 (B, 1, H_out, W_out)
        # 因为特征图比原图小，所以要把边缘图缩放对齐
        edge_resized = F.interpolate(edge, size=target_shape, mode='bilinear', align_corners=False)
        
        # 4. 生成注意力权重 (B, out_channels, H_out, W_out)
        # 1x1 卷积调整通道 -> Sigmoid 归一化到 (0, 1)
        # 维度变化: (B, 1, ...) -> (B, out_channels, ...)
        attention = torch.sigmoid(self.conv_adjust(edge_resized))
        
        return attention


@MODELS.register_module()
class DDRNetLaplacianAttentionSZero(BaseModule):
    """DDRNet backbone.

    This backbone is the implementation of `Deep Dual-resolution Networks for
    Real-time and Accurate Semantic Segmentation of Road Scenes
    <http://arxiv.org/abs/2101.06085>`_.
    Modified from https://github.com/ydhongHIT/DDRNet.

    Args:
        in_channels (int): Number of input image channels. Default: 3.
        channels: (int): The base channels of DDRNet. Default: 32.
        ppm_channels (int): The channels of PPM module. Default: 128.
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False.
        norm_cfg (dict): Config dict to build norm layer.
            Default: dict(type='BN', requires_grad=True).
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU', inplace=True).
        init_cfg (dict, optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 in_channels: int = 3,
                 channels: int = 32,
                 ppm_channels: int = 128,
                 align_corners: bool = False,
                 norm_cfg: OptConfigType = dict(type='BN', requires_grad=True),
                 act_cfg: OptConfigType = dict(type='ReLU', inplace=True),
                 init_cfg: OptConfigType = None):
        super().__init__(init_cfg)

        self.in_channels = in_channels
        self.ppm_channels = ppm_channels

        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.align_corners = align_corners

        # stage 0-2
        self.stem = self._make_stem_layer(in_channels, channels, num_blocks=2)
        self.relu = nn.ReLU()

        # low resolution(context) branch
        self.context_branch_layers = nn.ModuleList()
        for i in range(3):
            self.context_branch_layers.append(
                self._make_layer(
                    block=BasicBlock if i < 2 else Bottleneck,
                    inplanes=channels * 2**(i + 1),
                    planes=channels * 8 if i > 0 else channels * 4,
                    num_blocks=2 if i < 2 else 1,
                    stride=2))

        # bilateral fusion
        self.compression_1 = ConvModule(
            channels * 4,
            channels * 2,
            kernel_size=1,
            norm_cfg=self.norm_cfg,
            act_cfg=None)
        self.down_1 = ConvModule(
            channels * 2,
            channels * 4,
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=self.norm_cfg,
            act_cfg=None)

        self.compression_2 = ConvModule(
            channels * 8,
            channels * 2,
            kernel_size=1,
            norm_cfg=self.norm_cfg,
            act_cfg=None)
        self.down_2 = nn.Sequential(
            ConvModule(
                channels * 2,
                channels * 4,
                kernel_size=3,
                stride=2,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg),
            ConvModule(
                channels * 4,
                channels * 8,
                kernel_size=3,
                stride=2,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=None))

        # high resolution(spatial) branch
        self.spatial_branch_layers = nn.ModuleList()
        for i in range(3):
            self.spatial_branch_layers.append(
                self._make_layer(
                    block=BasicBlock if i < 2 else Bottleneck,
                    inplanes=channels * 2,
                    planes=channels * 2,
                    num_blocks=2 if i < 2 else 1,
                ))

        # ==================================================================
        # 🔥 [修改点 2]: 初始化你的 LaplacianAttention
        # ==================================================================
        # 这里的 channels * 2 对应 Spatial 分支的通道数 (例如 32*2=64)
        self.lap_attn = LaplacianAttention(out_channels=channels * 2)
        # 手动调用初始化，确保开始时权重为0
        self.lap_attn.init_zero()

        self.spp = DAPPM(
            channels * 16, ppm_channels, channels * 4, num_scales=5)

    def _make_stem_layer(self, in_channels, channels, num_blocks):
        layers = [
            ConvModule(
                in_channels,
                channels,
                kernel_size=3,
                stride=2,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg),
            ConvModule(
                channels,
                channels,
                kernel_size=3,
                stride=2,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        ]

        layers.extend([
            self._make_layer(BasicBlock, channels, channels, num_blocks),
            nn.ReLU(),
            self._make_layer(
                BasicBlock, channels, channels * 2, num_blocks, stride=2),
            nn.ReLU(),
        ])

        return nn.Sequential(*layers)

    def _make_layer(self, block, inplanes, planes, num_blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False),
                build_norm_layer(self.norm_cfg, planes * block.expansion)[1])

        layers = [
            block(
                in_channels=inplanes,
                channels=planes,
                stride=stride,
                downsample=downsample)
        ]
        inplanes = planes * block.expansion
        for i in range(1, num_blocks):
            layers.append(
                block(
                    in_channels=inplanes,
                    channels=planes,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg_out=None if i == num_blocks - 1 else self.act_cfg))

        return nn.Sequential(*layers)

    def forward(self, x):
        """Forward function."""

        # ==================================================================
        # 🔥 [修改点 3]: 保存原始输入 x (raw image)
        # ==================================================================
        x_raw = x 

        out_size = (x.shape[-2] // 8, x.shape[-1] // 8)

        # stage 0-2
        x = self.stem(x)

        # stage3
        x_c = self.context_branch_layers[0](x)
        x_s = self.spatial_branch_layers[0](x)

        # ==================================================================
        # 🔥 [修改点 4]: 调用 LaplacianAttention 并增强特征
        # ==================================================================
        # 1. 传入原始图片 x_raw 和 目标尺寸
        attn_map = self.lap_attn(x_raw, target_shape=x_s.shape[2:])
        
        # 2. 增强特征：x_s * (1 + attention)
        # 这样边缘区域会被放大，非边缘区域保持原样 (或轻微放大)
        x_s = x_s * (1 + attn_map)

        comp_c = self.compression_1(self.relu(x_c))
        x_c += self.down_1(self.relu(x_s))
        x_s += resize(
            comp_c,
            size=out_size,
            mode='bilinear',
            align_corners=self.align_corners)
        if self.training:
            temp_context = x_s.clone()

        # stage4
        x_c = self.context_branch_layers[1](self.relu(x_c))
        x_s = self.spatial_branch_layers[1](self.relu(x_s))
        comp_c = self.compression_2(self.relu(x_c))
        x_c += self.down_2(self.relu(x_s))
        x_s += resize(
            comp_c,
            size=out_size,
            mode='bilinear',
            align_corners=self.align_corners)

        # stage5
        x_s = self.spatial_branch_layers[2](self.relu(x_s))
        x_c = self.context_branch_layers[2](self.relu(x_c))
        x_c = self.spp(x_c)
        x_c = resize(
            x_c,
            size=out_size,
            mode='bilinear',
            align_corners=self.align_corners)

        return (temp_context, x_s + x_c) if self.training else x_s + x_c
