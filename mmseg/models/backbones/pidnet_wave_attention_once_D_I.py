# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from mmengine.runner import CheckpointLoader
from torch import Tensor
import cv2
import numpy as np
import os

from mmseg.registry import MODELS
from mmseg.utils import OptConfigType
from ..utils import DAPPM, PAPPM, BasicBlock, Bottleneck

class WaveletSemanticAttention(nn.Module):
    def __init__(self):
        super().__init__()
        # 🌟 修复：改为 0.25，确保是均值滤波，与原图 x 尺度对齐
        weight_ll = torch.tensor(
            [[0.25, 0.25], [0.25, 0.25]], dtype=torch.float32
        ).view(1, 1, 2, 2)
        self.register_buffer('weight_ll', weight_ll)
        self.scale = nn.Parameter(torch.tensor(-4.0))

        self.step_counter = 0  # 🌟 改用计数器

    def forward(self, x):
        B, C, H, W = x.shape

        pad_h = H % 2
        pad_w = W % 2
        x_padded = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')
        x_reshaped = x_padded.reshape(B * C, 1, H + pad_h, W + pad_w)

        ll = F.conv2d(x_reshaped, self.weight_ll, stride=2)
        ll_up = F.interpolate(ll, size=(H, W), mode='bilinear', align_corners=False)
        ll_up = ll_up.reshape(B, C, H, W)

        diff = torch.abs(ll_up - x)
        
        mu = diff.mean(dim=(2, 3), keepdim=True)
        sigma = diff.std(dim=(2, 3), keepdim=True).clamp(min=1e-5)
        
        # diff 越小（越平坦），-(diff - mu)/sigma 越大，sigmoid 后越接近 1
        attn = torch.sigmoid(-(diff - mu) / sigma)

        # 🌟 新增：在训练模式下，且还没保存过时，执行保存
        if self.training:
            self.step_counter += 1
            if self.step_counter == 6000:  # 等网络学了一段时间后再看
                self._save_heatmap(attn, "wavelet_semantic_low_freq.png")

        return x + x * attn * torch.sigmoid(self.scale)
    
    def _save_heatmap(self, attn, filename):
        """将注意力张量转换为热力图并保存到本地"""
        # 取 Batch 的第 0 张图，在通道维度求平均，并转到 CPU 转 numpy
        attn_map = attn[0].mean(dim=0).detach().cpu().numpy()
        
        # 归一化到 0~1
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
        # 转换为 0~255 的 uint8 格式
        attn_map_uint8 = (attn_map * 255).astype(np.uint8)
        
        # 应用伪彩色 (Jet: 红高蓝低)
        heatmap = cv2.applyColorMap(attn_map_uint8, cv2.COLORMAP_JET)
        
        # 创建文件夹并保存
        save_dir = "wavelet_vis_results"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename)
        cv2.imwrite(save_path, heatmap)
        print(f"\n[可视化导出] 成功保存小波语义注意力图至: {save_path}\n")

class WaveletBoundaryAttention(nn.Module):
    def __init__(self):
        super().__init__()
        # 高通滤波器求和为 0，不受绝对尺度影响，保持 0.5 即可
        weight_lh = torch.tensor([[-0.5, -0.5], [0.5, 0.5]], dtype=torch.float32).view(1, 1, 2, 2)
        weight_hl = torch.tensor([[-0.5, 0.5], [-0.5, 0.5]], dtype=torch.float32).view(1, 1, 2, 2)
        weight_hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]], dtype=torch.float32).view(1, 1, 2, 2)

        self.register_buffer('weight_lh', weight_lh)
        self.register_buffer('weight_hl', weight_hl)
        self.register_buffer('weight_hh', weight_hh)
        self.scale = nn.Parameter(torch.tensor(-4.0))
        
        self.step_counter = 0  # 🌟 改用计数器

    def forward(self, x):
        B, C, H, W = x.shape

        pad_h = H % 2
        pad_w = W % 2
        x_padded = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')
        x_reshaped = x_padded.reshape(B * C, 1, H + pad_h, W + pad_w)

        lh = F.conv2d(x_reshaped, self.weight_lh, stride=2)
        hl = F.conv2d(x_reshaped, self.weight_hl, stride=2)
        hh = F.conv2d(x_reshaped, self.weight_hh, stride=2)

        mag = torch.sqrt(lh**2 + hl**2 + hh**2 + 1e-8)
        mag_up = F.interpolate(mag, size=(H, W), mode='bilinear', align_corners=False)
        mag_up = mag_up.reshape(B, C, H, W)

        mu = mag_up.mean(dim=(2, 3), keepdim=True)
        sigma = mag_up.std(dim=(2, 3), keepdim=True).clamp(min=1e-5)
        
        # mag 越大（边界越强），(mag - mu)/sigma 越大，sigmoid 后越接近 1
        attn = torch.sigmoid((mag_up - mu) / sigma)

        # 🌟 新增：在训练模式下，且还没保存过时，执行保存
        if self.training:
            self.step_counter += 1
            if self.step_counter == 6000:  # 等网络学了一段时间后再看
                self._save_heatmap(attn, "wavelet_boundary_high_freq.png")

        return x + x * attn * torch.sigmoid(self.scale)
    
    def _save_heatmap(self, attn, filename):
        """将注意力张量转换为热力图并保存到本地"""
        attn_map = attn[0].mean(dim=0).detach().cpu().numpy()
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
        attn_map_uint8 = (attn_map * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(attn_map_uint8, cv2.COLORMAP_JET)
        
        save_dir = "wavelet_vis_results"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename)
        cv2.imwrite(save_path, heatmap)
        print(f"\n[可视化导出] 成功保存小波边界注意力图至: {save_path}\n")


class PagFM(BaseModule):
    """Pixel-attention-guided fusion module.

    Args:
        in_channels (int): The number of input channels.
        channels (int): The number of channels.
        after_relu (bool): Whether to use ReLU before attention.
            Default: False.
        with_channel (bool): Whether to use channel attention.
            Default: False.
        upsample_mode (str): The mode of upsample. Default: 'bilinear'.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Default: dict(typ='ReLU', inplace=True).
        init_cfg (dict): Config dict for initialization. Default: None.
    """

    def __init__(self,
                 in_channels: int,
                 channels: int,
                 after_relu: bool = False,
                 with_channel: bool = False,
                 upsample_mode: str = 'bilinear',
                 norm_cfg: OptConfigType = dict(type='BN'),
                 act_cfg: OptConfigType = dict(typ='ReLU', inplace=True),
                 init_cfg: OptConfigType = None):
        super().__init__(init_cfg)
        self.after_relu = after_relu
        self.with_channel = with_channel
        self.upsample_mode = upsample_mode
        self.f_i = ConvModule(
            in_channels, channels, 1, norm_cfg=norm_cfg, act_cfg=None)
        self.f_p = ConvModule(
            in_channels, channels, 1, norm_cfg=norm_cfg, act_cfg=None)
        if with_channel:
            self.up = ConvModule(
                channels, in_channels, 1, norm_cfg=norm_cfg, act_cfg=None)
        if after_relu:
            self.relu = MODELS.build(act_cfg)

    def forward(self, x_p: Tensor, x_i: Tensor) -> Tensor:
        """Forward function.

        Args:
            x_p (Tensor): The featrue map from P branch.
            x_i (Tensor): The featrue map from I branch.

        Returns:
            Tensor: The feature map with pixel-attention-guided fusion.
        """
        if self.after_relu:
            x_p = self.relu(x_p)
            x_i = self.relu(x_i)

        f_i = self.f_i(x_i)
        f_i = F.interpolate(
            f_i,
            size=x_p.shape[2:],
            mode=self.upsample_mode,
            align_corners=False)

        f_p = self.f_p(x_p)

        if self.with_channel:
            sigma = torch.sigmoid(self.up(f_p * f_i))
        else:
            sigma = torch.sigmoid(torch.sum(f_p * f_i, dim=1).unsqueeze(1))

        x_i = F.interpolate(
            x_i,
            size=x_p.shape[2:],
            mode=self.upsample_mode,
            align_corners=False)

        out = sigma * x_i + (1 - sigma) * x_p
        return out


class Bag(BaseModule):
    """Boundary-attention-guided fusion module.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        kernel_size (int): The kernel size of the convolution. Default: 3.
        padding (int): The padding of the convolution. Default: 1.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU', inplace=True).
        conv_cfg (dict): Config dict for convolution layer.
            Default: dict(order=('norm', 'act', 'conv')).
        init_cfg (dict): Config dict for initialization. Default: None.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 padding: int = 1,
                 norm_cfg: OptConfigType = dict(type='BN'),
                 act_cfg: OptConfigType = dict(type='ReLU', inplace=True),
                 conv_cfg: OptConfigType = dict(order=('norm', 'act', 'conv')),
                 init_cfg: OptConfigType = None):
        super().__init__(init_cfg)

        self.conv = ConvModule(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            **conv_cfg)

    def forward(self, x_p: Tensor, x_i: Tensor, x_d: Tensor) -> Tensor:
        """Forward function.

        Args:
            x_p (Tensor): The featrue map from P branch.
            x_i (Tensor): The featrue map from I branch.
            x_d (Tensor): The featrue map from D branch.

        Returns:
            Tensor: The feature map with boundary-attention-guided fusion.
        """
        sigma = torch.sigmoid(x_d)
        return self.conv(sigma * x_p + (1 - sigma) * x_i)


class LightBag(BaseModule):
    """Light Boundary-attention-guided fusion module.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer. Default: None.
        init_cfg (dict): Config dict for initialization. Default: None.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 norm_cfg: OptConfigType = dict(type='BN'),
                 act_cfg: OptConfigType = None,
                 init_cfg: OptConfigType = None):
        super().__init__(init_cfg)
        self.f_p = ConvModule(
            in_channels,
            out_channels,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.f_i = ConvModule(
            in_channels,
            out_channels,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, x_p: Tensor, x_i: Tensor, x_d: Tensor) -> Tensor:
        """Forward function.
        Args:
            x_p (Tensor): The featrue map from P branch.
            x_i (Tensor): The featrue map from I branch.
            x_d (Tensor): The featrue map from D branch.

        Returns:
            Tensor: The feature map with light boundary-attention-guided
                fusion.
        """
        sigma = torch.sigmoid(x_d)

        f_p = self.f_p((1 - sigma) * x_i + x_p)
        f_i = self.f_i(x_i + sigma * x_p)

        return f_p + f_i


@MODELS.register_module()
class PIDNetWaveAttentionOnceDI(BaseModule):
    """PIDNet backbone.

    This backbone is the implementation of `PIDNet: A Real-time Semantic
    Segmentation Network Inspired from PID Controller
    <https://arxiv.org/abs/2206.02066>`_.
    Modified from https://github.com/XuJiacong/PIDNet.

    Licensed under the MIT License.

    Args:
        in_channels (int): The number of input channels. Default: 3.
        channels (int): The number of channels in the stem layer. Default: 64.
        ppm_channels (int): The number of channels in the PPM layer.
            Default: 96.
        num_stem_blocks (int): The number of blocks in the stem layer.
            Default: 2.
        num_branch_blocks (int): The number of blocks in the branch layer.
            Default: 3.
        align_corners (bool): The align_corners argument of F.interpolate.
            Default: False.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU', inplace=True).
        init_cfg (dict): Config dict for initialization. Default: None.
    """

    def __init__(self,
                 in_channels: int = 3,
                 channels: int = 64,
                 ppm_channels: int = 96,
                 num_stem_blocks: int = 2,
                 num_branch_blocks: int = 3,
                 align_corners: bool = False,
                 norm_cfg: OptConfigType = dict(type='BN'),
                 act_cfg: OptConfigType = dict(type='ReLU', inplace=True),
                 init_cfg: OptConfigType = None,
                 **kwargs):
        super().__init__(init_cfg)
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.align_corners = align_corners

        # stem layer
        self.stem = self._make_stem_layer(in_channels, channels,
                                          num_stem_blocks)
        self.relu = nn.ReLU()

        # =====================================================================
        # [新增] 实例化王炸组合：双频小波路由器
        # =====================================================================
        self.wlsa = WaveletSemanticAttention()  # 供 I 分支使用
        self.whba = WaveletBoundaryAttention()  # 供 D 分支使用

        # I Branch
        self.i_branch_layers = nn.ModuleList()
        for i in range(3):
            self.i_branch_layers.append(
                self._make_layer(
                    block=BasicBlock if i < 2 else Bottleneck,
                    in_channels=channels * 2**(i + 1),
                    channels=channels * 8 if i > 0 else channels * 4,
                    num_blocks=num_branch_blocks if i < 2 else 2,
                    stride=2))

        # P Branch
        self.p_branch_layers = nn.ModuleList()
        for i in range(3):
            self.p_branch_layers.append(
                self._make_layer(
                    block=BasicBlock if i < 2 else Bottleneck,
                    in_channels=channels * 2,
                    channels=channels * 2,
                    num_blocks=num_stem_blocks if i < 2 else 1))
        self.compression_1 = ConvModule(
            channels * 4,
            channels * 2,
            kernel_size=1,
            bias=False,
            norm_cfg=norm_cfg,
            act_cfg=None)
        self.compression_2 = ConvModule(
            channels * 8,
            channels * 2,
            kernel_size=1,
            bias=False,
            norm_cfg=norm_cfg,
            act_cfg=None)
        self.pag_1 = PagFM(channels * 2, channels)
        self.pag_2 = PagFM(channels * 2, channels)

        # D Branch
        if num_stem_blocks == 2:
            self.d_branch_layers = nn.ModuleList([
                self._make_single_layer(BasicBlock, channels * 2, channels),
                self._make_layer(Bottleneck, channels, channels, 1)
            ])
            channel_expand = 1
            spp_module = PAPPM
            dfm_module = LightBag
            act_cfg_dfm = None
        else:
            self.d_branch_layers = nn.ModuleList([
                self._make_single_layer(BasicBlock, channels * 2,
                                        channels * 2),
                self._make_single_layer(BasicBlock, channels * 2, channels * 2)
            ])
            channel_expand = 2
            spp_module = DAPPM
            dfm_module = Bag
            act_cfg_dfm = act_cfg

        self.diff_1 = ConvModule(
            channels * 4,
            channels * channel_expand,
            kernel_size=3,
            padding=1,
            bias=False,
            norm_cfg=norm_cfg,
            act_cfg=None)
        self.diff_2 = ConvModule(
            channels * 8,
            channels * 2,
            kernel_size=3,
            padding=1,
            bias=False,
            norm_cfg=norm_cfg,
            act_cfg=None)

        self.spp = spp_module(
            channels * 16, ppm_channels, channels * 4, num_scales=5)
        self.dfm = dfm_module(
            channels * 4, channels * 4, norm_cfg=norm_cfg, act_cfg=act_cfg_dfm)

        self.d_branch_layers.append(
            self._make_layer(Bottleneck, channels * 2, channels * 2, 1))

    def _make_stem_layer(self, in_channels: int, channels: int,
                         num_blocks: int) -> nn.Sequential:
        """Make stem layer.

        Args:
            in_channels (int): Number of input channels.
            channels (int): Number of output channels.
            num_blocks (int): Number of blocks.

        Returns:
            nn.Sequential: The stem layer.
        """

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

        layers.append(
            self._make_layer(BasicBlock, channels, channels, num_blocks))
        layers.append(nn.ReLU())
        layers.append(
            self._make_layer(
                BasicBlock, channels, channels * 2, num_blocks, stride=2))
        layers.append(nn.ReLU())

        return nn.Sequential(*layers)

    def _make_layer(self,
                    block: BasicBlock,
                    in_channels: int,
                    channels: int,
                    num_blocks: int,
                    stride: int = 1) -> nn.Sequential:
        """Make layer for PIDNet backbone.
        Args:
            block (BasicBlock): Basic block.
            in_channels (int): Number of input channels.
            channels (int): Number of output channels.
            num_blocks (int): Number of blocks.
            stride (int): Stride of the first block. Default: 1.

        Returns:
            nn.Sequential: The Branch Layer.
        """
        downsample = None
        if stride != 1 or in_channels != channels * block.expansion:
            downsample = ConvModule(
                in_channels,
                channels * block.expansion,
                kernel_size=1,
                stride=stride,
                norm_cfg=self.norm_cfg,
                act_cfg=None)

        layers = [block(in_channels, channels, stride, downsample)]
        in_channels = channels * block.expansion
        for i in range(1, num_blocks):
            layers.append(
                block(
                    in_channels,
                    channels,
                    stride=1,
                    act_cfg_out=None if i == num_blocks - 1 else self.act_cfg))
        return nn.Sequential(*layers)

    def _make_single_layer(self,
                           block: Union[BasicBlock, Bottleneck],
                           in_channels: int,
                           channels: int,
                           stride: int = 1) -> nn.Module:
        """Make single layer for PIDNet backbone.
        Args:
            block (BasicBlock or Bottleneck): Basic block or Bottleneck.
            in_channels (int): Number of input channels.
            channels (int): Number of output channels.
            stride (int): Stride of the first block. Default: 1.

        Returns:
            nn.Module
        """

        downsample = None
        if stride != 1 or in_channels != channels * block.expansion:
            downsample = ConvModule(
                in_channels,
                channels * block.expansion,
                kernel_size=1,
                stride=stride,
                norm_cfg=self.norm_cfg,
                act_cfg=None)
        return block(
            in_channels, channels, stride, downsample, act_cfg_out=None)

    def init_weights(self):
        """Initialize the weights in backbone.

        Since the D branch is not initialized by the pre-trained model, we
        initialize it with the same method as the ResNet.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if self.init_cfg is not None:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            ckpt = CheckpointLoader.load_checkpoint(
                self.init_cfg['checkpoint'], map_location='cpu')
            self.load_state_dict(ckpt, strict=False)

    def forward(self, x: Tensor) -> Union[Tensor, Tuple[Tensor]]:
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (B, C, H, W).

        Returns:
            Tensor or tuple[Tensor]: If self.training is True, return
                tuple[Tensor], else return Tensor.
        """
        w_out = x.shape[-1] // 8
        h_out = x.shape[-2] // 8

        # stage 0-2
        x = self.stem(x)

        # stage 3
        x_i = self.relu(self.i_branch_layers[0](x))
        
        # I 分支：提取特征后，立刻应用 WLSA (低频) 提纯语义
        x_i = self.wlsa(x_i)

        x_p = self.p_branch_layers[0](x)
        x_d = self.d_branch_layers[0](x)

        # D 分支：提取特征后，立刻应用 WHBA (高频) 提纯边界
        x_d = self.whba(x_d)

        comp_i = self.compression_1(x_i)
        x_p = self.pag_1(x_p, comp_i)
        diff_i = self.diff_1(x_i)
        x_d += F.interpolate(
            diff_i,
            size=[h_out, w_out],
            mode='bilinear',
            align_corners=self.align_corners)
        if self.training:
            temp_p = x_p.clone()

        # stage 4
        x_i = self.relu(self.i_branch_layers[1](x_i))
        # I 分支：增强低频语义
        # x_i = self.wlsa(x_i)

        x_p = self.p_branch_layers[1](self.relu(x_p))
        x_d = self.d_branch_layers[1](self.relu(x_d))

        # D 分支：增强高频边界
        # x_d = self.whba(x_d)

        comp_i = self.compression_2(x_i)
        x_p = self.pag_2(x_p, comp_i)
        diff_i = self.diff_2(x_i)
        x_d += F.interpolate(
            diff_i,
            size=[h_out, w_out],
            mode='bilinear',
            align_corners=self.align_corners)
        if self.training:
            temp_d = x_d.clone()

        # stage 5
        x_i = self.i_branch_layers[2](x_i)
        # I 分支：增强低频语义 (在进入 SPP 多尺度池化前进行提纯)
        # x_i = self.wlsa(x_i)

        x_p = self.p_branch_layers[2](self.relu(x_p))
        x_d = self.d_branch_layers[2](self.relu(x_d))
        # D 分支：增强高频边界
        # x_d = self.whba(x_d)

        x_i = self.spp(x_i)
        x_i = F.interpolate(
            x_i,
            size=[h_out, w_out],
            mode='bilinear',
            align_corners=self.align_corners)
        out = self.dfm(x_p, x_i, x_d)
        return (temp_p, out, temp_d) if self.training else out
