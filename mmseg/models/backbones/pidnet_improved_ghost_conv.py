# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from mmengine.runner import CheckpointLoader
from torch import Tensor

from mmseg.registry import MODELS
from mmseg.utils import OptConfigType
# 注意：这里保留了原始的引用，确保兼容性
from ..utils import DAPPM, PAPPM, BasicBlock, Bottleneck


# ==================================================================
# 1. 新增 GhostModule (核心轻量化组件)
# ==================================================================
class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]


# ==================================================================
# 2. 新增 GhostBasicBlock (用于替换 I 分支的 BasicBlock)
# ==================================================================
class GhostBasicBlock(BaseModule):
    """
    基于 GhostModule 的 BasicBlock。
    结构：GhostModule(3x3) -> BN -> ReLU -> GhostModule(3x3) -> BN -> Add -> ReLU
    """
    expansion = 1

    def __init__(self,
                 in_channels: int,
                 channels: int,
                 stride: int = 1,
                 downsample: nn.Module = None,
                 act_cfg_out: dict = dict(type='ReLU', inplace=True),
                 init_cfg: dict = None):
        super().__init__(init_cfg)
        
        # 将第一个 3x3 卷积替换为 GhostModule
        self.conv1 = GhostModule(
            in_channels, 
            channels, 
            kernel_size=3, 
            stride=stride, 
            relu=True
        )
        
        # 将第二个 3x3 卷积替换为 GhostModule (注意这里不立即做 ReLU，因为后面要 Add)
        self.conv2 = GhostModule(
            channels, 
            channels, 
            kernel_size=3, 
            stride=1, 
            relu=False
        )
        
        # 使用 GhostModule 自带的 BN，这里只需要定义最后的激活函数
        self.act_out = MODELS.build(act_cfg_out) if act_cfg_out else None
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        
        if self.act_out is not None:
            out = self.act_out(out)

        return out


# ==================================================================
# 3. 原始辅助模块 (保持不变)
# ==================================================================
class PagFM(BaseModule):
    """Pixel-attention-guided fusion module."""
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
    """Boundary-attention-guided fusion module."""
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
        sigma = torch.sigmoid(x_d)
        return self.conv(sigma * x_p + (1 - sigma) * x_i)


class LightBag(BaseModule):
    """Light Boundary-attention-guided fusion module."""
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
        sigma = torch.sigmoid(x_d)
        f_p = self.f_p((1 - sigma) * x_i + x_p)
        f_i = self.f_i(x_i + sigma * x_p)
        return f_p + f_i


# ==================================================================
# 4. 主类: PIDNetImprovedGhostConv
# ==================================================================
@MODELS.register_module()
class PIDNetImprovedGhostConv(BaseModule):
    """PIDNet with Ghost Convolutions.
    
    Improvements:
    - Replaced BasicBlock with GhostBasicBlock in the 'I Branch' (Context Branch)
      to reduce parameters and FLOPs while maintaining context information.
    - P Branch (Detail) and D Branch (Boundary) remain unchanged to preserve accuracy.
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

        # ----------------------------------------------------------
        # I Branch (Modified with Ghost Conv)
        # ----------------------------------------------------------
        self.i_branch_layers = nn.ModuleList()
        for i in range(3):
            # 策略：前两个阶段使用 GhostBasicBlock 替换 BasicBlock
            # 第三个阶段如果是 Bottleneck，保持原样 (或者也可以实现 GhostBottleneck，这里为稳健起见保留)
            if i < 2:
                block_type = GhostBasicBlock 
            else:
                block_type = Bottleneck
            
            self.i_branch_layers.append(
                self._make_layer(
                    block=block_type,
                    in_channels=channels * 2**(i + 1),
                    channels=channels * 8 if i > 0 else channels * 4,
                    num_blocks=num_branch_blocks if i < 2 else 2,
                    stride=2))

        # ----------------------------------------------------------
        # P Branch (Standard Conv to keep details)
        # ----------------------------------------------------------
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

        # ----------------------------------------------------------
        # D Branch (Standard Conv to keep edges)
        # ----------------------------------------------------------
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
        """Make stem layer."""
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
                    block: Union[BasicBlock, Bottleneck, GhostBasicBlock], # Update type hint
                    in_channels: int,
                    channels: int,
                    num_blocks: int,
                    stride: int = 1) -> nn.Sequential:
        """Make layer for PIDNet backbone."""
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
        """Make single layer for PIDNet backbone."""
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
        """Initialize the weights in backbone."""
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
        """Forward function."""
        w_out = x.shape[-1] // 8
        h_out = x.shape[-2] // 8

        # stage 0-2
        x = self.stem(x)

        # stage 3
        x_i = self.relu(self.i_branch_layers[0](x))
        x_p = self.p_branch_layers[0](x)
        x_d = self.d_branch_layers[0](x)

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
        x_p = self.p_branch_layers[1](self.relu(x_p))
        x_d = self.d_branch_layers[1](self.relu(x_d))

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
        x_p = self.p_branch_layers[2](self.relu(x_p))
        x_d = self.d_branch_layers[2](self.relu(x_d))

        x_i = self.spp(x_i)
        x_i = F.interpolate(
            x_i,
            size=[h_out, w_out],
            mode='bilinear',
            align_corners=self.align_corners)
        out = self.dfm(x_p, x_i, x_d)
        return (temp_p, out, temp_d) if self.training else out
