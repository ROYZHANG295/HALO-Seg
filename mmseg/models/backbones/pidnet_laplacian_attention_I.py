# Copyright (c) OpenMMLab. All rights reserved.
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
from ..utils import DAPPM, PAPPM, BasicBlock, Bottleneck

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
        # 🔥 初始化权重为 0
        # 这样初始的卷积输出为 0 -> Sigmoid(0)=0.5
        # 虽然初始会放大特征 1.5 倍，但分布是均匀的，不会破坏预训练的特征结构
        nn.init.constant_(self.conv_adjust.weight, 0)
        nn.init.constant_(self.conv_adjust.bias, 0)

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
class PIDNetLaplacianAttentionI(BaseModule):
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

        #########################################################################
        # ######## [修改点 2] 初始化注意力模块 ########
        #########################################################################
        # 我们要注入的位置是 I 分支的第一个阶段之后
        # 根据上面的代码：i=0 时，channels = channels * 4
        # 默认 channels=64，所以这里的通道数是 256
        target_channels = channels * 4
        
        self.edge_attention = LaplacianAttention(out_channels=target_channels)
        #########################################################################            

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

        #########################################################################
        # ######## [修改点 3] 强制初始化注意力权重为 0 ########
        #########################################################################
        # 必须放在上面的循环之后，否则会被 kaiming_normal 覆盖！
        if hasattr(self, 'edge_attention'):
            self.edge_attention.init_zero()
        #########################################################################        

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
        
        #########################################################################
        # ######## [修改点 4] 备份原始输入 ########
        #########################################################################
        x_raw = x # 备份原始图片 (B, 3, H, W)
        #########################################################################

        w_out = x.shape[-1] // 8
        h_out = x.shape[-2] // 8

        # stage 0-2
        x = self.stem(x)

        # stage 3
        x_i = self.relu(self.i_branch_layers[0](x))

        #########################################################################
        # ######## [修改点 5] 计算并应用注意力 (核心步骤) ########
        #########################################################################
        # 1. 计算注意力图
        # 输入: x_raw (B, 3, H, W)
        # 目标尺寸: x_i.shape[2:] 即 (H/8, W/8) 左右
        att = self.edge_attention(x_raw, x_i.shape[2:]) 
        # att 维度: (B, 256, H_i, W_i)
        
        # 2. 乘法增强 (Residual Attention)
        # 公式: Feature = Feature * (1 + Attention)
        # 含义: 在原有特征基础上，根据边缘强度进行信号放大
        x_i = x_i * (1 + att)
        #########################################################################

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
