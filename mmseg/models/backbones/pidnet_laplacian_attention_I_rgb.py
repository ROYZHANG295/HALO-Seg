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

class TrueRGBLaplacianAttention(nn.Module):
    """
    真实 RGB 物理边缘注意力模块 (反归一化版)
    作用：从网络输入中恢复真实像素，提取纯净物理边缘，并用最大池化保留锐利度
    """
    def __init__(self, out_channels):
        super().__init__()
        # 1. 拉普拉斯卷积核
        kernel = torch.tensor([[0.,  1., 0.],
                               [1., -4., 1.],
                               [0.,  1., 0.]], dtype=torch.float32)
        self.register_buffer('kernel', kernel.view(1, 1, 3, 3))
        
        # 2. 注册 ImageNet 的均值和标准差 (MMSeg 默认使用的配置)
        # 注意：这里假设通道顺序已经是 RGB
        mean = torch.tensor([123.675, 116.280, 103.530]).view(1, 3, 1, 1)
        std = torch.tensor([58.395, 57.120, 57.375]).view(1, 3, 1, 1)
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)
        
        # 3. 通道调整层
        self.conv_adjust = nn.Conv2d(1, out_channels, kernel_size=1)

    def init_zero(self):
        nn.init.constant_(self.conv_adjust.weight, 0)
        if self.conv_adjust.bias is not None:
            nn.init.constant_(self.conv_adjust.bias, -6.0)

    def forward(self, x_normalized, target_shape):
        """
        x_normalized: 网络最原始的输入 (经过预处理的张量) (B, 3, 1024, 2048)
        target_shape: 目标特征图的高和宽，例如 (128, 256)
        """
        # 1. 🔥 反归一化：强行恢复出 0~255 的真实物理 RGB 图像！
        x_raw = x_normalized * self.std + self.mean

        # ================= [DEBUG: 导出第一张图] =================
        # 加一个 hasattr 判断，确保整个训练/测试过程中只保存一次，防止把硬盘写满
        if not hasattr(self, '_debug_saved'):
            import cv2
            import numpy as np
            
            # 1. 提取 Batch 里的第一张图，放到 CPU 上，去掉梯度
            # 此时 img_tensor 的形状是 (3, H, W)
            img_tensor = x_raw[0].detach().cpu()
            
            # 2. PyTorch 的通道在前面 (C, H, W)，OpenCV 需要通道在后面 (H, W, C)
            img_np = img_tensor.numpy().transpose(1, 2, 0)
            
            # 3. 确保数值在 0~255 之间，并转为 uint8 类型
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)
            
            # 4. 因为网络里是 RGB 顺序，而 OpenCV 保存图片默认需要 BGR 顺序，所以翻转一下通道
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            
            # 5. 保存图片到项目根目录
            save_path = 'debug_recovered_rgb.jpg'
            cv2.imwrite(save_path, img_bgr)
            print(f"\n======> [Debug 成功] 物理 RGB 图像已导出至: {save_path} <======\n")
            
            # 打上标记，下次 forward 就不再执行这段代码了
            self._debug_saved = True
        # =========================================================
        
        # 2. 提取 RGB 通道并计算真实物理灰度
        R = x_raw[:, 0:1, :, :]
        G = x_raw[:, 1:2, :, :]
        B = x_raw[:, 2:3, :, :]
        x_gray = 0.299 * R + 0.587 * G + 0.114 * B

        # ================= [DEBUG: 导出灰度图] =================
        if not hasattr(self, '_debug_gray_saved'):
            import cv2
            import numpy as np
            
            # 1. 提取 Batch 里第一张图的灰度矩阵
            # x_gray 形状是 (B, 1, H, W)，取 [0, 0] 变成 (H, W)
            gray_tensor = x_gray[0, 0].detach().cpu()
            
            # 2. 转成 numpy 数组
            gray_np = gray_tensor.numpy()
            
            # 3. 限制范围在 0~255 并转为 uint8 (灰度图单通道即可，不需要转换 BGR)
            gray_np = np.clip(gray_np, 0, 255).astype(np.uint8)
            
            # 4. 保存图片到项目根目录
            save_path_gray = 'debug_recovered_gray.jpg'
            cv2.imwrite(save_path_gray, gray_np)
            print(f"\n======> [Debug 成功] 物理灰度图已导出至: {save_path_gray} <======\n")
            
            # 打上标记，保证只存一次
            self._debug_gray_saved = True
        # =========================================================
        
        # 3. 在 1024x2048 的原分辨率上提取极致细腻的物理边缘
        edge = F.conv2d(x_gray, self.kernel, padding=1)
        edge = torch.abs(edge)
        
        # 4. 🔥 核心下采样修复：用 AdaptiveMaxPool 代替 Bilinear！
        # 为什么？因为边缘是极细的亮线，Bilinear 会把亮线和周围的黑色平均掉（变糊）
        # MaxPool 会在 8x8 的感受野里挑出最亮的那根边缘线，完美保留轮廓！
        edge_down = F.adaptive_max_pool2d(edge, output_size=target_shape)
        
        # 5. 归一化防止爆炸
        B_dim = edge_down.shape[0]
        edge_flat = edge_down.view(B_dim, -1)
        edge_max = edge_flat.max(dim=1)[0].view(B_dim, 1, 1, 1).clamp(min=1e-8)
        edge_norm = edge_down / edge_max
        
        # 6. 生成注意力权重
        attention = torch.sigmoid(self.conv_adjust(edge_norm))
        
        # ================= [DEBUG: 导出纯物理边缘图 (绕过网络权重)] =================
        if not hasattr(self, '_debug_edge_saved'):
            import cv2
            import numpy as np
            
            # 1. 拿回原图 (转为 OpenCV 需要的 BGR 格式)
            img_tensor = x_raw[0].detach().cpu()
            img_np = img_tensor.numpy().transpose(1, 2, 0)
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            
            # 2. 🔥 核心改变：直接拿取未经网络缩放的纯物理边缘 (edge)，而不是 attention
            # edge 的形状是 (B, 1, 1024, 2048)，取 [0, 0] 变成 2D 矩阵
            edge_tensor = edge[0, 0].detach().cpu() 
            edge_np = edge_tensor.numpy()
            
            # 3. 将物理边缘的数值映射到 0~255
            # 拉普拉斯算出来的边缘值可能很大，我们做个简单的最大最小值归一化
            edge_min, edge_max = edge_np.min(), edge_np.max()
            if edge_max > edge_min:
                edge_norm_vis = (edge_np - edge_min) / (edge_max - edge_min)
            else:
                edge_norm_vis = edge_np
                
            # 为了让轮廓在视觉上更炸裂，我们把信号放大 2 倍再截断到 255 (类似于调高对比度)
            edge_gray_vis = np.clip(edge_norm_vis * 255 * 2.0, 0, 255).astype(np.uint8)
            
            # 涂上 JET 伪彩色 (边缘越强的地方越红，平坦的地方是深蓝)
            heatmap = cv2.applyColorMap(edge_gray_vis, cv2.COLORMAP_JET)
            
            # 4. 将边缘热力图以 50% 透明度盖在原图上
            overlay = cv2.addWeighted(img_bgr, 0.5, heatmap, 0.5, 0)
            
            # 5. 把三张图拼在一起 (左: 原图, 中: 拉普拉斯边缘, 右: 叠加图)
            concat_img = np.concatenate([img_bgr, heatmap, overlay], axis=1)
            
            # 6. 保存到硬盘
            save_path = 'debug_pure_laplacian_edge.jpg'
            cv2.imwrite(save_path, concat_img)
            
            print(f"\n======> [Debug 成功] 纯物理边缘叠加图已导出至: {save_path} <======")
            print(f"======> [Debug Info] 物理边缘极值 Min: {edge_min:.4f}, Max: {edge_max:.4f} <======\n")
            
            self._debug_edge_saved = True
        # =========================================================================

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
class PIDNetLaplacianAttentionIRgb(BaseModule):
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
        
        self.edge_attention = TrueRGBLaplacianAttention(out_channels=target_channels)
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
