import torch
import torch.nn as nn
from mmengine.model import BaseModule
from mmseg.registry import MODELS
from mmcv.cnn import ConvModule

# 尝试导入 PIDNet 父类
try:
    from .pidnet import PIDNet
    from ..utils import BasicBlock, Bottleneck
except ImportError:
    from mmseg.models.backbones.pidnet import PIDNet
    from mmseg.models.utils import BasicBlock, Bottleneck

# ==================================================================
# 1. SpaceOptiConv (修复 In-place 报错版)
# ==================================================================
class SpaceOptiConv(BaseModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 padding=1,
                 stride=1,
                 reduce_ratio=4,
                 spatial_channels=16,
                 # ⚠️ 修改 1: 默认关闭 inplace，防止梯度计算报错
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU', inplace=False)): 
        super().__init__()
        
        reduced_channels = max(in_channels // reduce_ratio, 8)
        
        # 1. Reduce
        self.reduce = ConvModule(
            in_channels,
            reduced_channels,
            kernel_size=1,
            stride=stride,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg # 传递 inplace=False
        )
        
        # 2. Spatial
        self.spatial = ConvModule(
            reduced_channels,
            spatial_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg # 传递 inplace=False
        )
        
        # 3. Fuse
        self.fuse = ConvModule(
            reduced_channels + spatial_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg # 传递 inplace=False
        )

    def forward(self, x):
        x_reduce = self.reduce(x)
        x_spatial = self.spatial(x_reduce)
        x_fused = torch.cat([x_reduce, x_spatial], dim=1)
        return self.fuse(x_fused)

# ==================================================================
# 2. ESBlock (修复 In-place 报错版)
# ==================================================================
class ESBlock(BaseModule):
    expansion = 1

    def __init__(self, in_channels, channels, stride=1, downsample=None, act_cfg_out=None):
        super().__init__()
        
        self.conv_block = SpaceOptiConv(
            in_channels=in_channels,
            out_channels=channels,
            stride=stride,
            reduce_ratio=4,
            spatial_channels=16,
            # 显式传入 inplace=False
            act_cfg=dict(type='ReLU', inplace=False) 
        )
        
        self.downsample = downsample
        self.stride = stride
        
        self.act_cfg_out = act_cfg_out
        if act_cfg_out is not None:
             # 如果外部传入了配置，我们需要确保它是非 inplace 的
             # 这里做一个防御性编程，虽然通常 PIDNet 传入的是 None
             if 'inplace' in act_cfg_out:
                 act_cfg_out['inplace'] = False
             self.relu_out = MODELS.build(act_cfg_out)
        else:
             # ⚠️ 修改 2: 这里的 ReLU 必须是 inplace=False
             self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        identity = x

        out = self.conv_block(x)

        if self.downsample is not None:
            identity = self.downsample(x)

        # ⚠️ ⚠️ ⚠️ 核心修复点 ⚠️ ⚠️ ⚠️
        # 原代码: out += identity  (这是 In-place 操作，会报错)
        # 新代码: out = out + identity (这是 Out-of-place 操作，安全)
        out = out + identity
        
        if self.act_cfg_out is not None:
            out = self.relu_out(out)
        else:
            out = self.relu(out)

        return out

# ==================================================================
# 3. ESPIDNet (主类 - 保持不变)
# ==================================================================
@MODELS.register_module()
class ESPIDNet(PIDNet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._replace_p_branch_with_esblock()

    def _replace_p_branch_with_esblock(self):
        if not hasattr(self, 'p_branch_layers'):
            return

        for stage_idx, stage in enumerate(self.p_branch_layers):
            for i, block in enumerate(stage):
                
                is_basic = isinstance(block, BasicBlock)
                is_bottleneck = isinstance(block, Bottleneck)
                
                if is_basic or is_bottleneck:
                    if is_basic:
                        in_c = block.conv1.in_channels
                        out_c = block.conv2.out_channels
                    else:
                        in_c = block.conv1.in_channels
                        out_c = block.conv3.out_channels

                    stride = block.stride if hasattr(block, 'stride') else 1
                    downsample = block.downsample
                    act_cfg_out = getattr(block, 'act_cfg_out', None)
                    
                    new_block = ESBlock(
                        in_channels=in_c,
                        channels=out_c,
                        stride=stride,
                        downsample=downsample,
                        act_cfg_out=act_cfg_out
                    )
                    
                    stage[i] = new_block
