import torch
import torch.nn as nn
from mmengine.model import BaseModule
from mmseg.registry import MODELS
from mmcv.cnn import ConvModule

# 导入 PIDNet 父类
# 注意：请根据你的文件结构调整这里的 import 路径
# 如果是在 mmseg/models/backbones/pidnet.py 同级目录下，可以直接 import
from .pidnet import PIDNet 
from ..utils import BasicBlock, Bottleneck # 确保能导入基础模块

# ==================================================================
# 1. PConv (部分卷积)
# ==================================================================
class PConv(nn.Module):
    def __init__(self, dim, n_div=4, kernel_size=3):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        # 只卷 1/4 通道
        self.partial_conv3 = nn.Conv2d(
            self.dim_conv3, 
            self.dim_conv3, 
            kernel_size, 
            stride=1, 
            padding=(kernel_size - 1) // 2, 
            bias=False
        )

    def forward(self, x):
        x1 = x[:, :self.dim_conv3, :, :]
        x2 = x[:, self.dim_conv3:, :, :]
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)
        return x

# ==================================================================
# 2. FasterBlock (用于替换 P 分支)
# ==================================================================
class FasterBlock(BaseModule):
    expansion = 1

    def __init__(self, in_channels, channels, stride=1, downsample=None, act_cfg_out=None):
        super(FasterBlock, self).__init__()
        
        # PConv 提取特征 (3x3)
        self.conv1 = PConv(in_channels, n_div=4, kernel_size=3)
        
        # PWConv 特征融合 (1x1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        
        self.relu = nn.ReLU(inplace=True)
        
        self.downsample = downsample
        self.stride = stride

        # 兼容 PIDNet 的输出激活配置
        self.act_cfg_out = act_cfg_out
        if act_cfg_out is not None:
             self.relu_out = MODELS.build(act_cfg_out)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        
        if self.act_cfg_out is not None:
            out = self.relu_out(out)
        else:
            out = self.relu(out)

        return out

# ==================================================================
# 3. FasterPIDNet (MMSegmentation 专用版)
# ==================================================================
@MODELS.register_module()
class FasterPIDNet_P(PIDNet):
    def __init__(self, **kwargs):
        # 1. 初始化原版 PIDNet (MMSegmentation 版本)
        super().__init__(**kwargs)
        
        # 2. 执行手术：替换 p_branch_layers
        self._replace_p_branch_layers()

    def _replace_p_branch_layers(self):
        """
        针对 MMSegmentation 的 PIDNet 结构进行替换。
        目标：self.p_branch_layers (通常是一个 nn.ModuleList)
        """
        # 安全检查：确保 p_branch_layers 存在
        if not hasattr(self, 'p_branch_layers'):
            print("⚠️ Warning: p_branch_layers not found in PIDNet! Skipping replacement.")
            return

        # 遍历 P 分支的每一个 Stage
        # p_branch_layers 通常包含 3 个 stage (对应 layer3, layer4, layer5)
        for stage_idx, stage in enumerate(self.p_branch_layers):
            
            # 遍历 Stage 里的每一个 Block
            for i, block in enumerate(stage):
                
                # 策略：跳过下采样层 (stride=2)，只替换 stride=1 的层
                # 这能保证最稳定的训练效果
                if block.downsample is not None:
                    continue

                # 检查 Block 类型
                is_basic = isinstance(block, BasicBlock)
                is_bottleneck = isinstance(block, Bottleneck)
                
                if is_basic or is_bottleneck:
                    # 获取输入输出通道
                    if is_basic:
                        in_c = block.conv1.in_channels
                        out_c = block.conv2.out_channels
                    else: # Bottleneck
                        in_c = block.conv1.in_channels
                        out_c = block.conv3.out_channels

                    # 获取激活配置
                    act_cfg_out = getattr(block, 'act_cfg_out', None)
                    
                    # 创建 FasterBlock
                    new_block = FasterBlock(
                        in_channels=in_c,
                        channels=out_c,
                        stride=1,
                        downsample=None, 
                        act_cfg_out=act_cfg_out
                    )
                    
                    # 原地替换
                    stage[i] = new_block
                    
                    # (调试用) 打印替换信息
                    # print(f"✅ Replaced p_branch_layers Stage {stage_idx} Block {i} with FasterBlock")

