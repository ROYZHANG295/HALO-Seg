import torch
import torch.nn as nn
from mmengine.model import BaseModule
from mmseg.registry import MODELS
from mmcv.cnn import ConvModule

# 导入原有的 PIDNet 和基础模块
from .pidnet import PIDNet
from ..utils import BasicBlock

# ==================================================================
# 1. PConv (部分卷积) - 核心加速算子
# ==================================================================
class PConv(nn.Module):
    def __init__(self, dim, n_div=4, kernel_size=3):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        # 只卷 1/4 的通道
        self.partial_conv3 = nn.Conv2d(
            self.dim_conv3, 
            self.dim_conv3, 
            kernel_size, 
            stride=1, 
            padding=(kernel_size - 1) // 2, 
            bias=False
        )

    def forward(self, x):
        # 切片 -> 卷积 -> 拼接
        x1 = x[:, :self.dim_conv3, :, :]
        x2 = x[:, self.dim_conv3:, :, :]
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)
        return x

# ==================================================================
# 2. FasterBlock (仅用于替换 Stride=1 的 BasicBlock)
# ==================================================================
class FasterBlock(BaseModule):
    expansion = 1

    def __init__(self, in_channels, channels, stride=1, downsample=None, act_cfg_out=None):
        super(FasterBlock, self).__init__()
        
        # 既然我们只替换 Stride=1 的层，这里直接写死逻辑
        # 1. PConv (3x3, 1/4 channels)
        self.conv1 = PConv(in_channels, n_div=4)
        
        # 2. PWConv (1x1, 全通道融合)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        
        self.relu = nn.ReLU(inplace=True)
        
        # 处理残差连接
        self.downsample = downsample
        self.stride = stride

        # 额外的激活函数配置 (兼容 PIDNet 接口)
        self.act_cfg_out = act_cfg_out
        if act_cfg_out is not None:
             self.relu_out = MODELS.build(act_cfg_out)

    def forward(self, x):
        identity = x

        # PConv 部分 (无 BN, 无 ReLU, 或者 ReLU 放后面)
        out = self.conv1(x)
        out = self.relu(out) # PConv 后加个 ReLU 增加非线性

        # PWConv 部分
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
# 3. FasterPIDNet (精准手术版)
# ==================================================================
@MODELS.register_module()
class FasterPIDNet(PIDNet):
    def __init__(self, **kwargs):
        # 1. 先让父类把整个网络建好 (全部是用标准卷积建的)
        super().__init__(**kwargs)
        
        # 2. 执行“外科手术”：只替换 I 分支里 Stride=1 的模块
        self._surgical_replace_i_branch()

    def _surgical_replace_i_branch(self):
        """
        修正后的逻辑：
        1. 跳过 Stage 1 (index 0) -> 保护低级特征。
        2. 修改 Stage 2 (index 1) -> 替换 BasicBlock。
        3. 强制修改 Stage 3 (index 2) -> 替换 Bottleneck (这是计算量大头！)。
        """
        from ..utils import BasicBlock, Bottleneck # 确保导入了 Bottleneck
        
        for stage_idx, stage in enumerate(self.i_branch_layers):
            
            # 策略：保护第一个 Stage，只改后面两个
            if stage_idx == 0:
                continue

            # 遍历这个 Stage 里的每一个 Block
            for i, block in enumerate(stage):
                
                # 跳过下采样层 (stride=2 的不要动)
                if block.downsample is not None:
                    continue

                # ---------------------------------------------------
                # 核心修改：同时处理 BasicBlock 和 Bottleneck
                # ---------------------------------------------------
                is_basic = isinstance(block, BasicBlock)
                is_bottleneck = isinstance(block, Bottleneck)
                
                if is_basic or is_bottleneck:
                    # 获取输入输出通道数
                    # BasicBlock 的结构是 conv1 -> conv2
                    # Bottleneck 的结构是 conv1 -> conv2 -> conv3
                    # 我们统一取第一层输入和最后一层输出
                    
                    if is_basic:
                        in_c = block.conv1.in_channels
                        out_c = block.conv2.out_channels
                    else: # Bottleneck
                        in_c = block.conv1.in_channels
                        out_c = block.conv3.out_channels

                    # 获取激活函数配置 (兼容性)
                    act_cfg_out = getattr(block, 'act_cfg_out', None)
                    
                    # ⚠️ 强制替换为 FasterBlock
                    # 即使原版是 3 层的 Bottleneck，我们换成 2 层的 FasterBlock 也是没问题的，
                    # 这样反而更轻量，速度更快。
                    new_block = FasterBlock(
                        in_channels=in_c,
                        channels=out_c,
                        stride=1,
                        downsample=None,
                        act_cfg_out=act_cfg_out
                    )
                    
                    # 原地替换
                    stage[i] = new_block
                    # print(f"✅ Stage {stage_idx} Block {i} ({'Bottleneck' if is_bottleneck else 'BasicBlock'}) -> FasterBlock")

