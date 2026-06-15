# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from torch import Tensor

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.losses import accuracy
from mmseg.models.utils import resize
from mmseg.registry import MODELS
from mmseg.utils import OptConfigType, SampleList

@MODELS.register_module()
class BiSeNetHALOHead(BaseDecodeHead):
    """
    ===========================================================================
    🏆 HALO: Harmonic Asymmetric Laplacian Optimization (谐波非对称拉普拉斯优化)
    ===========================================================================
    这是一个用于实时语义分割的“神级”动态训练插件。它包含了四大核心创新：
    
    1. 【非对称解耦 (Asymmetric Decoupling)】：
       不再把边界Loss和语义Loss混在一起算（会导致梯度打架）。我们专门把 Detail(细节) 分支
       抽出来算边界，把 Fused(融合) 分支拿来算语义。各司其职，互不干扰！
       
    2. 【拉普拉斯高频先验 (Laplacian Prior)】：
       用拉普拉斯算子从真实的语义标签(GT)中，实时提取极其锐利的物理边界，作为“私人教练”
       去死死盯住 Detail 分支，逼它学好高频细节。
       
    3. 【自动黄金分割 (Golden Ratio Scheduling)】：
       告别硬编码！无论你训练 16万步还是 8万步，自动按 38.2% - 23.6% - 38.2% 的黄金比例
       切分为三个训练阶段，极其符合深度学习的收敛规律。
       
    4. 【由粗到细空间课程 (Coarse-to-Fine Curriculum)】：
       边界的粗细(Dilation)按 5 -> 4 -> 3 阶梯式下降。就像教小孩画画，先画粗犷的轮廓，
       再描绘中等线条，最后精雕细琢微观细节。不仅防过拟合，还能让 mIoU 稳步飙升！
       
    ★ 最爽的是：在测试(推理)阶段，这个 Head 里的边界计算会全部丢弃，FPS 不掉一帧 (0 FLOPs)！
    ===========================================================================
    """
    def __init__(self,
                 in_channels,            # 接收一个列表，比如 [128, 128]，分别对应 Detail 和 Fused 特征
                 channels: int,          # Head 内部的通道数，比如 1024
                 num_classes: int,       # 类别数，Cityscapes 是 19
                 num_convs: int = 2,     # 语义分支用的卷积层数
                 concat_input: bool = True,
                 max_iters: int = 160000, # 【创新点3】传入总训练步数，用于自动算黄金分割点
                 **kwargs):
                 
        # 告诉底层的 MMSegmentation 框架：“别慌，我要接收的是多个特征图（列表），不是单个！”
        kwargs['input_transform'] = 'multiple_select'
        
        super().__init__(
            in_channels=in_channels, 
            channels=channels, 
            num_classes=num_classes, 
            **kwargs)
        
        self.in_channels_list = in_channels 
        self.num_convs = num_convs
        self.concat_input = concat_input
        self.max_iters = max_iters  # 记住总步数

        # ---------------------------------------------------------------------
        # 模块 A：全局语义分支 (Context/Fused) -> 负责“看懂大局” (比如认出那是辆车)
        # ---------------------------------------------------------------------
        self.context_convs = self._make_fcn_blocks(self.in_channels_list[1], self.channels)
        if self.concat_input:
            self.context_cat = ConvModule(
                self.in_channels_list[1] + self.channels, self.channels,
                kernel_size=3, padding=1, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)

        # ---------------------------------------------------------------------
        # 模块 B：高频细节分支 (Detail) -> 负责“死磕边缘” (纯粹的高频特征提取器)
        # ---------------------------------------------------------------------
        self.bd_conv = ConvModule(
            self.in_channels_list[0], self.channels, kernel_size=3, padding=1,
            norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
        self.bd_cls_seg = nn.Conv2d(self.channels, 1, kernel_size=1) # 输出单通道的边界预测图

        # 注册一个在 GPU 上的步数计数器，随模型一起保存和加载
        self.register_buffer('local_step', torch.tensor(0, dtype=torch.long))
        
        # 自动构建黄金分割调度表
        self.dynamic_schedule = self._build_golden_schedule(self.max_iters)

    def _make_fcn_blocks(self, in_c, out_c):
        """构建连续的卷积层，用于处理语义特征"""
        if self.num_convs == 0:
            return nn.Identity()
        convs = [ConvModule(in_c, out_c, kernel_size=3, padding=1, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)]
        for _ in range(self.num_convs - 1):
            convs.append(ConvModule(out_c, out_c, kernel_size=3, padding=1, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg))
        return nn.Sequential(*convs)

    def _build_golden_schedule(self, max_iters: int) -> dict:
        """
        【HALO 核心引擎】：自动计算黄金分割点
        """
        t1 = int(max_iters * 0.382)  # 第一个黄金点 (比如 160k 的 61120 步)
        t2 = int(max_iters * 0.618)  # 第二个黄金点 (比如 160k 的 98880 步)
        
        schedule = {
            # Phase 1: 0 -> 38.2% (严苛打底期)
            # 策略：极粗的边界(5) + 极重的惩罚(3.0)。强迫网络先把物体的宏观大框架搭好。
            0:      {'dilation': 5, 'dice_w': 3.0},
            t1:     {'dilation': 5, 'dice_w': 3.0},
            
            # Phase 2: 38.2% -> 61.8% (平滑过渡期)
            # 策略：边界变细(4) + 惩罚缓慢降到1.0。开始剥离粗糙边缘，向高精度过渡。
            t1 + 1: {'dilation': 4, 'dice_w': 3.0}, 
            t2:     {'dilation': 4, 'dice_w': 1.0},
            
            # Phase 3: 61.8% -> 100% (算力全开冲刺期)
            # 策略：极细的边界(3) + 极轻的惩罚(0.5)。彻底卸下沙袋，网络算力全开，狂刷 mIoU！
            t2 + 1: {'dilation': 3, 'dice_w': 0.5},
            max_iters: {'dilation': 3, 'dice_w': 0.5} 
        }
        return schedule

    def _get_dynamic_params(self, current_step: int) -> Tuple[int, float]:
        """根据当前步数，去调度表里查出当前的边界粗细(dilation)和损失权重(dice_w)"""
        schedule = self.dynamic_schedule
        milestones = sorted(schedule.keys())
        
        # 边界保护
        if current_step <= milestones[0]: return schedule[milestones[0]]['dilation'], schedule[milestones[0]]['dice_w']
        if current_step >= milestones[-1]: return schedule[milestones[-1]]['dilation'], schedule[milestones[-1]]['dice_w']
            
        # 找到当前步数所在的区间
        start_step, end_step = milestones[0], milestones[-1]
        for i in range(len(milestones) - 1):
            if milestones[i] <= current_step < milestones[i+1]:
                start_step, end_step = milestones[i], milestones[i+1]
                break
                
        # 计算在这个区间内的进度 (0.0 到 1.0)
        progress = (current_step - start_step) / float(end_step - start_step)
        
        # Dilation 保持整数台阶下降 (5 -> 4 -> 3)
        cur_dilation = schedule[start_step]['dilation']
        # Dice Weight 进行平滑线性插值衰减 (3.0 -> ... -> 0.5)
        cur_dice_w = schedule[start_step]['dice_w'] + progress * (schedule[end_step]['dice_w'] - schedule[start_step]['dice_w'])
        
        return cur_dilation, cur_dice_w

    def forward(self, inputs):
        """前向传播"""
        # 自动根据 config 里的 in_index=[5, 0] 把我们要的两个特征图抓出来
        inputs = self._transform_inputs(inputs)
        detail_feat, fused_feat = inputs[0], inputs[1]

        # 1. 正常计算语义分类 (所有人都要做这一步)
        x_c = self.context_convs(fused_feat)
        if self.concat_input:
            x_c = self.context_cat(torch.cat([fused_feat, x_c], dim=1))
        x_c_logit = self.cls_seg(x_c)

        # 2. 【HALO 零开销魔法】：只在训练时算边界，测试时直接跳过！
        if self.training:
            x_s = self.bd_conv(detail_feat)
            bd_logit = self.bd_cls_seg(x_s)
            return x_c_logit, bd_logit
        else:
            return x_c_logit

    def _generate_laplacian_boundary(self, semantic_gt: Tensor, num_classes: int, 
                                     ignore_index: int = 255, dilation_size: int = 3) -> Tensor:
        """用拉普拉斯算子(Laplacian)动态生成物理边界标签"""
        # 过滤掉 ignore_index (比如背景或无效像素)
        valid_mask = (semantic_gt != ignore_index).float().unsqueeze(1)
        clean_gt = torch.where(semantic_gt == ignore_index, torch.zeros_like(semantic_gt), semantic_gt)
        
        # 把标签变成 one-hot 格式，方便卷积
        gt_onehot = F.one_hot(clean_gt, num_classes=num_classes).permute(0, 3, 1, 2).float()
        
        # 核心：拉普拉斯边缘检测算子 (中心为 -8，四周为 1)
        laplacian_kernel = torch.tensor([[1.0, 1.0, 1.0], [1.0, -8.0, 1.0], [1.0, 1.0, 1.0]], 
                                        device=semantic_gt.device, dtype=torch.float32).view(1, 1, 3, 3).repeat(num_classes, 1, 1, 1)
        
        # 卷一下，边缘就出来了
        edge = F.conv2d(gt_onehot, laplacian_kernel, padding=1, groups=num_classes)
        edge = (torch.abs(edge) > 0.1).float()
        
        # 把所有类别的边缘合并成一张单通道的图
        boundary_map = torch.max(edge, dim=1, keepdim=True)[0]
        
        # 【动态课程】：根据当前的 dilation_size 把边缘变粗 (Phase 1 粗，Phase 3 细)
        if dilation_size > 1:
            pad = dilation_size // 2
            boundary_map = F.max_pool2d(boundary_map, kernel_size=dilation_size, stride=1, padding=pad)
            boundary_map = boundary_map[:, :, :valid_mask.shape[2], :valid_mask.shape[3]]
            
        return (boundary_map * valid_mask).squeeze(1)

    def loss_by_feat(self, seg_logits: Tuple[Tensor], batch_data_samples: SampleList) -> dict:
        """计算最终的 Loss"""
        if self.training:
            self.local_step += 1
        current_step = self.local_step.item()
        
        # 获取当前步数对应的动态参数 (粗细 & 权重)
        cur_dilation, cur_dice_w = self._get_dynamic_params(current_step)

        loss = dict()
        context_logit, bd_logit = seg_logits
        seg_label = self._stack_batch_gt(batch_data_samples)

        # 把预测结果放大到和原始图片一样大，方便算 Loss
        context_logit = resize(context_logit, size=seg_label.shape[2:], mode='bilinear', align_corners=self.align_corners)
        bd_logit = resize(bd_logit, size=seg_label.shape[2:], mode='bilinear', align_corners=self.align_corners)
        seg_label = seg_label.squeeze(1)

        # 实时生成拉普拉斯边界标签 (根据当前的 cur_dilation 决定粗细)
        bd_label = self._generate_laplacian_boundary(
            seg_label, num_classes=self.num_classes, ignore_index=self.ignore_index, dilation_size=cur_dilation
        )

        # -------------------------------------------------------------
        # Loss 1: 语义分类 Loss (CE Loss 等，挂在 Context 分支上)
        # 兼容了 MMSegmentation 的多种 Loss 组合写法
        # -------------------------------------------------------------
        if isinstance(self.loss_decode, nn.ModuleList):
            for loss_module in self.loss_decode:
                loss[loss_module.loss_name] = loss_module(context_logit, seg_label, ignore_index=self.ignore_index)
        else:
            loss['loss_ce'] = self.loss_decode(context_logit, seg_label, ignore_index=self.ignore_index)
        
        # -------------------------------------------------------------
        # Loss 2: 高频边界 Loss (BCE + Dice，专门挂在 Detail 分支上)
        # -------------------------------------------------------------
        bce_loss = F.binary_cross_entropy_with_logits(bd_logit.squeeze(1), bd_label)
        
        pred_sigmoid = torch.sigmoid(bd_logit[:, 0, :, :])
        valid_mask = (seg_label != self.ignore_index).float()
        intersection = (pred_sigmoid * bd_label * valid_mask).sum(dim=(1, 2))
        union = (pred_sigmoid * valid_mask).sum(dim=(1, 2)) + (bd_label * valid_mask).sum(dim=(1, 2))
        dice_loss = (1.0 - (2.0 * intersection + 1e-5) / (union + 1e-5)).mean()
        
        # 动态加权：核心创新点发力的地方
        loss['loss_bd_laplacian'] = bce_loss + cur_dice_w * dice_loss
        
        # 记录一下当前的分割准确率，方便在日志里看
        loss['acc_seg'] = accuracy(context_logit, seg_label, ignore_index=self.ignore_index)

        return loss
