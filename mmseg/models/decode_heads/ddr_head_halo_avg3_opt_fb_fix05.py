# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_activation_layer, build_norm_layer
from torch import Tensor

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.losses import accuracy
from mmseg.models.utils import resize
from mmseg.registry import MODELS
from mmseg.utils import OptConfigType, SampleList
from mmengine.logging import MMLogger


@MODELS.register_module()
class DDRHeadHALOAvg3OptFbFix05(BaseDecodeHead):
    """
    ===========================================================================
    🏆 HALO: Holistic Asymmetric Laplacian Oracle (全局非对称拉普拉斯先知)
    ===========================================================================
    【论文核心故事线 (The Grand Story) - 跨分支反哺解耦版】：
    
    1. 痛点 (The Trap of Two-Branch Networks)：
       DDRNet 这种实时网络在推理时【仅保留 Context 主干分支】，完全丢弃 Spatial 分支。
       如果将边界 Loss 仅加在 Spatial 分支上，会导致“训练时花里胡哨，推理时原形毕露”，
       测试 mIoU 与 Baseline 完美重合。
       
    2. 创新 1 (Cross-Branch Oracle Feedback 跨分支先知反哺)：
       打破分支壁垒！我们让高分辨率的 Spatial 分支充当“边界先知(Oracle)”，
       预测出高频边界。然后，利用该预测结果掩蔽(Mask)真值标签，生成极其严苛的
       【纯边界语义标签】，并将其直接作为强监督信号【砸向 Context 分支】！
       迫使主干网络在低分辨率下也能学到极其锐利的物理边界。推理时 0 FLOPs 增加！
       
    3. 创新 2 (Decoupled Dynamic Weighting 解耦动态权重调度)：
       【核心】：将边界自监督权重 (dice_w) 与跨分支反哺权重 (fb_w) 物理层面解耦！
       虽然在 DDRNet 中两者数值同步衰减，但这一架构完美支持了 PIDNet 等大容量网络
       对边界分支施加极高初始权重(如 3.0)，同时使用稳压器(fb_w=1.0)保护主干网络的策略。
       
       本代码（DDRNet配置）采用 1.0 -> 0.5 -> 0.1 的均分三阶段衰减：
       - 阶段 1 (0~33.3%): 强先验期 (dice_w=1.0, fb_w=1.0)。
       - 阶段 2 (33.3%~66.7%): 平滑衰减期 (dice_w=0.5, fb_w=0.5)，防止冲垮主干。
       - 阶段 3 (66.7%~100%): 极速冲刺期 (dice_w=0.1, fb_w=0.1)，彻底抚平震荡。
    ===========================================================================
    """

    def __init__(self,
                 in_channels: int,
                 channels: int,
                 num_classes: int,
                 norm_cfg: OptConfigType = dict(type='BN'),
                 act_cfg: OptConfigType = dict(type='ReLU', inplace=True),
                 max_iters: int = 120000,  # <--- 【新增】传入总训练步数，触发均分引擎
                 **kwargs):
        super().__init__(
            in_channels,
            channels,
            num_classes=num_classes,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            **kwargs)

        self.head = self._make_base_head(self.in_channels, self.channels)
        self.aux_head = self._make_base_head(self.in_channels // 2, self.channels)
        self.aux_cls_seg = nn.Conv2d(self.channels, self.out_channels, kernel_size=1)
        
        # 记录总步数，用于计算均分点
        self.max_iters = max_iters
        
        # =====================================================================
        # 【修改点 1】：新增计步器与轻量化边界预测头
        # =====================================================================
        self.register_buffer('local_step', torch.tensor(0, dtype=torch.long))
        self.bd_cls_seg = nn.Conv2d(self.channels, 1, kernel_size=1)

        # =====================================================================
        # 【HALO 核心引擎】：初始化时自动计算三阶段均分解耦调度表
        # =====================================================================
        self.dynamic_schedule = self._build_avg3_schedule(self.max_iters)
        # 获取当前的全局 logger
        self.logger = MMLogger.get_current_instance()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_base_head(self, in_channels: int, channels: int) -> nn.Sequential:
        layers = [
            ConvModule(
                in_channels, channels, kernel_size=3, padding=1,
                norm_cfg=self.norm_cfg, act_cfg=self.act_cfg,
                order=('norm', 'act', 'conv')),
            build_norm_layer(self.norm_cfg, channels)[1],
            build_activation_layer(self.act_cfg),
        ]
        return nn.Sequential(*layers)

    def forward(self, inputs: Union[Tensor, Tuple[Tensor]]) -> Union[Tensor, Tuple[Tensor]]:
        if self.training:
            # DDRNet 典型的双分支输入: c3 (空间细节), c5 (全局上下文)
            c3_feat, c5_feat = inputs
            
            x_c = self.head(c5_feat)
            x_c_logit = self.cls_seg(x_c)
            
            x_s = self.aux_head(c3_feat)
            x_s_logit = self.aux_cls_seg(x_s)
            
            # =================================================================
            # 【修改点 2】：训练期提取边界 logits
            # =================================================================
            bd_logit = self.bd_cls_seg(x_s)

            return x_c_logit, x_s_logit, bd_logit
        else:
            # 推理阶段：完全丢弃边界计算，0 FLOPs 增加！
            # 注意：这里返回的 x_c 就是 Context 分支，所以必须在训练时反哺给它！
            x_c = self.head(inputs)
            x_c = self.cls_seg(x_c)
            return x_c

    # =========================================================================
    # 【修改点 3】：在线拉普拉斯边界提取 (On-the-fly Laplacian Boundary Extraction)
    # =========================================================================
    def _generate_laplacian_boundary(self, semantic_gt: Tensor, num_classes: int, 
                                     ignore_index: int = 255, dilation_size: int = 3) -> Tensor:
        valid_mask = (semantic_gt != ignore_index).float().unsqueeze(1)
        clean_gt = torch.where(semantic_gt == ignore_index, torch.zeros_like(semantic_gt), semantic_gt)
        gt_onehot = F.one_hot(clean_gt, num_classes=num_classes).permute(0, 3, 1, 2).float()
        
        laplacian_kernel = torch.tensor([
            [1.0,  1.0, 1.0],
            [1.0, -8.0, 1.0],
            [1.0,  1.0, 1.0]
        ], device=semantic_gt.device, dtype=torch.float32).view(1, 1, 3, 3).repeat(num_classes, 1, 1, 1)
        
        edge = F.conv2d(gt_onehot, laplacian_kernel, padding=1, groups=num_classes)
        edge = (torch.abs(edge) > 0.1).float()
        boundary_map = torch.max(edge, dim=1, keepdim=True)[0]
        
        if dilation_size > 1:
            pad = dilation_size // 2
            boundary_map = F.max_pool2d(boundary_map, kernel_size=dilation_size, stride=1, padding=pad)
            boundary_map = boundary_map[:, :, :valid_mask.shape[2], :valid_mask.shape[3]]
            
        return (boundary_map * valid_mask).squeeze(1)

    # =========================================================================
    # 【修改点 4】：自适应均分三阶段课程学习调度器 (解耦版)
    # =========================================================================
    def _build_avg3_schedule(self, max_iters: int) -> dict:
        """根据总训练步数，自动计算三阶段平均分割点，完美解耦内部监督与外部反哺！"""
        t1 = int(max_iters / 3.0)      # 三等分点 1 (33.3%)
        t2 = int(max_iters * 2.0 / 3.0)  # 三等分点 2 (66.7%)
        
        # 成功 77.95
        schedule = {
            # 阶段1：强先验期
            # dice_w 控制内部边界学习，fb_w 控制外部反哺力度
            0:      {'dilation': 5, 'dice_w': 1.0, 'fb_w': 0.5},
            t1:     {'dilation': 5, 'dice_w': 1.0, 'fb_w': 0.5},
            
            # 阶段2：平滑衰减期
            t1 + 1: {'dilation': 4, 'dice_w': 0.5, 'fb_w': 0.5},
            t2:     {'dilation': 4, 'dice_w': 0.5, 'fb_w': 0.5},
            
            # 阶段3：极速冲刺期 (语义解放)
            t2 + 1: {'dilation': 3, 'dice_w': 0.1, 'fb_w': 0.5},
            max_iters: {'dilation': 3, 'dice_w': 0.1, 'fb_w': 0.5}
        }

        # fb_w=1.0 恒定1.0，追求大一统 
        # 失败：77.66
        # 版本二，xx_fb_w_1
        # schedule = {
        #     # 阶段1：强先验期
        #     # dice_w 控制内部边界学习，fb_w 控制外部反哺力度
        #     0:      {'dilation': 5, 'dice_w': 1.0, 'fb_w': 1.0},
        #     t1:     {'dilation': 5, 'dice_w': 1.0, 'fb_w': 1.0},
            
        #     # 阶段2：平滑衰减期
        #     t1 + 1: {'dilation': 4, 'dice_w': 0.5, 'fb_w': 1.0},
        #     t2:     {'dilation': 4, 'dice_w': 0.5, 'fb_w': 1.0},
            
        #     # 阶段3：极速冲刺期 (语义解放)
        #     t2 + 1: {'dilation': 3, 'dice_w': 0.1, 'fb_w': 1.0},
        #     max_iters: {'dilation': 3, 'dice_w': 0.1, 'fb_w': 1.0}
        # }

        # 版本三，发现版本二最后40k不如版本1，推测仍然是dice_w的问题，所以讲dice_w提高到0.3
        # 失败：77.23
        # schedule = {
        #     # 阶段1：强先验期
        #     # dice_w 控制内部边界学习，fb_w 控制外部反哺力度
        #     0:      {'dilation': 5, 'dice_w': 1.0, 'fb_w': 1.0},
        #     t1:     {'dilation': 5, 'dice_w': 1.0, 'fb_w': 1.0},
            
        #     # 阶段2：平滑衰减期
        #     t1 + 1: {'dilation': 4, 'dice_w': 0.5, 'fb_w': 1.0},
        #     t2:     {'dilation': 4, 'dice_w': 0.5, 'fb_w': 1.0},
            
        #     # 重点在这里！阶段3不再降到0.1，而是保持 0.5 的反哺，内部稍微降一点到 0.3
        #     t2 + 1: {'dilation': 3, 'dice_w': 0.3, 'fb_w': 0.5},
        #     max_iters: {'dilation': 3, 'dice_w': 0.3, 'fb_w': 0.5}
        # }

        # 版本四，发现版本二最后40k不如版本1，推测仍然是dice_w的问题，所以讲dice_w提高到0.3， fb_w仍然=1.0
        # 失败：77.38
        # schedule = {
        #     # 阶段1：强先验期
        #     # dice_w 控制内部边界学习，fb_w 控制外部反哺力度
        #     0:      {'dilation': 5, 'dice_w': 1.0, 'fb_w': 1.0},
        #     t1:     {'dilation': 5, 'dice_w': 1.0, 'fb_w': 1.0},
            
        #     # 阶段2：平滑衰减期
        #     t1 + 1: {'dilation': 4, 'dice_w': 0.5, 'fb_w': 1.0},
        #     t2:     {'dilation': 4, 'dice_w': 0.5, 'fb_w': 1.0},
            
        #     # 重点在这里！阶段3不再降到0.1，而是保持 0.5 的反哺，内部稍微降一点到 0.3
        #     t2 + 1: {'dilation': 3, 'dice_w': 0.3, 'fb_w': 1.0},
        #     max_iters: {'dilation': 3, 'dice_w': 0.3, 'fb_w': 1.0}
        # }

        # 版本五 在版本1基础上小改
        # 77.84
        # ddrnet_workdir/halo-ddrnet_23-slim_in1k-pre-halo_avg3-opt_1xb12-120k_cityscapes-1024x1024-dilation-4
        # schedule = {
        #     # 阶段1：强先验期
        #     # dice_w 控制内部边界学习，fb_w 控制外部反哺力度
        #     0:      {'dilation': 5, 'dice_w': 1.0, 'fb_w': 1.0},
        #     t1:     {'dilation': 5, 'dice_w': 1.0, 'fb_w': 1.0},
            
        #     # 阶段2：平滑衰减期
        #     t1 + 1: {'dilation': 4, 'dice_w': 0.5, 'fb_w': 0.5},
        #     t2:     {'dilation': 4, 'dice_w': 0.5, 'fb_w': 0.5},
            
        #     # 阶段3：极速冲刺期 (语义解放) - 不要降到 3！让 DDRNet 停留在 Dilation=4。略粗一点的边界对轻量级网络更友好，梯度更平滑。
        #     t2 + 1: {'dilation': 4, 'dice_w': 0.1, 'fb_w': 0.1},
        #     max_iters: {'dilation': 4, 'dice_w': 0.1, 'fb_w': 0.1}
        # }

        # 版本六 在版本1基础上修改fb_w=0.3
        # 77.67
        # schedule = {
        #     # 阶段1：强先验期
        #     # dice_w 控制内部边界学习，fb_w 控制外部反哺力度
        #     0:      {'dilation': 5, 'dice_w': 1.0, 'fb_w': 1.0},
        #     t1:     {'dilation': 5, 'dice_w': 1.0, 'fb_w': 1.0},
            
        #     # 阶段2：平滑衰减期
        #     t1 + 1: {'dilation': 4, 'dice_w': 0.5, 'fb_w': 0.5},
        #     t2:     {'dilation': 4, 'dice_w': 0.5, 'fb_w': 0.5},
            
        #     # 阶段3：极速冲刺期 (语义解放)
        #     t2 + 1: {'dilation': 3, 'dice_w': 0.1, 'fb_w': 0.3},
        #     max_iters: {'dilation': 3, 'dice_w': 0.1, 'fb_w': 0.3}
        # }

        # 版本7，看80k左右的趋势，仍然压着baseline走，所以直接用40k-80k的参数
        # 77.63
        # schedule = {
        #     # 阶段1：强先验期
        #     # dice_w 控制内部边界学习，fb_w 控制外部反哺力度
        #     0:      {'dilation': 5, 'dice_w': 1.0, 'fb_w': 1.0},
        #     t1:     {'dilation': 5, 'dice_w': 1.0, 'fb_w': 1.0},
            
        #     # 阶段2：平滑衰减期
        #     t1 + 1: {'dilation': 4, 'dice_w': 0.5, 'fb_w': 0.5},
        #     t2:     {'dilation': 4, 'dice_w': 0.5, 'fb_w': 0.5},
            
        #     # 阶段3：极速冲刺期 (语义解放)
        #     # t2 + 1: {'dilation': 3, 'dice_w': 0.1, 'fb_w': 0.3},
        #     # max_iters: {'dilation': 3, 'dice_w': 0.1, 'fb_w': 0.3}
        #     t2 + 1: {'dilation': 4, 'dice_w': 0.5, 'fb_w': 0.5},
        #     max_iters: {'dilation': 4, 'dice_w': 0.5, 'fb_w': 0.5}
        # }

        # 版本8 在版本1基础上修改fb_w=0.3 dice_w=0.3
        # 77.65
        # schedule = {
        #     # 阶段1：强先验期
        #     # dice_w 控制内部边界学习，fb_w 控制外部反哺力度
        #     0:      {'dilation': 5, 'dice_w': 1.0, 'fb_w': 1.0},
        #     t1:     {'dilation': 5, 'dice_w': 1.0, 'fb_w': 1.0},
            
        #     # 阶段2：平滑衰减期
        #     t1 + 1: {'dilation': 4, 'dice_w': 0.5, 'fb_w': 0.5},
        #     t2:     {'dilation': 4, 'dice_w': 0.5, 'fb_w': 0.5},
            
        #     # 阶段3：极速冲刺期 (语义解放)
        #     t2 + 1: {'dilation': 3, 'dice_w': 0.3, 'fb_w': 0.3},
        #     max_iters: {'dilation': 3, 'dice_w': 0.3, 'fb_w': 0.3}
        # }

        # 版本9: 提取 Spatial 分支 (先知) 预测的边界掩码 bd_pred_mask = (pred_sigmoid > 0.8)
        # 版本9 bd_pred_mask = (pred_sigmoid > 0.8) 之前是0.5
        # ddr_head_halo_avg3_opt_mask08.py
        # 77.53
        # schedule = {
        #     # 阶段1：强先验期
        #     # dice_w 控制内部边界学习，fb_w 控制外部反哺力度
        #     0:      {'dilation': 5, 'dice_w': 1.0, 'fb_w': 1.0},
        #     t1:     {'dilation': 5, 'dice_w': 1.0, 'fb_w': 1.0},
            
        #     # 阶段2：平滑衰减期
        #     t1 + 1: {'dilation': 4, 'dice_w': 0.5, 'fb_w': 0.5},
        #     t2:     {'dilation': 4, 'dice_w': 0.5, 'fb_w': 0.5},
            
        #     # 阶段3：极速冲刺期 (语义解放)
        #     t2 + 1: {'dilation': 3, 'dice_w': 0.1, 'fb_w': 0.1},
        #     max_iters: {'dilation': 3, 'dice_w': 0.1, 'fb_w': 0.1}
        # }

        # 版本10 开始动手stage2 
        # 提前解放版：压缩 Stage 1 和 Stage 2 的时间
        # t1 = 30000  # 原来可能是 40000
        # t2 = 60000  # 原来可能是 80000
        # # 77.51
        # schedule = {
        #     0:         {'dilation': 5, 'dice_w': 1.0, 'fb_w': 1.0},
        #     t1:        {'dilation': 5, 'dice_w': 1.0, 'fb_w': 1.0},
        #     t1 + 1:    {'dilation': 4, 'dice_w': 0.5, 'fb_w': 0.5},
        #     t2:        {'dilation': 4, 'dice_w': 0.5, 'fb_w': 0.5},
        #     # 从 60k 到 120k，整整一半的时间都在进行 Semantic Liberation！
        #     t2 + 1:    {'dilation': 3, 'dice_w': 0.1, 'fb_w': 0.1},
        #     max_iters: {'dilation': 3, 'dice_w': 0.1, 'fb_w': 0.1}
        # }

        print(schedule)

        return schedule

    def _get_dynamic_params(self, current_step: int) -> Tuple[int, float, float]:
        """读取动态参数，执行平滑衰减，返回完全解耦的参数"""
        schedule = self.dynamic_schedule
        milestones = sorted(schedule.keys())
        
        if current_step <= milestones[0]: 
            cfg = schedule[milestones[0]]
            return cfg['dilation'], cfg['dice_w'], cfg['fb_w']
        if current_step >= milestones[-1]: 
            cfg = schedule[milestones[-1]]
            return cfg['dilation'], cfg['dice_w'], cfg['fb_w']
            
        start_step, end_step = milestones[0], milestones[-1]
        for i in range(len(milestones) - 1):
            if milestones[i] <= current_step < milestones[i+1]:
                start_step, end_step = milestones[i], milestones[i+1]
                break
                
        start_cfg, end_cfg = schedule[start_step], schedule[end_step]
        progress = (current_step - start_step) / float(end_step - start_step)
        
        cur_dilation = start_cfg['dilation']
        # 独立执行线性插值，彻底解耦
        cur_dice_w = start_cfg['dice_w'] + progress * (end_cfg['dice_w'] - start_cfg['dice_w'])
        cur_fb_w = start_cfg['fb_w'] + progress * (end_cfg['fb_w'] - start_cfg['fb_w'])
        
        return cur_dilation, cur_dice_w, cur_fb_w

    def loss_by_feat(self, seg_logits: Tuple[Tensor], batch_data_samples: SampleList) -> dict:
        
        # =====================================================================
        # 【修改点 5】：最终损失函数计算逻辑 (跨分支反哺解耦版)
        # =====================================================================
        
        if self.training:
            self.local_step += 1
        current_step = self.local_step.item()
        
        # 获取解耦后的动态参数：cur_fb_w 现已独立！
        cur_dilation, cur_dice_w, cur_fb_w = self._get_dynamic_params(current_step)


        if current_step % 50 == 0:
            self.logger.info('cur_dilation=' + str(cur_dilation) + ',cur_dice_w=' + str(cur_dice_w) + ', cur_fb_w=' + str(cur_fb_w))
        
        loss = dict()
        context_logit, spatial_logit, bd_logit = seg_logits
        seg_label = self._stack_batch_gt(batch_data_samples)

        context_logit = resize(context_logit, size=seg_label.shape[2:], mode='bilinear', align_corners=self.align_corners)
        spatial_logit = resize(spatial_logit, size=seg_label.shape[2:], mode='bilinear', align_corners=self.align_corners)
        bd_logit = resize(bd_logit, size=seg_label.shape[2:], mode='bilinear', align_corners=self.align_corners)
        
        seg_label = seg_label.squeeze(1)

        # 实时生成拉普拉斯边界标签 (基于动态 Dilation)
        bd_label = self._generate_laplacian_boundary(
            seg_label, num_classes=self.num_classes, ignore_index=self.ignore_index, dilation_size=cur_dilation
        )

        # 1. 计算原版 DDRNet 的全局语义损失 (让它们看全图，绝不遮挡！)
        loss['loss_context'] = self.loss_decode[0](context_logit, seg_label)
        loss['loss_spatial'] = self.loss_decode[1](spatial_logit, seg_label)
        
        # 2. 边界分支内部 Loss (使用 cur_dice_w)
        bce_loss = F.binary_cross_entropy_with_logits(bd_logit.squeeze(1), bd_label)
        
        pred_sigmoid = torch.sigmoid(bd_logit[:, 0, :, :])
        valid_mask = (seg_label != self.ignore_index).float()
        intersection = (pred_sigmoid * bd_label * valid_mask).sum(dim=(1, 2))
        union = (pred_sigmoid * valid_mask).sum(dim=(1, 2)) + (bd_label * valid_mask).sum(dim=(1, 2))
        dice_loss = (1.0 - (2.0 * intersection + 1e-5) / (union + 1e-5)).mean()
        
        # 核心：内部监督使用独立加权
        loss['loss_bd_laplacian'] = bce_loss + cur_dice_w * dice_loss
        
        # =====================================================================
        # 🚀 绝杀机制：跨分支先知反哺 (Cross-Branch Oracle Feedback) 🚀
        # =====================================================================
        # 提取 Spatial 分支 (先知) 预测的边界掩码
        bd_pred_mask = (pred_sigmoid > 0.5)
        
        # 制作“纯边界语义标签”：只保留先知认为是边界的地方，其他地方全部填 255 (忽略)
        filler = torch.ones_like(seg_label) * self.ignore_index
        halo_label = torch.where(bd_pred_mask, seg_label, filler)
        
        # 【致胜一击】：将严苛的边界标签砸向 Context 分支！
        # 使用独立的 cur_fb_w 控制反哺力度，彻底解耦！
        loss['loss_halo_feedback'] = self.loss_decode[0](context_logit, halo_label) * cur_fb_w
        
        # 3. 算个准确率汇报一下 (基于主干 Context 分支的预测)
        loss['acc_seg'] = accuracy(context_logit, seg_label, ignore_index=self.ignore_index)

        return loss

