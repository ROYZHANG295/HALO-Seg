# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F  
from mmcv.cnn import ConvModule, build_activation_layer, build_norm_layer
from mmengine.model import BaseModule
from torch import Tensor

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.losses import accuracy
from mmseg.models.utils import resize
from mmseg.registry import MODELS
from mmseg.utils import OptConfigType, SampleList
from mmengine.logging import MMLogger


class BasePIDHead(BaseModule):
    """Base class for PID head."""
    
    def __init__(self,
                 in_channels: int,
                 channels: int,
                 norm_cfg: OptConfigType = dict(type='BN'),
                 act_cfg: OptConfigType = dict(type='ReLU', inplace=True),
                 init_cfg: OptConfigType = None):
        super().__init__(init_cfg)
        self.conv = ConvModule(
            in_channels,
            channels,
            kernel_size=3,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            order=('norm', 'act', 'conv')
        )
        _, self.norm = build_norm_layer(norm_cfg, num_features=channels)
        self.act = build_activation_layer(act_cfg)

    def forward(self, x: Tensor, cls_seg: Optional[nn.Module]) -> Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        if cls_seg is not None:
            x = cls_seg(x)
        return x


@MODELS.register_module()
class PIDHeadHALOSameDDRAvg3Opt(BaseDecodeHead):
    """
    ===========================================================================
    🏆 HALO: Topology-Aware Boundary Supervision (通用解耦框架版 - PIDNet)
    ===========================================================================
    【Universal Framework 跨架构大一统】
    
    为了证明 HALO 框架的绝对普适性，PIDNet 采用了与 DDRNet 完全一致的【权重解耦架构】，
    但根据 PIDNet D分支容量极大的物理特性，配置了专属的“架构感知(Architecture-Aware)”参数：
    
    1. 内部监督 (dice_w): 采用 3.0 -> 1.0 -> 0.5 的激进衰减。
       前期下猛药(3.0)瞬间唤醒庞大的 D 分支，迫使其画出极其锐利的物理边界；
       后期平滑降权，防止过拟合。
       
    2. 外部反哺 (fb_w): 采用 1.0 -> 1.0 -> 1.0 的恒定稳压器。
       无论 D 分支内部的 loss 权重有多高，砸向 I 分支(主干)的反馈权重始终锁定在 1.0。
       这完美解决了前期 3.0 权重带来的“毒反馈(Toxic Feedback)”和“语义坍塌”问题！
       
    3. 三阶段硬跳变调度 (Universal Piecewise-Constant Scheduler):
       - 阶段 1 (0~33.3%): Dilation=5, dice_w=3.0, fb_w=1.0
       - 阶段 2 (33.3%~66.7%): Dilation=4, dice_w=1.0, fb_w=1.0
       - 阶段 3 (66.7%~100%): Dilation=3, dice_w=0.5, fb_w=1.0
    ===========================================================================
    """
    
    def __init__(self,
                 in_channels: int,
                 channels: int,
                 num_classes: int,
                 norm_cfg: OptConfigType = dict(type='BN'),
                 act_cfg: OptConfigType = dict(type='ReLU', inplace=True),
                 max_iters: int = 120000, 
                 **kwargs):
        super().__init__(
            in_channels,
            channels,
            num_classes=num_classes,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            **kwargs
        )
        
        self.register_buffer('local_step', torch.tensor(0, dtype=torch.long))
        self.max_iters = max_iters

        self.i_head = BasePIDHead(in_channels, channels, norm_cfg, act_cfg)
        self.p_head = BasePIDHead(in_channels // 2, channels, norm_cfg, act_cfg)
        self.d_head = BasePIDHead(in_channels // 2, in_channels // 4, norm_cfg)
        
        self.p_cls_seg = nn.Conv2d(channels, self.out_channels, kernel_size=1)
        self.d_cls_seg = nn.Conv2d(in_channels // 4, 1, kernel_size=1)

        # 【统一化】：采用解耦后的三阶段均分调度表
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

    def forward(self, inputs: Union[Tensor, Tuple[Tensor]]) -> Union[Tensor, Tuple[Tensor]]:
        if self.training:
            x_p, x_i, x_d = inputs
            x_p = self.p_head(x_p, self.p_cls_seg)
            x_i = self.i_head(x_i, self.cls_seg)
            x_d = self.d_head(x_d, self.d_cls_seg)
            return x_p, x_i, x_d
        else:
            return self.i_head(inputs, self.cls_seg)

    def _stack_batch_gt(self, batch_data_samples: SampleList) -> Tensor:
        gt_semantic_segs = [
            data_sample.gt_sem_seg.data for data_sample in batch_data_samples
        ]
        return torch.stack(gt_semantic_segs, dim=0)

    # =========================================================================
    # 1. 在线拉普拉斯边界提取
    # =========================================================================
    def _generate_laplacian_boundary(self, 
                                     semantic_gt: Tensor, 
                                     num_classes: int, 
                                     ignore_index: int = 255, 
                                     dilation_size: int = 3) -> Tensor:
        
        valid_mask = (semantic_gt != ignore_index).float().unsqueeze(1)
        clean_gt = torch.where(
            semantic_gt == ignore_index, 
            torch.zeros_like(semantic_gt), 
            semantic_gt
        )
        
        gt_onehot = F.one_hot(clean_gt, num_classes=num_classes)
        gt_onehot = gt_onehot.permute(0, 3, 1, 2).float()
        
        laplacian_kernel = torch.tensor([
            [1.0,  1.0, 1.0],
            [1.0, -8.0, 1.0],
            [1.0,  1.0, 1.0]
        ], device=semantic_gt.device, dtype=torch.float32).view(1, 1, 3, 3)
        
        laplacian_kernel = laplacian_kernel.repeat(num_classes, 1, 1, 1)
        
        edge = F.conv2d(gt_onehot, laplacian_kernel, padding=1, groups=num_classes)
        edge = (torch.abs(edge) > 0.1).float()
        boundary_map = torch.max(edge, dim=1, keepdim=True)[0]
        
        if dilation_size > 1:
            pad = dilation_size // 2
            boundary_map = F.max_pool2d(
                boundary_map, 
                kernel_size=dilation_size, 
                stride=1, 
                padding=pad
            )
            boundary_map = boundary_map[:, :, :valid_mask.shape[2], :valid_mask.shape[3]]
            
        boundary_map = boundary_map * valid_mask
        return boundary_map.squeeze(1)

    # =========================================================================
    # 2. 大一统的分阶段硬跳变解耦调度器 (Decoupled Piecewise-Constant Scheduler)
    # =========================================================================
    def _build_avg3_schedule(self, max_iters: int) -> dict:
        t1 = int(max_iters / 3.0)
        t2 = int(max_iters * 2.0 / 3.0)

        # 版本1【解耦化】：dice_w 采用 3.0->1.0->0.5，fb_w 恒定 1.0 保护主干
        # schedule = {
        #     0:         {'dilation': 5, 'dice_w': 3.0, 'fb_w': 1.0},
        #     t1:        {'dilation': 5, 'dice_w': 3.0, 'fb_w': 1.0},
        #     t1 + 1:    {'dilation': 4, 'dice_w': 1.0, 'fb_w': 0.5},
        #     t2:        {'dilation': 4, 'dice_w': 1.0, 'fb_w': 0.5},
        #     t2 + 1:    {'dilation': 3, 'dice_w': 0.5, 'fb_w': 0.1},
        #     max_iters: {'dilation': 3, 'dice_w': 0.5, 'fb_w': 0.1}
        # }

        # 版本2 跑出了 78.61,但是后劲不足
        # schedule = {
        #     0:         {'dilation': 5, 'dice_w': 3.0, 'fb_w': 1.0},
        #     t1:        {'dilation': 5, 'dice_w': 3.0, 'fb_w': 1.0},
        #     t1 + 1:    {'dilation': 4, 'dice_w': 1.0, 'fb_w': 1.0},
        #     t2:        {'dilation': 4, 'dice_w': 1.0, 'fb_w': 1.0},
        #     t2 + 1:    {'dilation': 3, 'dice_w': 0.5, 'fb_w': 1.0},
        #     max_iters: {'dilation': 3, 'dice_w': 0.5, 'fb_w': 1.0}
        # }

        # 版本3 虽然上一个版本跑出 78.61，但是明显后劲不足
        # 策略：减缓 Dice Loss 的衰减，全程保持较高的边界约束
        # pidnet_workdir/halo-pidnet-s-halo-same-ddr-1xb12-120k_1024x1024-cityscapes-FULL-fb_w-1_dice_w15-10
        # pidnet_workdir/halo-pidnet-s-halo-same-ddr-1xb12-120k_1024x1024-cityscapes-FULL-fb_w-1_dice_w15-10-best-run2
        # 成功跑出 79.08的高分
        schedule = {
            0:         {'dilation': 5, 'dice_w': 3.0, 'fb_w': 1.0},
            t1:        {'dilation': 5, 'dice_w': 3.0, 'fb_w': 1.0},
            t1 + 1:    {'dilation': 4, 'dice_w': 1.5, 'fb_w': 1.0},  # 从 1.0 提高到 1.5
            t2:        {'dilation': 4, 'dice_w': 1.5, 'fb_w': 1.0},
            t2 + 1:    {'dilation': 3, 'dice_w': 1.0, 'fb_w': 1.0},  # 尾段保底 1.0，而不是 0.5
            max_iters: {'dilation': 3, 'dice_w': 1.0, 'fb_w': 1.0}
        }

        # 版本4 消融实验，动态膨胀5-4-3，dice_w固定3.0，fb_w固定1.0，证明 动态膨胀有效果
        # pidnet_workdir/halo-pidnet-s-halo-same-ddr-1xb12-120k_1024x1024-cityscapes-FULL-dilation543-dice3-fb1-v4
        # schedule = {
        #     0:         {'dilation': 5, 'dice_w': 3.0, 'fb_w': 1.0},
        #     t1:        {'dilation': 5, 'dice_w': 3.0, 'fb_w': 1.0},
        #     t1 + 1:    {'dilation': 4, 'dice_w': 3.0, 'fb_w': 1.0}, 
        #     t2:        {'dilation': 4, 'dice_w': 3.0, 'fb_w': 1.0},
        #     t2 + 1:    {'dilation': 3, 'dice_w': 3.0, 'fb_w': 1.0},  
        #     max_iters: {'dilation': 3, 'dice_w': 3.0, 'fb_w': 1.0}
        # }
        return schedule

    def _get_dynamic_params(self, current_step: int) -> Tuple[int, float, float]:
        schedule = self.dynamic_schedule
        milestones = sorted(schedule.keys())
        
        start_step = milestones[0]
        for i in range(len(milestones) - 1):
            if milestones[i] <= current_step <= milestones[i+1]:
                start_step = milestones[i]
                break
                
        cfg = schedule[start_step]
        # 返回完全解耦的三个参数
        return cfg['dilation'], cfg['dice_w'], cfg['fb_w']

    def loss_by_feat(self, seg_logits: Tuple[Tensor], batch_data_samples: SampleList) -> dict:
        
        if self.training:
            self.local_step += 1
        current_step = self.local_step.item()
        
        # 接收解耦后的三个参数
        cur_dilation, cur_dice_w, cur_fb_w = self._get_dynamic_params(current_step)

        if current_step % 50 == 0:
            self.logger.info('cur_dilation=' + str(cur_dilation) + ',cur_dice_w=' + str(cur_dice_w) + ', cur_fb_w=' + str(cur_fb_w))

        loss = dict()
        p_logit, i_logit, d_logit = seg_logits
        sem_label = self._stack_batch_gt(batch_data_samples)

        p_logit = resize(input=p_logit, size=sem_label.shape[2:], mode='bilinear', align_corners=self.align_corners)
        i_logit = resize(input=i_logit, size=sem_label.shape[2:], mode='bilinear', align_corners=self.align_corners)
        d_logit = resize(input=d_logit, size=sem_label.shape[2:], mode='bilinear', align_corners=self.align_corners)
        
        sem_label = sem_label.squeeze(1)

        bd_label = self._generate_laplacian_boundary(
            sem_label, num_classes=self.num_classes, ignore_index=self.ignore_index, dilation_size=cur_dilation
        )

        # 1. 基础全局语义损失 (P 分支与 I 分支)
        loss['loss_sem_p'] = self.loss_decode[0](p_logit, sem_label, ignore_index=self.ignore_index)
        loss['loss_sem_i'] = self.loss_decode[1](i_logit, sem_label)
        
        # 2. 联合边界损失 (D 分支内部监督)
        bce_loss = self.loss_decode[2](d_logit, bd_label)
        pred_sigmoid = torch.sigmoid(d_logit[:, 0, :, :])
        
        valid_mask = (sem_label != self.ignore_index).float()
        intersection = (pred_sigmoid * bd_label * valid_mask).sum(dim=(1, 2))
        union = (pred_sigmoid * valid_mask).sum(dim=(1, 2)) + (bd_label * valid_mask).sum(dim=(1, 2))
        dice_loss = (1.0 - (2.0 * intersection + 1e-5) / (union + 1e-5)).mean()
        
        # 内部监督使用激进的 cur_dice_w
        loss['loss_bd_laplacian'] = bce_loss + cur_dice_w * dice_loss
        
        # =====================================================================
        # 🚀 绝杀机制：跨分支先知反哺 (Cross-Branch Oracle Feedback) 🚀
        # =====================================================================
        # 提取 D 分支 (先知) 预测的边界掩码 (统一固定阈值 0.5)
        bd_pred_mask = (pred_sigmoid > 0.5)
        
        # 制作“纯边界语义标签”：只保留先知认为是边界的地方，其他地方填 255
        filler = torch.ones_like(sem_label) * self.ignore_index
        halo_label = torch.where(bd_pred_mask, sem_label, filler)
        
        # 将严苛的边界标签砸向 I 分支！
        # 【核心修改】：乘以独立的稳压器 cur_fb_w，完美保护主干网络！
        loss['loss_halo_feedback'] = self.loss_decode[3](i_logit, halo_label) * cur_fb_w
        
        # 4. 准确率统计
        loss['acc_seg'] = accuracy(i_logit, sem_label, ignore_index=self.ignore_index)
            
        return loss
