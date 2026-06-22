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
     PIDNet 版本 HALO 解码头（解耦权重框架）。

     设计要点：
     1. 在线构造拉普拉斯边界监督。
     2. 使用 D 分支边界预测掩码，构造边界语义标签，
         对 I 分支执行跨分支语义反馈监督。
     3. 将边界内部监督权重（dice_w）与反馈权重（fb_w）解耦，
         便于单独控制训练稳定性与边界学习强度。

     当前默认调度：
     - dilation: 5 -> 4 -> 3
     - dice_w: 3.0 -> 1.5 -> 1.0
     - fb_w: 1.0（固定）
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

        # 初始化三阶段解耦调度表
        self.dynamic_schedule = self._build_avg3_schedule(self.max_iters)
        # 当前全局日志器
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

    def _generate_binary_boundary(self,
                               semantic_gt: Tensor,
                               ignore_index: int = 255,
                               dilation_size: int = 3) -> Tensor:
        """
        简单二值边界：检测相邻像素标签跳变，不区分类别。
        对应消融项：Binary Target。
        """
        valid_mask = (semantic_gt != ignore_index).float()
        
        # 水平方向标签跳变
        h_diff = (semantic_gt[:, :, 1:] != semantic_gt[:, :, :-1]).float()
        # 垂直方向标签跳变
        v_diff = (semantic_gt[:, 1:, :] != semantic_gt[:, :-1, :]).float()
        
        # 对齐尺寸
        boundary_map = torch.zeros_like(semantic_gt).float()
        boundary_map[:, :, :-1] += h_diff
        boundary_map[:, :, 1:]  += h_diff
        boundary_map[:, :-1, :] += v_diff
        boundary_map[:, 1:, :]  += v_diff
        boundary_map = (boundary_map > 0).float()
        
        # 膨胀（保持和 OLB 一致）
        if dilation_size > 1:
            pad = dilation_size // 2
            boundary_map = F.max_pool2d(
                boundary_map.unsqueeze(1),
                kernel_size=dilation_size,
                stride=1,
                padding=pad
            ).squeeze(1)
        
        boundary_map = boundary_map * valid_mask
        return boundary_map

    # 在线拉普拉斯边界提取
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

    # 三阶段硬跳变解耦调度器
    def _build_avg3_schedule(self, max_iters: int) -> dict:
        t1 = int(max_iters / 3.0)
        t2 = int(max_iters * 2.0 / 3.0)

        # 主版本：三阶段硬跳变，解耦 D 分支内部监督与跨分支反馈权重
        # 记录结果：79.08 
        # 策略：放缓 Dice 权重衰减，保持较强边界约束
        schedule = {
            0:         {'dilation': 5, 'dice_w': 3.0, 'fb_w': 1.0},
            t1:        {'dilation': 5, 'dice_w': 3.0, 'fb_w': 1.0},
            t1 + 1:    {'dilation': 4, 'dice_w': 1.5, 'fb_w': 1.0},  
            t2:        {'dilation': 4, 'dice_w': 1.5, 'fb_w': 1.0},
            t2 + 1:    {'dilation': 3, 'dice_w': 1.0, 'fb_w': 1.0},  
            max_iters: {'dilation': 3, 'dice_w': 1.0, 'fb_w': 1.0}
        }

        # 消融 1
        # OLB Only	Class-wise	✓	Fixed 5	Fixed 3.0	78.6
        # 78.58
        # schedule = {
        #     0:         {'dilation': 5, 'dice_w': 3.0, 'fb_w': 1.0},
        #     t1:        {'dilation': 5, 'dice_w': 3.0, 'fb_w': 1.0},
        #     t1 + 1:    {'dilation': 4, 'dice_w': 3.0, 'fb_w': 1.0}, 
        #     t2:        {'dilation': 4, 'dice_w': 3.0, 'fb_w': 1.0},
        #     t2 + 1:    {'dilation': 3, 'dice_w': 3.0, 'fb_w': 1.0},  
        #     max_iters: {'dilation': 3, 'dice_w': 3.0, 'fb_w': 1.0}
        # }

        # 消融 2
        # OLB+DD	Class-wise	✓	5-4-3	Fixed 3.0	78.8
        # 78.84
        # schedule = {
        #     0:         {'dilation': 5, 'dice_w': 3.0, 'fb_w': 1.0},
        #     t1:        {'dilation': 5, 'dice_w': 3.0, 'fb_w': 1.0},
        #     t1 + 1:    {'dilation': 4, 'dice_w': 3.0, 'fb_w': 1.0}, 
        #     t2:        {'dilation': 4, 'dice_w': 3.0, 'fb_w': 1.0},
        #     t2 + 1:    {'dilation': 3, 'dice_w': 3.0, 'fb_w': 1.0},  
        #     max_iters: {'dilation': 3, 'dice_w': 3.0, 'fb_w': 1.0}
        # }

        # 消融 3
        # OLB+AAS	Class-wise	✓	Fixed 5	3.0-1.5-1.0	78.7
        # 78.72
        # schedule = {
        #     0:         {'dilation': 5, 'dice_w': 3.0, 'fb_w': 1.0},
        #     t1:        {'dilation': 5, 'dice_w': 3.0, 'fb_w': 1.0},
        #     t1 + 1:    {'dilation': 5, 'dice_w': 1.5, 'fb_w': 1.0}, 
        #     t2:        {'dilation': 5, 'dice_w': 1.5, 'fb_w': 1.0},
        #     t2 + 1:    {'dilation': 5, 'dice_w': 1.0, 'fb_w': 1.0},  
        #     max_iters: {'dilation': 5, 'dice_w': 1.0, 'fb_w': 1.0}
        # }

        # 消融 4
        # Binary Target	Binary	✓	5-4-3	3.0-1.5-1.0	78.7
        # 注意：需要启用 _generate_binary_boundary
        # 78.67
        # schedule = {
        #     0:         {'dilation': 5, 'dice_w': 3.0, 'fb_w': 1.0},
        #     t1:        {'dilation': 5, 'dice_w': 3.0, 'fb_w': 1.0},
        #     t1 + 1:    {'dilation': 4, 'dice_w': 1.5, 'fb_w': 1.0},  
        #     t2:        {'dilation': 4, 'dice_w': 1.5, 'fb_w': 1.0},
        #     t2 + 1:    {'dilation': 3, 'dice_w': 1.0, 'fb_w': 1.0},  
        #     max_iters: {'dilation': 3, 'dice_w': 1.0, 'fb_w': 1.0}
        # }

        # 消融 5
        # w/o Semantic Re-sup.	Class-wise	-	5-4-3	3.0-1.5-1.0	78.4
        # 78.35
        # schedule = {
        #     0:         {'dilation': 5, 'dice_w': 3.0, 'fb_w': 0.0},
        #     t1:        {'dilation': 5, 'dice_w': 3.0, 'fb_w': 0.0},
        #     t1 + 1:    {'dilation': 4, 'dice_w': 1.5, 'fb_w': 0.0},  
        #     t2:        {'dilation': 4, 'dice_w': 1.5, 'fb_w': 0.0},
        #     t2 + 1:    {'dilation': 3, 'dice_w': 1.0, 'fb_w': 0.0},  
        #     max_iters: {'dilation': 3, 'dice_w': 1.0, 'fb_w': 0.0}
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
        # 返回解耦后的三项参数
        return cfg['dilation'], cfg['dice_w'], cfg['fb_w']

    def loss_by_feat(self, seg_logits: Tuple[Tensor], batch_data_samples: SampleList) -> dict:
        
        if self.training:
            self.local_step += 1
        current_step = self.local_step.item()
        
        # 获取当前步的解耦参数
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

        # 消融 4：改用二值边界标签时启用
        # bd_label = self._generate_binary_boundary(
        #     sem_label,
        #     ignore_index=self.ignore_index,
        #     dilation_size=cur_dilation
        # )

        # 1) 全局语义损失（P 分支与 I 分支）
        loss['loss_sem_p'] = self.loss_decode[0](p_logit, sem_label, ignore_index=self.ignore_index)
        loss['loss_sem_i'] = self.loss_decode[1](i_logit, sem_label)
        
        # 2) 边界内部监督（D 分支）
        bce_loss = self.loss_decode[2](d_logit, bd_label)
        pred_sigmoid = torch.sigmoid(d_logit[:, 0, :, :])
        
        valid_mask = (sem_label != self.ignore_index).float()
        intersection = (pred_sigmoid * bd_label * valid_mask).sum(dim=(1, 2))
        union = (pred_sigmoid * valid_mask).sum(dim=(1, 2)) + (bd_label * valid_mask).sum(dim=(1, 2))
        dice_loss = (1.0 - (2.0 * intersection + 1e-5) / (union + 1e-5)).mean()
        
        # 使用动态 dice_w 加权边界内部监督
        loss['loss_bd_laplacian'] = bce_loss + cur_dice_w * dice_loss
        
        # 3) 跨分支语义反馈（固定阈值 0.5）
        # 提取 D 分支预测边界掩码
        bd_pred_mask = (pred_sigmoid > 0.5)
        
        # 构造边界语义标签：非边界位置填 ignore_index
        filler = torch.ones_like(sem_label) * self.ignore_index
        halo_label = torch.where(bd_pred_mask, sem_label, filler)
        
        # 对 I 分支施加反馈监督，并用 fb_w 独立加权
        loss['loss_halo_feedback'] = self.loss_decode[3](i_logit, halo_label) * cur_fb_w
        
        # 4) 分割准确率统计
        loss['acc_seg'] = accuracy(i_logit, sem_label, ignore_index=self.ignore_index)
            
        return loss
