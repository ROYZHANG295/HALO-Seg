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
     HALO 版 DDRHead（解耦反馈权重）。

     设计目标：
     1. 基于语义标签在线生成拉普拉斯边界监督。
     2. 利用 Spatial 分支预测的边界掩码，构造边界语义标签，
         对 Context 分支进行跨分支语义反馈监督。
     3. 将边界内部监督权重（dice_w）与反馈权重（fb_w）解耦，
         便于做稳定性与性能的独立消融。

     当前默认调度（本文件主版本）：
     - dilation: 5 -> 4 -> 3
     - dice_w: 1.0 -> 0.5 -> 0.1
     - fb_w: 固定为 0.5
    """

    def __init__(self,
                 in_channels: int,
                 channels: int,
                 num_classes: int,
                 norm_cfg: OptConfigType = dict(type='BN'),
                 act_cfg: OptConfigType = dict(type='ReLU', inplace=True),
                 max_iters: int = 120000,  # 总训练步数，用于构建三阶段调度
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
        
        # 记录总步数，用于计算调度里程碑
        self.max_iters = max_iters
        
        # 训练步计数器 + 轻量化边界预测头
        self.register_buffer('local_step', torch.tensor(0, dtype=torch.long))
        self.bd_cls_seg = nn.Conv2d(self.channels, 1, kernel_size=1)

        # 初始化解耦调度表
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
            # DDRNet 训练期双分支输入：c3（空间细节）与 c5（全局上下文）
            c3_feat, c5_feat = inputs
            
            x_c = self.head(c5_feat)
            x_c_logit = self.cls_seg(x_c)
            
            x_s = self.aux_head(c3_feat)
            x_s_logit = self.aux_cls_seg(x_s)
            
            # 训练期额外输出边界 logits
            bd_logit = self.bd_cls_seg(x_s)

            return x_c_logit, x_s_logit, bd_logit
        else:
            # 推理期仅保留 Context 分支，不引入额外边界计算
            x_c = self.head(inputs)
            x_c = self.cls_seg(x_c)
            return x_c

    # 在线拉普拉斯边界提取
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

    # 三阶段解耦调度器
    def _build_avg3_schedule(self, max_iters: int) -> dict:
        """根据总步数自动生成三阶段调度里程碑。"""
        t1 = int(max_iters / 3.0)      # 三等分点 1（约 33.3%）
        t2 = int(max_iters * 2.0 / 3.0)  # 三等分点 2（约 66.7%）

        # 主版本：fb_w 固定为 0.5
        # 记录结果：78.23
        schedule = {
            # 阶段 1
            0:      {'dilation': 5, 'dice_w': 1.0, 'fb_w': 0.5},
            t1:     {'dilation': 5, 'dice_w': 1.0, 'fb_w': 0.5},
            
            # 阶段 2
            t1 + 1: {'dilation': 4, 'dice_w': 0.5, 'fb_w': 0.5},
            t2:     {'dilation': 4, 'dice_w': 0.5, 'fb_w': 0.5},
            
            # 阶段 3
            t2 + 1: {'dilation': 3, 'dice_w': 0.1, 'fb_w': 0.5},
            max_iters: {'dilation': 3, 'dice_w': 0.1, 'fb_w': 0.5}
        }

        # 消融 1：fb_w 固定为 1.0
        # 记录结果：77.64
        # schedule = {
        #     # 阶段 1
        #     0:      {'dilation': 5, 'dice_w': 1.0, 'fb_w': 1.0},
        #     t1:     {'dilation': 5, 'dice_w': 1.0, 'fb_w': 1.0},
            
        #     # 阶段 2
        #     t1 + 1: {'dilation': 4, 'dice_w': 0.5, 'fb_w': 1.0},
        #     t2:     {'dilation': 4, 'dice_w': 0.5, 'fb_w': 1.0},
            
        #     # 阶段 3
        #     t2 + 1: {'dilation': 3, 'dice_w': 0.1, 'fb_w': 1.0},
        #     max_iters: {'dilation': 3, 'dice_w': 0.1, 'fb_w': 1.0}
        # }

        # 消融 2：fb_w 固定为 0.3
        # 记录结果：77.81
        # schedule = {
        #     # 阶段 1
        #     0:      {'dilation': 5, 'dice_w': 1.0, 'fb_w': 0.3},
        #     t1:     {'dilation': 5, 'dice_w': 1.0, 'fb_w': 0.3},
            
        #     # 阶段 2
        #     t1 + 1: {'dilation': 4, 'dice_w': 0.5, 'fb_w': 0.3},
        #     t2:     {'dilation': 4, 'dice_w': 0.5, 'fb_w': 0.3},
            
        #     # 阶段 3
        #     t2 + 1: {'dilation': 3, 'dice_w': 0.1, 'fb_w': 0.3},
        #     max_iters: {'dilation': 3, 'dice_w': 0.1, 'fb_w': 0.3}
        # }

        # 消融 3：fb_w 固定为 0.1
        # 记录结果：77.48
        # schedule = {
        #     # 阶段 1
        #     0:      {'dilation': 5, 'dice_w': 1.0, 'fb_w': 0.1},
        #     t1:     {'dilation': 5, 'dice_w': 1.0, 'fb_w': 0.1},
            
        #     # 阶段 2
        #     t1 + 1: {'dilation': 4, 'dice_w': 0.5, 'fb_w': 0.1},
        #     t2:     {'dilation': 4, 'dice_w': 0.5, 'fb_w': 0.1},
            
        #     # 阶段 3
        #     t2 + 1: {'dilation': 3, 'dice_w': 0.1, 'fb_w': 0.1},
        #     max_iters: {'dilation': 3, 'dice_w': 0.1, 'fb_w': 0.1}
        # }

        # 消融 4：fb_w 固定为 0.0（关闭反馈监督）
        # 记录结果：77.56
        # schedule = {
        #     # 阶段 1
        #     0:      {'dilation': 5, 'dice_w': 1.0, 'fb_w': 0.0},
        #     t1:     {'dilation': 5, 'dice_w': 1.0, 'fb_w': 0.0},
            
        #     # 阶段 2
        #     t1 + 1: {'dilation': 4, 'dice_w': 0.5, 'fb_w': 0.0},
        #     t2:     {'dilation': 4, 'dice_w': 0.5, 'fb_w': 0.0},
            
        #     # 阶段 3
        #     t2 + 1: {'dilation': 3, 'dice_w': 0.1, 'fb_w': 0.0},
        #     max_iters: {'dilation': 3, 'dice_w': 0.1, 'fb_w': 0.0}
        # }

        print(schedule)

        return schedule

    def _get_dynamic_params(self, current_step: int) -> Tuple[int, float, float]:
        """读取当前步对应的动态参数，并在阶段内做线性插值。"""
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
        # 对 dice_w 与 fb_w 分别线性插值
        cur_dice_w = start_cfg['dice_w'] + progress * (end_cfg['dice_w'] - start_cfg['dice_w'])
        cur_fb_w = start_cfg['fb_w'] + progress * (end_cfg['fb_w'] - start_cfg['fb_w'])
        
        return cur_dilation, cur_dice_w, cur_fb_w

    def loss_by_feat(self, seg_logits: Tuple[Tensor], batch_data_samples: SampleList) -> dict:
        
        # 最终损失：全局语义损失 + 边界内部监督 + 跨分支反馈监督
        
        if self.training:
            self.local_step += 1
        current_step = self.local_step.item()
        
        # 获取当前步的解耦参数
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

        # 基于当前 dilation 在线生成拉普拉斯边界标签
        bd_label = self._generate_laplacian_boundary(
            seg_label, num_classes=self.num_classes, ignore_index=self.ignore_index, dilation_size=cur_dilation
        )

        # 1) 全局语义损失（Context / Spatial）
        loss['loss_context'] = self.loss_decode[0](context_logit, seg_label)
        loss['loss_spatial'] = self.loss_decode[1](spatial_logit, seg_label)
        
        # 2) 边界分支内部损失（BCE + Dice）
        bce_loss = F.binary_cross_entropy_with_logits(bd_logit.squeeze(1), bd_label)
        
        pred_sigmoid = torch.sigmoid(bd_logit[:, 0, :, :])
        valid_mask = (seg_label != self.ignore_index).float()
        intersection = (pred_sigmoid * bd_label * valid_mask).sum(dim=(1, 2))
        union = (pred_sigmoid * valid_mask).sum(dim=(1, 2)) + (bd_label * valid_mask).sum(dim=(1, 2))
        dice_loss = (1.0 - (2.0 * intersection + 1e-5) / (union + 1e-5)).mean()
        
        # 使用动态 dice_w 加权边界内部监督
        loss['loss_bd_laplacian'] = bce_loss + cur_dice_w * dice_loss
        
        # 3) 跨分支反馈：由 Spatial 边界预测构造边界语义监督
        # 提取 Spatial 分支预测的边界掩码
        bd_pred_mask = (pred_sigmoid > 0.5)
        
        # 构造边界语义标签：非边界位置填 ignore_index
        filler = torch.ones_like(seg_label) * self.ignore_index
        halo_label = torch.where(bd_pred_mask, seg_label, filler)
        
        # 对 Context 分支施加反馈损失，并用 fb_w 独立加权
        loss['loss_halo_feedback'] = self.loss_decode[0](context_logit, halo_label) * cur_fb_w
        
        # 4) 记录 Context 分支分割准确率
        loss['acc_seg'] = accuracy(context_logit, seg_label, ignore_index=self.ignore_index)

        return loss

