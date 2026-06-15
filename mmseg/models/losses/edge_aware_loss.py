import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.registry import MODELS

@MODELS.register_module()
class EdgeAwareOhemCrossEntropy(nn.Module):
    def __init__(self, 
                 thres=0.9, 
                 min_kept=131072, 
                 target_edge_weight=2.0,
                 delay_iters=4000,
                 warmup_iters=10000,
                 loss_weight=1.0, 
                 ignore_index=255, 
                 loss_name='loss_edge_ohem'):
        super().__init__()
        self.thres = thres
        self.min_kept = min_kept
        self.target_edge_weight = target_edge_weight
        self.delay_iters = delay_iters
        self.warmup_iters = warmup_iters
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index
        self._loss_name = loss_name

        self.register_buffer('step_counter', torch.zeros(1, dtype=torch.long))

    def forward(self, cls_score, label, weight=None, **kwargs):
        # 1. 更新计步器
        if self.training:
            self.step_counter += 1
            
        current_step = self.step_counter.item()

        # 2. 计算当前权重
        if current_step <= self.delay_iters:
            cur_edge_weight = 1.0
        elif current_step >= self.warmup_iters:
            cur_edge_weight = self.target_edge_weight
        else:
            alpha = (current_step - self.delay_iters) / (self.warmup_iters - self.delay_iters)
            cur_edge_weight = 1.0 + alpha * (self.target_edge_weight - 1.0)

        # 3. 基础交叉熵
        loss = F.cross_entropy(
            cls_score, label, ignore_index=self.ignore_index, reduction='none')

        # 4. 边缘加权
        valid_mask = (label != self.ignore_index).float()
        
        if cur_edge_weight > 1.0:
            label_float = label.clone().float()
            label_float[label == self.ignore_index] = 0
            label_unsqueeze = label_float.unsqueeze(1)
            
            max_pool = F.max_pool2d(label_unsqueeze, kernel_size=3, stride=1, padding=1)
            min_pool = -F.max_pool2d(-label_unsqueeze, kernel_size=3, stride=1, padding=1)
            edge_mask = (max_pool != min_pool).squeeze(1).float()

            pixel_weights = torch.ones_like(loss) + edge_mask * (cur_edge_weight - 1.0)
            pixel_weights = pixel_weights * valid_mask
        else:
            pixel_weights = valid_mask

        if weight is not None:
            pixel_weights = pixel_weights * weight
            
        weighted_loss = loss * pixel_weights

        # ---------------------------------------------------------
        # 5. 修复后的 OHEM 核心逻辑 (严格对齐 mmseg 官方)
        # ---------------------------------------------------------
        valid_loss = weighted_loss[valid_mask == 1]
        
        if valid_loss.numel() == 0:
            return weighted_loss.sum() * 0.0

        # 降序排列
        valid_loss, _ = valid_loss.sort(descending=True)
        
        if valid_loss.numel() > self.min_kept:
            # 取出第 min_kept 个像素的 loss 作为当前批次的阈值参考
            threshold = valid_loss[self.min_kept - 1].item()
            
            if threshold > self.thres:
                # 如果第 min_kept 个像素的 loss 已经大于 thres，
                # 说明有超过 min_kept 数量的像素 loss 大于 thres，保留所有大于 thres 的像素！
                valid_loss = valid_loss[valid_loss >= self.thres]
            else:
                # 如果大于 thres 的像素不足 min_kept 个，
                # 强制保留前 min_kept 个像素，防止梯度饿死！
                valid_loss = valid_loss[:self.min_kept]

        return valid_loss.mean() * self.loss_weight

    @property
    def loss_name(self):
        return self._loss_name
