import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.registry import MODELS # 如果是 mmseg v0.x 版本，这里用 from mmseg.models.builder import LOSSES

@MODELS.register_module() # 如果是 mmseg v0.x，用 @LOSSES.register_module()
class KLDivergence(nn.Module):
    """KL Divergence Loss for Knowledge Distillation."""
    
    def __init__(self, tau=1.0, reduction='mean', loss_weight=1.0, loss_name='loss_kd'):
        super(KLDivergence, self).__init__()
        self.tau = tau
        self.reduction = reduction
        self.loss_weight = loss_weight
        self._loss_name = loss_name

    def forward(self, pred, target, **kwargs):
        """
        Args:
            pred (Tensor): Student logits (N, C, H, W)
            target (Tensor): Teacher logits (N, C, H, W)
        """
        # 1. 调整温度
        pred = pred / self.tau
        target = target / self.tau
        
        # 2. 计算 LogSoftmax (Student) 和 Softmax (Teacher)
        pred_log_softmax = F.log_softmax(pred, dim=1)
        target_softmax = F.softmax(target, dim=1)
        
        # 3. 计算 KL 散度
        # PyTorch 的 kl_div 期望输入是 log_probs，目标是 probs
        loss = F.kl_div(pred_log_softmax, target_softmax, reduction=self.reduction)
        
        # 4. 乘以温度的平方 (标准蒸馏公式)
        loss = loss * (self.tau ** 2)
        
        return self.loss_weight * loss

    @property
    def loss_name(self):
        return self._loss_name

@MODELS.register_module()
class ChannelWiseDivergence(nn.Module):
    """Channel-wise Distillation Loss."""
    
    def __init__(self, student_channels, teacher_channels, tau=1.0, loss_weight=1.0, loss_name='loss_cwd'):
        super(ChannelWiseDivergence, self).__init__()
        self.tau = tau
        self.loss_weight = loss_weight
        self._loss_name = loss_name
        
        # 如果通道数不一致，需要一个 1x1 卷积来对齐
        if student_channels != teacher_channels:
            self.align = nn.Conv2d(student_channels, teacher_channels, kernel_size=1, bias=False)
        else:
            self.align = None

    def forward(self, pred, target, **kwargs):
        """
        pred: Student features (N, C_s, H, W)
        target: Teacher features (N, C_t, H, W)
        """
        # 1. 通道对齐
        if self.align is not None:
            # 注意：这里需要处理 device 问题，确保 align 模块在正确的 device 上
            if next(self.align.parameters()).device != pred.device:
                self.align = self.align.to(pred.device)
            pred = self.align(pred)
            
        N, C, H, W = pred.shape
        
        # 2. 归一化 (Softmax over spatial dimensions H*W)
        # CWD 的核心是把每个 Channel 看作一个分布
        softmax_pred = F.softmax(pred.view(N, C, -1) / self.tau, dim=2)
        softmax_target = F.softmax(target.view(N, C, -1) / self.tau, dim=2)
        
        # 3. 计算 KL 散度
        # sum over spatial (dim=2), mean over channel (dim=1)
        loss = torch.sum(softmax_target * (torch.log(softmax_target + 1e-8) - torch.log(softmax_pred + 1e-8)), dim=2)
        
        return self.loss_weight * (self.tau ** 2) * loss.mean()

    @property
    def loss_name(self):
        return self._loss_name
