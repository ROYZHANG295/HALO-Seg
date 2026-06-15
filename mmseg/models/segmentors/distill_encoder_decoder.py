import torch
import torch.nn.functional as F
from mmengine.config import Config
from mmengine.model import BaseModel
from mmseg.registry import MODELS
from mmseg.utils import sample_to_img
from .encoder_decoder import EncoderDecoder

@MODELS.register_module()
class DistillPIDNet(EncoderDecoder):
    """
    针对 MMSegmentation v1.x 的 PIDNet 蒸馏封装类
    """
    def __init__(self, 
                 teacher_config, 
                 teacher_ckpt, 
                 distill_temp=4.0, 
                 distill_weight=3.0,
                 **kwargs):
        super().__init__(**kwargs)
        
        # 1. 构建 Teacher 模型
        t_cfg = Config.fromfile(teacher_config)
        # 确保 Teacher 的预处理和 Student 一致，或者在 forward 中处理
        # 这里我们直接构建 Teacher 的模型部分
        self.teacher = MODELS.build(t_cfg.model)
        
        # 2. 加载 Teacher 权重
        from mmengine.runner import load_checkpoint
        load_checkpoint(self.teacher, teacher_ckpt, map_location='cpu')
        
        # 3. 冻结 Teacher
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
            
        self.distill_temp = distill_temp
        self.distill_weight = distill_weight
        print(f"[Distill] Teacher loaded. T={distill_temp}, W={distill_weight}")

    def _forward_teacher(self, inputs):
        """Teacher 前向推理"""
        with torch.no_grad():
            # Teacher 提取特征
            t_feat = self.teacher.extract_feat(inputs)
            # Teacher 解码头输出 (PIDNet Head 输出通常是 logits)
            t_logits = self.teacher.decode_head.forward(t_feat)
        return t_logits

    def kl_loss(self, s_logits, t_logits):
        """计算 KL 散度 Loss"""
        T = self.distill_temp
        
        # 尺寸对齐
        if s_logits.shape != t_logits.shape:
            t_logits = F.interpolate(
                t_logits, size=s_logits.shape[2:], mode='bilinear', align_corners=False)

        # 展平 [B, C, H, W] -> [N, C]
        B, C, H, W = s_logits.shape
        s = s_logits.permute(0, 2, 3, 1).reshape(-1, C)
        t = t_logits.permute(0, 2, 3, 1).reshape(-1, C)

        p_s = F.log_softmax(s / T, dim=1)
        p_t = F.softmax(t / T, dim=1)
        
        loss = F.kl_div(p_s, p_t, reduction='batchmean') * (T**2)
        return loss

    def loss(self, inputs, data_samples):
        """
        覆写 loss 方法：
        1. 计算 Student 原有 Loss
        2. 计算 Distill Loss
        """
        # --- Student 正常流程 ---
        x = self.extract_feat(inputs)
        
        # 获取 Student Logits (PIDNet Head forward 返回的是 logits)
        s_logits = self.decode_head.forward(x)
        
        # 计算原本的 CrossEntropy + OHEM + Boundary Loss
        losses = self.decode_head.loss_by_feat(s_logits, data_samples)
        
        # --- Teacher 流程 ---
        t_logits = self._forward_teacher(inputs)
        
        # --- 蒸馏 Loss ---
        loss_distill = self.kl_loss(s_logits, t_logits)
        losses['loss_distill'] = loss_distill * self.distill_weight
        
        return losses

    # 确保 Teacher 始终 Eval
    def train(self, mode=True):
        super().train(mode)
        if hasattr(self, 'teacher'):
            self.teacher.eval()
        return self
