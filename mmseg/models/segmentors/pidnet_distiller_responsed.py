import torch
import torch.nn.functional as F
from mmseg.registry import MODELS
from mmseg.utils import ConfigType
from .base import BaseSegmentor
from mmseg.models import build_segmentor

@MODELS.register_module()
class PIDNetDistiller(BaseSegmentor):
    def __init__(self,
                 teacher_cfg: ConfigType,
                 student_cfg: ConfigType,
                 teacher_pretrained: str = None,
                 distill_weight: float = 5.0,
                 temperature: float = 4.0,
                 **kwargs):
        super().__init__(**kwargs)

        # 1. 构建 Student
        self.student = build_segmentor(student_cfg)
        
        # 2. 构建 Teacher
        self.teacher = build_segmentor(teacher_cfg)
        
        # 3. 加载 Teacher 权重
        if teacher_pretrained:
            from mmengine.runner import load_checkpoint
            # print(f"Loading Teacher checkpoint from {teacher_pretrained}...")
            load_checkpoint(self.teacher, teacher_pretrained, map_location='cpu')
        
        # 4. 冻结 Teacher
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

        self.distill_weight = distill_weight
        self.temperature = temperature

    # =======================================================
    # ⚠️ 修正点 1: 移除 rescale 参数，符合 v1.x 规范
    # =======================================================
    def predict(self, inputs, data_samples):
        """推理接口"""
        return self.student.predict(inputs, data_samples)

    def _forward(self, inputs, data_samples):
        """用于获取 FLOPs 等"""
        return self.student._forward(inputs, data_samples)

    def encode_decode(self, inputs, img_metas):
        return self.student.encode_decode(inputs, img_metas)

    def extract_feat(self, inputs):
        return self.student.extract_feat(inputs)

    def loss(self, inputs, data_samples):
        """
        训练核心逻辑
        """
        # 1. 计算 Student 原有的 Loss
        student_losses = self.student.loss(inputs, data_samples)

        # 2. 提取 Logits 进行蒸馏
        # 我们需要再次前向传播 Head 来获取 Logits
        x_s = self.student.extract_feat(inputs)
        student_logits_all = self.student.decode_head.forward(x_s)
        
        with torch.no_grad():
            x_t = self.teacher.extract_feat(inputs)
            teacher_logits_all = self.teacher.decode_head.forward(x_t)

        # =========================================================
        # ⚠️ 修正点 2: 解包 Tuple/List，只取 Main Branch
        # PIDNetHead 返回: [Aux, Main, Boundary] -> 对应索引 [0, 1, 2]
        # 我们只蒸馏 Main (索引 1)
        # =========================================================
        
        # 处理 Student
        if isinstance(student_logits_all, (tuple, list)):
            student_main_logit = student_logits_all[1]
        else:
            student_main_logit = student_logits_all

        # 处理 Teacher
        if isinstance(teacher_logits_all, (tuple, list)):
            teacher_main_logit = teacher_logits_all[1]
        else:
            teacher_main_logit = teacher_logits_all

        # =========================================================

        # 3. 计算 KD Loss (KL Divergence)
        # 尺寸对齐 (以防 Teacher 和 Student 分辨率不同)
        if student_main_logit.shape != teacher_main_logit.shape:
            student_main_logit = F.interpolate(
                student_main_logit, 
                size=teacher_main_logit.shape[2:], 
                mode='bilinear', 
                align_corners=True)

        kd_loss = self.calc_distill_loss(student_main_logit, teacher_main_logit)
        
        student_losses['loss_kd'] = kd_loss * self.distill_weight
        
        return student_losses

    def calc_distill_loss(self, s_logits, t_logits):
        """KL 散度计算"""
        # 1. NaN 防护 (保留之前的)
        if torch.isnan(t_logits).any() or torch.isnan(s_logits).any():
            return (t_logits.sum() * 0.0) + (s_logits.sum() * 0.0)

        # 2. 软化分布
        s_soft = F.log_softmax(s_logits / self.temperature, dim=1)
        t_soft = F.softmax(t_logits / self.temperature, dim=1)
        
        # 3. 计算 KL 散度
        # ⚠️ 关键修改：
        # 'batchmean' 会把 1024*1024 个像素的误差全加起来，导致 Loss 几十万。
        # 'mean' 会除以像素数量，把 Loss 拉回 0.x ~ 5.0 的正常范围。
        loss = F.kl_div(s_soft, t_soft, reduction='mean')
        
        # 4. 再次检查结果
        if torch.isnan(loss):
            return loss * 0.0
            
        return loss * (self.temperature ** 2)


