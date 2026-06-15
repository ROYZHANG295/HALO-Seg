import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.registry import MODELS
from mmseg.models.segmentors import BaseSegmentor

@MODELS.register_module()
class PIDNetDistillerWrapper(BaseSegmentor):
    def __init__(self, 
                 student_cfg, 
                 teacher_cfg,
                 student_ckpt=None,         
                 teacher_ckpt=None,
                 temperature=4.0,
                 kd_weight=1.0,
                 boundary_weight=5.0,
                 data_preprocessor=None,
                 init_cfg=None):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        
        self.teacher = MODELS.build(teacher_cfg)
        self.student = MODELS.build(student_cfg)
        
        # 仅仅保存路径，不要在这里加载权重！
        self.student_ckpt = student_ckpt
        self.teacher_ckpt = teacher_ckpt
            
        for param in self.teacher.parameters():
            param.requires_grad = False
            
        self.temp = temperature
        self.kd_weight = kd_weight
        self.boundary_weight = boundary_weight

    # ================= 【绝杀修复 1】接管 MMEngine 的初始化生命周期 =================
    def init_weights(self):
        # 1. 先让 MMEngine 跑完它默认的初始化（ImageNet等）
        self.student.init_weights()
        self.teacher.init_weights()
        
        # 2. 在最后一步，强势覆盖我们需要的权重！绝对不会再被顶掉！
        from mmengine.runner import load_checkpoint
        if self.student_ckpt:
            print(f"\n{'='*50}\n[Distiller] 正在强势加载学生权重: {self.student_ckpt}\n{'='*50}\n")
            load_checkpoint(self.student, self.student_ckpt, map_location='cpu')
        if self.teacher_ckpt:
            print(f"\n{'='*50}\n[Distiller] 正在强势加载老师权重: {self.teacher_ckpt}\n{'='*50}\n")
            load_checkpoint(self.teacher, self.teacher_ckpt, map_location='cpu')

    def extract_feat(self, inputs):
        return self.student.extract_feat(inputs)

    def encode_decode(self, inputs, batch_img_metas):
        return self.student.encode_decode(inputs, batch_img_metas)

    def predict(self, inputs, data_samples):
        return self.student.predict(inputs, data_samples)

    def _forward(self, inputs, data_samples):
        return self.student._forward(inputs, data_samples)

    def loss(self, inputs, data_samples):
        # 1. 学生模型前向传播 (严格只执行一次，防止 SyncBN 崩溃)
        s_feats = self.student.extract_feat(inputs)
        s_head_out = self.student.decode_head(s_feats)
        s_boundary_logits, s_semantic_logits = s_head_out[0], s_head_out[1]

        # 2. 计算学生 Task Loss
        s_loss_raw = self.student.decode_head.loss_by_feat(s_head_out, data_samples)
        s_loss = {f'decode.{k}': v for k, v in s_loss_raw.items()}

        # 3. 教师模型前向传播
        with torch.no_grad():
            self.teacher.eval() 
            self.teacher.backbone.training = True
            self.teacher.decode_head.training = True 
            
            t_feats = self.teacher.extract_feat(inputs)
            t_head_out = self.teacher.decode_head(t_feats)
            t_boundary_logits, t_semantic_logits = t_head_out[0], t_head_out[1]
            
            self.teacher.backbone.training = False
            self.teacher.decode_head.training = False

        # 4. 维度对齐
        if s_semantic_logits.shape[2:] != t_semantic_logits.shape[2:]:
            t_semantic_logits = F.interpolate(
                t_semantic_logits, size=s_semantic_logits.shape[2:], mode='bilinear', align_corners=False)
            t_boundary_logits = F.interpolate(
                t_boundary_logits, size=s_boundary_logits.shape[2:], mode='bilinear', align_corners=False)

        distill_losses = {}
        
        # ================= 【绝杀修复 2】数学安全锁 (防数值溢出) =================
        # 强制将数值限制在 -50 到 50 之间，彻底杜绝 exp(极大值) 产生的 NaN
        s_sem_safe = torch.clamp(s_semantic_logits / self.temp, min=-50.0, max=50.0)
        t_sem_safe = torch.clamp(t_semantic_logits / self.temp, min=-50.0, max=50.0)
        
        t_logits_soft = F.softmax(t_sem_safe, dim=1)
        s_logits_log = F.log_softmax(s_sem_safe, dim=1)
        
        loss_kd = F.kl_div(s_logits_log, t_logits_soft, reduction='none')
        distill_losses['loss_kd_logits'] = loss_kd.sum(dim=1).mean() * (self.temp**2) * self.kd_weight

        # ================= 【绝杀修复 3】边界蒸馏加上 float() 强制转换 =================
        # 防止在使用 AMP (混合精度 fp16) 时 sigmoid 产生下溢出
        distill_losses['loss_kd_boundary'] = F.mse_loss(
            torch.sigmoid(s_boundary_logits.float()), 
            torch.sigmoid(t_boundary_logits.float())
        ) * self.boundary_weight

        s_loss.update(distill_losses)
        return s_loss
