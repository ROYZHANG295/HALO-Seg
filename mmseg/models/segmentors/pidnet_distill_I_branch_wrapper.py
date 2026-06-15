import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.registry import MODELS
from mmseg.models.segmentors import BaseSegmentor

@MODELS.register_module()
class PIDNetDistillerIBranchWrapper(BaseSegmentor):
    def __init__(self, 
                 student_cfg, 
                 teacher_cfg,
                 student_ckpt=None,         
                 teacher_ckpt=None,
                 temperature=4.0,
                 kd_weight=1.0,
                 boundary_weight=5.0,
                 feat_weight=2.0,       # <--- 新增：特征蒸馏权重
                 s_i_channels=128,      # <--- 新增：学生 I 分支通道数 (看你的 config)
                 t_i_channels=256,      # <--- 新增：老师 I 分支通道数 (通常是 256 或 128)
                 data_preprocessor=None,
                 init_cfg=None):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        
        self.teacher = MODELS.build(teacher_cfg)
        self.student = MODELS.build(student_cfg)
        
        self.student_ckpt = student_ckpt
        self.teacher_ckpt = teacher_ckpt
            
        for param in self.teacher.parameters():
            param.requires_grad = False
            
        self.temp = temperature
        self.kd_weight = kd_weight
        self.boundary_weight = boundary_weight
        self.feat_weight = feat_weight

        # ================= 【新增 1】 特征适配器 =================
        # 作用：将学生的 I 分支特征映射到老师的维度，并学习对齐特征空间
        # 即使通道数相同，加一个 1x1 卷积层也是有益的（提供缓冲）
        self.feat_adapter = nn.Conv2d(s_i_channels, t_i_channels, kernel_size=1)
        
        # 初始化适配器 (防止初始 Loss 爆炸)
        nn.init.kaiming_normal_(self.feat_adapter.weight, mode='fan_out', nonlinearity='relu')
        if self.feat_adapter.bias is not None:
            nn.init.constant_(self.feat_adapter.bias, 0)

    def init_weights(self):
        self.student.init_weights()
        self.teacher.init_weights()
        
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
        # 1. 学生模型前向传播
        # PIDNet 的 extract_feat 返回元组 (x_p, x_i, x_d)
        # 索引 0: Detail (P)
        # 索引 1: Context (I) <--- 我们要蒸馏这个
        # 索引 2: Boundary (D)
        s_feats = self.student.extract_feat(inputs)
        s_head_out = self.student.decode_head(s_feats)
        s_boundary_logits, s_semantic_logits = s_head_out[0], s_head_out[1]

        # 计算学生 Task Loss
        s_loss_raw = self.student.decode_head.loss_by_feat(s_head_out, data_samples)
        s_loss = {f'decode.{k}': v for k, v in s_loss_raw.items()}

        # 2. 教师模型前向传播
        with torch.no_grad():
            self.teacher.eval() 
            self.teacher.backbone.training = True # 保持 BN 统计
            self.teacher.decode_head.training = True 
            
            t_feats = self.teacher.extract_feat(inputs)
            t_head_out = self.teacher.decode_head(t_feats)
            t_boundary_logits, t_semantic_logits = t_head_out[0], t_head_out[1]
            
            self.teacher.backbone.training = False
            self.teacher.decode_head.training = False

        # 3. 维度对齐 (Logits)
        if s_semantic_logits.shape[2:] != t_semantic_logits.shape[2:]:
            t_semantic_logits = F.interpolate(
                t_semantic_logits, size=s_semantic_logits.shape[2:], mode='bilinear', align_corners=False)
            t_boundary_logits = F.interpolate(
                t_boundary_logits, size=s_boundary_logits.shape[2:], mode='bilinear', align_corners=False)

        distill_losses = {}
        
        # --- Loss A: Logits KD ---
        s_sem_safe = torch.clamp(s_semantic_logits / self.temp, min=-50.0, max=50.0)
        t_sem_safe = torch.clamp(t_semantic_logits / self.temp, min=-50.0, max=50.0)
        t_logits_soft = F.softmax(t_sem_safe, dim=1)
        s_logits_log = F.log_softmax(s_sem_safe, dim=1)
        loss_kd = F.kl_div(s_logits_log, t_logits_soft, reduction='none')
        distill_losses['loss_kd_logits'] = loss_kd.sum(dim=1).mean() * (self.temp**2) * self.kd_weight

        # --- Loss B: Boundary KD ---
        distill_losses['loss_kd_boundary'] = F.mse_loss(
            torch.sigmoid(s_boundary_logits.float()), 
            torch.sigmoid(t_boundary_logits.float())
        ) * self.boundary_weight

        # ================= 【新增 2】 I-Branch 特征蒸馏逻辑 =================
        # 提取 I 分支特征 (通常是 index 1)
        s_i_feat = s_feats[1] 
        t_i_feat = t_feats[1]

        # 1. 适配通道 (Student -> Teacher)
        s_i_adapted = self.feat_adapter(s_i_feat)

        # 2. 适配空间尺寸 (如果分辨率不同，通常 I 分支是 1/8)
        if s_i_adapted.shape[2:] != t_i_feat.shape[2:]:
            s_i_adapted = F.interpolate(
                s_i_adapted, 
                size=t_i_feat.shape[2:], 
                mode='bilinear', 
                align_corners=False
            )

        # 3. 计算 MSE Loss (归一化有助于训练稳定)
        # 使用 std 进行归一化可以防止量级差异过大
        loss_feat = F.mse_loss(s_i_adapted, t_i_feat) 
        
        distill_losses['loss_kd_feat_i'] = loss_feat * self.feat_weight

        s_loss.update(distill_losses)
        return s_loss
