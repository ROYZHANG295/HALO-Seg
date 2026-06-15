import torch
import torch.nn as nn
from mmengine.model import BaseModel
from mmseg.registry import MODELS
from mmseg.utils import ConfigType, OptConfigType, OptMultiConfig
from mmseg.models.segmentors import EncoderDecoder
from mmengine.runner import load_checkpoint
from mmengine.config import Config

@MODELS.register_module()
class EncoderDecoderKD(EncoderDecoder):
    """支持知识蒸馏的 EncoderDecoder (修复 Tuple 输出问题版)"""

    def __init__(self, 
                 teacher_config: ConfigType,
                 teacher_ckpt: str,
                 distill_losses: list = [],
                 **kwargs):
        super().__init__(**kwargs)
        
        # 初始化 Teacher
        if isinstance(teacher_config, str):
            t_cfg = Config.fromfile(teacher_config)
        else:
            t_cfg = teacher_config
            
        self.teacher = MODELS.build(t_cfg.model)
        
        print(f'正在加载 Teacher 权重: {teacher_ckpt} ...')
        load_checkpoint(self.teacher, teacher_ckpt, map_location='cpu')
        
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
            
        self.distill_losses = nn.ModuleList()
        for loss_cfg in distill_losses:
            self.distill_losses.append(MODELS.build(loss_cfg))

    def loss(self, inputs: torch.Tensor, data_samples: list) -> dict:
        # 1. 计算 Student 原始 Loss
        losses = super().loss(inputs, data_samples)
        
        # 2. 计算 Teacher 输出
        with torch.no_grad():
            t_feat = self.teacher.extract_feat(inputs)
            t_out = self.teacher.decode_head.forward(t_feat)
            
        # 3. 计算 Student 输出
        s_feat = self.extract_feat(inputs)
        s_out = self.decode_head.forward(s_feat)
        
        # 4. 计算蒸馏 Loss
        for loss_module in self.distill_losses:
            loss_name = loss_module.loss_name
            
            # =======================================================
            # 核心修复：处理 PIDNet 输出的 Tuple (Main, Aux, Boundary)
            # =======================================================
            if 'logits' in loss_name:
                # 如果是 Tuple，取第一个元素 (Main Logits)
                s_logits = s_out[0] if isinstance(s_out, tuple) else s_out
                t_logits = t_out[0] if isinstance(t_out, tuple) else t_out
                
                d_loss = loss_module(s_logits, t_logits)
                
            elif 'cwd' in loss_name or 'feat' in loss_name:
                # 特征蒸馏 (CWD)
                # PIDNet Backbone 输出通常是 (P分支, I分支, D分支)
                # 我们通常蒸馏 I 分支 (索引 1) 用来恢复纹理，或者 P 分支 (索引 0)
                # 这里默认取索引 1 (I-Branch)，因为它是主要的语义特征
                
                # 检查是否越界，防止报错
                s_idx = 1 if len(s_feat) > 1 else 0
                t_idx = 1 if len(t_feat) > 1 else 0
                
                d_loss = loss_module(s_feat[s_idx], t_feat[t_idx])
            else:
                # 兜底逻辑
                d_loss = loss_module(s_out, t_out)
                
            losses[loss_name] = d_loss
            
        return losses
