# 文件路径: mmseg/models/losses/laplacian_edge_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.registry import MODELS # 注册机制，让配置能找到这个类

@MODELS.register_module()
class LaplacianEdgeLoss(nn.Module):
    def __init__(self, 
                 loss_weight=1.0, 
                 kernel_size=5, 
                 sigma=1.0,
                 use_soft_label=True):
        """
        PIDNet 专用：拉普拉斯边缘感知损失
        Args:
            loss_weight (float): Loss 的权重，控制它对总 Loss 的贡献
            kernel_size (int): 卷积核大小，5x5 比较适合捕捉边缘
            sigma (float): 高斯模糊的程度。值越小边缘越细，值越大边缘越粗(容易学)
            use_soft_label (bool): True=使用概率图(软边缘)，False=使用0/1(硬边缘)
        """
        super(LaplacianEdgeLoss, self).__init__()
        self.loss_weight = loss_weight
        self.use_soft_label = use_soft_label
        
        # 1. 初始化时，自动生成一个 LoG (Laplacian of Gaussian) 卷积核
        # 这个核是固定的，不需要训练
        self.kernel = self._get_log_kernel(kernel_size, sigma)

    def _get_log_kernel(self, kernel_size, sigma):
        # 这是一个标准的数学公式，生成 LoG 算子
        pad = kernel_size // 2
        # 生成网格坐标
        x = torch.arange(-pad, pad + 1).float()
        y = torch.arange(-pad, pad + 1).float()
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        
        # LoG 公式计算
        r2 = xx**2 + yy**2
        sigma2 = sigma**2
        kernel = -(1 - r2 / (2 * sigma2)) * torch.exp(-r2 / (2 * sigma2)) / (3.14159 * sigma2**2)
        
        # 归一化：保证所有参数加起来为 0 (这是拉普拉斯算子的特性)
        kernel = kernel - kernel.mean()
        
        # 调整形状以适配卷积操作: (Out=1, In=1, H, W)
        return kernel.view(1, 1, kernel_size, kernel_size)

    def forward(self, pred, target):
        """
        计算 Loss 的过程
        pred:   PIDNet 边界分支的预测输出 (B, 1, H, W)
        target: 真实的语义分割标签 (B, H, W)
        """
        # 确保卷积核在同一个设备上 (CPU/GPU)
        if self.kernel.device != pred.device:
            self.kernel = self.kernel.to(pred.device)

        # -------------------------------------------------
        # 核心步骤：根据语义标签(Target)自动生成边缘真值(Edge GT)
        # -------------------------------------------------
        with torch.no_grad(): # 生成 GT 不需要梯度
            # 1. 预处理 Target
            # 变成 float 类型，增加一个通道维度: (B, H, W) -> (B, 1, H, W)
            target_float = target.unsqueeze(1).float()
            
            # 2. 卷积提取边缘
            # 在标签图上滑窗，数值变化剧烈的地方(不同类别交界处)会有大响应
            edge_gt = F.conv2d(target_float, self.kernel, padding=self.kernel.shape[-1]//2)
            
            # 3. 处理边缘强度
            edge_gt = torch.abs(edge_gt) # 拉普拉斯有负数，取绝对值
            
            # 4. 归一化到 [0, 1]
            # 找出每张图的最大值，除以它
            max_val = edge_gt.view(edge_gt.shape[0], -1).max(dim=1)[0]
            max_val = max_val.view(-1, 1, 1, 1) + 1e-6 # 加个极小值防除零
            edge_gt = edge_gt / max_val
            
            # 5. (可选) 转回硬标签
            if not self.use_soft_label:
                edge_gt = (edge_gt > 0.1).float()

        # -------------------------------------------------
        # 计算 Loss：加权二元交叉熵 (Weighted BCE)
        # -------------------------------------------------
        # 为什么要加权？因为边缘像素很少(正样本少)，背景像素很多(负样本多)
        # 如果不加权，模型会倾向于把所有像素都预测为“非边缘”
        
        n_pos = edge_gt.sum()
        n_neg = edge_gt.numel() - n_pos
        pos_weight = n_neg / (n_pos + 1e-6) # 正样本的权重 = 负样本数量 / 正样本数量
        
        # 限制权重上限，防止梯度爆炸 (比如全是黑图时)
        pos_weight = torch.clamp(pos_weight, max=20.0)

        # 计算 BCE Loss
        loss = F.binary_cross_entropy_with_logits(
            pred,           # 模型的预测 (Logits)
            edge_gt,        # 我们刚才生成的拉普拉斯边缘
            pos_weight=pos_weight
        )
        
        return loss * self.loss_weight
