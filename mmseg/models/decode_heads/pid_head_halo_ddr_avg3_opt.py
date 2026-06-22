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
    HALO decode head for PIDNet with decoupled boundary supervision.

    Overview
    --------
    This head extends the standard PIDNet three-branch structure (P / I / D)
    with two HALO components:

    1. Online Laplacian Boundary Targets
       Boundary masks are generated on-the-fly from the semantic GT using a
       Laplacian kernel followed by optional max-pool dilation.  This avoids
       storing pre-computed edge maps and keeps the target resolution aligned
       with the current training stage.

    2. Cross-Branch Semantic Re-supervision
       The D branch acts as a boundary oracle.  Its sigmoid predictions are
       thresholded at 0.5 to build a binary boundary mask, which is then used
       to select a sparse set of boundary pixels from the semantic GT.  The
       resulting boundary-only label is fed back to supervise the I branch,
       forcing the main backbone to encode sharp boundary semantics.

    3. Decoupled Weight Scheduling
       The boundary-internal supervision weight (dice_w) and the cross-branch
       feedback weight (fb_w) are controlled by independent schedules.  This
       allows the D branch to use an aggressive dice_w (e.g. 3.0) in the early
       training phase without destabilising the I branch, because fb_w can be
       kept at a stable fixed value.

    Default schedule (active in this file)
    ---------------------------------------
    Stage 1  (0 ~ 33.3%)  :  dilation=5, dice_w=3.0, fb_w=1.0
    Stage 2  (33.3 ~ 66.7%):  dilation=4, dice_w=1.5, fb_w=1.0
    Stage 3  (66.7 ~ 100%) :  dilation=3, dice_w=1.0, fb_w=1.0

    Reported result: 79.08 mIoU on Cityscapes val.
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

        # Build the three-stage piecewise-constant decoupled schedule.
        self.dynamic_schedule = self._build_avg3_schedule(self.max_iters)
        # Retrieve the global MMEngine logger for training-step logging.
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
        Generate a class-agnostic binary boundary map by detecting label
        discontinuities between adjacent pixels (no per-class distinction).

        This method corresponds to the ablation variant 'Binary Target'.
        Swap in this function instead of _generate_laplacian_boundary when
        running ablation 4.

        Args:
            semantic_gt:  Semantic segmentation GT, shape (B, H, W).
            ignore_index: Pixels with this label are excluded from the output.
            dilation_size: Kernel size for max-pool boundary dilation.

        Returns:
            Binary boundary map of shape (B, H, W) with values in {0, 1}.
        """
        valid_mask = (semantic_gt != ignore_index).float()

        # Detect horizontal label transitions.
        h_diff = (semantic_gt[:, :, 1:] != semantic_gt[:, :, :-1]).float()
        # Detect vertical label transitions.
        v_diff = (semantic_gt[:, 1:, :] != semantic_gt[:, :-1, :]).float()

        # Accumulate transition signals and mark any triggered pixel as boundary.
        boundary_map = torch.zeros_like(semantic_gt).float()
        boundary_map[:, :, :-1] += h_diff
        boundary_map[:, :, 1:]  += h_diff
        boundary_map[:, :-1, :] += v_diff
        boundary_map[:, 1:, :]  += v_diff
        boundary_map = (boundary_map > 0).float()

        # Dilate to widen boundary regions (consistent with OLB dilation).
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

    # -------------------------------------------------------------------------
    # Online Laplacian Boundary Extraction
    # -------------------------------------------------------------------------
    def _generate_laplacian_boundary(self,
                                     semantic_gt: Tensor,
                                     num_classes: int,
                                     ignore_index: int = 255,
                                     dilation_size: int = 3) -> Tensor:
        """
        Compute per-class boundary masks from semantic GT using a Laplacian
        filter, then take the per-pixel union across all classes.

        Steps:
          1. One-hot encode the GT (ignore_index pixels mapped to class 0
             temporarily, then masked out via valid_mask).
          2. Apply a depth-wise Laplacian convolution per class to detect
             regions with non-zero gradient magnitude.
          3. Threshold at 0.1 to obtain binary per-class edge maps.
          4. Collapse class dimension with max-pooling to get a single map.
          5. Optionally dilate with max-pool to widen thin boundary lines.

        Args:
            semantic_gt:  Semantic segmentation GT, shape (B, H, W).
            num_classes:  Total number of semantic classes.
            ignore_index: Pixels with this label are masked to 0 in output.
            dilation_size: Kernel size for boundary dilation (1 = no dilation).

        Returns:
            Float boundary map of shape (B, H, W) with values in [0, 1].
        """
        
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

    # -------------------------------------------------------------------------
    # Three-Stage Piecewise-Constant Decoupled Scheduler
    # -------------------------------------------------------------------------
    def _build_avg3_schedule(self, max_iters: int) -> dict:
        """
        Build a piecewise-constant schedule that partitions training into three
        equal stages and assigns independent (dilation, dice_w, fb_w) values
        to each stage.

        Milestone layout (milestones mark the START of each segment pair):

            [0, t1]        Stage 1 – large dilation, high dice weight
            [t1+1, t2]     Stage 2 – medium dilation, moderate dice weight
            [t2+1, max]    Stage 3 – small dilation, low dice weight

        dice_w  controls the Dice loss weight for D-branch internal supervision.
        fb_w    controls the feedback loss weight applied to the I branch.
        """
        t1 = int(max_iters / 3.0)         # end of stage 1 (~33.3%)
        t2 = int(max_iters * 2.0 / 3.0)  # end of stage 2 (~66.7%)

        # ------------------------------------------------------------------ #
        # Default (active) schedule                                           #
        # Strategy: gradual dice_w decay (3.0 -> 1.5 -> 1.0) with dilation   #
        # shrinkage to progressively tighten boundary supervision.            #
        # Reported result: 79.08 mIoU on Cityscapes val.                      #
        # ------------------------------------------------------------------ #
        schedule = {
            0:         {'dilation': 5, 'dice_w': 3.0, 'fb_w': 1.0},
            t1:        {'dilation': 5, 'dice_w': 3.0, 'fb_w': 1.0},
            t1 + 1:    {'dilation': 4, 'dice_w': 1.5, 'fb_w': 1.0},  
            t2:        {'dilation': 4, 'dice_w': 1.5, 'fb_w': 1.0},
            t2 + 1:    {'dilation': 3, 'dice_w': 1.0, 'fb_w': 1.0},  
            max_iters: {'dilation': 3, 'dice_w': 1.0, 'fb_w': 1.0}
        }

        # ------------------------------------------------------------------ #
        # Ablation 1: OLB Only – fixed dilation=5, fixed dice_w=3.0          #
        # OLB Only  Class-wise  ✓  Fixed 5  Fixed 3.0  →  78.58 mIoU         #
        # ------------------------------------------------------------------ #
        # schedule = {
        #     0:         {'dilation': 5, 'dice_w': 3.0, 'fb_w': 1.0},
        #     t1:        {'dilation': 5, 'dice_w': 3.0, 'fb_w': 1.0},
        #     t1 + 1:    {'dilation': 4, 'dice_w': 3.0, 'fb_w': 1.0}, 
        #     t2:        {'dilation': 4, 'dice_w': 3.0, 'fb_w': 1.0},
        #     t2 + 1:    {'dilation': 3, 'dice_w': 3.0, 'fb_w': 1.0},  
        #     max_iters: {'dilation': 3, 'dice_w': 3.0, 'fb_w': 1.0}
        # }

        # ------------------------------------------------------------------ #
        # Ablation 2: OLB + Dynamic Dilation – dilation 5->4->3,             #
        #             fixed dice_w=3.0                                        #
        # OLB+DD  Class-wise  ✓  5-4-3  Fixed 3.0  →  78.84 mIoU            #
        # ------------------------------------------------------------------ #
        # schedule = {
        #     0:         {'dilation': 5, 'dice_w': 3.0, 'fb_w': 1.0},
        #     t1:        {'dilation': 5, 'dice_w': 3.0, 'fb_w': 1.0},
        #     t1 + 1:    {'dilation': 4, 'dice_w': 3.0, 'fb_w': 1.0}, 
        #     t2:        {'dilation': 4, 'dice_w': 3.0, 'fb_w': 1.0},
        #     t2 + 1:    {'dilation': 3, 'dice_w': 3.0, 'fb_w': 1.0},  
        #     max_iters: {'dilation': 3, 'dice_w': 3.0, 'fb_w': 1.0}
        # }

        # ------------------------------------------------------------------ #
        # Ablation 3: OLB + Adaptive weight Schedule – fixed dilation=5,     #
        #             dice_w decays 3.0 -> 1.5 -> 1.0                        #
        # OLB+AAS  Class-wise  ✓  Fixed 5  3.0-1.5-1.0  →  78.72 mIoU      #
        # ------------------------------------------------------------------ #
        # schedule = {
        #     0:         {'dilation': 5, 'dice_w': 3.0, 'fb_w': 1.0},
        #     t1:        {'dilation': 5, 'dice_w': 3.0, 'fb_w': 1.0},
        #     t1 + 1:    {'dilation': 5, 'dice_w': 1.5, 'fb_w': 1.0}, 
        #     t2:        {'dilation': 5, 'dice_w': 1.5, 'fb_w': 1.0},
        #     t2 + 1:    {'dilation': 5, 'dice_w': 1.0, 'fb_w': 1.0},  
        #     max_iters: {'dilation': 5, 'dice_w': 1.0, 'fb_w': 1.0}
        # }

        # ------------------------------------------------------------------ #
        # Ablation 4: Binary boundary target instead of class-wise OLB.      #
        # NOTE: also switch _generate_laplacian_boundary -> _generate_binary  #
        #       _boundary in loss_by_feat.                                    #
        # Binary Target  Binary  ✓  5-4-3  3.0-1.5-1.0  →  78.67 mIoU      #
        # ------------------------------------------------------------------ #
        # schedule = {
        #     0:         {'dilation': 5, 'dice_w': 3.0, 'fb_w': 1.0},
        #     t1:        {'dilation': 5, 'dice_w': 3.0, 'fb_w': 1.0},
        #     t1 + 1:    {'dilation': 4, 'dice_w': 1.5, 'fb_w': 1.0},  
        #     t2:        {'dilation': 4, 'dice_w': 1.5, 'fb_w': 1.0},
        #     t2 + 1:    {'dilation': 3, 'dice_w': 1.0, 'fb_w': 1.0},  
        #     max_iters: {'dilation': 3, 'dice_w': 1.0, 'fb_w': 1.0}
        # }

        # ------------------------------------------------------------------ #
        # Ablation 5: Disable cross-branch semantic re-supervision (fb_w=0). #
        # w/o Re-sup.  Class-wise  –  5-4-3  3.0-1.5-1.0  →  78.35 mIoU    #
        # ------------------------------------------------------------------ #
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
        """
        Look up the active schedule entry for the given training step and
        return the three decoupled parameters.

        Returns:
            Tuple of (dilation_size, dice_w, fb_w) for the current stage.
        """
        schedule = self.dynamic_schedule
        milestones = sorted(schedule.keys())

        # Find the milestone that the current step falls into.
        start_step = milestones[0]
        for i in range(len(milestones) - 1):
            if milestones[i] <= current_step <= milestones[i+1]:
                start_step = milestones[i]
                break

        cfg = schedule[start_step]
        # Return the three decoupled parameters for this stage.
        return cfg['dilation'], cfg['dice_w'], cfg['fb_w']

    def loss_by_feat(self, seg_logits: Tuple[Tensor], batch_data_samples: SampleList) -> dict:
        
        if self.training:
            self.local_step += 1
        current_step = self.local_step.item()
        
        # Fetch decoupled schedule parameters for the current step.
        cur_dilation, cur_dice_w, cur_fb_w = self._get_dynamic_params(current_step)

        # Periodic status log every 50 steps.
        if current_step % 50 == 0:
            self.logger.info(
                f'[HALO] step={current_step}/{self.max_iters} | '
                f'dilation={cur_dilation}, dice_w={cur_dice_w:.3f}, fb_w={cur_fb_w:.3f}'
            )

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

        # Ablation 4: uncomment below to swap in class-agnostic binary boundary.
        # bd_label = self._generate_binary_boundary(
        #     sem_label,
        #     ignore_index=self.ignore_index,
        #     dilation_size=cur_dilation
        # )

        # ------------------------------------------------------------------
        # 1) Global semantic losses for the P branch and the I branch.
        # ------------------------------------------------------------------
        loss['loss_sem_p'] = self.loss_decode[0](p_logit, sem_label, ignore_index=self.ignore_index)
        loss['loss_sem_i'] = self.loss_decode[1](i_logit, sem_label)

        # ------------------------------------------------------------------
        # 2) D-branch internal boundary supervision  (BCE + weighted Dice).
        #    dice_w is decoupled from fb_w and decays across training stages.
        # ------------------------------------------------------------------
        bce_loss = self.loss_decode[2](d_logit, bd_label)
        pred_sigmoid = torch.sigmoid(d_logit[:, 0, :, :])

        valid_mask = (sem_label != self.ignore_index).float()
        intersection = (pred_sigmoid * bd_label * valid_mask).sum(dim=(1, 2))
        union = (pred_sigmoid * valid_mask).sum(dim=(1, 2)) + (bd_label * valid_mask).sum(dim=(1, 2))
        dice_loss = (1.0 - (2.0 * intersection + 1e-5) / (union + 1e-5)).mean()

        # Combine BCE and Dice with the dynamic dice_w weight.
        loss['loss_bd_laplacian'] = bce_loss + cur_dice_w * dice_loss

        # ------------------------------------------------------------------
        # 3) Cross-branch semantic re-supervision  (feedback to I branch).
        #    The D branch's sigmoid predictions are thresholded at 0.5 to
        #    select boundary pixels, which are then used to construct a
        #    sparse boundary-semantic label for supervising the I branch.
        # ------------------------------------------------------------------

        # Binarise D-branch predictions to obtain the boundary pixel mask.
        bd_pred_mask = (pred_sigmoid > 0.5)

        # Build the boundary-semantic label: keep GT only at boundary pixels;
        # all other positions are masked out with ignore_index.
        filler = torch.ones_like(sem_label) * self.ignore_index
        halo_label = torch.where(bd_pred_mask, sem_label, filler)

        # Apply the feedback loss to the I branch, scaled by independent fb_w.
        loss['loss_halo_feedback'] = self.loss_decode[3](i_logit, halo_label) * cur_fb_w

        # ------------------------------------------------------------------
        # 4) Segmentation accuracy metric (based on I-branch predictions).
        # ------------------------------------------------------------------
        loss['acc_seg'] = accuracy(i_logit, sem_label, ignore_index=self.ignore_index)
            
        return loss
