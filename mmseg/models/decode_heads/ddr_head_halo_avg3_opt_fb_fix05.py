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
    HALO decode head for DDRNet with decoupled boundary supervision.

    Overview
    --------
    This head augments the standard DDRNet two-branch design (Context + Spatial)
    with two HALO components:

    1. Online Laplacian Boundary Targets
       Boundary masks are computed on-the-fly from the semantic GT using a
       Laplacian filter and optional max-pool dilation, so no pre-computed edge
       maps are required.  The dilation radius shrinks across training stages to
       produce progressively tighter supervision.

    2. Cross-Branch Semantic Re-supervision
       The Spatial branch's sigmoid boundary predictions are thresholded at 0.5
       to obtain a boundary pixel mask.  Semantic GT labels are then masked to
       boundary pixels only, and this sparse boundary-semantic label is used to
       supervise the Context branch.  At inference time the Spatial branch is
       discarded, so this feedback incurs zero extra FLOPs.

    3. Decoupled Weight Scheduling
       The boundary-internal supervision weight (dice_w) and the cross-branch
       feedback weight (fb_w) are controlled by independent piecewise-linear
       schedules, enabling separate ablation of each component.

    Default schedule (active in this file)
    ----------------------------------------
    Stage 1  (0 ~ 33.3%)  :  dilation=5, dice_w=1.0, fb_w=0.5
    Stage 2  (33.3 ~ 66.7%):  dilation=4, dice_w=0.5, fb_w=0.5
    Stage 3  (66.7 ~ 100%) :  dilation=3, dice_w=0.1, fb_w=0.5

    Reported result: 78.23 mIoU on Cityscapes val.
    """

    def __init__(self,
                 in_channels: int,
                 channels: int,
                 num_classes: int,
                 norm_cfg: OptConfigType = dict(type='BN'),
                 act_cfg: OptConfigType = dict(type='ReLU', inplace=True),
                 max_iters: int = 120000,  # Total training iterations used to build schedule milestones.
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
        
        # Store total iterations so schedule milestones can be computed once.
        self.max_iters = max_iters

        # Local step counter incremented each training iteration.
        self.register_buffer('local_step', torch.tensor(0, dtype=torch.long))
        # Lightweight 1×1 conv that predicts a single-channel boundary map
        # from Spatial branch features during training.
        self.bd_cls_seg = nn.Conv2d(self.channels, 1, kernel_size=1)

        # Pre-compute the three-stage decoupled parameter schedule.
        self.dynamic_schedule = self._build_avg3_schedule(self.max_iters)
        # Retrieve the global MMEngine logger for periodic training-step logs.
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
            # DDRNet training receives two feature maps:
            #   c3_feat – high-resolution Spatial branch features
            #   c5_feat – low-resolution Context branch features
            c3_feat, c5_feat = inputs

            x_c = self.head(c5_feat)
            x_c_logit = self.cls_seg(x_c)

            x_s = self.aux_head(c3_feat)
            x_s_logit = self.aux_cls_seg(x_s)

            # Produce boundary logits from Spatial features (training only).
            bd_logit = self.bd_cls_seg(x_s)

            return x_c_logit, x_s_logit, bd_logit
        else:
            # At inference time only the Context branch is used;
            # no boundary head is executed, adding zero extra FLOPs.
            x_c = self.head(inputs)
            x_c = self.cls_seg(x_c)
            return x_c

    # -------------------------------------------------------------------------
    # Online Laplacian Boundary Extraction
    # -------------------------------------------------------------------------
    def _generate_laplacian_boundary(
        self,
        semantic_gt: Tensor,
        num_classes: int,
        ignore_index: int = 255,
        dilation_size: int = 3,
    ) -> Tensor:
        """
        Generate per-class boundary masks from the semantic GT on-the-fly using
        a Laplacian convolution, then collapse to a single binary boundary map.

        Steps
        -----
        1. Mask out ignore_index pixels with a valid_mask.
        2. One-hot encode the GT after temporarily zeroing ignore pixels.
        3. Apply a depth-wise Laplacian kernel per class to detect class-edge
           regions (pixels where label transitions occur).
        4. Threshold absolute response > 0.1 to get binary per-class edges.
        5. Reduce across the class dimension using element-wise max.
        6. Optionally dilate the boundary map with max-pool to widen thin lines.

        Args:
            semantic_gt:  Semantic segmentation GT tensor, shape (B, H, W).
            num_classes:  Total number of semantic classes.
            ignore_index: Pixels with this label value are excluded from output.
            dilation_size: Kernel size for max-pool dilation (1 = no dilation).

        Returns:
            Float boundary map of shape (B, H, W) with values in [0, 1].
        """
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

    # -------------------------------------------------------------------------
    # Three-Stage Piecewise-Linear Decoupled Scheduler
    # -------------------------------------------------------------------------
    def _build_avg3_schedule(self, max_iters: int) -> dict:
        """
        Partition training into three equal stages and assign independent
        (dilation, dice_w, fb_w) parameter pairs to each stage boundary.

        Milestone layout
        ----------------
        [0,    t1]       Stage 1 – coarse dilation, high dice weight
        [t1+1, t2]       Stage 2 – medium dilation, moderate dice weight
        [t2+1, max]      Stage 3 – fine dilation, low dice weight

        Parameters within each stage are linearly interpolated by
        _get_dynamic_params; parameters between stage boundaries are constant
        (identical start/end values per segment).

        dice_w  scales the Dice loss weight for Spatial-branch internal supervision.
        fb_w    scales the feedback loss applied to the Context branch.
        """
        t1 = int(max_iters / 3.0)          # stage 1/2 boundary (~33.3 %)
        t2 = int(max_iters * 2.0 / 3.0)   # stage 2/3 boundary (~66.7 %)

        # ------------------------------------------------------------------ #
        # Default (active) schedule – fixed fb_w = 0.5                       #
        # Strategy: linearly decay dice_w while keeping fb_w constant.       #
        # Reported result: 78.23 mIoU on Cityscapes val.                      #
        # ------------------------------------------------------------------ #
        schedule = {
            # Stage 1
            0:      {'dilation': 5, 'dice_w': 1.0, 'fb_w': 0.5},
            t1:     {'dilation': 5, 'dice_w': 1.0, 'fb_w': 0.5},

            # Stage 2
            t1 + 1: {'dilation': 4, 'dice_w': 0.5, 'fb_w': 0.5},
            t2:     {'dilation': 4, 'dice_w': 0.5, 'fb_w': 0.5},

            # Stage 3
            t2 + 1: {'dilation': 3, 'dice_w': 0.1, 'fb_w': 0.5},
            max_iters: {'dilation': 3, 'dice_w': 0.1, 'fb_w': 0.5}
        }

        # ------------------------------------------------------------------ #
        # Ablation 1 – fixed fb_w = 1.0                                      #
        # Stronger feedback throughout training; Context branch over-perturbed #
        # in the final 40k steps, preventing stable boundary learning.        #
        # Reported result: 77.64 mIoU.                                        #
        # ------------------------------------------------------------------ #
        # schedule = {
        #     # Stage 1
        #     0:      {'dilation': 5, 'dice_w': 1.0, 'fb_w': 1.0},
        #     t1:     {'dilation': 5, 'dice_w': 1.0, 'fb_w': 1.0},
        #
        #     # Stage 2
        #     t1 + 1: {'dilation': 4, 'dice_w': 0.5, 'fb_w': 1.0},
        #     t2:     {'dilation': 4, 'dice_w': 0.5, 'fb_w': 1.0},
        #
        #     # Stage 3
        #     t2 + 1: {'dilation': 3, 'dice_w': 0.1, 'fb_w': 1.0},
        #     max_iters: {'dilation': 3, 'dice_w': 0.1, 'fb_w': 1.0}
        # }

        # ------------------------------------------------------------------ #
        # Ablation 2 – fixed fb_w = 0.3                                      #
        # Weaker feedback; boundary semantics underrepresented in I branch.  #
        # Reported result: 77.81 mIoU.                                        #
        # ------------------------------------------------------------------ #
        # schedule = {
        #     # Stage 1
        #     0:      {'dilation': 5, 'dice_w': 1.0, 'fb_w': 0.3},
        #     t1:     {'dilation': 5, 'dice_w': 1.0, 'fb_w': 0.3},
        #
        #     # Stage 2
        #     t1 + 1: {'dilation': 4, 'dice_w': 0.5, 'fb_w': 0.3},
        #     t2:     {'dilation': 4, 'dice_w': 0.5, 'fb_w': 0.3},
        #
        #     # Stage 3
        #     t2 + 1: {'dilation': 3, 'dice_w': 0.1, 'fb_w': 0.3},
        #     max_iters: {'dilation': 3, 'dice_w': 0.1, 'fb_w': 0.3}
        # }

        # ------------------------------------------------------------------ #
        # Ablation 3 – fixed fb_w = 0.1                                      #
        # Very weak feedback; minimal improvement over no-feedback baseline.  #
        # Reported result: 77.48 mIoU.                                        #
        # ------------------------------------------------------------------ #
        # schedule = {
        #     # Stage 1
        #     0:      {'dilation': 5, 'dice_w': 1.0, 'fb_w': 0.1},
        #     t1:     {'dilation': 5, 'dice_w': 1.0, 'fb_w': 0.1},
        #
        #     # Stage 2
        #     t1 + 1: {'dilation': 4, 'dice_w': 0.5, 'fb_w': 0.1},
        #     t2:     {'dilation': 4, 'dice_w': 0.5, 'fb_w': 0.1},
        #
        #     # Stage 3
        #     t2 + 1: {'dilation': 3, 'dice_w': 0.1, 'fb_w': 0.1},
        #     max_iters: {'dilation': 3, 'dice_w': 0.1, 'fb_w': 0.1}
        # }

        # ------------------------------------------------------------------ #
        # Ablation 4 – fixed fb_w = 0.0 (cross-branch feedback disabled)    #
        # Establishes the no-feedback baseline for cross-branch supervision.  #
        # Reported result: 77.56 mIoU.                                        #
        # ------------------------------------------------------------------ #
        # schedule = {
        #     # Stage 1
        #     0:      {'dilation': 5, 'dice_w': 1.0, 'fb_w': 0.0},
        #     t1:     {'dilation': 5, 'dice_w': 1.0, 'fb_w': 0.0},
        #
        #     # Stage 2
        #     t1 + 1: {'dilation': 4, 'dice_w': 0.5, 'fb_w': 0.0},
        #     t2:     {'dilation': 4, 'dice_w': 0.5, 'fb_w': 0.0},
        #
        #     # Stage 3
        #     t2 + 1: {'dilation': 3, 'dice_w': 0.1, 'fb_w': 0.0},
        #     max_iters: {'dilation': 3, 'dice_w': 0.1, 'fb_w': 0.0}
        # }

        return schedule

    def _get_dynamic_params(self, current_step: int) -> Tuple[int, float, float]:
        """
        Look up the active schedule segment for the current training step and
        return piecewise-linearly interpolated parameters.

        Because adjacent stage-boundary entries in the schedule dict share the
        same parameter values (e.g. t1 and t1+1 are in different segments), the
        interpolation is effectively constant within each stage and produces a
        smooth transition only when a segment spans a genuine range.

        Args:
            current_step: Current global training iteration.

        Returns:
            Tuple of (dilation_size, dice_w, fb_w) for the current step.
        """
        schedule = self.dynamic_schedule
        milestones = sorted(schedule.keys())

        # Clamp to schedule boundaries.
        if current_step <= milestones[0]:
            cfg = schedule[milestones[0]]
            return cfg['dilation'], cfg['dice_w'], cfg['fb_w']
        if current_step >= milestones[-1]:
            cfg = schedule[milestones[-1]]
            return cfg['dilation'], cfg['dice_w'], cfg['fb_w']

        # Find the segment [milestones[i], milestones[i+1]] that contains the
        # current step.
        start_step, end_step = milestones[0], milestones[-1]
        for i in range(len(milestones) - 1):
            if milestones[i] <= current_step < milestones[i+1]:
                start_step, end_step = milestones[i], milestones[i+1]
                break

        start_cfg, end_cfg = schedule[start_step], schedule[end_step]
        progress = (current_step - start_step) / float(end_step - start_step)

        # dilation is kept constant within each stage (no interpolation needed).
        cur_dilation = start_cfg['dilation']
        # dice_w and fb_w are independently linearly interpolated.
        cur_dice_w = start_cfg['dice_w'] + progress * (end_cfg['dice_w'] - start_cfg['dice_w'])
        cur_fb_w   = start_cfg['fb_w']   + progress * (end_cfg['fb_w']   - start_cfg['fb_w'])

        return cur_dilation, cur_dice_w, cur_fb_w

    def loss_by_feat(self, seg_logits: Tuple[Tensor], batch_data_samples: SampleList) -> dict:
        
        # Advance the local step counter during training only.
        if self.training:
            self.local_step += 1
        current_step = self.local_step.item()

        # Fetch the three decoupled schedule parameters for this step.
        cur_dilation, cur_dice_w, cur_fb_w = self._get_dynamic_params(current_step)

        # Periodic status log every 50 steps.
        if current_step % 50 == 0:
            self.logger.info(
                f'[HALO-DDR] step={current_step}/{self.max_iters} | '
                f'dilation={cur_dilation}, dice_w={cur_dice_w:.3f}, fb_w={cur_fb_w:.3f}'
            )
        
        loss = dict()
        context_logit, spatial_logit, bd_logit = seg_logits
        seg_label = self._stack_batch_gt(batch_data_samples)

        context_logit = resize(context_logit, size=seg_label.shape[2:], mode='bilinear', align_corners=self.align_corners)
        spatial_logit = resize(spatial_logit, size=seg_label.shape[2:], mode='bilinear', align_corners=self.align_corners)
        bd_logit = resize(bd_logit, size=seg_label.shape[2:], mode='bilinear', align_corners=self.align_corners)
        
        seg_label = seg_label.squeeze(1)

        # Generate online Laplacian boundary targets using the current dilation.
        bd_label = self._generate_laplacian_boundary(
            seg_label, num_classes=self.num_classes,
            ignore_index=self.ignore_index, dilation_size=cur_dilation
        )

        # ------------------------------------------------------------------
        # 1) Global semantic losses for Context and Spatial branches.
        #    Both branches see the full-image labels without any masking.
        # ------------------------------------------------------------------
        loss['loss_context'] = self.loss_decode[0](context_logit, seg_label)
        loss['loss_spatial'] = self.loss_decode[1](spatial_logit, seg_label)

        # ------------------------------------------------------------------
        # 2) Spatial-branch internal boundary supervision  (BCE + Dice).
        #    dice_w is decoupled from fb_w and decreases across training stages
        #    to relax boundary constraints as the model converges.
        # ------------------------------------------------------------------
        bce_loss = F.binary_cross_entropy_with_logits(bd_logit.squeeze(1), bd_label)

        pred_sigmoid = torch.sigmoid(bd_logit[:, 0, :, :])
        valid_mask = (seg_label != self.ignore_index).float()
        intersection = (pred_sigmoid * bd_label * valid_mask).sum(dim=(1, 2))
        union = (pred_sigmoid * valid_mask).sum(dim=(1, 2)) + (bd_label * valid_mask).sum(dim=(1, 2))
        dice_loss = (1.0 - (2.0 * intersection + 1e-5) / (union + 1e-5)).mean()

        # Combine BCE and Dice with the dynamic dice_w weight.
        loss['loss_bd_laplacian'] = bce_loss + cur_dice_w * dice_loss

        # ------------------------------------------------------------------
        # 3) Cross-branch semantic re-supervision  (feedback to Context).
        #    Threshold Spatial boundary predictions at 0.5 to obtain a binary
        #    boundary mask, then build a sparse boundary-semantic label by
        #    keeping GT only at predicted boundary pixels.
        # ------------------------------------------------------------------

        # Binarise Spatial boundary predictions to obtain the boundary mask.
        bd_pred_mask = (pred_sigmoid > 0.5)

        # Construct boundary-semantic label: GT at boundary pixels,
        # ignore_index everywhere else.
        filler = torch.ones_like(seg_label) * self.ignore_index
        halo_label = torch.where(bd_pred_mask, seg_label, filler)

        # Apply the feedback loss to the Context branch, scaled by fb_w.
        loss['loss_halo_feedback'] = self.loss_decode[0](context_logit, halo_label) * cur_fb_w

        # ------------------------------------------------------------------
        # 4) Segmentation accuracy metric (based on Context-branch logits).
        # ------------------------------------------------------------------
        loss['acc_seg'] = accuracy(context_logit, seg_label, ignore_index=self.ignore_index)

        return loss

