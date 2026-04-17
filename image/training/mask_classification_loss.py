# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
#
# Portions of this file are adapted from the Hugging Face Transformers library,
# specifically from the Mask2Former loss implementation, which itself is based on
# Mask2Former and DETR by Facebook, Inc. and its affiliates.
# Used under the Apache 2.0 License.
# ---------------------------------------------------------------


from typing import List, Optional
import torch.distributed as dist
import torch
import torch.nn as nn
from torchvision.ops import box_convert, generalized_box_iou_loss
from transformers.models.mask2former.modeling_mask2former import (
    Mask2FormerLoss,
    Mask2FormerHungarianMatcher,
)


class MaskClassificationLoss(Mask2FormerLoss):
    def __init__(
        self,
        num_points: int,
        oversample_ratio: float,
        importance_sample_ratio: float,
        mask_coefficient: float,
        dice_coefficient: float,
        class_coefficient: float,
        num_labels: int,
        no_object_coefficient: float,
        bbox_l1_coefficient: float = 5.0,
        bbox_giou_coefficient: float = 2.0,
    ):
        nn.Module.__init__(self)
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        self.mask_coefficient = mask_coefficient
        self.dice_coefficient = dice_coefficient
        self.class_coefficient = class_coefficient
        self.bbox_l1_coefficient = bbox_l1_coefficient
        self.bbox_giou_coefficient = bbox_giou_coefficient
        self.num_labels = num_labels
        self.eos_coef = no_object_coefficient
        empty_weight = torch.ones(self.num_labels + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

        self.matcher = Mask2FormerHungarianMatcher(
            num_points=num_points,
            cost_mask=mask_coefficient,
            cost_dice=dice_coefficient,
            cost_class=class_coefficient,
        )

    def get_num_instances(self, class_labels: List[torch.Tensor], device: torch.device) -> torch.Tensor:
        """
        Normalize exactly like mask loss should be normalized:
        average number of GT instances across all ranks, clamped to >= 1.
        """
        num_instances = sum(len(labels) for labels in class_labels)
        num_instances = torch.as_tensor(num_instances, dtype=torch.float, device=device)

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(num_instances)
            num_instances = num_instances / dist.get_world_size()

        return torch.clamp(num_instances, min=1.0)
    
    @torch.compiler.disable
    def forward(
        self,
        masks_queries_logits: torch.Tensor,
        targets: List[dict],
        class_queries_logits: Optional[torch.Tensor] = None,
        box_queries_preds: Optional[torch.Tensor] = None,
    ):  
        mask_labels = [
            target["masks"].to(masks_queries_logits[0].dtype) for target in targets
        ]
        class_labels = [target["labels"].long() for target in targets]

        indices = self.matcher(
            masks_queries_logits=masks_queries_logits,
            mask_labels=mask_labels,
            class_queries_logits=class_queries_logits,
            class_labels=class_labels,
        )

        loss_masks = self.loss_masks(masks_queries_logits, mask_labels, indices)
        loss_classes = self.loss_labels(class_queries_logits, class_labels, indices)
        loss_boxes = {}
        if box_queries_preds is not None:
            num_instances = self.get_num_instances(class_labels, device=masks_queries_logits.device)
            loss_boxes = self.loss_boxes(
                box_queries_preds,
                [target["boxes"] for target in targets],
                indices,num_instances,
            )

        return {**loss_masks, **loss_classes, **loss_boxes}

    def loss_masks(self, masks_queries_logits, mask_labels, indices):
        loss_masks = super().loss_masks(masks_queries_logits, mask_labels, indices, 1)

        num_masks = sum(len(tgt) for (_, tgt) in indices)
        num_masks_tensor = torch.as_tensor(
            num_masks, dtype=torch.float, device=masks_queries_logits.device
        )

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(num_masks_tensor)
            world_size = dist.get_world_size()
        else:
            world_size = 1

        num_masks = torch.clamp(num_masks_tensor / world_size, min=1)

        for key in loss_masks.keys():
            loss_masks[key] = loss_masks[key] / num_masks

        return loss_masks

    def loss_total(self, losses_all_layers, log_fn) -> torch.Tensor:
        loss_total = None
        for loss_key, loss in losses_all_layers.items():
            log_fn(f"losses/train_{loss_key}", loss, sync_dist=True)

            if "mask" in loss_key:
                weighted_loss = loss * self.mask_coefficient
            elif "dice" in loss_key:
                weighted_loss = loss * self.dice_coefficient
            elif "cross_entropy" in loss_key:
                weighted_loss = loss * self.class_coefficient
            elif "bbox" in loss_key:
                weighted_loss = loss * self.bbox_l1_coefficient
            elif "giou" in loss_key:
                weighted_loss = loss * self.bbox_giou_coefficient
            else:
                raise ValueError(f"Unknown loss key: {loss_key}")

            if loss_total is None:
                loss_total = weighted_loss
            else:
                loss_total = torch.add(loss_total, weighted_loss)

        log_fn("losses/train_loss_total", loss_total, sync_dist=True, prog_bar=True)

        return loss_total  # type: ignore

    def loss_boxes(
        self,
        bbox_queries_preds: torch.Tensor,   # [B, Q, 4], normalized cxcywh
        box_labels: list[torch.Tensor],     # list[[Ni, 4]], normalized cxcywh
        indices,
        num_instances: torch.Tensor,
    ):
        src_idx = self._get_predictions_permutation_indices(indices)
        zero = bbox_queries_preds.sum() * 0.0

        if src_idx[0].numel() == 0:
            return {"loss_bbox": zero, "loss_giou": zero}

        src_boxes = bbox_queries_preds[src_idx]
        target_boxes = torch.cat(
            [target[j] for target, (_, j) in zip(box_labels, indices)],
            dim=0,
        ).to(src_boxes)

        # 1) Drop pairs that are already NaN / Inf in cxcywh
        finite_cxcywh = (
            torch.isfinite(src_boxes).all(dim=1) &
            torch.isfinite(target_boxes).all(dim=1)
        )

        if not finite_cxcywh.any():
            return {"loss_bbox": zero, "loss_giou": zero}

        src_boxes = src_boxes[finite_cxcywh]
        target_boxes = target_boxes[finite_cxcywh]

        # 2) Convert to xyxy and clamp to image range
        src_xyxy = box_convert(src_boxes, "cxcywh", "xyxy").clamp(0, 1)
        tgt_xyxy = box_convert(target_boxes, "cxcywh", "xyxy").clamp(0, 1)

        # 3) Keep only valid xyxy pairs
        valid_xyxy = (
            torch.isfinite(src_xyxy).all(dim=1) &
            torch.isfinite(tgt_xyxy).all(dim=1) &
            (src_xyxy[:, 2] > src_xyxy[:, 0]) &
            (src_xyxy[:, 3] > src_xyxy[:, 1]) &
            (tgt_xyxy[:, 2] > tgt_xyxy[:, 0]) &
            (tgt_xyxy[:, 3] > tgt_xyxy[:, 1])
        )

        if not valid_xyxy.any():
            return {"loss_bbox": zero, "loss_giou": zero}

        # If you want to ignore bad pairs entirely, use the same mask for both losses
        src_boxes = src_boxes[valid_xyxy]
        target_boxes = target_boxes[valid_xyxy]
        src_xyxy = src_xyxy[valid_xyxy]
        tgt_xyxy = tgt_xyxy[valid_xyxy]

        # L1 in cxcywh
        loss_bbox = torch.nn.functional.l1_loss(src_boxes, target_boxes, reduction="sum")

        # GIoU in xyxy
        loss_giou_vec = generalized_box_iou_loss(
            src_xyxy.float(),
            tgt_xyxy.float(),
            reduction="none",
            eps=1e-7,
        )

        # extra safety
        finite_giou = torch.isfinite(loss_giou_vec)
        loss_giou = loss_giou_vec[finite_giou].sum() if finite_giou.any() else zero

        denom = num_instances.clamp_min(1)
        return {
            "loss_bbox": loss_bbox / denom,
            "loss_giou": loss_giou / denom,
        }