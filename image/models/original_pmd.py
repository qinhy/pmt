# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
#
# Portions of this file are adapted from the timm library by Ross Wightman,
# used under the Apache 2.0 License.
# ---------------------------------------------------------------

from typing import Optional, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.ops import box_convert, masks_to_boxes

from models.layers import ScaleBlock, DINOv3ViTLayer
from models.pmd import PlainMaskDecoder

def masks_to_boxes_cxcywh(mask_logits: torch.Tensor):
    B, Q, H, W = mask_logits.shape
    N = B * Q
    with torch.no_grad():
        masks = mask_logits.reshape(N, H, W) > 0.0
        empty_mask = ~masks.flatten(1).any(dim=1)
        base_boxes = torch.zeros(
            (N, 4),
            device=mask_logits.device,
            dtype=mask_logits.dtype,
        )
        if (~empty_mask).any():
            non_empty_boxes = masks_to_boxes(masks[~empty_mask])
            base_boxes[~empty_mask] = non_empty_boxes.to(
                mask_logits.device,
                mask_logits.dtype,
            )
        base_boxes[:, [0, 2]] /= max(W - 1, 1)
        base_boxes[:, [1, 3]] /= max(H - 1, 1)
        boxes_cxcywh = box_convert(
            base_boxes,
            in_fmt="xyxy",
            out_fmt="cxcywh",
        ).clamp(0, 1)
        return boxes_cxcywh.reshape(B, Q, 4)
    
class PlainMaskDecoder(PlainMaskDecoder):
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: Optional[int], 
        num_prefix_tokens: int,
        grid_size: Tuple[int, int],
        patch_size: Tuple[int, int],
        num_classes: int,
        num_q: int,
        num_blocks: int = 4,
        masked_attn_enabled: bool = True,
        interaction_indices: List[int] | torch.Tensor = [5, 11, 17, 23],
        lateral_projection: str = "mlp",
        residual_projection: bool = True,
        num_heads: int = 16,
    ):
        super().__init__(embed_dim, hidden_dim, num_prefix_tokens,
                         grid_size, patch_size, num_classes,
                         num_q, num_blocks, masked_attn_enabled,
                         interaction_indices, lateral_projection, residual_projection, num_heads)
        
        self.bbox_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim//2),
            nn.GELU(),
            nn.Linear(embed_dim//2, 4),
        )

    def _predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mask_logits, class_logits = super()._predict(x)
        q = x[:, : self.num_q, :]
        bbox_logits = self.bbox_head(q)*0.1 + masks_to_boxes_cxcywh(mask_logits)
        return mask_logits, class_logits, bbox_logits

    def forward(
        self,
        outputs: List[torch.Tensor],
        pe: Optional[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        
        lateral_features = self._lateral_projections_forward(outputs)

        x = torch.stack(lateral_features, dim=0).sum(dim=0)

        x = torch.cat(
            (self.q.weight[None, :, :].expand(x.shape[0], -1, -1), x), dim=1
        )

        attn_mask = None
        mask_logits_per_layer, class_logits_per_layer, bbox_logits_per_layer = [], [], []

        for i, block in enumerate(self.blocks):
            if self.masked_attn_enabled:
                mask_logits, class_logits, bbox_logits = self._predict(self.decoder_norm(x))
                mask_logits_per_layer.append(mask_logits)
                class_logits_per_layer.append(class_logits)
                bbox_logits_per_layer.append(bbox_logits)

                attn_mask = self._attn_mask(x, mask_logits, i)
                
            x = self.block_forward(x, block, attn_mask, pe)

        mask_logits, class_logits, bbox_logits = self._predict(self.decoder_norm(x))
        mask_logits_per_layer.append(mask_logits)
        class_logits_per_layer.append(class_logits)
        bbox_logits_per_layer.append(bbox_logits)

        return (
            mask_logits_per_layer,
            class_logits_per_layer,
            bbox_logits_per_layer
        )
