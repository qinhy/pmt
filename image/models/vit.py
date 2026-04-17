# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------


import logging
import os
import torch
import torch.nn as nn

import timm
from transformers import AutoModel

def parameter_to_buffer(
    module: nn.Module,
    name: str,
    *,
    clone: bool = True,
    persistent: bool = True,
):
    p = getattr(module, name)

    if not isinstance(p, nn.Parameter):
        raise TypeError(f"{name!r} is not an nn.Parameter")

    t = p.detach()
    if clone:
        t = t.clone()

    # Remove from registered parameters
    delattr(module, name)

    # Re-register under the same name as a buffer
    module.register_buffer(name, t, persistent=persistent)
    return module


def freeze_module_as_buffers(
    module: nn.Module,
    *,
    clone: bool = True,
    persistent: bool = True,
):
    """
    Convert every parameter in `module` (including all child submodules)
    into a buffer under the same attribute name.

    Returns:
        list[str]: parameter names (relative to `module`) that were converted
    """
    # Snapshot names first, because we are going to mutate registrations
    names = [name for name, _ in module.named_parameters(recurse=True)]

    for full_name in names:
        *parts, leaf = full_name.split(".")
        submod = module
        for part in parts:
            submod = getattr(submod, part)
        parameter_to_buffer(submod, leaf, clone=clone, persistent=persistent)

    return names

class ViT(nn.Module):
    def __init__(
        self,
        img_size: tuple[int, int],
        patch_size: int = 16,
        backbone_name: str = "vit_large_patch14_reg4_dinov2",
        frozen_encoder: bool = False,
    ):
        super().__init__()

        self.image_size = img_size

        self.is_dinov3 = False
        
        if "/" in backbone_name:
            self.backbone = self.transformers_to_timm(
                AutoModel.from_pretrained(backbone_name),
                img_size,
            )
            self.is_dinov3 = True
        else:
            self.backbone = timm.create_model(
                backbone_name,
                pretrained=True,
                img_size=img_size,
                patch_size=patch_size,
                num_classes=0,
            )

        pixel_mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, -1, 1, 1)
        pixel_std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, -1, 1, 1)

        self.register_buffer("pixel_mean", pixel_mean)
        self.register_buffer("pixel_std", pixel_std)

        self.is_frozen = frozen_encoder
        if frozen_encoder:
            self.freeze_encoder()

    def transformers_to_timm(self, backbone, img_size: tuple[int, int]):
        backbone.patch_embed = backbone.embeddings
        backbone.patch_embed.patch_size = (
            backbone.embeddings.config.patch_size,
            backbone.embeddings.config.patch_size,
        )
        backbone.patch_embed.grid_size = (
            img_size[0] // backbone.embeddings.config.patch_size,
            img_size[1] // backbone.embeddings.config.patch_size,
        )

        backbone.embed_dim = backbone.embeddings.config.hidden_size
        backbone.num_prefix_tokens = backbone.patch_embed.config.num_register_tokens + 1
        backbone.blocks = backbone.layer

        del (
            backbone.patch_embed.mask_token,
            backbone.embeddings,
            backbone.layer,
        )

        return backbone

    def freeze_encoder(self):
        """Freeze the backbone encoder parameters."""
        # for param in self.backbone.parameters():
        #     param.requires_grad = False

        # self.backbone.eval()
        # freeze_module_as_buffers(self)

        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            logging.info("Backbone encoder frozen.")
