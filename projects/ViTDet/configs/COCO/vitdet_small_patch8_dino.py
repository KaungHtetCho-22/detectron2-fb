# Save this as: detectron2/projects/ViTDet/configs/COCO/vitdet_small_patch8_dino.py

import torch
from detectron2.config import LazyCall as L
from detectron2.modeling.backbone.vit import get_vit_lr_decay_rate
from detectron2.solver import WarmupParamScheduler

from ..common.coco_schedule import lr_multiplier_1x
from .vitdet_base_cascade_mask_rcnn import (
    dataloader,
    model,
    train,
    optimizer,
)

# Model modifications for ViT-Small with patch_size=8
model.backbone.net.img_size = 224  # Adjust based on your training
model.backbone.net.patch_size = 8  # Your DINO patch size
model.backbone.net.embed_dim = 384  # ViT-Small embedding dimension
model.backbone.net.depth = 12       # ViT-Small depth
model.backbone.net.num_heads = 6    # ViT-Small attention heads
model.backbone.net.mlp_ratio = 4.0  # Standard ViT MLP ratio

# Adjust backbone settings for smaller patches
model.backbone.square_pad = 224  # Should match your training image size
model.backbone.out_features = ["x3_nocls", "x7_nocls", "x11_nocls"]

# FPN settings - might need adjustment for patch_size=8
model.backbone.scale_factors = (4.0, 2.0, 1.0, 0.5)

# Training settings optimized for DINO pre-trained weights
train.max_iter = 90000  # Adjust based on your dataset size
train.eval_period = 5000

# Optimizer settings for fine-tuning pre-trained model
optimizer.lr = 1e-4  # Lower learning rate for pre-trained backbone
optimizer.weight_decay = 0.1

# Learning rate schedule with layer decay (important for ViT fine-tuning)
lr_multiplier = WarmupParamScheduler(
    scheduler=L(torch.optim.lr_scheduler.MultiStepLR)(
        milestones=[75000, 85000],
        gamma=0.1,
    ),
    warmup_length=1000 / train.max_iter,
    warmup_method="linear",
    warmup_factor=0.001,
)

# Layer-wise learning rate decay (optional but recommended)
def get_vit_lr_decay_rate_small(name, lr_decay_rate=0.7, num_layers=12):
    """
    Calculate layer-wise lr decay rate for ViT-Small.
    """
    return get_vit_lr_decay_rate(name, lr_decay_rate, num_layers)

# Apply layer-wise lr decay
optimizer.params.lr_factor_func = lambda module_name: get_vit_lr_decay_rate_small(
    module_name, lr_decay_rate=0.8, num_layers=12
)

# Data loading settings
dataloader.train.total_batch_size = 16  # Adjust based on your GPU memory
dataloader.test.num_workers = 4