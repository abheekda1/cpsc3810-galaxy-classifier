"""
models/resnet_transfer.py
ResNet-18 fine-tuned for galaxy morphology classification.
This is the main model — use after establishing the baseline.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from data.dataset import NUM_CLASSES


def build_resnet(
    num_classes: int = NUM_CLASSES,
    freeze_backbone: bool = True,
    pretrained: bool = True,
) -> nn.Module:
    """
    Load a pretrained ResNet-18 and replace the final FC layer.

    Training strategy (two phases):
      Phase A — freeze_backbone=True:
        Only the new FC head is trained for a few epochs.
        This avoids corrupting pretrained features with large gradients
        before the head has converged.

      Phase B — freeze_backbone=False (call unfreeze_backbone()):
        All layers are unfrozen and fine-tuned end-to-end at a low LR.

    Args:
        num_classes:      Number of output classes.
        freeze_backbone:  If True, freeze all layers except the FC head.
        pretrained:       Use ImageNet pretrained weights.

    Returns:
        model: nn.Module ready for training.
    """
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)

    # Replace final classification head
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, num_classes),
    )

    if freeze_backbone:
        _freeze_backbone(model)

    return model


def _freeze_backbone(model: nn.Module) -> None:
    """Freeze all parameters except the FC head."""
    for name, param in model.named_parameters():
        if not name.startswith("fc"):
            param.requires_grad = False


def unfreeze_backbone(model: nn.Module, lr_scale: float = 0.1) -> None:
    """
    Unfreeze all parameters for end-to-end fine-tuning.
    Call this after the head has warmed up (Phase B).
    The backbone should be trained at a much lower LR than the head
    — pass lr_scale to your optimizer construction code.
    """
    for param in model.parameters():
        param.requires_grad = True
    print(f"[model] Backbone unfrozen — use lr × {lr_scale} for backbone params")


def get_param_groups(model: nn.Module, head_lr: float, backbone_lr: float) -> list[dict]:
    """
    Return parameter groups with different learning rates for
    the backbone vs. the classification head.

    Usage:
        optimizer = torch.optim.Adam(
            get_param_groups(model, head_lr=1e-3, backbone_lr=1e-4)
        )
    """
    backbone_params = [p for n, p in model.named_parameters() if not n.startswith("fc")]
    head_params     = [p for n, p in model.named_parameters() if     n.startswith("fc")]
    return [
        {"params": backbone_params, "lr": backbone_lr},
        {"params": head_params,     "lr": head_lr},
    ]


def count_params(model: nn.Module) -> None:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] Total params: {total:,}  |  Trainable: {trainable:,}")
