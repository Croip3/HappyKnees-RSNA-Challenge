"""Backbone registry with a common interface.

    backbone = build_backbone(cfg)            # cfg.backbone.name, weights_path, image size
    out = backbone(x)                         # x: (N, 3, H, W), ImageNet-normalized
    out["cls"]: (N, D); out["patch"]: (N, h*w, D) or None

Weights are loaded from a local file/folder (Kaggle dataset) when `weights_path` is set, so no
network access is needed at inference time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

REGISTRY: dict[str, Callable[..., "Backbone"]] = {}


def register(name: str, **kwargs):
    def deco(cls):
        REGISTRY[name] = lambda **kw: cls(**{**kwargs, **kw})
        return cls
    return deco


def _resolve_weights(weights_path: str | None, name: str | None = None) -> str | None:
    """A weights file, a folder with one, or a parent folder with a sub-folder named after the backbone."""
    if not weights_path:
        return None
    p = Path(weights_path)
    if p.is_dir() and name and (p / name).is_dir():
        p = p / name
    if p.is_dir():
        for name in ["model.safetensors", "pytorch_model.bin", "open_clip_pytorch_model.bin"]:
            if (p / name).exists():
                return str(p / name)
        files = sorted(list(p.glob("*.safetensors")) + list(p.glob("*.pth")) + list(p.glob("*.bin")))
        if not files:
            raise FileNotFoundError(f"no weights file in {p}")
        return str(files[0])
    if not p.exists():
        raise FileNotFoundError(f"backbone weights not found: {p}")
    return str(p)


class Backbone(nn.Module):
    out_dim: int
    patch_grid: tuple[int, int] | None

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor | None]:  # pragma: no cover - interface
        raise NotImplementedError


class TimmViT(Backbone):
    """DINOv2 (and other timm ViTs): CLS token + patch tokens (register tokens dropped)."""

    def __init__(self, timm_name: str, weights_path: str | None = None, img_size: int = 224, pretrained: bool = True,
                 name: str | None = None):
        super().__init__()
        import timm

        weights = _resolve_weights(weights_path, name)
        kwargs = dict(pretrained=pretrained, num_classes=0, img_size=img_size)
        if weights:
            kwargs["pretrained_cfg_overlay"] = dict(file=weights)
        self.model = timm.create_model(timm_name, **kwargs)
        self.out_dim = int(self.model.num_features)
        self.n_prefix = int(getattr(self.model, "num_prefix_tokens", 1))
        ps = self.model.patch_embed.patch_size
        ps = ps[0] if isinstance(ps, (tuple, list)) else ps
        self.patch_grid = (img_size // ps, img_size // ps)

    def forward(self, x):
        tokens = self.model.forward_features(x)
        return {"cls": tokens[:, 0], "patch": tokens[:, self.n_prefix:]}


class TimmCNN(Backbone):
    """CNNs: global average pool as 'cls', the final feature map as patch tokens."""

    def __init__(self, timm_name: str, weights_path: str | None = None, img_size: int = 224, pretrained: bool = True,
                 name: str | None = None):
        super().__init__()
        import timm

        weights = _resolve_weights(weights_path, name)
        kwargs = dict(pretrained=pretrained, num_classes=0)
        if weights:
            kwargs["pretrained_cfg_overlay"] = dict(file=weights)
        self.model = timm.create_model(timm_name, **kwargs)
        self.out_dim = int(self.model.num_features)
        reduction = self.model.feature_info[-1]["reduction"] if hasattr(self.model, "feature_info") else 32
        self.patch_grid = (img_size // reduction, img_size // reduction)

    def forward(self, x):
        fmap = self.model.forward_features(x)
        patch = fmap.flatten(2).transpose(1, 2)
        return {"cls": patch.mean(1), "patch": patch}


register("dinov2_vits14_reg", timm_name="vit_small_patch14_reg4_dinov2.lvd142m")(TimmViT)
register("dinov2_vitb14_reg", timm_name="vit_base_patch14_reg4_dinov2.lvd142m")(TimmViT)
register("dinov2_vits14", timm_name="vit_small_patch14_dinov2.lvd142m")(TimmViT)
register("dinov2_vitb14", timm_name="vit_base_patch14_dinov2.lvd142m")(TimmViT)
register("convnext_tiny", timm_name="convnext_tiny.fb_in22k_ft_in1k")(TimmCNN)
register("convnext_small", timm_name="convnext_small.fb_in22k_ft_in1k")(TimmCNN)
register("efficientnet_b2", timm_name="efficientnet_b2.ra_in1k")(TimmCNN)
register("resnet50", timm_name="resnet50.a1_in1k")(TimmCNN)


def build_backbone(cfg, pretrained: bool = True) -> Backbone:
    b = cfg.backbone
    if b.name not in REGISTRY:
        raise KeyError(f"unknown backbone '{b.name}', available: {sorted(REGISTRY)}")
    img_size = int(b.image_size or cfg.preprocess.image_size)
    model = REGISTRY[b.name](weights_path=b.weights_path, img_size=img_size, pretrained=pretrained, name=b.name)
    if b.frozen:
        for p in model.parameters():
            p.requires_grad_(False)
        model.eval()
    return model


def to_input(img_u8: torch.Tensor, channel_mode: str = "replicate") -> torch.Tensor:
    """(S, H, W) uint8 slice stack -> (S, 3, H, W) ImageNet-normalized float.

    replicate: grayscale copied to 3 channels; neighbors: [s-1, s, s+1] as RGB (2.5D).
    """
    x = img_u8.float().div_(255.0)
    if channel_mode == "replicate":
        x = x[:, None].expand(-1, 3, -1, -1)
    elif channel_mode == "neighbors":
        idx = torch.arange(x.shape[0], device=x.device)
        prev, nxt = (idx - 1).clamp(min=0), (idx + 1).clamp(max=x.shape[0] - 1)
        x = torch.stack([x[prev], x, x[nxt]], dim=1)
    else:
        raise ValueError(f"unknown channel_mode '{channel_mode}'")
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std
