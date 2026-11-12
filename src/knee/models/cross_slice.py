"""Cross-slice attention classifier.

Tokens: one per slice = projected backbone features + slice-position + plane + sequence-type
embeddings. A within-plane transformer lets each slice attend to its `slice_context` nearest
slices (band mask; "all" = full attention), a fusion transformer attends across all planes, and
a per-finding attention-pooling head produces 12 logits.

Masks are boolean "allowed" matrices. Every query may always attend to itself, so padded slots
and missing planes never produce NaNs; they are excluded as keys everywhere else.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from knee.models.heads import build_head


def parse_context(slice_context) -> int | None:
    """None for full attention ("all"), otherwise an odd int >= 1."""
    if slice_context is None or str(slice_context).lower() == "all":
        return None
    k = int(slice_context)
    if k < 1 or k % 2 == 0:
        raise ValueError(f"slice_context must be an odd integer >= 1 or 'all', got {slice_context}")
    return k


def band_allow(S: int, slice_context, device=None) -> torch.Tensor:
    """(S, S) bool: query i may attend key j iff |i - j| <= (k - 1) / 2 (all True for 'all')."""
    k = parse_context(slice_context)
    if k is None:
        return torch.ones(S, S, dtype=torch.bool, device=device)
    i = torch.arange(S, device=device)
    return (i[:, None] - i[None, :]).abs() <= (k - 1) // 2


class SinusoidalPosition(nn.Module):
    def __init__(self, d_model: int, scale: float = 100.0):
        super().__init__()
        half = d_model // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half) / half)
        self.register_buffer("freqs", freqs * scale, persistent=False)
        self.d_model = d_model

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        ang = pos.float().unsqueeze(-1) * self.freqs
        emb = torch.cat([ang.sin(), ang.cos()], dim=-1)
        if emb.shape[-1] < self.d_model:
            emb = F.pad(emb, (0, self.d_model - emb.shape[-1]))
        return emb


class BinnedPosition(nn.Module):
    def __init__(self, d_model: int, n_bins: int = 16):
        super().__init__()
        self.n_bins = n_bins
        self.emb = nn.Embedding(n_bins, d_model)

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        return self.emb((pos.float() * self.n_bins).long().clamp(0, self.n_bins - 1))


class Block(nn.Module):
    """Pre-norm transformer block using scaled_dot_product_attention with a boolean allow-mask."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, mlp_ratio: float = 4.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(d_model, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, d_model))
        self.drop = nn.Dropout(dropout)
        self.attn_dropout = dropout

    def forward(self, x: torch.Tensor, allow: torch.Tensor) -> torch.Tensor:
        N, T, D = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(N, T, 3, self.n_heads, D // self.n_heads).permute(2, 0, 3, 1, 4)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=allow[:, None],
                                             dropout_p=self.attn_dropout if self.training else 0.0)
        x = x + self.drop(self.proj(out.transpose(1, 2).reshape(N, T, D)))
        return x + self.drop(self.mlp(self.norm2(x)))


class PatchCrossSliceAttention(nn.Module):
    """CSA-Net style patch-level attention (Kumar et al. 2024), pooled into the slice token.

    For each slice s: neighbor slices' pooled patch tokens (within slice_context) are queries and
    the center slice's patch tokens are keys/values; plus in-slice self-attention over the center
    slice's patch tokens. Both outputs are mean-pooled and projected to d_model.
    """

    def __init__(self, grid_dim: int, d_model: int, n_heads: int, slice_context, dropout: float = 0.1):
        super().__init__()
        k = parse_context(slice_context)
        self.half = 2 if k is None else (k - 1) // 2  # "all" falls back to +-2 neighbors here
        self.in_norm = nn.LayerNorm(grid_dim)
        self.in_proj = nn.Linear(grid_dim, d_model)
        self.cross = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.out = nn.Sequential(nn.LayerNorm(2 * d_model), nn.Linear(2 * d_model, d_model))

    def forward(self, grid: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        # grid: (N, S, G, Dg) for N stacks; valid: (N, S)
        N, S, G, _ = grid.shape
        g = self.in_proj(self.in_norm(grid))                                      # (N, S, G, d)
        d = g.shape[-1]
        offsets = [o for o in range(-self.half, self.half + 1) if o != 0]
        if not offsets:
            offsets = [0]
        idx = torch.arange(S, device=grid.device)
        nb_idx = torch.stack([(idx + o).clamp(0, S - 1) for o in offsets], dim=1)  # (S, n_nb)
        nb_ok = torch.stack([((idx + o) >= 0) & ((idx + o) < S) for o in offsets], dim=1)
        nb_valid = valid[:, nb_idx] & nb_ok[None]                                 # (N, S, n_nb)
        q = g[:, nb_idx].reshape(N * S, len(offsets) * G, d)                     # neighbor patches as queries
        kv = g.reshape(N * S, G, d)                                               # center patches as keys/values
        cross, _ = self.cross(q, kv, kv, need_weights=False)
        w = nb_valid.reshape(N * S, len(offsets), 1).expand(-1, -1, G).reshape(N * S, -1, 1).float()
        cross = (cross * w).sum(1) / w.sum(1).clamp(min=1.0)                      # (N*S, d)
        selfo, _ = self.self_attn(kv, kv, kv, need_weights=False)
        out = self.out(torch.cat([cross, selfo.mean(1)], dim=-1)).view(N, S, d)
        return out * valid[..., None].float()


class CrossSliceModel(nn.Module):
    def __init__(self, in_dim: int, n_labels: int, mcfg, grid_dim: int = 0):
        super().__init__()
        d = int(mcfg.d_model)
        self.slice_context = mcfg.slice_context
        parse_context(self.slice_context)  # validate early
        self.in_norm = nn.LayerNorm(in_dim)
        self.in_proj = nn.Linear(in_dim, d)
        self.pos = (SinusoidalPosition(d) if mcfg.pos_encoding == "sinusoidal"
                    else BinnedPosition(d, int(mcfg.pos_bins)))
        self.plane_emb = nn.Embedding(3, d)
        self.seq_emb = nn.Embedding(int(mcfg.n_seq_types), d)
        self.patch = (PatchCrossSliceAttention(grid_dim, d, int(mcfg.n_heads), self.slice_context, float(mcfg.dropout))
                      if mcfg.patch_attention and grid_dim > 0 else None)
        self.slice_blocks = nn.ModuleList([Block(d, int(mcfg.n_heads), float(mcfg.dropout))
                                           for _ in range(int(mcfg.n_layers))])
        self.fusion_blocks = nn.ModuleList([Block(d, int(mcfg.n_heads), float(mcfg.dropout))
                                            for _ in range(int(mcfg.fusion_layers))])
        self.norm = nn.LayerNorm(d)
        self.head = build_head(mcfg.head, d, n_labels, int(mcfg.n_heads), float(mcfg.dropout))
        self.d_model = d

    def forward(self, feats, pos, valid, plane, seq, grid=None, return_attn: bool = False):
        """feats (B, K, S, F), pos (B, K, S), valid (B, K, S) bool, plane/seq (B, K) long,
        grid (B, K, S, G, Dg) optional. Returns logits (B, n_labels) [, attention (B, L, K*S)]."""
        B, K, S, _ = feats.shape
        x = self.in_proj(self.in_norm(feats))
        if self.patch is not None and grid is not None:
            x = x + self.patch(grid.reshape(B * K, S, *grid.shape[-2:]), valid.reshape(B * K, S)).view(B, K, S, -1)
        x = x + self.pos(pos) + self.plane_emb(plane)[:, :, None] + self.seq_emb(seq)[:, :, None]
        eye_s = torch.eye(S, dtype=torch.bool, device=x.device)
        v = valid.reshape(B * K, S)
        allow = (band_allow(S, self.slice_context, x.device)[None] & v[:, None, :]) | eye_s[None]
        x = x.reshape(B * K, S, -1)
        for blk in self.slice_blocks:
            x = blk(x, allow)
        x = x.reshape(B, K * S, -1)
        v = valid.reshape(B, K * S)
        if self.fusion_blocks:
            allow_f = v[:, None, :] | torch.eye(K * S, dtype=torch.bool, device=x.device)[None]
            for blk in self.fusion_blocks:
                x = blk(x, allow_f)
        x = self.norm(x)
        v_head = v.clone()
        v_head[~v.any(1), 0] = True  # a study without any slice: attend to a dummy token
        return self.head(x, v_head, return_attn)


def token_dropout(valid: torch.Tensor, p: float) -> torch.Tensor:
    """Randomly drop valid slice tokens (training only), keeping at least one per stack."""
    if p <= 0:
        return valid
    keep = (torch.rand_like(valid, dtype=torch.float) >= p) & valid
    empty = valid.any(-1) & ~keep.any(-1)
    if empty.any():
        first = valid.float().argmax(-1)
        keep[empty, first[empty]] = True
    return keep


class ImageKneeModel(nn.Module):
    """End-to-end mode (backbone.frozen=false): uint8 slice stacks -> backbone -> CrossSliceModel.

    Uses the same CrossSliceModel as the frozen-feature path; only the per-slice features are
    computed on the fly (and receive gradients).
    """

    def __init__(self, backbone, cross_slice: CrossSliceModel, feature_mode: str = "cls_mean",
                 channel_mode: str = "replicate", input_size: int | None = None):
        super().__init__()
        self.backbone = backbone
        self.cross_slice = cross_slice
        self.feature_mode = feature_mode
        self.channel_mode = channel_mode
        self.input_size = input_size

    def slice_features(self, images: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """images (B, K, S, H, W) uint8 -> (B, K, S, F) features (zeros at padded slots)."""
        from knee.models.backbones import to_input

        B, K, S, H, W = images.shape
        flat = images.reshape(B * K, S, H, W)
        lengths = valid.reshape(B * K, S).sum(1).tolist()
        xs = []
        for i, n in enumerate(lengths):
            if n == 0:
                continue
            x = flat[i, :n]
            if self.input_size and x.shape[-1] != self.input_size:
                x = F.interpolate(x[:, None].float(), size=(self.input_size, self.input_size), mode="bilinear",
                                  antialias=True, align_corners=False)[:, 0].round().clamp(0, 255).to(torch.uint8)
            xs.append(to_input(x, self.channel_mode))
        out = self.backbone(torch.cat(xs))
        parts = {"cls": out["cls"], "mean": out["patch"].mean(1) if out["patch"] is not None else out["cls"]}
        f = select_features(parts, self.feature_mode)
        feats = torch.zeros(B * K * S, f.shape[-1], device=f.device, dtype=f.dtype)
        feats[valid.reshape(-1)] = f  # valid slots are a prefix of each stack -> same order as torch.cat
        return feats.view(B, K, S, -1)

    def forward(self, images, pos, valid, plane, seq, return_attn: bool = False):
        feats = self.slice_features(images, valid)
        return self.cross_slice(feats.float(), pos, valid, plane, seq, return_attn=return_attn)


def select_features(parts: dict[str, torch.Tensor], feature_mode: str) -> torch.Tensor:
    if feature_mode == "cls_mean":
        return torch.cat([parts["cls"], parts["mean"]], dim=-1)
    if feature_mode in ("cls", "mean"):
        return parts[feature_mode]
    raise ValueError(f"unknown feature_mode '{feature_mode}'")


def feature_dim(backbone_dim: int, feature_mode: str) -> int:
    return 2 * backbone_dim if feature_mode == "cls_mean" else backbone_dim
