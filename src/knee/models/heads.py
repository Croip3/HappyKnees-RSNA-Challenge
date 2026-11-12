"""Study-level heads: per-finding attention pooling (default) or masked mean pooling (ablation)."""

from __future__ import annotations

import torch
import torch.nn as nn


class AttentionPoolHead(nn.Module):
    """One learned query per finding cross-attends over all slice tokens -> one logit per finding."""

    def __init__(self, d_model: int, n_labels: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.queries = nn.Parameter(torch.randn(n_labels, d_model) * 0.02)
        self.norm_kv = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.kv_proj = nn.Linear(d_model, 2 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm_out = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.cls_w = nn.Parameter(torch.zeros(n_labels, d_model))
        self.cls_b = nn.Parameter(torch.zeros(n_labels))
        nn.init.normal_(self.cls_w, std=0.02)

    def forward(self, tokens: torch.Tensor, valid: torch.Tensor, return_attn: bool = False):
        B, T, D = tokens.shape
        L, H = self.queries.shape[0], self.n_heads
        kv = self.kv_proj(self.norm_kv(tokens)).view(B, T, 2, H, D // H).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]                                                  # (B, H, T, dh)
        q = self.q_proj(self.queries).view(1, L, H, D // H).transpose(1, 2).expand(B, -1, -1, -1)
        logits_attn = (q @ k.transpose(-1, -2)) / (D // H) ** 0.5            # (B, H, L, T)
        logits_attn = logits_attn.masked_fill(~valid[:, None, None, :], float("-inf"))
        attn = torch.softmax(logits_attn.float(), dim=-1).to(v.dtype)
        pooled = (self.dropout(attn) @ v).transpose(1, 2).reshape(B, L, D)
        pooled = self.norm_out(self.queries[None] + self.out_proj(pooled))
        logits = (pooled * self.cls_w[None]).sum(-1) + self.cls_b
        return (logits, attn.mean(1)) if return_attn else logits


class MeanPoolHead(nn.Module):
    def __init__(self, d_model: int, n_labels: int, dropout: float = 0.1, **_):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, n_labels)

    def forward(self, tokens: torch.Tensor, valid: torch.Tensor, return_attn: bool = False):
        w = valid.float().unsqueeze(-1)
        pooled = (tokens * w).sum(1) / w.sum(1).clamp(min=1.0)
        logits = self.fc(self.dropout(self.norm(pooled)))
        if return_attn:
            attn = (w.squeeze(-1) / w.sum(1).clamp(min=1.0))[:, None].expand(-1, logits.shape[1], -1)
            return logits, attn
        return logits


def build_head(kind: str, d_model: int, n_labels: int, n_heads: int, dropout: float) -> nn.Module:
    if kind == "attention":
        return AttentionPoolHead(d_model, n_labels, n_heads, dropout)
    if kind in ("mean", "cls"):
        return MeanPoolHead(d_model, n_labels, dropout)
    raise ValueError(f"unknown head '{kind}' (attention | mean)")

