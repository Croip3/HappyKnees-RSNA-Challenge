"""AUC metrics with per-label masks (labels without both classes give NaN)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from knee.constants import LABELS


def per_label_auc(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """y_true (N, L) in {0, 1}; mask (N, L) bool selects the entries used for each label."""
    aucs = np.full(y_true.shape[1], np.nan)
    for j in range(y_true.shape[1]):
        m = np.ones(len(y_true), bool) if mask is None else mask[:, j].astype(bool)
        y, p = y_true[m, j], y_pred[m, j]
        if len(y) and 0 < y.sum() < len(y):
            aucs[j] = roc_auc_score(y, p)
    return aucs


def macro(aucs: np.ndarray) -> float:
    return float(np.nanmean(aucs)) if np.isfinite(aucs).any() else float("nan")


def llm_masks(sources: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hard labels and mask from LLM positive/negative sources only."""
    mask = np.isin(sources, ["llm_positive", "llm_negative"])
    return (sources == "llm_positive").astype(int), mask


def gold_masks(sources: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = sources == "gold"
    return (targets >= 0.5).astype(int), mask


def evaluate(preds: np.ndarray, targets: np.ndarray, sources: np.ndarray) -> dict:
    """LLM-label AUCs (positive/negative only) and gold AUCs per label + macro + collapse stats."""
    y_llm, m_llm = llm_masks(sources)
    y_gold, m_gold = gold_masks(sources, targets)
    auc_llm = per_label_auc(y_llm, preds, m_llm)
    auc_gold = per_label_auc(y_gold, preds, m_gold)
    return {
        "auc_llm": auc_llm, "macro_llm": macro(auc_llm),
        "auc_gold": auc_gold, "macro_gold": macro(auc_gold),
        "n_llm": m_llm.sum(0), "n_gold": m_gold.sum(0),
        "pred_std": preds.std(0), "mean_pred_std": float(preds.std(0).mean()),
    }


def worst_labels(aucs: np.ndarray, k: int = 3) -> list[tuple[str, float]]:
    order = np.argsort(np.where(np.isfinite(aucs), aucs, np.inf))
    return [(LABELS[i], float(aucs[i])) for i in order[:k] if np.isfinite(aucs[i])]
