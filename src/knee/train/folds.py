"""Study-level K-fold split, multi-label stratified on hard labels; gold studies spread evenly."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from knee.constants import LABELS, STUDY


def _stratified_folds(y: np.ndarray, n_folds: int, seed: int) -> np.ndarray:
    folds = np.full(len(y), -1, dtype=int)
    if len(y) == 0:
        return folds
    if len(y) < n_folds:
        return np.arange(len(y)) % n_folds
    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

        splitter = MultilabelStratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for k, (_, val) in enumerate(splitter.split(np.zeros(len(y)), y)):
            folds[val] = k
    except ImportError:
        # fallback: stratify on the number of positive labels (a coarse proxy)
        from sklearn.model_selection import StratifiedKFold

        strata = np.minimum(y.sum(1), 6)
        counts = np.bincount(strata)
        if counts[counts > 0].min() < n_folds:
            from sklearn.model_selection import KFold

            for k, (_, val) in enumerate(KFold(n_folds, shuffle=True, random_state=seed).split(y)):
                folds[val] = k
        else:
            for k, (_, val) in enumerate(StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(y, strata)):
                folds[val] = k
    return folds


def make_folds(studies: list[str], hard: np.ndarray, is_gold: np.ndarray, n_folds: int = 5,
               seed: int = 42) -> pd.DataFrame:
    """hard: (N, L) 0/1 labels (LLM positive, or gold where available). Gold studies are split
    separately so that every fold gets ~len(gold)/n_folds of them."""
    folds = np.full(len(studies), -1, dtype=int)
    for mask in [is_gold, ~is_gold]:
        idx = np.where(mask)[0]
        folds[idx] = _stratified_folds(hard[idx], n_folds, seed)
    return pd.DataFrame({STUDY: studies, "fold": folds, "is_gold": is_gold})


def load_or_make_folds(cfg, labels_final: pd.DataFrame, log, overwrite: bool = False) -> pd.DataFrame:
    n_folds, seed = int(cfg.train.folds), int(cfg.train.seed)
    path = Path(cfg.paths.labels_dir) / f"folds_{n_folds}_seed{seed}.csv"
    if path.exists() and not overwrite:
        folds = pd.read_csv(path)
        log.info(f"folds: loaded {path.name}")
        return folds
    wide_src = labels_final.pivot(index=STUDY, columns="label", values="source")[LABELS]
    wide_tgt = labels_final.pivot(index=STUDY, columns="label", values="target")[LABELS]
    studies = wide_src.index.tolist()
    is_gold = (wide_src == "gold").all(axis=1).values
    hard = np.where(is_gold[:, None], (wide_tgt.values >= 0.5), (wide_src.values == "llm_positive")).astype(int)
    folds = make_folds(studies, hard, is_gold, n_folds, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    folds.to_csv(path, index=False)
    log.info(f"folds: wrote {path.name} | studies per fold {folds.fold.value_counts().sort_index().tolist()} | "
             f"gold per fold {folds[folds.is_gold].fold.value_counts().sort_index().tolist()}")
    return folds
