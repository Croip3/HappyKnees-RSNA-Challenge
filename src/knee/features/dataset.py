"""In-RAM feature store + dataset/collate producing padded (B, K, S, F) batches for CrossSliceModel."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch

from knee.features.extract import load_features


def slice_features(f: dict[str, np.ndarray], feature_mode: str, mirrored: bool = False) -> np.ndarray:
    sfx = "_m" if mirrored else ""
    if feature_mode == "cls_mean":
        return np.concatenate([f["cls" + sfx], f["mean" + sfx]], axis=-1)
    if feature_mode in ("cls", "mean"):
        return f[feature_mode + sfx]
    raise ValueError(f"unknown feature_mode '{feature_mode}'")


class FeatureStore:
    """Loads per-study feature files once (fp16) and keeps them in memory."""

    def __init__(self, feat_dir: str | Path, studies: list[str], feature_mode: str = "cls_mean",
                 use_grid: bool = False, use_mirror: bool = False, threads: int = 8):
        self.feat_dir = Path(feat_dir)
        self.feature_mode = feature_mode
        self.use_grid = use_grid
        self.use_mirror = use_mirror
        keys = ["stack", "pos", "plane", "seq_type", "present", "cls", "mean"]
        if use_grid:
            keys.append("grid")
        if use_mirror:
            keys += ["cls_m", "mean_m"] + (["grid_m"] if use_grid else [])
        paths = {s: self.feat_dir / f"{s}.npz" for s in studies}
        missing = [s for s, p in paths.items() if not p.exists()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} studies have no feature file in {self.feat_dir}, "
                                    f"e.g. {missing[:3]}")

        def _load(s):
            f = load_features(paths[s], keys)
            item = {k: f[k] for k in ["stack", "pos", "plane", "seq_type", "present"]}
            item["x"] = slice_features(f, feature_mode)
            if use_mirror and "cls_m" in f:
                item["x_m"] = slice_features(f, feature_mode, mirrored=True)
            if use_grid and "grid" in f:
                item["grid"] = f["grid"]
                if use_mirror and "grid_m" in f:
                    item["grid_m"] = f["grid_m"]
            return s, item

        with ThreadPoolExecutor(threads) as ex:
            self.data: dict[str, dict[str, np.ndarray]] = dict(ex.map(_load, studies))
        first = next(iter(self.data.values()))
        self.in_dim = int(first["x"].shape[-1])
        self.grid_dim = int(first["grid"].shape[-1]) if use_grid and "grid" in first else 0
        self.n_stacks = int(len(first["plane"]))

    @classmethod
    def from_memory(cls, feats: dict[str, dict[str, np.ndarray]], feature_mode: str = "cls_mean",
                    use_grid: bool = False, use_mirror: bool = False) -> "FeatureStore":
        """Build a store from already computed features (inference path)."""
        self = cls.__new__(cls)
        self.feat_dir, self.feature_mode, self.use_grid, self.use_mirror = None, feature_mode, use_grid, use_mirror
        self.data = {}
        for s, f in feats.items():
            item = {k: f[k] for k in ["stack", "pos", "plane", "seq_type", "present"]}
            item["x"] = slice_features(f, feature_mode)
            if use_mirror and "cls_m" in f:
                item["x_m"] = slice_features(f, feature_mode, mirrored=True)
            if use_grid and "grid" in f:
                item["grid"] = f["grid"]
                if use_mirror and "grid_m" in f:
                    item["grid_m"] = f["grid_m"]
            self.data[s] = item
        first = next(iter(self.data.values()))
        self.in_dim = int(first["x"].shape[-1])
        self.grid_dim = int(first["grid"].shape[-1]) if use_grid and "grid" in first else 0
        self.n_stacks = int(len(first["plane"]))
        return self

    def __contains__(self, s: str) -> bool:
        return s in self.data

    def nbytes(self) -> int:
        return sum(a.nbytes for d in self.data.values() for a in d.values())


class StudyFeatureDataset(torch.utils.data.Dataset):
    def __init__(self, store: FeatureStore, studies: list[str], targets: np.ndarray | None = None,
                 weights: np.ndarray | None = None, train: bool = False, mirror_p: float = 0.0,
                 force_mirror: bool = False):
        self.store, self.studies = store, list(studies)
        self.targets, self.weights = targets, weights
        self.train, self.mirror_p, self.force_mirror = train, mirror_p, force_mirror
        if (mirror_p > 0 or force_mirror) and not store.use_mirror:
            raise ValueError("mirror augmentation/TTA needs mirrored features (features.mirror=true)")

    def __len__(self) -> int:
        return len(self.studies)

    def __getitem__(self, i: int) -> dict[str, Any]:
        s = self.studies[i]
        d = self.store.data[s]
        mirror = self.force_mirror or (self.train and self.mirror_p > 0 and np.random.rand() < self.mirror_p)
        x = d["x_m"] if mirror else d["x"]
        grid = (d.get("grid_m") if mirror else d.get("grid")) if self.store.use_grid else None
        stacks = []
        for k in range(len(d["plane"])):
            m = d["stack"] == k
            xs, ps = x[m], d["pos"][m]
            gs = grid[m] if grid is not None else None
            if mirror and int(d["plane"][k]) == 0:  # sagittal: mirror = reverse slice order
                xs, ps = xs[::-1], (1.0 - ps[::-1])
                gs = gs[::-1] if gs is not None else None
            stacks.append((xs, ps, gs))
        item = {"study": s, "stacks": stacks, "plane": d["plane"].astype(np.int64),
                "seq": d["seq_type"].astype(np.int64)}
        if self.targets is not None:
            item["target"] = self.targets[i].astype(np.float32)
            item["weight"] = self.weights[i].astype(np.float32)
        return item


def collate(items: list[dict[str, Any]]) -> dict[str, Any]:
    B, K = len(items), len(items[0]["stacks"])
    S = max(1, max(len(st[1]) for it in items for st in it["stacks"]))
    Fdim = items[0]["stacks"][0][0].shape[-1]
    feats = np.zeros((B, K, S, Fdim), np.float32)
    pos = np.zeros((B, K, S), np.float32)
    valid = np.zeros((B, K, S), bool)
    grid = None
    g0 = items[0]["stacks"][0][2]
    if g0 is not None:
        grid = np.zeros((B, K, S) + g0.shape[1:], np.float32)
    for b, it in enumerate(items):
        for k, (xs, ps, gs) in enumerate(it["stacks"]):
            n = len(ps)
            if n:
                feats[b, k, :n] = xs
                pos[b, k, :n] = ps
                valid[b, k, :n] = True
                if grid is not None and gs is not None:
                    grid[b, k, :n] = gs
    out = {
        "study": [it["study"] for it in items],
        "feats": torch.from_numpy(feats), "pos": torch.from_numpy(pos), "valid": torch.from_numpy(valid),
        "plane": torch.from_numpy(np.stack([it["plane"] for it in items])),
        "seq": torch.from_numpy(np.stack([it["seq"] for it in items])),
    }
    if grid is not None:
        out["grid"] = torch.from_numpy(grid)
    if "target" in items[0]:
        out["target"] = torch.from_numpy(np.stack([it["target"] for it in items]))
        out["weight"] = torch.from_numpy(np.stack([it["weight"] for it in items]))
    return out
