"""Phase 3: embed every cached slice once with the frozen backbone (fp16 .npz per study + manifest).

Per slice: CLS token, mean patch token, optional pooled patch grid (g x g tokens). With
features.mirror the width-flipped coronal/axial slices are embedded too (sagittal mirror = slice
order reversal, which needs no new features).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from knee.constants import PLANE_ID, STUDY
from knee.data.preprocess import load_study
from knee.models.backbones import build_backbone, to_input
from knee.utils import identity


def feature_tag(cfg) -> str:
    """Identifies a feature set: backbone, input size, channel mode, grid and preprocessing hash."""
    from knee.config import to_dict
    from knee.data.preprocess import preprocess_hash

    b = cfg.backbone
    size = int(b.image_size or cfg.preprocess.image_size)
    return (f"{b.name}_{size}_{b.channel_mode}_g{int(cfg.features.patch_grid)}"
            f"_p{preprocess_hash(to_dict(cfg.preprocess))[:6]}")


def features_dir(cfg, split: str) -> Path:
    return Path(cfg.paths.features_dir) / feature_tag(cfg) / split


@torch.no_grad()
def embed_stack(backbone, img_u8: np.ndarray, channel_mode: str, device: torch.device, batch_size: int,
                amp: bool, grid: int, size: int | None = None) -> dict[str, np.ndarray]:
    """(S, H, W) uint8 -> cls (S, D), mean (S, D), grid (S, g*g, D) as float16 numpy arrays.

    If `size` differs from the cached image size, slices are resized (antialiased) on the device.
    """
    x_all = torch.from_numpy(np.ascontiguousarray(img_u8)).to(device, non_blocking=True)
    if size is not None and x_all.shape[-1] != size:
        x_all = F.interpolate(x_all[:, None].float(), size=(size, size), mode="bilinear", antialias=True,
                              align_corners=False)[:, 0].round().clamp(0, 255).to(torch.uint8)
    x_all = to_input(x_all, channel_mode)  # neighbors need the whole stack, so convert before batching
    cls_l, mean_l, grid_l = [], [], []
    for i in range(0, x_all.shape[0], batch_size):
        x = x_all[i:i + batch_size]
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
            out = backbone(x)
        cls_l.append(out["cls"].float())
        patch = out["patch"]
        if patch is not None:
            patch = patch.float()
            mean_l.append(patch.mean(1))
            if grid > 0 and backbone.patch_grid is not None:
                h, w = backbone.patch_grid
                fmap = patch.transpose(1, 2).reshape(patch.shape[0], patch.shape[2], h, w)
                grid_l.append(F.adaptive_avg_pool2d(fmap, grid).flatten(2).transpose(1, 2))
        else:
            mean_l.append(out["cls"].float())
    res = {"cls": torch.cat(cls_l).half().cpu().numpy(), "mean": torch.cat(mean_l).half().cpu().numpy()}
    if grid_l:
        res["grid"] = torch.cat(grid_l).half().cpu().numpy()
    return res


def embed_study(backbone, study: dict[str, Any], cfg, device: torch.device) -> dict[str, np.ndarray]:
    fc = cfg.features
    grid = int(fc.patch_grid)
    D = backbone.out_dim
    size = int(cfg.backbone.image_size or cfg.preprocess.image_size)
    keys = ["cls", "mean"] + (["grid"] if grid > 0 else [])
    parts: dict[str, list[np.ndarray]] = {k: [] for k in keys}
    mparts: dict[str, list[np.ndarray]] = {k: [] for k in keys}
    stack_idx, pos = [], []
    for i, st in enumerate(study["stacks"]):
        img = st["img"]
        if not st["present"] or len(img) == 0:
            continue
        emb = embed_stack(backbone, img, cfg.backbone.channel_mode, device, int(fc.batch_size), bool(fc.amp), grid,
                          size)
        for k in keys:
            parts[k].append(emb[k])
        if fc.mirror:
            if st["plane"] == "sagittal":
                memb = emb
            else:
                memb = embed_stack(backbone, np.ascontiguousarray(img[..., ::-1]), cfg.backbone.channel_mode,
                                   device, int(fc.batch_size), bool(fc.amp), grid, size)
            for k in keys:
                mparts[k].append(memb[k])
        stack_idx.append(np.full(len(img), i, np.int16))
        pos.append(st["pos"].astype(np.float32))
    out: dict[str, np.ndarray] = {}
    for k in keys:
        shape = (0, D) if k != "grid" else (0, grid * grid, D)
        out[k] = np.concatenate(parts[k]) if parts[k] else np.zeros(shape, np.float16)
        if fc.mirror:
            out[f"{k}_m"] = np.concatenate(mparts[k]) if mparts[k] else np.zeros(shape, np.float16)
    out["stack"] = np.concatenate(stack_idx) if stack_idx else np.zeros(0, np.int16)
    out["pos"] = np.concatenate(pos) if pos else np.zeros(0, np.float32)
    st = study["stacks"]
    out["plane"] = np.asarray([PLANE_ID[s["plane"]] for s in st], np.int8)
    out["seq_type"] = np.asarray([s["seq_type"] for s in st], np.int8)
    out["present"] = np.asarray([s["present"] and len(s["img"]) > 0 for s in st], bool)
    out["n_orig"] = np.asarray([s["n_orig"] for s in st], np.int16)
    out["meta"] = np.asarray(json.dumps({"study_uid": study["study_uid"], "laterality": study.get("laterality", ""),
                                         "mirrored": bool(study.get("mirrored", False)),
                                         "feature_tag": feature_tag(cfg)}))
    return out


def save_features(path: str | os.PathLike, feats: dict[str, np.ndarray]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, **feats)
    os.replace(tmp, path)


def load_features(path: str | os.PathLike, keys: list[str] | None = None) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as z:
        names = keys or list(z.files)
        out = {k: z[k] for k in names if k in z.files}
        if "meta" in z.files:
            out["meta"] = json.loads(str(z["meta"]))
    return out


class _StudyFiles(torch.utils.data.Dataset):
    def __init__(self, paths: list[Path]):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            return self.paths[i].stem, load_study(self.paths[i]), None
        except Exception as e:  # noqa: BLE001
            return self.paths[i].stem, None, f"{type(e).__name__}: {e}"


def estimate_disk_gb(n_studies: int, slices_per_study: float, D: int, grid: int, mirror: bool) -> float:
    per_slice = 2 * (2 * D + grid * grid * D)  # fp16 bytes
    return n_studies * slices_per_study * per_slice * (2 if mirror else 1) / 1e9


def run_extract(cfg, split: str, log, limit: int | None = None, overwrite: bool = False,
                studies: list[str] | None = None) -> pd.DataFrame:
    from tqdm.auto import tqdm

    cache = Path(cfg.paths.cache_dir) / split
    out_dir = features_dir(cfg, split)
    out_dir.mkdir(parents=True, exist_ok=True)
    if studies is None:
        studies = sorted(p.stem for p in cache.glob("*.npz") if not p.stem.endswith(".tmp"))
    if limit is not None:
        studies = studies[:limit]
    todo = [s for s in studies if (cache / f"{s}.npz").exists() and (overwrite or not (out_dir / f"{s}.npz").exists())]
    missing_cache = [s for s in studies if not (cache / f"{s}.npz").exists()]
    if missing_cache:
        log.warning(f"{len(missing_cache)} studies have no image cache yet (run preprocess first)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = build_backbone(cfg).to(device)
    D, grid = backbone.out_dim, int(cfg.features.patch_grid)
    manifest_path = cache / "manifest.csv"
    sps = 3 * min(32.0, cfg.preprocess.max_slices_per_series)
    if manifest_path.exists():
        man = pd.read_csv(manifest_path)
        cols = [c for c in man.columns if c.endswith("_slices")]
        if cols and len(man):
            sps = float(man[cols].sum(axis=1).mean())
    est = estimate_disk_gb(len(studies), sps, D, grid, bool(cfg.features.mirror))
    log.info(f"features {feature_tag(cfg)} on {device}: {len(studies)} studies ({len(todo)} to embed), "
             f"~{sps:.0f} slices/study, D={D}, grid={grid}x{grid} -> estimated disk {est:.2f} GB in {out_dir}")
    if not todo:
        return pd.DataFrame()
    ds = _StudyFiles([cache / f"{s}.npz" for s in todo])
    nw = int(cfg.features.num_workers)
    loader = torch.utils.data.DataLoader(ds, batch_size=None, num_workers=nw, collate_fn=identity,
                                         prefetch_factor=4 if nw else None)
    rows, t0, n_slices = [], time.time(), 0
    for study_uid, study, err in tqdm(loader, total=len(ds), desc=f"embed {split}", mininterval=5):
        t1 = time.time()
        if study is None:
            rows.append({STUDY: study_uid, "status": "failed", "error": err})
            continue
        try:
            feats = embed_study(backbone, study, cfg, device)
            save_features(out_dir / f"{study_uid}.npz", feats)
            n_slices += len(feats["pos"])
            rows.append({STUDY: study_uid, "status": "ok", "n_slices": int(len(feats["pos"])),
                         "seconds": round(time.time() - t1, 3)})
        except Exception as e:  # noqa: BLE001
            rows.append({STUDY: study_uid, "status": "failed", "error": f"{type(e).__name__}: {e}"})
    el = time.time() - t0
    new = pd.DataFrame(rows)
    mpath = out_dir / "manifest.csv"
    if mpath.exists():
        old = pd.read_csv(mpath)
        new = pd.concat([old[~old[STUDY].isin(new[STUDY])], new], ignore_index=True)
    new.to_csv(mpath, index=False)
    n_fail = int((new.status != "ok").sum())
    log.info(f"embedded {len(rows)} studies / {n_slices} slices in {el:.0f}s "
             f"({n_slices / max(el, 1e-9):.0f} slices/s); failures: {n_fail}")
    return new


class _DicomStudies(torch.utils.data.Dataset):
    """Preprocess studies straight from DICOM inside DataLoader workers."""

    def __init__(self, studies: list[str], series: pd.DataFrame, image_root: str, pcfg: dict[str, Any]):
        from knee.data.preprocess import preprocess_study

        self._pre = preprocess_study
        self.studies, self.image_root, self.pcfg = studies, image_root, pcfg
        self.by_study = {s: g for s, g in series.groupby(STUDY)}
        self.empty = series.iloc[:0]

    def __len__(self):
        return len(self.studies)

    def __getitem__(self, i):
        s = self.studies[i]
        t0 = time.time()
        try:
            return s, self._pre(s, self.by_study.get(s, self.empty), self.image_root, self.pcfg), None, time.time() - t0
        except Exception as e:  # noqa: BLE001
            return s, None, f"{type(e).__name__}: {e}", time.time() - t0


def run_prep_embed(cfg, split: str, log, limit: int | None = None, overwrite: bool = False,
                   studies: list[str] | None = None, save_cache: bool = False,
                   variants: list[list[str]] | None = None) -> pd.DataFrame:
    """Streaming DICOM -> preprocess (CPU workers) -> embed (GPU) -> feature files; optionally also
    writes the uint8 .npz cache. `variants` are lists of config overrides (e.g. other backbones or
    input sizes) that are all embedded from the same preprocessed study. Resumable per variant."""
    import copy

    from tqdm.auto import tqdm

    from knee.config import parse_value, set_dotted, to_dict
    from knee.data.preprocess import preprocess_hash, save_study

    data_root = Path(cfg.paths.data_root)
    image_root = cfg.paths.train_images if split == "train" else cfg.paths.test_images
    series = pd.read_csv(data_root / f"{split}_series.csv")
    if studies is None:
        studies = pd.read_csv(data_root / f"{split}.csv")[STUDY].tolist()
    if limit is not None:
        studies = studies[:limit]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vcfgs = []
    for overrides in (variants or [[]]):
        vc = copy.deepcopy(cfg)
        for item in overrides:
            key, text = item.split("=", 1)
            set_dotted(vc, key.strip(), parse_value(text))
        if to_dict(vc.preprocess) != to_dict(cfg.preprocess):
            raise ValueError("variants may not change preprocess settings")
        vcfgs.append(vc)
    outs = [features_dir(vc, split) for vc in vcfgs]
    for o in outs:
        o.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(cfg.paths.cache_dir) / split
    todo = [s for s in studies if overwrite or not all((o / f"{s}.npz").exists() for o in outs)]
    pcfg = to_dict(cfg.preprocess)
    backbones = []
    for vc in vcfgs:
        bb = build_backbone(vc).to(device)
        backbones.append(bb)
        est = estimate_disk_gb(len(studies), 3 * 32, bb.out_dim, int(vc.features.patch_grid), bool(vc.features.mirror))
        log.info(f"variant {feature_tag(vc)}: est. feature disk {est:.2f} GB -> {features_dir(vc, split)}")
    log.info(f"prep+embed {split}: {len(studies)} studies ({len(todo)} to do) | preprocess hash {preprocess_hash(pcfg)} "
             f"| {len(vcfgs)} feature variant(s) on {device}"
             f"{' | saving uint8 cache to ' + str(cache_dir) if save_cache else ''}")
    if not todo:
        return pd.DataFrame()
    nw = int(cfg.preprocess.num_workers)
    loader = torch.utils.data.DataLoader(_DicomStudies(todo, series, image_root, pcfg), batch_size=None,
                                         num_workers=nw, collate_fn=identity, prefetch_factor=4 if nw else None)
    rows, t0, t_prep, t_embed, n_slices = [], time.time(), 0.0, 0.0, 0
    for i, (s, res, err, dt) in enumerate(tqdm(loader, total=len(todo), desc=f"prep+embed {split}", mininterval=30)):
        t_prep += dt
        row = {STUDY: s, "prep_seconds": round(dt, 2)}
        if res is None:
            row.update(status="failed", error=err)
        else:
            try:
                if save_cache:
                    save_study(cache_dir / f"{s}.npz", res, bool(pcfg.get("compress", True)))
                t1 = time.time()
                for vc, bb, o in zip(vcfgs, backbones, outs):
                    if overwrite or not (o / f"{s}.npz").exists():
                        feats = embed_study(bb, res, vc, device)
                        save_features(o / f"{s}.npz", feats)
                t_embed += time.time() - t1
                n = int(sum(len(st["img"]) for st in res["stacks"]))
                n_slices += n
                row.update(status="ok", n_slices=n, laterality=res["laterality"],
                           laterality_source=res["laterality_source"], mirrored=res["mirrored"],
                           errors=" | ".join(res["errors"]),
                           **{f"{st['plane']}_slices": int(len(st["img"])) for st in res["stacks"]})
            except Exception as e:  # noqa: BLE001
                row.update(status="failed", error=f"{type(e).__name__}: {e}")
        rows.append(row)
        if i + 1 == min(20, len(todo)) and all(r["status"] != "ok" for r in rows):
            raise RuntimeError(f"the first {i + 1} studies all failed, aborting: {rows[-1].get('error')}")
        if (i + 1) % 100 == 0:
            el = time.time() - t0
            log.info(f"[{i + 1}/{len(todo)}] {el:.0f}s | {(i + 1) / el:.2f} studies/s | ETA "
                     f"{(len(todo) - i - 1) / ((i + 1) / el) / 60:.1f} min | worker prep {t_prep / (i + 1):.2f}s/study,"
                     f" embed {t_embed / (i + 1):.3f}s/study")
    el = time.time() - t0
    new = pd.DataFrame(rows)
    for o in outs:
        mpath = o / "manifest.csv"
        man = new
        if mpath.exists():
            old = pd.read_csv(mpath)
            man = pd.concat([old[~old[STUDY].isin(new[STUDY])], new], ignore_index=True)
        man.to_csv(mpath, index=False)
        failed = man[man.status != "ok"]
        if len(failed):
            failed.to_csv(o / "failures.csv", index=False)
    ok = new[new.status == "ok"]
    log.info(f"done {len(rows)} studies in {el:.0f}s ({len(rows) / max(el, 1e-9):.2f}/s), {n_slices} slices, "
             f"{int((new.status != 'ok').sum())} failed | worker prep {t_prep / max(len(rows), 1):.2f}s/study, "
             f"embed {t_embed / max(len(rows), 1):.3f}s/study | laterality sources "
             f"{ok['laterality_source'].value_counts().to_dict() if 'laterality_source' in ok else {}}")
    return new
