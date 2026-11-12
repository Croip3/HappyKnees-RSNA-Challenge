"""Phase 6: test DICOMs -> preprocessing -> frozen-backbone embeddings -> fold heads -> submission.csv.

Runs offline. DICOM decoding/preprocessing happens in DataLoader workers (CPU) while the GPU
embeds the previous study. Prints a runtime breakdown per stage (efficiency track).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from knee.config import load_config, to_dict
from knee.constants import LABELS, STUDY
from knee.data.image_dataset import collate_images, stacks_to_item
from knee.data.preprocess import preprocess_hash, preprocess_study
from knee.features.dataset import FeatureStore, StudyFeatureDataset, collate
from knee.features.extract import embed_study, feature_tag
from knee.models.backbones import build_backbone
from knee.models.cross_slice import CrossSliceModel
from knee.utils import fmt_seconds, identity


def find_checkpoints(spec) -> list[Path]:
    if spec is None:
        raise ValueError("set infer.checkpoints to a run directory or a list of checkpoint files")
    specs = spec if isinstance(spec, (list, tuple)) else [spec]
    paths: list[Path] = []
    for s in specs:
        p = Path(s)
        if p.is_dir():
            paths += sorted(p.glob("fold*/best.pt")) or sorted(p.glob("**/best.pt"))
        elif p.exists():
            paths.append(p)
        else:
            raise FileNotFoundError(f"checkpoint not found: {p}")
    if not paths:
        raise FileNotFoundError(f"no checkpoints in {specs}")
    return paths


def load_heads(paths: list[Path], device: torch.device, log) -> tuple[list[torch.nn.Module], dict[str, Any]]:
    models, metas = [], []
    for p in paths:
        ck = torch.load(p, map_location="cpu", weights_only=False)
        ccfg = load_config(overrides=[])
        # rebuild the model with the training-time model (and, for end-to-end runs, backbone) config
        for section in ["model", "backbone", "train"]:
            for k, v in ck["config"][section].items():
                ccfg[section][k] = v
        if ck.get("e2e"):
            from knee.train.train import build_e2e_model

            ccfg.backbone["weights_path"] = None
            m = build_e2e_model(ccfg, pretrained=False)
        else:
            m = CrossSliceModel(ck["in_dim"], len(LABELS), ccfg.model, grid_dim=ck.get("grid_dim", 0))
        m.load_state_dict(ck["model"])
        models.append(m.to(device).eval())
        metas.append(ck)
        log.info(f"loaded {p} (fold {ck.get('fold')}, epoch {ck.get('epoch')}, metrics {ck.get('metrics')})")
    ref = metas[0]
    return models, {"config": ref["config"], "feature_tag": ref.get("feature_tag"), "e2e": bool(ref.get("e2e")),
                    "prior": np.mean([np.asarray(m["prior"], float) for m in metas], axis=0),
                    "use_grid": bool(ref["config"]["model"].get("patch_attention")) and ref.get("grid_dim", 0) > 0}


class _TestStudies(torch.utils.data.Dataset):
    def __init__(self, studies, series, image_root, pcfg):
        self.studies, self.image_root, self.pcfg = studies, image_root, pcfg
        self.by_study = {s: g for s, g in series.groupby(STUDY)}
        self.empty = series.iloc[:0]

    def __len__(self):
        return len(self.studies)

    def __getitem__(self, i):
        s = self.studies[i]
        t0 = time.time()
        try:
            res = preprocess_study(s, self.by_study.get(s, self.empty), self.image_root, self.pcfg)
            err = None if any(st["present"] for st in res["stacks"]) else "no usable series"
        except Exception as e:  # noqa: BLE001
            res, err = None, f"{type(e).__name__}: {e}"
        return s, res, err, time.time() - t0


@torch.no_grad()
def run_inference(cfg, log, dry_run: int = 0) -> pd.DataFrame:
    t_start = time.time()
    times = {"preprocess (worker sum)": 0.0, "wait for data": 0.0, "embed": 0.0, "heads": 0.0}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = Path(cfg.paths.data_root)
    test = pd.read_csv(data_root / "test.csv")
    series = pd.read_csv(data_root / "test_series.csv")
    studies = test[STUDY].tolist()
    if dry_run:
        studies = studies[:dry_run]
    log.info(f"inference on {len(studies)} studies | device {device} | data {data_root}")

    t0 = time.time()
    models, meta = load_heads(find_checkpoints(cfg.infer.checkpoints), device, log)
    tcfg = meta["config"]
    # preprocessing / backbone / features must match training
    for section in ["preprocess", "backbone", "features"]:
        for k, v in tcfg[section].items():
            if section == "backbone" and k == "weights_path":
                continue
            if section == "preprocess" and k == "num_workers":
                continue
            cfg[section][k] = v
    feature_mode = tcfg["model"]["feature_mode"]
    tta = bool(cfg.infer.tta_mirror)
    cfg.features["mirror"] = tta
    if meta["feature_tag"] and meta["feature_tag"] != feature_tag(cfg):
        log.warning(f"feature tag mismatch: checkpoint {meta['feature_tag']} vs inference {feature_tag(cfg)}")
    e2e = meta["e2e"]
    backbone = None if e2e else build_backbone(cfg).to(device).eval()
    times["load models"] = time.time() - t0
    pcfg = to_dict(cfg.preprocess)
    log.info(f"{len(models)} heads | backbone {cfg.backbone.name} | preprocess hash {preprocess_hash(pcfg)} | "
             f"TTA mirror {tta}")

    ds = _TestStudies(studies, series, cfg.paths.test_images, pcfg)
    nw = int(cfg.preprocess.num_workers)
    loader = torch.utils.data.DataLoader(ds, batch_size=None, num_workers=nw, collate_fn=identity,
                                         prefetch_factor=2 if nw else None)
    feats: dict[str, dict[str, np.ndarray]] = {}
    e2e_probs: dict[str, np.ndarray] = {}
    failed: dict[str, str] = {}
    t_wait = time.time()
    for i, (s, res, err, dt) in enumerate(loader):
        times["wait for data"] += time.time() - t_wait
        times["preprocess (worker sum)"] += dt
        if res is None or err:
            failed[s] = err
            log.warning(f"study {s} failed: {err}")
        elif e2e:  # end-to-end checkpoints: every fold runs its own fine-tuned backbone on the stacks
            t1 = time.time()
            try:
                views = [False, True] if tta else [False]
                p = 0.0
                for mirrored in views:
                    batch = collate_images([stacks_to_item(res, mirror=mirrored)])
                    inp = {k: batch[k].to(device) for k in ["images", "pos", "valid", "plane", "seq"]}
                    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                        p = p + np.mean([torch.sigmoid(m(**inp).float()).cpu().numpy()[0] for m in models], axis=0)
                e2e_probs[s] = p / len(views)
            except Exception as e:  # noqa: BLE001
                failed[s] = f"e2e: {type(e).__name__}: {e}"
                log.warning(f"study {s} failed in the end-to-end model: {e}")
            times["embed"] += time.time() - t1
        else:
            t1 = time.time()
            try:
                feats[s] = embed_study(backbone, res, cfg, device)
            except Exception as e:  # noqa: BLE001
                failed[s] = f"embed: {type(e).__name__}: {e}"
                log.warning(f"study {s} failed in embedding: {e}")
            if device.type == "cuda":
                torch.cuda.synchronize()
            times["embed"] += time.time() - t1
        if (i + 1) % 50 == 0 or i + 1 == len(ds):
            el = time.time() - t_start
            log.info(f"[{i + 1}/{len(ds)}] {el:.0f}s elapsed, ~{fmt_seconds(el / (i + 1) * (len(ds) - i - 1))} left")
        t_wait = time.time()

    probs = np.tile(meta["prior"], (len(studies), 1)).astype(np.float64)
    pos_of = {s: i for i, s in enumerate(studies)}
    for s, p in e2e_probs.items():
        probs[pos_of[s]] = p
    ok = [s for s in studies if s in feats]
    if ok:
        t1 = time.time()
        store = FeatureStore.from_memory({s: feats[s] for s in ok}, feature_mode, use_grid=meta["use_grid"],
                                         use_mirror=tta)
        views = [False, True] if tta else [False]
        acc = np.zeros((len(ok), len(LABELS)))
        row_of = {s: i for i, s in enumerate(ok)}
        for mirrored in views:
            dl = torch.utils.data.DataLoader(StudyFeatureDataset(store, ok, force_mirror=mirrored),
                                             batch_size=int(cfg.infer.batch_size), collate_fn=collate)
            for batch in dl:
                idx = [row_of[s] for s in batch["study"]]
                inp = {k: batch[k].to(device) for k in ["feats", "pos", "valid", "plane", "seq"]}
                if "grid" in batch:
                    inp["grid"] = batch["grid"].to(device)
                p = np.mean([torch.sigmoid(m(**inp).float()).cpu().numpy() for m in models], axis=0)
                acc[idx] += p / len(views)
        probs[[pos_of[s] for s in ok]] = acc
        times["heads"] = time.time() - t1
    if failed:
        log.warning(f"{len(failed)} studies fell back to the training prior: {list(failed)[:5]}")
    sub = pd.DataFrame(probs, columns=LABELS)
    sub.insert(0, STUDY, studies)
    total = time.time() - t_start
    times["total"] = total
    log.info("runtime breakdown: " + " | ".join(f"{k} {v:.1f}s" for k, v in times.items()))
    log.info(f"per study: {total / max(len(studies), 1):.2f}s | efficiency term RuntimeSeconds/32400 = "
             f"{total / 32400:.4f}")
    return sub


def write_submission(sub: pd.DataFrame, path: str | Path, sample_submission: str | Path | None = None, log=None):
    sub = sub[[STUDY] + LABELS]
    if sample_submission is not None and Path(sample_submission).exists():
        ss = pd.read_csv(sample_submission)
        assert list(ss.columns) == list(sub.columns), (list(ss.columns), list(sub.columns))
    assert sub[LABELS].notna().all().all(), "NaN in submission"
    sub.to_csv(path, index=False)
    if log:
        log.info(f"wrote {path} ({len(sub)} rows, columns {list(sub.columns)})")
    return sub
