"""Phase 5: K-fold training of the cross-slice attention head on cached frozen-backbone features."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from knee.config import dump_config, to_dict
from knee.constants import LABELS, STUDY
from knee.features.dataset import FeatureStore, StudyFeatureDataset, collate
from knee.features.extract import feature_tag, features_dir
from knee.labels.build_targets import load_targets
from knee.data.image_dataset import ImageStudyDataset, collate_images
from knee.models.cross_slice import CrossSliceModel, ImageKneeModel, feature_dim, token_dropout
from knee.train.folds import load_or_make_folds
from knee.train.logging import ETA, RunLogger
from knee.train.metrics import evaluate, worst_labels
from knee.utils import fmt_seconds, seed_everything


def build_model(cfg, in_dim: int, grid_dim: int = 0) -> CrossSliceModel:
    return CrossSliceModel(in_dim, len(LABELS), cfg.model, grid_dim=grid_dim)


def param_groups(model: torch.nn.Module, weight_decay: float, lr: float, backbone_lr_factor: float = 1.0
                 ) -> list[dict]:
    """AdamW groups: no decay for norms/biases/embeddings/queries; backbone params get lr * factor."""
    groups: dict[tuple[bool, bool], list] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        no_decay = p.ndim < 2 or "norm" in name or "emb" in name or name.endswith("queries")
        groups.setdefault((name.startswith("backbone."), no_decay), []).append(p)
    return [{"params": ps, "weight_decay": 0.0 if nd else weight_decay, "lr": lr * (backbone_lr_factor if bb else 1.0)}
            for (bb, nd), ps in groups.items()]


def cosine_with_warmup(warmup: int, total: int):
    def f(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        prog = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))
    return f


def weighted_bce(logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits.float(), target, reduction="none")
    return (loss * weight).sum() / weight.sum().clamp(min=1e-6)


def model_inputs(batch: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    keys = (["images"] if "images" in batch else ["feats"]) + ["pos", "valid", "plane", "seq"]
    keys += ["grid"] if "grid" in batch else []
    return {k: batch[k].to(device, non_blocking=True) for k in keys}


def build_e2e_model(cfg, pretrained: bool = True) -> ImageKneeModel:
    from knee.models.backbones import build_backbone

    backbone = build_backbone(cfg, pretrained=pretrained)
    if cfg.train.grad_checkpointing and hasattr(backbone.model, "set_grad_checkpointing"):
        backbone.model.set_grad_checkpointing(True)
    head = build_model(cfg, feature_dim(backbone.out_dim, cfg.model.feature_mode))
    return ImageKneeModel(backbone, head, cfg.model.feature_mode, cfg.backbone.channel_mode,
                          input_size=int(cfg.backbone.image_size) if cfg.backbone.image_size else None)


@torch.no_grad()
def predict(model, loader, device, amp: bool) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Returns (logits, probabilities, study ids)."""
    model.eval()
    logits_all, studies = [], []
    for batch in loader:
        inp = model_inputs(batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
            logits = model(**inp)
        logits_all.append(logits.float().cpu().numpy())
        studies += batch["study"]
    logits = np.concatenate(logits_all) if logits_all else np.zeros((0, len(LABELS)))
    return logits, 1.0 / (1.0 + np.exp(-logits)), studies


def label_stats_table(sources: pd.DataFrame, weights: pd.DataFrame, targets: pd.DataFrame) -> str:
    lines = [f"{'label':<18}{'gold':>6}{'pos':>7}{'neg':>7}{'unc':>7}{'n/m':>7}{'masked':>8}{'pos%':>7}"]
    for label in LABELS:
        s, w, t = sources[label], weights[label], targets[label]
        lines.append(f"{label:<18}{(s == 'gold').sum():>6}{(s == 'llm_positive').sum():>7}"
                     f"{(s == 'llm_negative').sum():>7}{(s == 'llm_uncertain').sum():>7}"
                     f"{(s == 'llm_not_mentioned').sum():>7}{(w <= 0).sum():>8}"
                     f"{100 * (t[w > 0] >= 0.5).mean():>7.1f}")
    return "\n".join(lines)


def train_fold(cfg, fold: int, train_ids: list[str], val_ids: list[str], store: FeatureStore | None,
               tgt: pd.DataFrame, wgt: pd.DataFrame, src: pd.DataFrame, rl: RunLogger, device) -> dict[str, Any]:
    """Frozen mode trains CrossSliceModel on `store`; end-to-end mode (store=None) trains
    ImageKneeModel on the uint8 image cache."""
    tc = cfg.train
    seed_everything(int(tc.seed) + fold)
    e2e = store is None
    nw = int(tc.num_workers)
    if e2e:
        cache = Path(cfg.paths.cache_dir) / "train"
        mirror_p = float(tc.mirror_aug)
        ds_tr = ImageStudyDataset(cache, train_ids, tgt.loc[train_ids].values, wgt.loc[train_ids].values,
                                  train=True, mirror_p=mirror_p, max_slices=int(tc.e2e_max_slices))
        ds_va = ImageStudyDataset(cache, val_ids, tgt.loc[val_ids].values, wgt.loc[val_ids].values)
        coll, val_bs = collate_images, int(tc.batch_size)
        model = build_e2e_model(cfg).to(device)
        in_dim, grid_dim = model.cross_slice.in_proj.in_features, 0
    else:
        mirror_p = float(tc.mirror_aug) if store.use_mirror else 0.0
        ds_tr = StudyFeatureDataset(store, train_ids, tgt.loc[train_ids].values, wgt.loc[train_ids].values,
                                    train=True, mirror_p=mirror_p)
        ds_va = StudyFeatureDataset(store, val_ids, tgt.loc[val_ids].values, wgt.loc[val_ids].values)
        coll, val_bs = collate, int(tc.batch_size) * 2
        model = build_model(cfg, store.in_dim, store.grid_dim).to(device)
        in_dim, grid_dim = store.in_dim, store.grid_dim
    dl_tr = torch.utils.data.DataLoader(ds_tr, batch_size=int(tc.batch_size), shuffle=True, drop_last=True,
                                        collate_fn=coll, num_workers=nw, persistent_workers=nw > 0)
    dl_va = torch.utils.data.DataLoader(ds_va, batch_size=val_bs, shuffle=False,
                                        collate_fn=coll, num_workers=nw, persistent_workers=nw > 0)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.AdamW(param_groups(model, float(tc.weight_decay), float(tc.lr),
                                         float(tc.backbone_lr_factor)), lr=float(tc.lr))
    steps_per_epoch = len(dl_tr)
    total = steps_per_epoch * int(tc.epochs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, cosine_with_warmup(int(float(tc.warmup_epochs) * steps_per_epoch), total))
    amp = bool(tc.amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    rl.info(f"[fold {fold}] train {len(train_ids)} / val {len(val_ids)} studies "
            f"(gold in val: {(src.loc[val_ids] == 'gold').all(axis=1).sum()}) | {n_params / 1e6:.2f}M params | "
            f"{steps_per_epoch} steps/epoch x {tc.epochs} epochs | mirror_aug {mirror_p}"
            f"{' | END-TO-END (backbone trainable)' if e2e else ''}")

    fold_dir = rl.run_dir / f"fold{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    best = {"score": -np.inf, "epoch": -1}
    va_src, va_tgt = src.loc[val_ids].values, tgt.loc[val_ids].values
    eta = ETA(int(tc.epochs))
    patience, global_step = 0, 0
    for epoch in range(int(tc.epochs)):
        model.train()
        t_ep, t_log = time.time(), time.time()
        loss_sum, n_seen, n_since = 0.0, 0, 0
        for step, batch in enumerate(dl_tr):
            inp = model_inputs(batch, device)
            if not e2e:  # e2e: random slice subsampling is the augmentation (valid must stay a prefix)
                inp["valid"] = token_dropout(inp["valid"], float(cfg.model.token_dropout))
            target, weight = batch["target"].to(device), batch["weight"].to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                logits = model(**inp)
            loss = weighted_bce(logits, target, weight)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(tc.grad_clip))
            scaler.step(opt)
            scaler.update()
            sched.step()
            global_step += 1
            bs = len(batch["study"])
            loss_sum += loss.item() * bs
            n_seen += bs
            n_since += bs
            if int(tc.log_every) > 0 and (step + 1) % int(tc.log_every) == 0:
                dt = time.time() - t_log
                mem = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0
                rl.info(f"  fold {fold} ep {epoch} step {step + 1}/{steps_per_epoch} | loss {loss.item():.4f} | "
                        f"lr {sched.get_last_lr()[0]:.2e} | gnorm {float(gnorm):.3f} | "
                        f"{n_since / max(dt, 1e-9):.0f} samples/s | gpu mem {mem:.2f} GB")
                t_log, n_since = time.time(), 0
        train_loss = loss_sum / max(n_seen, 1)

        logits, probs, _ = predict(model, dl_va, device, amp)
        val_loss = float(weighted_bce(torch.from_numpy(logits), torch.from_numpy(va_tgt.astype(np.float32)),
                                      torch.from_numpy(wgt.loc[val_ids].values.astype(np.float32))))
        ev = evaluate(probs, va_tgt, va_src)
        score = ev["macro_llm"]
        improved = np.isfinite(score) and score > best["score"]
        ep_time = time.time() - t_ep
        rl.label_table(
            f"[fold {fold}] epoch {epoch}: train loss {train_loss:.4f} | val loss {val_loss:.4f} | "
            f"macro AUC llm {score:.4f} | gold {ev['macro_gold']:.4f} | pred std {ev['mean_pred_std']:.4f} | "
            f"{fmt_seconds(ep_time)} | {eta.step()}{'  *best*' if improved else ''}",
            {"AUC llm": ev["auc_llm"], "AUC gold": ev["auc_gold"], "pred std": ev["pred_std"]},
            {"n llm": ev["n_llm"], "n gold": ev["n_gold"]},
        )
        if ev["mean_pred_std"] < 0.02:
            rl.warning(f"!!! COLLAPSE WARNING: mean per-label std of val predictions is {ev['mean_pred_std']:.4f} "
                       "(< 0.02): predictions are collapsing to the base rate")
        rl.metrics_row({"fold": fold, "epoch": epoch, "global_step": global_step, "train_loss": train_loss,
                        "val_loss": val_loss, "macro_auc_llm": score, "macro_auc_gold": ev["macro_gold"],
                        "mean_pred_std": ev["mean_pred_std"], "lr": sched.get_last_lr()[0], "epoch_time": ep_time,
                        **{f"auc_llm[{l}]": v for l, v in zip(LABELS, ev["auc_llm"])},
                        **{f"auc_gold[{l}]": v for l, v in zip(LABELS, ev["auc_gold"])}})
        if improved:
            best = {"score": score, "epoch": epoch, "macro_gold": ev["macro_gold"], "probs": probs,
                    "logits": logits}
            patience = 0
            torch.save({
                "model": model.state_dict(), "config": to_dict(cfg), "in_dim": in_dim,
                "grid_dim": grid_dim, "labels": LABELS, "fold": fold, "epoch": epoch, "e2e": e2e,
                "metrics": {"macro_auc_llm": score, "macro_auc_gold": ev["macro_gold"]},
                "feature_tag": feature_tag(cfg),
                "prior": (tgt.loc[train_ids].values * (wgt.loc[train_ids].values > 0)).sum(0)
                / np.maximum((wgt.loc[train_ids].values > 0).sum(0), 1),
            }, fold_dir / "best.pt")
        else:
            patience += 1
            if patience >= int(tc.early_stopping_patience):
                rl.info(f"[fold {fold}] early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    oof = pd.DataFrame(best["probs"], columns=LABELS)
    oof.insert(0, STUDY, val_ids)
    oof.insert(1, "fold", fold)
    oof.to_csv(fold_dir / "oof.csv", index=False)
    rl.info(f"[fold {fold}] best epoch {best['epoch']} | macro AUC llm {best['score']:.4f} | "
            f"gold {best.get('macro_gold', float('nan')):.4f} | checkpoint {fold_dir / 'best.pt'} | "
            f"OOF {fold_dir / 'oof.csv'}")
    return {"fold": fold, "best_epoch": best["epoch"], "score": best["score"], "oof": oof}


def run_training(cfg, args_limit: int | None = None) -> Path:
    tc = cfg.train
    name = tc.run_name or (f"{time.strftime('%Y%m%d-%H%M%S')}_{cfg.backbone.name}_ctx{cfg.model.slice_context}"
                           f"_{cfg.model.head}")
    rl = RunLogger(Path(cfg.paths.runs_dir) / name, tensorboard=bool(tc.tensorboard))
    dump_config(cfg, rl.run_dir / "config.yaml")
    rl.start(to_dict(cfg))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    labels_path = Path(cfg.paths.labels_dir) / "labels_final.csv"
    labels_final = pd.read_csv(labels_path)
    e2e = not bool(cfg.backbone.frozen)
    fdir = Path(cfg.paths.cache_dir) / "train" if e2e else features_dir(cfg, "train")
    have = {p.stem for p in fdir.glob("*.npz") if not p.stem.endswith(".tmp")}
    studies = [s for s in labels_final[STUDY].unique() if s in have]
    if args_limit:
        studies = studies[:args_limit]
    missing = labels_final[STUDY].nunique() - len(studies)
    rl.info(f"labels: {labels_path} | {'image cache (end-to-end)' if e2e else 'features'}: {fdir}"
            f"{'' if e2e else ' (' + feature_tag(cfg) + ')'} | {len(studies)} studies with inputs ({missing} without)")
    tgt, wgt, src = load_targets(labels_path, studies)
    informative = wgt.sum(axis=1) > 0  # no LLM output yet / llm_error everywhere -> nothing to learn
    if not informative.all():
        rl.info(f"skipping {int((~informative).sum())} studies without any weighted label")
        studies = [s for s in studies if informative[s]]
        tgt, wgt, src = tgt.loc[studies], wgt.loc[studies], src.loc[studies]
    folds = load_or_make_folds(cfg, labels_final, rl.log)
    folds = folds.set_index(STUDY).loc[studies].reset_index()
    rl.info("label statistics (training studies):\n" + label_stats_table(src, wgt, tgt))
    t0 = time.time()
    store = None
    if not e2e:
        store = FeatureStore(fdir, studies, feature_mode=cfg.model.feature_mode,
                             use_grid=bool(cfg.model.patch_attention),
                             use_mirror=bool(cfg.features.mirror) and float(tc.mirror_aug) > 0)
        rl.info(f"loaded features for {len(studies)} studies in {time.time() - t0:.1f}s "
                f"({store.nbytes() / 2**30:.2f} GB in RAM, in_dim {store.in_dim}, grid_dim {store.grid_dim}, "
                f"{store.n_stacks} stacks)")
    n_folds = int(tc.folds)
    for k in range(n_folds):
        n_tr = int((folds.fold != k).sum())
        rl.info(f"fold {k}: train {n_tr}, val {int((folds.fold == k).sum())} "
                f"(gold {int(((folds.fold == k) & folds.is_gold).sum())}), "
                f"~{n_tr // int(tc.batch_size)} steps/epoch")

    run_folds = list(tc.run_folds) if tc.run_folds is not None else list(range(n_folds))
    results = []
    for k in run_folds:
        tr = folds.loc[folds.fold != k, STUDY].tolist()
        va = folds.loc[folds.fold == k, STUDY].tolist()
        results.append(train_fold(cfg, k, tr, va, store, tgt, wgt, src, rl, device))

    oof = pd.concat([r["oof"] for r in results], ignore_index=True)
    oof.to_csv(rl.run_dir / "oof.csv", index=False)
    ids = oof[STUDY].tolist()
    ev = evaluate(oof[LABELS].values, tgt.loc[ids].values, src.loc[ids].values)
    rl.label_table(f"OOF over folds {run_folds}: macro AUC llm {ev['macro_llm']:.4f} | gold {ev['macro_gold']:.4f}",
                   {"AUC llm": ev["auc_llm"], "AUC gold": ev["auc_gold"]},
                   {"n llm": ev["n_llm"], "n gold": ev["n_gold"]})
    rl.info("fold scores: " + ", ".join(f"fold {r['fold']}: {r['score']:.4f} (ep {r['best_epoch']})" for r in results))
    rl.info("worst 3 labels (LLM OOF AUC): " + ", ".join(f"{l} {v:.3f}" for l, v in worst_labels(ev["auc_llm"])))
    rl.info("worst 3 labels (gold OOF AUC): " + ", ".join(f"{l} {v:.3f}" for l, v in worst_labels(ev["auc_gold"])))
    pd.DataFrame({"label": LABELS, "auc_llm": ev["auc_llm"], "auc_gold": ev["auc_gold"], "n_llm": ev["n_llm"],
                  "n_gold": ev["n_gold"]}).to_csv(rl.run_dir / "oof_auc.csv", index=False)
    rl.close()
    return rl.run_dir

