"""End-to-end smoke test on synthetic features: targets -> folds -> train -> checkpoint -> heads inference."""

import json

import numpy as np
import pandas as pd
import pytest
import torch

from knee.config import load_config
from knee.constants import LABELS, STUDY
from knee.features.dataset import FeatureStore, StudyFeatureDataset, collate
from knee.features.extract import features_dir, save_features
from knee.infer.predict import find_checkpoints, load_heads
from knee.train.train import run_training
from knee.utils import get_logger

D = 16


def _fake_features(rng, n_slices=(12, 9, 0), signal=0.0, mirror=True):
    stacks, pos, cls, mean = [], [], [], []
    for k, n in enumerate(n_slices):
        stacks.append(np.full(n, k, np.int16))
        pos.append(np.linspace(0, 1, n, dtype=np.float32) if n else np.zeros(0, np.float32))
        c = rng.normal(size=(n, D)) + signal
        cls.append(c.astype(np.float16))
        mean.append((c * 0.5).astype(np.float16))
    f = {"cls": np.concatenate(cls), "mean": np.concatenate(mean), "stack": np.concatenate(stacks),
         "pos": np.concatenate(pos), "plane": np.array([0, 1, 2], np.int8), "seq_type": np.array([3, 3, 0], np.int8),
         "present": np.array([n > 0 for n in n_slices]), "n_orig": np.array(n_slices, np.int16),
         "grid": np.zeros((sum(n_slices), 4, D), np.float16), "meta": np.asarray(json.dumps({"study_uid": "x"}))}
    if mirror:
        f["cls_m"], f["mean_m"], f["grid_m"] = f["cls"], f["mean"], f["grid"]
    return f


def test_collate_and_store_from_memory():
    rng = np.random.default_rng(0)
    feats = {"a": _fake_features(rng, (5, 3, 0)), "b": _fake_features(rng, (2, 7, 4))}
    store = FeatureStore.from_memory(feats, "cls_mean", use_mirror=True)
    assert store.in_dim == 2 * D and store.n_stacks == 3
    ds = StudyFeatureDataset(store, ["a", "b"], force_mirror=True)
    batch = collate([ds[0], ds[1]])
    assert batch["feats"].shape == (2, 3, 7, 2 * D)
    assert batch["valid"].sum().item() == 5 + 3 + 2 + 7 + 4
    # sagittal mirror reverses slice order
    assert torch.allclose(batch["pos"][0, 0, :5], torch.linspace(0, 1, 5))


@pytest.mark.parametrize("ctx", [3, "all"])
def test_train_and_infer_synthetic(tmp_path, ctx):
    rng = np.random.default_rng(0)
    cfg = load_config(overrides=[
        f"paths.features_dir={tmp_path / 'features'}", f"paths.labels_dir={tmp_path / 'labels'}",
        f"paths.runs_dir={tmp_path / 'runs'}", "train.folds=2", "train.epochs=2", "train.batch_size=8",
        "train.num_workers=0", "train.log_every=2", "model.d_model=32", "model.n_heads=4",
        f"model.slice_context={ctx}", "train.run_name=smoke", "train.mirror_aug=0.5", "features.mirror=true",
    ])
    fdir = features_dir(cfg, "train")
    studies = [f"s{i:03d}" for i in range(40)]
    rows = []
    for i, s in enumerate(studies):
        y = rng.integers(0, 2, len(LABELS))
        save_features(fdir / f"{s}.npz", _fake_features(rng, (rng.integers(3, 15), rng.integers(3, 15), 0),
                                                        signal=float(y[0])))
        gold = i < 6
        for j, label in enumerate(LABELS):
            src = "gold" if gold else ("llm_positive" if y[j] else "llm_negative")
            rows.append({STUDY: s, "label": label, "target": float(y[j]), "weight": 4.0 if gold else 1.0,
                         "source": src, "llm_state": "positive" if y[j] else "negative", "postop": False})
    (tmp_path / "labels").mkdir()
    pd.DataFrame(rows).to_csv(tmp_path / "labels" / "labels_final.csv", index=False)

    run_dir = run_training(cfg)
    assert (run_dir / "oof.csv").exists() and (run_dir / "metrics.csv").exists()
    oof = pd.read_csv(run_dir / "oof.csv")
    assert len(oof) == 40 and set(oof.fold) == {0, 1}
    metrics = pd.read_csv(run_dir / "metrics.csv")
    assert len(metrics) >= 2 and np.isfinite(metrics.train_loss).all()

    ckpts = find_checkpoints(run_dir)
    assert len(ckpts) == 2
    models, meta = load_heads(ckpts, torch.device("cpu"), get_logger("test"))
    assert meta["prior"].shape == (len(LABELS),)
    store = FeatureStore(fdir, studies[:4], use_mirror=True)
    batch = collate([StudyFeatureDataset(store, studies[:4])[i] for i in range(4)])
    with torch.no_grad():
        out = models[0](batch["feats"], batch["pos"], batch["valid"], batch["plane"], batch["seq"])
    assert out.shape == (4, len(LABELS)) and torch.isfinite(out).all()
