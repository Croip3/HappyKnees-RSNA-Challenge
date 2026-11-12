"""End-to-end (unfrozen backbone) mode on a synthetic uint8 cache with a tiny random-init ViT."""

import numpy as np
import pandas as pd
import pytest
import torch

from knee.config import load_config
from knee.constants import LABELS, STUDY
from knee.data.image_dataset import ImageStudyDataset, collate_images
from knee.data.preprocess import save_study
from knee.infer.predict import find_checkpoints, load_heads
from knee.models.backbones import REGISTRY, TimmViT, register
from knee.train.train import run_training
from knee.utils import get_logger

if "tiny_vit_test" not in REGISTRY:
    register("tiny_vit_test", timm_name="vit_tiny_patch16_224")(TimmViT)


def _study(rng, uid, n=(6, 5, 0), size=32):
    stacks = []
    for plane, k in zip(["sagittal", "coronal", "axial"], n):
        stacks.append({"plane": plane, "series_uid": f"{uid}.{plane}", "seq_type": 3, "present": k > 0,
                       "img": rng.integers(0, 255, (k, size, size), dtype=np.uint8),
                       "pos": np.linspace(0, 1, k, dtype=np.float32), "n_orig": k,
                       "spacing": np.ones(3, np.float32)})
    return {"study_uid": uid, "stacks": stacks, "laterality": "L", "laterality_source": "tag", "mirrored": False,
            "errors": [], "cfg_hash": "test"}


def test_collate_images_prefix_valid(tmp_path):
    rng = np.random.default_rng(0)
    save_study(tmp_path / "a.npz", _study(rng, "a", (6, 5, 0)))
    save_study(tmp_path / "b.npz", _study(rng, "b", (3, 7, 2)))
    ds = ImageStudyDataset(tmp_path, ["a", "b"], train=True, max_slices=4)
    batch = collate_images([ds[0], ds[1]])
    assert batch["images"].shape == (2, 3, 4, 32, 32) and batch["images"].dtype == torch.uint8
    v = batch["valid"]
    # valid slots form a prefix of every stack
    assert torch.equal(v, torch.cumprod(v.int(), dim=-1).bool())


@pytest.mark.parametrize("ctx", [3, "all"])
def test_e2e_train_and_reload(tmp_path, ctx):
    pytest.importorskip("timm")
    rng = np.random.default_rng(0)
    cfg = load_config(overrides=[
        f"paths.cache_dir={tmp_path / 'cache'}", f"paths.labels_dir={tmp_path / 'labels'}",
        f"paths.runs_dir={tmp_path / 'runs'}", "backbone.name=tiny_vit_test", "backbone.frozen=false",
        "backbone.image_size=32", "preprocess.image_size=32", "train.folds=2", "train.epochs=1",
        "train.batch_size=2", "train.num_workers=0", "train.e2e_max_slices=4", "train.log_every=0",
        "train.amp=false", "train.grad_checkpointing=true", "model.d_model=32", "model.n_heads=4",
        f"model.slice_context={ctx}", "train.run_name=e2e",
    ])
    studies = [f"s{i}" for i in range(8)]
    rows = []
    for s in studies:
        save_study(tmp_path / "cache" / "train" / f"{s}.npz", _study(rng, s, (int(rng.integers(2, 8)), 4, 3)))
        y = rng.integers(0, 2, len(LABELS))
        for j, label in enumerate(LABELS):
            rows.append({STUDY: s, "label": label, "target": float(y[j]), "weight": 1.0,
                         "source": "llm_positive" if y[j] else "llm_negative", "llm_state": "x", "postop": False})
    (tmp_path / "labels").mkdir()
    pd.DataFrame(rows).to_csv(tmp_path / "labels" / "labels_final.csv", index=False)
    # build_backbone(pretrained=True) would download; random init is enough for the code path
    import knee.models.backbones as bb

    orig = bb.build_backbone
    bb.build_backbone = lambda cfg, pretrained=True: orig(cfg, pretrained=False)
    try:
        run_dir = run_training(cfg)
        models, meta = load_heads(find_checkpoints(run_dir), torch.device("cpu"), get_logger("test"))
    finally:
        bb.build_backbone = orig
    assert meta["e2e"] and len(models) == 2
    ds = ImageStudyDataset(tmp_path / "cache" / "train", studies[:2])
    batch = collate_images([ds[0], ds[1]])
    with torch.no_grad():
        out = models[0](batch["images"], batch["pos"], batch["valid"], batch["plane"], batch["seq"])
    assert out.shape == (2, len(LABELS)) and torch.isfinite(out).all()
