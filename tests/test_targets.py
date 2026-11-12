import json

import numpy as np
import pandas as pd
import pytest

from knee.config import load_config
from knee.constants import LABELS, REPORT, STUDY
from knee.labels.build_targets import build_targets, load_targets
from knee.labels.validate import cohen_kappa, run_validation
from knee.utils import get_logger


def _setup(tmp_path, kill_kappa=0.4, nm_target="zero"):
    rng = np.random.default_rng(0)
    data = tmp_path / "raw"
    data.mkdir()
    studies = [f"s{i}" for i in range(30)]
    train = pd.DataFrame({STUDY: studies, REPORT: [f"report {i}" for i in range(30)]})
    for label in LABELS:
        train[label] = np.nan
    gold = rng.integers(0, 2, (12, len(LABELS)))
    train.loc[:11, LABELS] = gold
    train.to_csv(data / "train.csv", index=False)
    recs = []
    for i, s in enumerate(studies):
        if i == 29:
            recs.append({STUDY: s, "status": "llm_error", "error": "bad"})
            continue
        labels = {}
        for j, label in enumerate(LABELS):
            if i < 12 and label != "Fracture":
                state = "positive" if gold[i, j] else ("negative" if j % 2 else "not_mentioned")
            else:
                state = ["positive", "negative", "uncertain", "not_mentioned"][(i + j) % 4]
            labels[label] = {"state": state, "evidence": f"ev {label}"}
        recs.append({STUDY: s, "status": "ok", "labels": labels, "postop_labels": ["ACL"] if i == 20 else []})
    lab = tmp_path / "labels"
    lab.mkdir()
    with open(lab / "llm_labels.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    cfg = load_config(overrides=[f"paths.data_root={data}", f"paths.labels_dir={lab}",
                                 f"paths.results_dir={tmp_path / 'results'}", f"targets.min_kappa={kill_kappa}",
                                 f"targets.not_mentioned_target={nm_target}", "targets.gold_rate_min_count=3"])
    return cfg, studies


def test_kappa():
    assert cohen_kappa(np.array([1, 0, 1, 0]), np.array([1, 0, 1, 0])) == pytest.approx(1.0)
    assert cohen_kappa(np.array([1, 0, 1, 0]), np.array([0, 1, 0, 1])) == pytest.approx(-1.0)


def test_validation_and_targets(tmp_path):
    cfg, studies = _setup(tmp_path)
    log = get_logger("test")
    table = run_validation(cfg, log)
    assert table is not None
    acl = table.set_index("label").loc["ACL"]
    assert acl["precision_pos"] == pytest.approx(1.0) and acl["recall_pos"] == pytest.approx(1.0)
    assert (tmp_path / "results" / "labels" / "disagreements.csv").exists()

    out = build_targets(cfg, log)
    assert len(out) == 30 * len(LABELS)
    g = out[(out[STUDY] == "s0")]
    assert (g.source == "gold").all() and (g.weight == cfg.targets.gold_weight).all()
    err = out[out[STUDY] == "s29"]
    assert (err.source == "llm_error").all() and (err.weight == 0).all()
    s = out[(out[STUDY] == "s13")].set_index("label")
    for label, row in s.iterrows():
        st = row.llm_state
        if label == "Fracture":  # Fracture LLM states disagree with gold -> kill switch (downweight)
            continue
        exp = {"positive": (1.0, 1.0), "negative": (0.0, 1.0), "uncertain": (0.6, 0.5), "not_mentioned": (0.0, 0.3)}[st]
        assert (row.target, row.weight) == pytest.approx(exp), (label, st)
    fr = out[(out.label == "Fracture") & out.source.str.startswith("llm_") & (out.source != "llm_error")]
    base = fr.llm_state.map({"positive": 1.0, "negative": 1.0, "uncertain": 0.5, "not_mentioned": 0.3})
    assert np.allclose(fr.weight, base * cfg.targets.kill_weight_factor)

    tgt, wgt, src = load_targets(tmp_path / "labels" / "labels_final.csv", studies)
    assert tgt.shape == (30, len(LABELS)) and list(tgt.columns) == LABELS
    assert (src.loc["s0"] == "gold").all()


def test_gold_rate_not_mentioned(tmp_path):
    cfg, _ = _setup(tmp_path, kill_kappa=-1.0, nm_target="gold_rate")
    out = build_targets(cfg, get_logger("test"))
    nm = out[out.source == "llm_not_mentioned"]
    assert nm.target.between(0, 1).all() and (nm.weight == cfg.targets.not_mentioned.weight).all()
