"""Final soft targets + weights + source for every study x label (labels_final.csv, long format)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from knee.constants import LABELS, STUDY
from knee.labels.validate import agreement_table, gold_long_table, llm_long_table


def _gold_rate_not_mentioned(merged: pd.DataFrame, min_count: int) -> dict[str, float]:
    """Empirical gold positive rate among gold studies the LLM called not_mentioned (pooled if rare)."""
    nm = merged[merged.llm_state == "not_mentioned"]
    pooled = float(nm.gold.mean()) if len(nm) else 0.0
    rates = {}
    for label in LABELS:
        g = nm[nm.label == label]
        rates[label] = float(g.gold.mean()) if len(g) >= min_count else pooled
    return rates


def _calibrated_targets(merged: pd.DataFrame, strength: float) -> dict[tuple[str, str], float]:
    """P(gold=1 | label, state) with a Beta prior centred on the pooled (all-label) rate of the state."""
    out = {}
    for state in ["positive", "negative", "uncertain", "not_mentioned"]:
        cell = merged[merged.llm_state == state]
        pooled = float(cell.gold.mean()) if len(cell) else {"positive": 1.0, "uncertain": 0.6}.get(state, 0.0)
        for label in LABELS:
            g = cell[cell.label == label].gold
            out[(label, state)] = float((g.sum() + strength * pooled) / (len(g) + strength))
    return out


def build_targets(cfg, log) -> pd.DataFrame:
    t = cfg.targets
    labels_dir = Path(cfg.paths.labels_dir)
    train = pd.read_csv(Path(cfg.paths.data_root) / "train.csv")
    llm = llm_long_table(labels_dir / cfg.llm.output_jsonl)
    gold = gold_long_table(train)

    grid = pd.MultiIndex.from_product([train[STUDY], LABELS], names=[STUDY, "label"]).to_frame(index=False)
    df = grid.merge(llm, on=[STUDY, "label"], how="left").merge(gold, on=[STUDY, "label"], how="left")
    df["llm_state"] = df["llm_state"].fillna("missing")

    merged = df[df.gold.notna() & ~df.llm_state.isin(["missing", "llm_error"])].assign(gold=lambda d: d.gold.astype(int))
    kill = set()
    if len(merged):
        table = agreement_table(merged).set_index("label")
        for label in LABELS:
            k = table.loc[label, "kappa"]
            if not np.isfinite(k) or k < t.min_kappa:
                kill.add(label)
        log.info(f"per-label kappa (gold, n={merged[STUDY].nunique()} studies): "
                 + ", ".join(f"{label}={table.loc[label, 'kappa']:.2f}" for label in LABELS))
    else:
        log.warning("no gold/LLM overlap: kill switch and gold_rate mapping disabled")
    if kill:
        log.warning(f"kill switch (kappa < {t.min_kappa}) -> {t.kill_action} LLM labels for: {sorted(kill)}")
    else:
        log.info(f"kill switch: no label below min_kappa={t.min_kappa}")

    nm_target = {label: t.not_mentioned.target for label in LABELS}
    if t.not_mentioned_target == "gold_rate" and len(merged):
        nm_target = _gold_rate_not_mentioned(merged, t.gold_rate_min_count)
        log.info("not_mentioned targets (gold rate): " + ", ".join(f"{k}={v:.2f}" for k, v in nm_target.items()))

    state_map = {"positive": t.positive, "negative": t.negative, "uncertain": t.uncertain}
    calib = None
    if t.calibrate_all_states and len(merged):
        calib = _calibrated_targets(merged, float(t.calibration_strength))
        log.info("calibrated targets P(gold=1 | state): " + "; ".join(
            f"{label}: " + ", ".join(f"{st[:3]}={calib[(label, st)]:.2f}"
                                     for st in ["positive", "negative", "uncertain", "not_mentioned"])
            for label in LABELS))
    target = np.full(len(df), np.nan)
    weight = np.zeros(len(df))
    source = np.empty(len(df), dtype=object)
    postop_rule = _postop_rule(cfg)
    for i, r in enumerate(df.itertuples(index=False)):
        if not pd.isna(r.gold):
            target[i], weight[i], source[i] = float(r.gold), t.gold_weight, "gold"
            continue
        state = r.llm_state
        if state in ("missing", "llm_error"):
            source[i] = state if state == "llm_error" else "missing"
            continue
        if r.postop is True and postop_rule != "as_reported":
            if postop_rule == "mask":
                target[i], weight[i], source[i] = 0.0, 0.0, f"llm_{state}"
                continue
            state = "uncertain"
        if state == "not_mentioned":
            target[i], weight[i] = nm_target[r.label], t.not_mentioned.weight
        else:
            target[i], weight[i] = state_map[state].target, state_map[state].weight
        if calib is not None:
            target[i] = calib[(r.label, state)]
        if r.label in kill:
            weight[i] = 0.0 if t.kill_action == "mask" else weight[i] * t.kill_weight_factor
        source[i] = f"llm_{state}"
    out = pd.DataFrame({STUDY: df[STUDY], "label": df["label"], "target": target, "weight": weight,
                        "source": source, "llm_state": df["llm_state"], "postop": df["postop"].fillna(False)})
    out_path = labels_dir / "labels_final.csv"
    out.to_csv(out_path, index=False)
    log.info(f"wrote {out_path} ({out[STUDY].nunique()} studies x {len(LABELS)} labels)")
    n_postop = int(out.postop.astype(bool).sum())
    if n_postop:
        log.info(f"post-operative mentions (rule={postop_rule}): {n_postop} study-labels, "
                 f"by label {out[out.postop.astype(bool)].label.value_counts().to_dict()}")
    print_label_stats(out, log)
    return out


def _postop_rule(cfg) -> str:
    import yaml

    with open(cfg.paths.label_definitions, encoding="utf-8") as f:
        return (yaml.safe_load(f).get("postoperative") or {}).get("rule", "as_reported")


def print_label_stats(out: pd.DataFrame, log) -> pd.DataFrame:
    rows = []
    for label in LABELS:
        g = out[out.label == label]
        n = len(g)
        rows.append({
            "label": label,
            "gold": int((g.source == "gold").sum()),
            "positive": int((g.source == "llm_positive").sum()),
            "negative": int((g.source == "llm_negative").sum()),
            "uncertain%": 100 * (g.source == "llm_uncertain").mean(),
            "not_mentioned%": 100 * (g.source == "llm_not_mentioned").mean(),
            "masked%": 100 * (g.weight <= 0).mean(),
            "pos_rate(w>0)": float((g.target[g.weight > 0] >= 0.5).mean()) if (g.weight > 0).any() else np.nan,
            "mean_weight": float(g.weight.mean()),
            "n": n,
        })
    stats = pd.DataFrame(rows)
    log.info("final label statistics:\n" + stats.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    return stats


def load_targets(path: str | Path, studies: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Wide (study x label) target, weight and source frames, indexed by StudyInstanceUID."""
    df = pd.read_csv(path)
    tgt = df.pivot(index=STUDY, columns="label", values="target")[LABELS]
    wgt = df.pivot(index=STUDY, columns="label", values="weight")[LABELS]
    src = df.pivot(index=STUDY, columns="label", values="source")[LABELS]
    if studies is not None:
        tgt, wgt, src = tgt.reindex(studies), wgt.reindex(studies), src.reindex(studies)
    return tgt.fillna(0.0), wgt.fillna(0.0), src.fillna("missing")
