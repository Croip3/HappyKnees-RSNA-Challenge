"""Compare LLM states with the gold labels and write a disagreement report for manual review."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from knee.constants import LABELS, LLM_STATES, REPORT, STUDY
from knee.labels.llm_extract import load_results

STATES_ALL = LLM_STATES + ["llm_error", "missing"]


def llm_long_table(jsonl: Path) -> pd.DataFrame:
    """One row per study x label with the LLM state and evidence."""
    rows = []
    for study, rec in load_results(jsonl).items():
        postop = set(rec.get("postop_labels") or [])
        for label in LABELS:
            if rec.get("status") == "ok":
                item = rec["labels"][label]
                rows.append({STUDY: study, "label": label, "llm_state": item["state"],
                             "evidence": item.get("evidence", ""), "postop": label in postop})
            else:
                rows.append({STUDY: study, "label": label, "llm_state": "llm_error", "evidence": "",
                             "postop": False})
    return pd.DataFrame(rows, columns=[STUDY, "label", "llm_state", "evidence", "postop"])


def gold_long_table(train: pd.DataFrame) -> pd.DataFrame:
    gold = train[train[LABELS].notna().all(axis=1)]
    return gold.melt(id_vars=[STUDY], value_vars=LABELS, var_name="label", value_name="gold").astype({"gold": int})


def cohen_kappa(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, int), np.asarray(b, int)
    if len(a) == 0:
        return float("nan")
    po = float((a == b).mean())
    pe = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
    return float("nan") if pe >= 1.0 else (po - pe) / (1 - pe)


def binarize(states: pd.Series, uncertain_as: int) -> np.ndarray:
    return np.where(states == "positive", 1, np.where(states == "uncertain", uncertain_as, 0))


def agreement_table(merged: pd.DataFrame, uncertain_positive: bool = True) -> pd.DataFrame:
    """Per-label agreement metrics. `merged` has columns label, gold, llm_state."""
    rows = []
    groups = [(label, merged[merged.label == label]) for label in LABELS] + [("ALL", merged)]
    for label, g in groups:
        st = g.llm_state
        pred_pos = (st == "positive").astype(int).values
        gold = g.gold.values
        tp = int(((pred_pos == 1) & (gold == 1)).sum())
        fp = int(((pred_pos == 1) & (gold == 0)).sum())
        fn = int(((pred_pos == 0) & (gold == 1)).sum())
        binary = binarize(st, int(uncertain_positive))
        row = {
            "label": label, "n": len(g), "gold_pos": int(gold.sum()),
            "precision_pos": tp / (tp + fp) if tp + fp else np.nan,
            "recall_pos": tp / (tp + fn) if tp + fn else np.nan,
            "agreement": float((binary == gold).mean()) if len(g) else np.nan,
            "kappa": cohen_kappa(binary, gold),
            "kappa_uncertain_as_neg": cohen_kappa(binarize(st, 0), gold),
        }
        for s in STATES_ALL:
            m = st == s
            row[f"n_{s}"] = int(m.sum())
            row[f"gold_rate_{s}"] = float(g.gold[m].mean()) if m.any() else np.nan
        # confusion counts gold x state
        for s in LLM_STATES + ["llm_error"]:
            row[f"g1_{s}"] = int(((st == s) & (g.gold == 1)).sum())
            row[f"g0_{s}"] = int(((st == s) & (g.gold == 0)).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def excerpt(report: str, evidence: str, width: int = 300) -> str:
    report = " ".join(str(report).split())
    if evidence:
        i = report.lower().find(evidence.lower()[:40])
        if i >= 0:
            a = max(0, i - width // 2)
            return ("..." if a > 0 else "") + report[a:a + width + len(evidence)] + "..."
    return report[:2 * width] + ("..." if len(report) > 2 * width else "")


def run_validation(cfg, log, uncertain_positive: bool = True) -> pd.DataFrame | None:
    labels_dir = Path(cfg.paths.labels_dir)
    out_dir = Path(cfg.paths.results_dir) / "labels"
    out_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(Path(cfg.paths.data_root) / "train.csv")
    llm = llm_long_table(labels_dir / cfg.llm.output_jsonl)
    gold = gold_long_table(train)
    merged = gold.merge(llm, on=[STUDY, "label"], how="left")
    merged["llm_state"] = merged["llm_state"].fillna("missing")
    merged["evidence"] = merged["evidence"].fillna("")
    n_cov = merged.loc[merged.llm_state != "missing", STUDY].nunique()
    log.info(f"gold studies: {gold[STUDY].nunique()}, with LLM output: {n_cov}")
    if n_cov == 0:
        log.warning("no LLM results for gold studies yet")
        return None
    merged = merged[merged.llm_state != "missing"]
    table = agreement_table(merged, uncertain_positive)
    table.to_csv(out_dir / "gold_agreement.csv", index=False)

    cols = ["label", "n", "gold_pos", "n_positive", "n_negative", "n_uncertain", "n_not_mentioned", "n_llm_error",
            "precision_pos", "recall_pos", "agreement", "kappa", "gold_rate_not_mentioned", "gold_rate_uncertain"]
    try:
        from rich.console import Console
        from rich.table import Table

        t = Table(title=f"LLM vs gold ({n_cov} studies; kappa: uncertain counted as "
                        f"{'positive' if uncertain_positive else 'negative'})")
        for c in cols:
            t.add_column(c.replace("gold_rate_", "gold%|").replace("n_", "#"), justify="right")
        for _, r in table[cols].iterrows():
            t.add_row(*[f"{v:.2f}" if isinstance(v, float) else str(v) for v in r.values])
        Console(width=220).print(t)
    except ImportError:
        log.info("\n" + table[cols].to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    # Disagreements for manual review
    reports = dict(zip(train[STUDY], train[REPORT].fillna("")))
    bin_pred = binarize(merged.llm_state, int(uncertain_positive))
    dis = merged[(bin_pred != merged.gold.values) | (merged.llm_state == "uncertain")].copy()
    dis["type"] = np.select(
        [dis.llm_state == "uncertain", (dis.gold == 1) & (dis.llm_state == "not_mentioned"),
         (dis.gold == 1) & (dis.llm_state == "negative"), dis.gold == 0],
        ["uncertain", "missed(not_mentioned)", "missed(negative)", "false_positive"], default="other")
    dis["report_excerpt"] = [excerpt(reports[s], e) for s, e in zip(dis[STUDY], dis.evidence)]
    dis = dis[[STUDY, "label", "gold", "llm_state", "type", "evidence", "report_excerpt"]]
    dis.sort_values(["label", "type"]).to_csv(out_dir / "disagreements.csv", index=False)
    log.info(f"wrote {out_dir / 'gold_agreement.csv'} and disagreements.csv ({len(dis)} rows: "
             f"{dis['type'].value_counts().to_dict()})")
    return table
