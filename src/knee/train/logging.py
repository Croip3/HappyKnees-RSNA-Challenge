"""Verbose run logging: console + log file + metrics.csv (+ optional TensorBoard)."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from knee.constants import LABELS
from knee.utils import device_info, fmt_seconds, get_logger, git_hash


class RunLogger:
    def __init__(self, run_dir: Path, tensorboard: bool = False):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log = get_logger(f"knee.train.{self.run_dir.name}", self.run_dir / "train.log")
        self.metrics_path = self.run_dir / "metrics.csv"
        self._metrics_header: list[str] | None = None
        self.tb = None
        if tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.tb = SummaryWriter(str(self.run_dir / "tb"))
            except Exception as e:  # noqa: BLE001
                self.log.warning(f"tensorboard disabled: {e}")

    def info(self, msg: str) -> None:
        self.log.info(msg)

    def warning(self, msg: str) -> None:
        self.log.warning(msg)

    def start(self, cfg_dict: dict) -> None:
        self.info(f"run dir: {self.run_dir}")
        self.info(f"git: {git_hash()} | device: {json.dumps(device_info())}")
        self.info("config:\n" + json.dumps(cfg_dict, indent=1, default=str))

    def metrics_row(self, row: dict[str, Any]) -> None:
        if self._metrics_header is None:
            self._metrics_header = list(row.keys())
            new = not self.metrics_path.exists()
            with open(self.metrics_path, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=self._metrics_header, extrasaction="ignore")
                if new:
                    w.writeheader()
        with open(self.metrics_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self._metrics_header, extrasaction="ignore").writerow(row)
        if self.tb is not None:
            step = int(row.get("global_step", row.get("epoch", 0)))
            for k, v in row.items():
                if isinstance(v, (int, float)) and np.isfinite(v):
                    self.tb.add_scalar(f"fold{row.get('fold', 0)}/{k}", v, step)

    def label_table(self, title: str, columns: dict[str, np.ndarray], counts: dict[str, np.ndarray] | None = None) -> None:
        """Per-label table, e.g. columns={'AUC llm': ..., 'AUC gold': ...}."""
        names = list(columns)
        header = f"{'label':<18}" + "".join(f"{n:>11}" for n in names)
        if counts:
            header += "".join(f"{n:>9}" for n in counts)
        lines = [title, header]
        for j, label in enumerate(LABELS):
            line = f"{label:<18}" + "".join(
                f"{columns[n][j]:>11.3f}" if np.isfinite(columns[n][j]) else f"{'-':>11}" for n in names)
            if counts:
                line += "".join(f"{int(counts[n][j]):>9}" for n in counts)
            lines.append(line)
        macro = "".join(f"{np.nanmean(columns[n]) if np.isfinite(columns[n]).any() else float('nan'):>11.3f}"
                        for n in names)
        lines.append(f"{'MACRO':<18}" + macro)
        self.info("\n".join(lines))

    def close(self) -> None:
        if self.tb is not None:
            self.tb.close()


class ETA:
    def __init__(self, total: int):
        self.total, self.t0, self.done = total, time.time(), 0

    def step(self, n: int = 1) -> str:
        self.done += n
        el = time.time() - self.t0
        rem = el / max(self.done, 1) * (self.total - self.done)
        return f"elapsed {fmt_seconds(el)}, ETA {fmt_seconds(rem)}"
