"""Small shared helpers: logging, seeding, timing, environment info."""

from __future__ import annotations

import json
import logging
import os
import platform
import random
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

_CONSOLE_HANDLER: logging.Handler | None = None


def get_logger(name: str = "knee", log_file: str | os.PathLike | None = None) -> logging.Logger:
    """Console logger (rich if available) plus an optional log file."""
    global _CONSOLE_HANDLER
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if _CONSOLE_HANDLER is None:
        try:
            if not sys.stdout.isatty():
                raise ImportError("plain logging when not attached to a terminal (files, Kaggle logs)")
            from rich.logging import RichHandler

            _CONSOLE_HANDLER = RichHandler(show_path=False, rich_tracebacks=True, markup=False)
            _CONSOLE_HANDLER.setFormatter(logging.Formatter("%(message)s", datefmt="%H:%M:%S"))
        except ImportError:
            _CONSOLE_HANDLER = logging.StreamHandler(sys.stdout)
            _CONSOLE_HANDLER.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))
    if _CONSOLE_HANDLER not in logger.handlers:
        logger.addHandler(_CONSOLE_HANDLER)
    if log_file is not None:
        log_file = str(log_file)
        if not any(isinstance(h, logging.FileHandler) and h.baseFilename == os.path.abspath(log_file)
                   for h in logger.handlers):
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S"))
            logger.addHandler(fh)
    return logger


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def git_hash() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                             cwd=Path(__file__).parent, timeout=5)
        return out.stdout.strip() or "n/a"
    except Exception:
        return "n/a"


def device_info() -> dict[str, Any]:
    info: dict[str, Any] = {"python": platform.python_version(), "platform": platform.platform()}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpus"] = [
                f"{torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / 2**30:.1f} GB)"
                for i in range(torch.cuda.device_count())
            ]
            info["bf16_supported"] = torch.cuda.is_bf16_supported()
    except ImportError:
        info["torch"] = "not installed"
    return info


@contextmanager
def timer(store: dict[str, float], key: str):
    """Accumulate wall time of a block into store[key]."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        store[key] = store.get(key, 0.0) + time.perf_counter() - t0


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {(k if isinstance(k, (str, int, float, bool)) or k is None else str(k)): _json_safe(v)
                for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def save_json(obj: Any, path: str | os.PathLike) -> None:
    obj = _json_safe(obj)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)
    os.replace(tmp, path)


def fmt_seconds(s: float) -> str:
    s = int(round(s))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}h{m:02d}m{sec:02d}s" if h else f"{m}m{sec:02d}s"


def identity(x):
    """collate_fn for DataLoader(batch_size=None) that keeps numpy arrays as they are."""
    return x
