"""YAML config with strict keys, dotted CLI overrides and path resolution.

Usage in a script:
    parser = add_common_args(argparse.ArgumentParser())
    args = parser.parse_args()
    cfg = config_from_args(args)
    cfg.preprocess.image_size  # attribute access
"""

from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
from typing import Any, Iterable

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"
COMPETITION = "rsna-knee-abnormality-detection"


class Cfg(dict):
    """A dict with attribute access. Nested dicts are Cfg as well."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(f"config has no key '{key}'") from e

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __deepcopy__(self, memo):
        return _wrap(copy.deepcopy(to_dict(self), memo))


def _wrap(obj: Any) -> Any:
    if isinstance(obj, dict):
        return Cfg({k: _wrap(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_wrap(v) for v in obj]
    return obj


def to_dict(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: to_dict(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_dict(v) for v in obj]
    return obj


def _merge(base: dict, update: dict, strict: bool, prefix: str = "") -> None:
    for key, value in update.items():
        dotted = f"{prefix}{key}"
        # Dicts that are empty in the defaults are free-form.
        if strict and key not in base and len(base) > 0:
            raise KeyError(f"Unknown config key '{dotted}'")
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value, strict, dotted + ".")
        else:
            base[key] = value


def parse_value(text: str) -> Any:
    """Parse a CLI override value with YAML rules (plus plain floats like 1e-4)."""
    value = yaml.safe_load(text) if text.strip() else ""
    if isinstance(value, str):
        try:
            return float(value) if any(c in value for c in ".eE") else int(value)
        except ValueError:
            return value
    return value


def set_dotted(cfg: dict, dotted: str, value: Any, strict: bool = True) -> None:
    keys = dotted.split(".")
    node = cfg
    for i, key in enumerate(keys[:-1]):
        if key not in node or not isinstance(node[key], dict):
            if strict:
                raise KeyError(f"Unknown config section '{'.'.join(keys[: i + 1])}'")
            node[key] = {}
        node = node[key]
    if strict and keys[-1] not in node and len(node) > 0:
        raise KeyError(f"Unknown config key '{dotted}'")
    node[keys[-1]] = value


def get_dotted(cfg: dict, dotted: str, default: Any = None) -> Any:
    node: Any = cfg
    for key in dotted.split("."):
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def resolve_path(p: str | os.PathLike | None) -> Path | None:
    if p is None:
        return None
    p = Path(os.path.expanduser(str(p)))
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def find_data_root() -> Path:
    """Locate the competition folder (contains test.csv) on Kaggle or locally."""
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for cand in [
            kaggle_input / COMPETITION,
            kaggle_input / "competitions" / COMPETITION,
        ]:
            if (cand / "test.csv").exists():
                return cand
        for csv in sorted(kaggle_input.glob("*/test.csv")) + sorted(kaggle_input.glob("*/*/test.csv")):
            if (csv.parent / "sample_submission.csv").exists():
                return csv.parent
    return PROJECT_ROOT / "data" / "raw"


def _resolve_paths(cfg: Cfg) -> None:
    paths = cfg.paths
    root = find_data_root() if paths.data_root in (None, "auto") else resolve_path(paths.data_root)
    paths.data_root = str(root)
    paths.train_images = str(resolve_path(paths.train_images) or root / "train_series")
    paths.test_images = str(resolve_path(paths.test_images) or root / "test_series")
    for key in ["cache_dir", "features_dir", "labels_dir", "results_dir", "runs_dir", "label_definitions"]:
        paths[key] = str(resolve_path(paths[key]))


def load_config(path: str | os.PathLike | None = None, overrides: Iterable[str] = ()) -> Cfg:
    with open(DEFAULT_CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if path:
        with open(resolve_path(path), encoding="utf-8") as f:
            _merge(cfg, yaml.safe_load(f) or {}, strict=True)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must look like key=value, got '{item}'")
        key, text = item.split("=", 1)
        set_dotted(cfg, key.strip(), parse_value(text))
    cfg = _wrap(cfg)
    _resolve_paths(cfg)
    return cfg


def dump_config(cfg: Cfg, path: str | os.PathLike) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(to_dict(cfg), f, sort_keys=False, allow_unicode=True)


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--config", default=None, help="YAML merged over configs/default.yaml")
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
        help="override a config value, e.g. --set model.slice_context=all (repeatable)",
    )
    parser.add_argument("--overwrite", action="store_true", help="recompute outputs that already exist")
    parser.add_argument("--limit", type=int, default=None, help="only process the first N studies")
    return parser


def config_from_args(args: argparse.Namespace) -> Cfg:
    return load_config(args.config, args.overrides)
