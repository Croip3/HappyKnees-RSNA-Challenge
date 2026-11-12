"""Phase 2c: build labels_final.csv (target, weight, source per study x label) from gold + LLM states."""

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from knee.config import add_common_args, config_from_args
from knee.labels.build_targets import build_targets
from knee.utils import get_logger


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args()
    cfg = config_from_args(args)
    log = get_logger("knee.labels", Path(cfg.paths.labels_dir) / "build_targets.log")
    build_targets(cfg, log)


if __name__ == "__main__":
    main()
