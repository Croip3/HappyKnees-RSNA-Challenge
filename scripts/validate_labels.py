"""Phase 2b: compare LLM states with the gold labels; writes gold_agreement.csv and disagreements.csv."""

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from knee.config import add_common_args, config_from_args
from knee.labels.validate import run_validation
from knee.utils import get_logger


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--uncertain-negative", action="store_true",
                        help="count 'uncertain' as negative for agreement/kappa (default: positive)")
    args = parser.parse_args()
    cfg = config_from_args(args)
    log = get_logger("knee.labels", Path(cfg.paths.results_dir) / "labels" / "validate.log")
    run_validation(cfg, log, uncertain_positive=not args.uncertain_negative)


if __name__ == "__main__":
    main()
