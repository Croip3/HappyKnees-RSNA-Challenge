"""Phase 5: K-fold training of the cross-slice attention classifier on cached features.

Examples:
  python scripts/train.py --set train.run_folds=[0] --set train.epochs=2      # smoke test
  python scripts/train.py --set model.slice_context=all --set train.run_name=ctx_all
"""

import argparse

import _bootstrap  # noqa: F401

from knee.config import add_common_args, config_from_args
from knee.train.train import run_training


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__,
                                                     formatter_class=argparse.RawDescriptionHelpFormatter))
    args = parser.parse_args()
    cfg = config_from_args(args)
    run_dir = run_training(cfg, args_limit=args.limit)
    print(f"run directory: {run_dir}")


if __name__ == "__main__":
    main()
