"""Phase 6: offline inference -> submission.csv (also used by the Kaggle notebook).

Examples:
  python scripts/predict.py --set infer.checkpoints=runs/<run> --set infer.dry_run=5
  python scripts/predict.py --set infer.checkpoints=runs/<run> --set infer.tta_mirror=true
"""

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from knee.config import add_common_args, config_from_args
from knee.infer.predict import run_inference, write_submission
from knee.utils import get_logger


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__,
                                                     formatter_class=argparse.RawDescriptionHelpFormatter))
    args = parser.parse_args()
    cfg = config_from_args(args)
    out = Path(cfg.infer.output)
    log = get_logger("knee.infer", out.with_suffix(".log"))
    dry = args.limit or int(cfg.infer.dry_run or 0)
    sub = run_inference(cfg, log, dry_run=dry)
    write_submission(sub, out, Path(cfg.paths.data_root) / "sample_submission.csv", log)


if __name__ == "__main__":
    main()
