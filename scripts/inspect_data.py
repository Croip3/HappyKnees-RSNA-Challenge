"""Phase 0: inspect CSVs, reports, series and DICOM headers.

Examples:
  python scripts/inspect_data.py                          # local: CSV checks (images not present)
  python scripts/inspect_data.py --header-scan -1         # on Kaggle: + one header per series for all series
"""

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from knee.config import add_common_args, config_from_args
from knee.data.inspect import run_inspection
from knee.utils import get_logger


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--out", default=None, help="output dir (default: {results_dir}/inspect)")
    parser.add_argument("--dicom-series", type=int, default=20, help="random series with full header read")
    parser.add_argument("--header-scan", type=int, default=0,
                        help="one-file header scan over N random series (-1 = all, 0 = skip)")
    parser.add_argument("--no-count", action="store_true", help="skip counting files per series")
    parser.add_argument("--skip-languages", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    cfg = config_from_args(args)
    out = Path(args.out) if args.out else Path(cfg.paths.results_dir) / "inspect"
    log = get_logger("knee.inspect", out / "inspect.log")
    run_inspection(cfg, out, log, dicom_series=args.dicom_series, header_scan_n=args.header_scan,
                   count=not args.no_count, seed=args.seed, skip_languages=args.skip_languages)


if __name__ == "__main__":
    main()
