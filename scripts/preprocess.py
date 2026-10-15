"""Phase 1: preprocess DICOM studies into the uint8 cache (one .npz per study + manifest.csv).

Examples:
  python scripts/preprocess.py --split train --limit 50
  python scripts/preprocess.py --split test --set preprocess.num_workers=4
"""

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from knee.config import add_common_args, config_from_args, dump_config
from knee.data.preprocess import run_preprocess
from knee.utils import get_logger


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__,
                                                     formatter_class=argparse.RawDescriptionHelpFormatter))
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--studies", nargs="*", default=None, help="only these StudyInstanceUIDs")
    parser.add_argument("--studies-file", default=None, help="text file with one StudyInstanceUID per line")
    args = parser.parse_args()
    cfg = config_from_args(args)
    out_dir = Path(cfg.paths.cache_dir) / args.split
    log = get_logger("knee.preprocess", out_dir / "preprocess.log")
    dump_config(cfg, out_dir / "preprocess_config.yaml")
    studies = args.studies
    if args.studies_file:
        studies = [s.strip() for s in Path(args.studies_file).read_text().split() if s.strip()]
    run_preprocess(cfg, args.split, log, limit=args.limit, overwrite=args.overwrite, studies=studies)


if __name__ == "__main__":
    main()
