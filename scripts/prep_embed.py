"""Phases 1+3 in one streaming pass (used on Kaggle where the DICOMs live): DICOM -> preprocess
(CPU workers) -> frozen-backbone embedding (GPU) -> feature files; --save-cache also keeps the
uint8 .npz image cache. Resumable.

Examples:
  python scripts/prep_embed.py --split train --studies-file data/smoke_studies.txt --save-cache
  python scripts/prep_embed.py --split train --set preprocess.num_workers=4
"""

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from knee.config import add_common_args, config_from_args, dump_config
from knee.features.extract import features_dir, run_prep_embed
from knee.utils import get_logger


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__,
                                                     formatter_class=argparse.RawDescriptionHelpFormatter))
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--studies-file", default=None)
    parser.add_argument("--save-cache", action="store_true")
    parser.add_argument("--variant", action="append", default=None, metavar="OVERRIDES",
                        help="extra feature variant as space separated KEY=VALUE overrides, e.g. "
                             "'backbone.image_size=448' (repeatable; the base config is always included)")
    args = parser.parse_args()
    cfg = config_from_args(args)
    out = features_dir(cfg, args.split)
    log = get_logger("knee.prep_embed", out / "prep_embed.log")
    dump_config(cfg, out / "prep_embed_config.yaml")
    studies = None
    if args.studies_file:
        studies = [s.strip() for s in Path(args.studies_file).read_text().split() if s.strip()]
    variants = [[]] + [v.split() for v in (args.variant or [])]
    run_prep_embed(cfg, args.split, log, limit=args.limit, overwrite=args.overwrite, studies=studies,
                   save_cache=args.save_cache, variants=variants)


if __name__ == "__main__":
    main()
