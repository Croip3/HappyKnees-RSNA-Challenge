"""Phase 3: embed cached slices with the frozen backbone (fp16 .npz per study + manifest).

Examples:
  python scripts/extract_features.py --split train --limit 50
  python scripts/extract_features.py --set backbone.name=dinov2_vitb14_reg --set backbone.weights_path=weights/vitb
"""

import argparse

import _bootstrap  # noqa: F401

from knee.config import add_common_args, config_from_args, dump_config
from knee.features.extract import features_dir, run_extract
from knee.utils import get_logger


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__,
                                                     formatter_class=argparse.RawDescriptionHelpFormatter))
    parser.add_argument("--split", choices=["train", "test"], default="train")
    args = parser.parse_args()
    cfg = config_from_args(args)
    out = features_dir(cfg, args.split)
    log = get_logger("knee.features", out / "extract.log")
    dump_config(cfg, out / "extract_config.yaml")
    run_extract(cfg, args.split, log, limit=args.limit, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
