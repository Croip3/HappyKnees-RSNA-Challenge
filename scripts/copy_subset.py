"""Pack the DICOMs of a few train studies into a tar.gz (run on Kaggle, download for local dev).

Example:
  python scripts/copy_subset.py --n-gold 2 --n-other 2 --out /kaggle/working/subset.tar.gz
"""

import argparse
import tarfile
from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd

from knee.config import add_common_args, config_from_args
from knee.constants import LABELS, STUDY


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--n-gold", type=int, default=2)
    parser.add_argument("--n-other", type=int, default=2)
    parser.add_argument("--studies", nargs="*", default=[], help="explicit StudyInstanceUIDs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    cfg = config_from_args(args)
    train = pd.read_csv(Path(cfg.paths.data_root) / "train.csv")
    gold = train[train[LABELS].notna().all(axis=1)]
    other = train[train[LABELS].isna().all(axis=1)]
    studies = list(args.studies)
    studies += gold.sample(n=args.n_gold, random_state=args.seed)[STUDY].tolist() if args.n_gold else []
    studies += other.sample(n=args.n_other, random_state=args.seed)[STUDY].tolist() if args.n_other else []
    root = Path(cfg.paths.train_images)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out, "w:gz", compresslevel=1) as tar:
        for s in studies:
            tar.add(root / s, arcname=f"train_series/{s}")
            print("added", s, flush=True)
    print(f"wrote {out} ({out.stat().st_size / 2**20:.0f} MB) with {len(studies)} studies")


if __name__ == "__main__":
    main()
