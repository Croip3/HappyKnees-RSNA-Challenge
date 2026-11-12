"""Pack folders from attached kernel outputs into single tar files (one download instead of thousands).

Example (Kaggle, with the knee-full-prep kernel attached as a kernel source):
  python scripts/pack_outputs.py --pattern "features/*_g0_*" --out /kaggle/working/packed
"""

import argparse
import os
import tarfile
from pathlib import Path


def find_roots(base: Path, max_depth: int = 4) -> list[Path]:
    """Folders under /kaggle/input that contain a 'features' folder (skips the DICOM trees)."""
    roots = []
    for dirpath, dirnames, _ in os.walk(base):
        depth = len(Path(dirpath).relative_to(base).parts)
        dirnames[:] = [d for d in dirnames if d not in ("train_series", "test_series") and depth < max_depth]
        if "features" in dirnames:
            roots.append(Path(dirpath))
    return roots


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", default="/kaggle/input")
    p.add_argument("--pattern", required=True, help="glob relative to the output root, e.g. 'features/*_g0_*'")
    p.add_argument("--out", required=True)
    p.add_argument("--files-per-tar", type=int, default=800, help="split big folders into several tars")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    roots = find_roots(Path(args.base))
    print("output roots:", roots, flush=True)
    for root in roots:
        for folder in sorted(root.glob(args.pattern)):
            if not folder.is_dir():
                continue
            files = sorted(f for f in folder.rglob("*") if f.is_file())
            n = args.files_per_tar
            for k in range(0, len(files), n):
                tar_path = out / f"{folder.name}.part{k // n:02d}.tar"
                with tarfile.open(tar_path, "w") as tar:
                    for f in files[k:k + n]:
                        tar.add(f, arcname=f.relative_to(root).as_posix())
                print(f"packed {len(files[k:k + n])} files of {folder} -> {tar_path} "
                      f"({tar_path.stat().st_size / 2**20:.0f} MB)", flush=True)


if __name__ == "__main__":
    main()
