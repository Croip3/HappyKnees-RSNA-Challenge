"""Build the offline Kaggle inference inputs and notebook.

Creates (under build/):
  kaggle_datasets/knee-weights/    backbone weights, <backbone_name>/model.safetensors (+ marker)
  kaggle_datasets/knee-inference/  package source + trained fold checkpoints (+ marker)
  kaggle/knee-inference/           notebook + kernel-metadata.json (internet off, GPU)
and writes notebooks/kaggle_inference.ipynb.

Examples:
  python scripts/make_kaggle_bundle.py --run runs/<run> --weights weights          # build only
  python scripts/make_kaggle_bundle.py --run runs/<run> --push-bundle --push-kernel # upload + run
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMPETITION = "rsna-knee-abnormality-detection"

NOTEBOOK_CELLS = [
    ("markdown", """# RSNA Knee: offline inference (cross-slice attention on frozen DINOv2 features)

Inputs: competition data, `knee-inference` (package source + fold checkpoints), `knee-weights` (backbone).
Internet is not needed. Set `KNEE_DRY_RUN=5` to run on the first 5 test studies only."""),
    ("code", """import os, sys, time, json
from pathlib import Path

T0 = time.time()
DRY_RUN = int(os.environ.get("KNEE_DRY_RUN", "0"))


def find_dir(marker, root="/kaggle/input", max_depth=6, skip=("train_series", "test_series")):
    \"\"\"First folder under /kaggle/input containing `marker` (skips the big DICOM trees).\"\"\"
    root = Path(root)
    for dirpath, dirnames, filenames in os.walk(root):
        depth = len(Path(dirpath).relative_to(root).parts)
        dirnames[:] = sorted(d for d in dirnames if d not in skip and depth < max_depth)
        if marker in filenames:
            return Path(dirpath)
    raise FileNotFoundError(f"{marker} not found under {root}")


BUNDLE = find_dir("knee_bundle.json")
WEIGHTS = find_dir("knee_weights.json")
print("bundle:", BUNDLE, json.loads((BUNDLE / "knee_bundle.json").read_text()))
print("weights:", WEIGHTS, sorted(p.name for p in WEIGHTS.iterdir()))
sys.path.insert(0, str(BUNDLE / "src"))"""),
    ("code", """from knee.config import load_config
from knee.infer.predict import run_inference, write_submission
from knee.utils import get_logger

cfg = load_config(overrides=[
    f"backbone.weights_path={WEIGHTS}",
    f"infer.checkpoints={BUNDLE / 'models'}",
    "infer.output=/kaggle/working/submission.csv",
    "preprocess.num_workers=4",
    "infer.tta_mirror=__TTA__",
])
log = get_logger("knee.infer", "/kaggle/working/inference.log")
sub = run_inference(cfg, log, dry_run=DRY_RUN)
write_submission(sub, "/kaggle/working/submission.csv", Path(cfg.paths.data_root) / "sample_submission.csv", log)
print(f"notebook wall time: {time.time() - T0:.1f}s")
sub.head()"""),
]


def make_notebook(tta: bool) -> dict:
    cells = []
    for i, (kind, src) in enumerate(NOTEBOOK_CELLS):
        src = src.replace("__TTA__", "true" if tta else "false")
        cell = {"cell_type": kind, "id": f"cell-{i:02d}", "metadata": {}, "source": src.splitlines(keepends=True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                                         "name": "python3"},
                                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


def kaggle_username() -> str:
    out = subprocess.run(["kaggle", "config", "view"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if "username:" in line:
            return line.split("username:", 1)[1].strip()
    raise RuntimeError("could not determine Kaggle username")


def dataset_exists(ref: str) -> bool:
    r = subprocess.run(["kaggle", "datasets", "status", ref], capture_output=True, text=True)
    return r.returncode == 0 and "ready" in (r.stdout + r.stderr).lower()


def push_dataset(folder: Path, ref: str, message: str) -> None:
    cmd = (["kaggle", "datasets", "version", "-p", str(folder), "-m", message, "-r", "zip"] if dataset_exists(ref)
           else ["kaggle", "datasets", "create", "-p", str(folder), "-r", "zip"])
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default=None, help="training run dir with fold*/best.pt")
    p.add_argument("--weights", default="weights", help="folder with <backbone_name>/model.safetensors")
    p.add_argument("--user", default=None)
    p.add_argument("--tta", action="store_true")
    p.add_argument("--cpu", action="store_true", help="run the notebook on CPU (no GPU queue)")
    p.add_argument("--kernel-name", default="knee-infer")
    p.add_argument("--push-weights", action="store_true")
    p.add_argument("--push-bundle", action="store_true")
    p.add_argument("--push-kernel", action="store_true")
    p.add_argument("--message", default="update")
    args = p.parse_args()
    user = args.user or kaggle_username()
    build = ROOT / "build"

    # backbone weights dataset
    wdir = build / "kaggle_datasets" / "knee-weights"
    src_w = (ROOT / args.weights) if not Path(args.weights).is_absolute() else Path(args.weights)
    if src_w.exists():
        if wdir.exists():
            shutil.rmtree(wdir)
        shutil.copytree(src_w, wdir, ignore=shutil.ignore_patterns(".cache", "*.log", "*.lock"))
        (wdir / "knee_weights.json").write_text(json.dumps({"backbones": sorted(p.name for p in wdir.iterdir()
                                                                                if p.is_dir())}))
        (wdir / "dataset-metadata.json").write_text(json.dumps(
            {"title": "knee-weights", "id": f"{user}/knee-weights", "licenses": [{"name": "Apache 2.0"}]}, indent=1))
        print(f"built {wdir}")

    # code + checkpoints dataset
    bdir = build / "kaggle_datasets" / "knee-inference"
    if args.run:
        run = Path(args.run)
        if bdir.exists():
            shutil.rmtree(bdir)
        for sub in ["src", "configs"]:
            shutil.copytree(ROOT / sub, bdir / sub, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        ckpts = sorted(run.glob("fold*/best.pt"))
        if not ckpts:
            raise FileNotFoundError(f"no fold*/best.pt in {run}")
        for ck in ckpts:
            (bdir / "models" / ck.parent.name).mkdir(parents=True, exist_ok=True)
            shutil.copy2(ck, bdir / "models" / ck.parent.name / "best.pt")
        for extra in ["config.yaml", "oof_auc.csv"]:
            if (run / extra).exists():
                shutil.copy2(run / extra, bdir / "models" / extra)
        (bdir / "knee_bundle.json").write_text(json.dumps({"run": run.name, "folds": [c.parent.name for c in ckpts]}))
        (bdir / "dataset-metadata.json").write_text(json.dumps(
            {"title": "knee-inference", "id": f"{user}/knee-inference", "licenses": [{"name": "CC0-1.0"}]}, indent=1))
        print(f"built {bdir} with {len(ckpts)} checkpoints from {run}")

    # notebook + kernel metadata
    nb = make_notebook(args.tta)
    (ROOT / "notebooks").mkdir(exist_ok=True)
    (ROOT / "notebooks" / "kaggle_inference.ipynb").write_text(json.dumps(nb, indent=1))
    kdir = build / "kaggle" / args.kernel_name
    kdir.mkdir(parents=True, exist_ok=True)
    (kdir / f"{args.kernel_name}.ipynb").write_text(json.dumps(nb, indent=1))
    (kdir / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{user}/{args.kernel_name}", "title": args.kernel_name, "code_file": f"{args.kernel_name}.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": not args.cpu,
        "enable_tpu": False, "enable_internet": False, **({} if args.cpu else {"machine_shape": "NvidiaTeslaT4"}),
        "dataset_sources": [f"{user}/knee-inference", f"{user}/knee-weights"],
        "competition_sources": [COMPETITION], "kernel_sources": [], "model_sources": [],
    }, indent=1))
    print(f"built {kdir} and notebooks/kaggle_inference.ipynb")

    if args.push_weights:
        push_dataset(wdir, f"{user}/knee-weights", args.message)
    if args.push_bundle:
        push_dataset(bdir, f"{user}/knee-inference", args.message)
    if args.push_kernel:
        cmd = ["kaggle", "kernels", "push", "-p", str(kdir)]
        subprocess.run(cmd if args.cpu else cmd + ["--accelerator", "NvidiaTeslaT4"], check=True)


if __name__ == "__main__":
    main()
