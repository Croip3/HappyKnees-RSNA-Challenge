"""Publish the training notebook (notebooks/kaggle_training.ipynb) and its inputs to Kaggle.

Builds (under build/):
  kaggle_datasets/knee-src/      package source (src, configs, scripts) + marker knee_src.json
  kaggle_datasets/knee-labels/   LLM label JSONL + marker knee_labels.json
  kaggle/knee-train/             notebook + kernel-metadata.json (GPU, internet off); the output of
                                 --features-kernel is attached so precomputed features are reused

Parameters can be injected without editing the notebook (papermill style: a cell after the
"parameters" cell), e.g. for a quick check:
  python scripts/push_training_notebook.py --push-src --push-labels --push-kernel \
      --param RUN_FOLDS=[0] --param EPOCHS=2 --param RUN_NAME='"kaggle_check"'
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from make_kaggle_bundle import COMPETITION, ROOT, kaggle_username, push_dataset

NOTEBOOK = ROOT / "notebooks" / "kaggle_training.ipynb"


def inject_parameters(nb: dict, params: list[str]) -> dict:
    if not params:
        return nb
    idx = next((i for i, c in enumerate(nb["cells"]) if "parameters" in c.get("metadata", {}).get("tags", [])), None)
    if idx is None:
        raise ValueError("notebook has no cell tagged 'parameters'")
    src = "# Injected parameters\n" + "\n".join(params) + "\n"
    cell = {"cell_type": "code", "id": "injected-parameters", "metadata": {"tags": ["injected-parameters"]},
            "execution_count": None, "outputs": [], "source": src.splitlines(keepends=True)}
    nb = dict(nb)
    nb["cells"] = nb["cells"][:idx + 1] + [cell] + nb["cells"][idx + 1:]
    return nb


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--user", default=None)
    p.add_argument("--labels", default="data/labels/llm_labels.jsonl", help="LLM label JSONL to publish")
    p.add_argument("--features-kernel", default="knee-full-prep",
                   help="kernel whose output holds precomputed features ('' = none)")
    p.add_argument("--kernel-name", default="knee-train")
    p.add_argument("--param", action="append", default=[], metavar="NAME=VALUE",
                   help="python assignment injected after the parameters cell (repeatable)")
    p.add_argument("--push-src", action="store_true")
    p.add_argument("--push-labels", action="store_true")
    p.add_argument("--push-kernel", action="store_true")
    p.add_argument("--message", default="update")
    args = p.parse_args()
    user = args.user or kaggle_username()
    build = ROOT / "build"

    sdir = build / "kaggle_datasets" / "knee-src"
    if sdir.exists():
        shutil.rmtree(sdir)
    for sub in ["src", "configs", "scripts"]:
        shutil.copytree(ROOT / sub, sdir / sub, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (sdir / "knee_src.json").write_text(json.dumps({"package": "knee"}))
    (sdir / "dataset-metadata.json").write_text(json.dumps(
        {"title": "knee-src", "id": f"{user}/knee-src", "licenses": [{"name": "CC0-1.0"}]}, indent=1))

    ldir = build / "kaggle_datasets" / "knee-labels"
    labels = (ROOT / args.labels) if not Path(args.labels).is_absolute() else Path(args.labels)
    if ldir.exists():
        shutil.rmtree(ldir)
    ldir.mkdir(parents=True)
    shutil.copy2(labels, ldir / "llm_labels.jsonl")
    n = sum(1 for _ in open(labels, encoding="utf-8"))
    (ldir / "knee_labels.json").write_text(json.dumps({"source": str(args.labels), "records": n}))
    (ldir / "dataset-metadata.json").write_text(json.dumps(
        {"title": "knee-labels", "id": f"{user}/knee-labels", "licenses": [{"name": "CC0-1.0"}]}, indent=1))
    print(f"built {sdir} and {ldir} ({n} label records)")

    kdir = build / "kaggle" / args.kernel_name
    if kdir.exists():
        shutil.rmtree(kdir)
    kdir.mkdir(parents=True)
    nb = inject_parameters(json.loads(NOTEBOOK.read_text()), args.param)
    (kdir / f"{args.kernel_name}.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    (kdir / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{user}/{args.kernel_name}", "title": args.kernel_name, "code_file": f"{args.kernel_name}.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True,
        "enable_tpu": False, "enable_internet": False, "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [f"{user}/knee-src", f"{user}/knee-weights", f"{user}/knee-labels"],
        "competition_sources": [COMPETITION],
        "kernel_sources": [f"{user}/{args.features_kernel}"] if args.features_kernel else [],
        "model_sources": [],
    }, indent=1))
    print(f"built {kdir}" + (f" with injected parameters {args.param}" if args.param else ""))

    if args.push_src:
        push_dataset(sdir, f"{user}/knee-src", args.message)
    if args.push_labels:
        push_dataset(ldir, f"{user}/knee-labels", args.message)
    if args.push_kernel:
        subprocess.run(["kaggle", "kernels", "push", "-p", str(kdir), "--accelerator", "NvidiaTeslaT4"], check=True)


if __name__ == "__main__":
    main()
