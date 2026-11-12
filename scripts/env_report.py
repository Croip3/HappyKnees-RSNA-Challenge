"""Print the runtime environment (package versions, GPU, CPU, RAM, disk, /kaggle/input)."""

import importlib
import json
import os
import shutil
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from knee.utils import device_info


def main() -> None:
    info = device_info()
    versions = {}
    for mod in ["numpy", "pandas", "torch", "torchvision", "timm", "pydicom", "cv2", "sklearn", "pylibjpeg",
                "gdcm", "iterstrat", "transformers", "vllm", "rich", "yaml", "langdetect"]:
        try:
            m = importlib.import_module(mod)
            versions[mod] = getattr(m, "__version__", "installed")
        except Exception:
            versions[mod] = None
    info["packages"] = versions
    info["cpu_count"] = os.cpu_count()
    try:
        with open("/proc/meminfo") as f:
            info["mem_total_gb"] = round(int(f.readline().split()[1]) / 2**20, 1)
    except OSError:
        pass
    for d in ["/kaggle/working", "/tmp", "."]:
        if Path(d).exists():
            info[f"disk_free_gb[{d}]"] = round(shutil.disk_usage(d).free / 2**30, 1)
    if Path("/kaggle/input").exists():
        info["kaggle_input"] = sorted(str(p) for p in Path("/kaggle/input").glob("*")) + sorted(
            str(p) for p in Path("/kaggle/input").glob("*/*"))[:40]
    info["executable"] = sys.executable
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
