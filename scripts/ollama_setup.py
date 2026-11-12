"""Install and start Ollama (e.g. on a Kaggle GPU machine with internet), pull a model, warm it up.

The server keeps running after this script exits, so the next command can use it:
  python scripts/ollama_setup.py --model qwen3.8:27b --parallel 8
  python scripts/llm_extract.py --set llm.concurrency=8 ...
"""

import argparse
import json
import os
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path


def _get(url: str, timeout: float = 5.0):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _post(url: str, body: dict, timeout: float = 900.0):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="qwen3.8:27b")
    p.add_argument("--parallel", type=int, default=8)
    p.add_argument("--ctx", type=int, default=8192, help="context per request (must match llm.num_ctx)")
    p.add_argument("--log", default="/kaggle/working/ollama_serve.log" if Path("/kaggle/working").exists()
                   else "ollama_serve.log")
    args = p.parse_args()

    if not shutil.which("ollama"):
        if not shutil.which("zstd"):  # the installer needs zstd; the Kaggle image does not ship it
            subprocess.run("apt-get -qq update && apt-get -qq install -y zstd", shell=True, check=False)
        t0 = time.time()
        subprocess.run("curl -fsSL https://ollama.com/install.sh | sh", shell=True, check=True)
        print(f"installed ollama in {time.time() - t0:.0f}s", flush=True)
    env = dict(os.environ, OLLAMA_NUM_PARALLEL=str(args.parallel), OLLAMA_FLASH_ATTENTION="1",
               OLLAMA_KV_CACHE_TYPE="q8_0", OLLAMA_KEEP_ALIVE="24h", OLLAMA_CONTEXT_LENGTH=str(args.ctx),
               OLLAMA_SCHED_SPREAD="1", OLLAMA_MAX_QUEUE="1024")
    log = open(args.log, "w")
    subprocess.Popen(["ollama", "serve"], env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    for _ in range(120):
        try:
            print("ollama version:", _get("http://127.0.0.1:11434/api/version"), flush=True)
            break
        except Exception:
            time.sleep(1)
    else:
        raise RuntimeError("ollama serve did not start; see " + args.log)
    t0 = time.time()
    subprocess.run(["ollama", "pull", args.model], check=True)
    print(f"pulled {args.model} in {time.time() - t0:.0f}s", flush=True)
    t0 = time.time()
    out = _post("http://127.0.0.1:11434/api/chat", {
        "model": args.model, "stream": False, "think": False,
        "messages": [{"role": "user", "content": "Reply with the word ready."}],
        "options": {"temperature": 0, "num_ctx": args.ctx, "num_predict": 8}})
    print(f"warm-up answer {out.get('message', {}).get('content')!r} in {time.time() - t0:.0f}s", flush=True)
    print(subprocess.run(["ollama", "ps"], capture_output=True, text=True).stdout, flush=True)
    print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total", "--format=csv"],
                         capture_output=True, text=True).stdout, flush=True)


if __name__ == "__main__":
    main()
