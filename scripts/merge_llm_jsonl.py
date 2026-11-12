"""Merge LLM label JSONL files (e.g. from a local run and a Kaggle run) into one file.

Per study the last successful record wins; an llm_error record is kept only if no file has a
successful one. Records from other prompt versions are reported (and skipped with --same-prompt).

Example:
  python scripts/merge_llm_jsonl.py data/labels/llm_labels.jsonl results/kaggle/llm/labels/llm_labels.jsonl \
      --out data/labels/llm_labels.jsonl
"""

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--same-prompt", default=None, help="keep only this prompt_version")
    args = p.parse_args()
    best: dict[str, dict] = {}
    versions = Counter()
    for path in args.inputs:
        if not Path(path).exists():
            print(f"skip missing {path}")
            continue
        n = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                versions[rec.get("prompt_version")] += 1
                if args.same_prompt and rec.get("prompt_version") != args.same_prompt:
                    continue
                s = rec["StudyInstanceUID"]
                if s not in best or rec.get("status") == "ok" or best[s].get("status") != "ok":
                    best[s] = rec
                n += 1
        print(f"{path}: {n} records")
    out = Path(args.out)
    tmp = out.with_suffix(".merge.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for rec in best.values():
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tmp.replace(out)
    status = Counter(r.get("status") for r in best.values())
    print(f"wrote {out}: {len(best)} studies {dict(status)} | prompt versions seen {dict(versions)}")


if __name__ == "__main__":
    main()
