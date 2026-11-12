"""Phase 2a: extract per-finding states from the radiology reports with an LLM (resumable JSONL).

Examples:
  python scripts/llm_extract.py --limit 50                       # throughput estimate + ETA
  python scripts/llm_extract.py                                  # all remaining reports
  python scripts/llm_extract.py --set llm.backend=openai --set llm.base_url=http://127.0.0.1:8080 \
      --set llm.model=qwen3.8-27b --set llm.concurrency=4
"""

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from knee.config import add_common_args, config_from_args, dump_config
from knee.labels.llm_extract import run_extraction
from knee.utils import get_logger


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__,
                                                     formatter_class=argparse.RawDescriptionHelpFormatter))
    parser.add_argument("--retry-errors", action="store_true", help="re-run studies marked llm_error")
    parser.add_argument("--studies", nargs="*", default=None, help="only these StudyInstanceUIDs")
    parser.add_argument("--studies-file", default=None, help="text file with one StudyInstanceUID per line")
    parser.add_argument("--reverse", action="store_true", help="process the study list from the end")
    parser.add_argument("--max-minutes", type=float, default=None, help="stop submitting after N minutes")
    args = parser.parse_args()
    cfg = config_from_args(args)
    labels_dir = Path(cfg.paths.labels_dir)
    log = get_logger("knee.llm", labels_dir / "llm_extract.log")
    dump_config(cfg, labels_dir / "llm_extract_config.yaml")
    studies = args.studies
    if args.studies_file:
        studies = [x.strip() for x in Path(args.studies_file).read_text().split() if x.strip()]
    if args.reverse and studies:
        studies = studies[::-1]
    run_extraction(cfg, log, limit=args.limit, overwrite=args.overwrite, retry_errors=args.retry_errors,
                   studies=studies, max_minutes=args.max_minutes)


if __name__ == "__main__":
    main()
