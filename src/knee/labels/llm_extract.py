"""Report -> per-finding states with an LLM. Resumable: results are appended to a JSONL file.

Backends (cfg.llm.backend):
  ollama  - native Ollama API (/api/chat), think=false, JSON-schema constrained output
  openai  - any OpenAI-compatible server (llama.cpp `llama-server`, `vllm serve`, ...)
  vllm    - in-process vLLM (offline batching)
"""

from __future__ import annotations

import hashlib
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

from knee.constants import LABELS, REPORT, STUDY
from knee.labels import prompts
from knee.utils import fmt_seconds


class Backend:
    name = "base"
    concurrent = True

    def chat(self, messages: list[dict[str, str]]) -> tuple[str, dict[str, Any]]:
        raise NotImplementedError

    def chat_batch(self, batch: list[list[dict[str, str]]]) -> list[tuple[str, dict[str, Any]]]:
        return [self.chat(m) for m in batch]


class OllamaBackend(Backend):
    name = "ollama"

    def __init__(self, cfg, model: str):
        import requests

        self.session = requests.Session()
        self.url = cfg.llm.base_url.rstrip("/") + "/api/chat"
        self.model = model
        self.cfg = cfg.llm
        self.schema = prompts.json_schema()

    def chat(self, messages):
        body = {
            "model": self.model, "messages": messages, "stream": False, "think": bool(self.cfg.think),
            "format": self.schema,
            "options": {"temperature": self.cfg.temperature, "num_ctx": self.cfg.num_ctx,
                        "num_predict": self.cfg.max_tokens, "seed": 0},
        }
        r = self.session.post(self.url, json=body, timeout=self.cfg.timeout_s)
        r.raise_for_status()
        d = r.json()
        usage = {"prompt_tokens": d.get("prompt_eval_count"), "completion_tokens": d.get("eval_count")}
        return d.get("message", {}).get("content", ""), usage


class OpenAIBackend(Backend):
    name = "openai"

    def __init__(self, cfg, model: str):
        import requests

        self.session = requests.Session()
        base = cfg.llm.base_url.rstrip("/")
        self.url = base + ("/chat/completions" if base.endswith("/v1") else "/v1/chat/completions")
        self.model = model
        self.cfg = cfg.llm
        self.headers = {"Authorization": f"Bearer {cfg.llm.api_key}"}
        self.schema = prompts.json_schema()

    def chat(self, messages):
        body = {
            "model": self.model, "messages": messages, "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens, "seed": 0,
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": "knee_findings", "schema": self.schema, "strict": True}},
            "chat_template_kwargs": {"enable_thinking": bool(self.cfg.think)},
        }
        r = self.session.post(self.url, json=body, headers=self.headers, timeout=self.cfg.timeout_s)
        r.raise_for_status()
        d = r.json()
        usage = d.get("usage", {}) or {}
        timings = d.get("timings", {}) or {}
        if timings:
            usage = {**usage, "cached_tokens": timings.get("cache_n")}
        return d["choices"][0]["message"].get("content") or "", usage


class VLLMBackend(Backend):
    name = "vllm"
    concurrent = False

    def __init__(self, cfg, model: str):
        from vllm import LLM, SamplingParams

        v = cfg.llm.vllm
        kwargs = dict(model=model, tensor_parallel_size=v.tensor_parallel_size, max_model_len=v.max_model_len,
                      gpu_memory_utilization=v.gpu_memory_utilization, dtype=v.dtype, enable_prefix_caching=True,
                      limit_mm_per_prompt={"image": 0, "video": 0})
        if cfg.llm.quantization:
            kwargs["quantization"] = cfg.llm.quantization
        self.llm = LLM(**kwargs)
        schema = prompts.json_schema()
        try:  # vLLM >= 0.10.2
            from vllm.sampling_params import StructuredOutputsParams

            structured = {"structured_outputs": StructuredOutputsParams(json=schema)}
        except ImportError:
            from vllm.sampling_params import GuidedDecodingParams

            structured = {"guided_decoding": GuidedDecodingParams(json=schema)}
        self.params = SamplingParams(temperature=cfg.llm.temperature, max_tokens=cfg.llm.max_tokens, seed=0,
                                     **structured)
        self.think = bool(cfg.llm.think)

    def chat_batch(self, batch):
        outs = self.llm.chat(batch, self.params, use_tqdm=False,
                             chat_template_kwargs={"enable_thinking": self.think})
        return [(o.outputs[0].text, {"prompt_tokens": len(o.prompt_token_ids),
                                     "completion_tokens": len(o.outputs[0].token_ids)}) for o in outs]

    def chat(self, messages):
        return self.chat_batch([messages])[0]


def make_backend(cfg) -> Backend:
    model = cfg.llm.fallback_model if cfg.llm.use_fallback and cfg.llm.fallback_model else cfg.llm.model
    backends = {"ollama": OllamaBackend, "openai": OpenAIBackend, "vllm": VLLMBackend}
    if cfg.llm.backend not in backends:
        raise ValueError(f"unknown llm.backend '{cfg.llm.backend}' (choose from {list(backends)})")
    return backends[cfg.llm.backend](cfg, model)


def prompt_version(system_prompt: str, model: str) -> str:
    return hashlib.sha1((model + "\n" + system_prompt).encode()).hexdigest()[:10]


def load_results(path: str | Path) -> dict[str, dict[str, Any]]:
    """Latest record per study from the JSONL file (later lines win)."""
    path = Path(path)
    results: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return results
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # a partially written last line after a crash
            results[rec[STUDY]] = rec
    return results


def extract_one(backend: Backend, system_prompt: str, report: str, max_chars: int) -> dict[str, Any]:
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": prompts.build_user_prompt(report, max_chars)}]
    t0 = time.time()
    rec: dict[str, Any] = {"attempts": 0}
    text, usage, err = "", {}, None
    for attempt in range(2):
        rec["attempts"] = attempt + 1
        try:
            text, usage = backend.chat(messages)
        except Exception as e:  # noqa: BLE001 - network/server errors count as a failed attempt
            text, err = "", f"request failed: {type(e).__name__}: {e}"[:500]
            continue
        parsed, err = prompts.parse_answer(text)
        if parsed is not None:
            rec.update(status="ok", labels={k: parsed[k] for k in LABELS}, postop_labels=parsed["postop_labels"])
            break
        messages = messages + [{"role": "assistant", "content": text},
                               {"role": "user", "content": prompts.build_repair_prompt(text, err)}]
    if rec.get("status") != "ok":
        rec.update(status="llm_error", error=err, raw=text[:4000])
    rec["latency_s"] = round(time.time() - t0, 2)
    rec["usage"] = usage
    return rec


def order_studies(train: pd.DataFrame, seed: int = 0) -> list[str]:
    """Gold studies first (agreement can be checked early), then the rest in a fixed random order."""
    gold_mask = train[LABELS].notna().all(axis=1)
    gold = train.loc[gold_mask, STUDY].tolist()
    rest = train.loc[~gold_mask, STUDY].tolist()
    random.Random(seed).shuffle(rest)
    return gold + rest


def run_extraction(cfg, log, limit: int | None = None, overwrite: bool = False, retry_errors: bool = False,
                   studies: list[str] | None = None, max_minutes: float | None = None) -> Path:
    """Process reports not yet in the JSONL. With max_minutes, stop submitting new reports after that
    time (in-flight requests finish), so a time-limited session (e.g. Kaggle) ends cleanly."""
    labels_dir = Path(cfg.paths.labels_dir)
    labels_dir.mkdir(parents=True, exist_ok=True)
    out_path = labels_dir / cfg.llm.output_jsonl
    if overwrite and out_path.exists():
        bak = out_path.with_suffix(f".{time.strftime('%Y%m%d-%H%M%S')}.bak")
        out_path.rename(bak)
        log.info(f"--overwrite: moved existing results to {bak.name}")

    train = pd.read_csv(Path(cfg.paths.data_root) / "train.csv")
    reports = dict(zip(train[STUDY], train[REPORT].fillna("")))
    order = studies or order_studies(train)
    done = load_results(out_path)
    skip = {s for s, r in done.items() if r.get("status") == "ok" or not retry_errors}
    todo = [s for s in order if s not in skip]
    if limit is not None:
        todo = todo[:limit]
    defs = prompts.load_definitions(cfg.paths.label_definitions)
    system_prompt = prompts.build_system_prompt(defs)
    model = cfg.llm.fallback_model if cfg.llm.use_fallback and cfg.llm.fallback_model else cfg.llm.model
    version = prompt_version(system_prompt, model)
    log.info(f"LLM extraction: backend={cfg.llm.backend} model={model} prompt_version={version} "
             f"system_prompt_chars={len(system_prompt)}")
    log.info(f"studies: {len(order)} total, {len(done)} already in {out_path.name}, {len(todo)} to process now")
    if not todo:
        return out_path

    backend = make_backend(cfg)
    lock = threading.Lock()
    n_done, n_err, t_start = 0, 0, time.time()
    total_remaining_all = len([s for s in order if s not in skip])
    tok_out = 0

    def _write(study: str, rec: dict[str, Any]) -> None:
        nonlocal n_done, n_err, tok_out
        rec = {STUDY: study, "model": model, "backend": cfg.llm.backend, "prompt_version": version,
               "time": time.strftime("%Y-%m-%d %H:%M:%S"), **rec}
        with lock:
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_done += 1
            n_err += rec["status"] != "ok"
            tok_out += int((rec.get("usage") or {}).get("completion_tokens") or 0)
            if n_done % 10 == 0 or n_done == len(todo):
                el = time.time() - t_start
                rpm = n_done / el * 60
                eta_all = (total_remaining_all - n_done) / max(rpm, 1e-9) * 60
                log.info(f"[{n_done}/{len(todo)}] {rpm:.1f} reports/min | {tok_out / el:.0f} out tok/s | "
                         f"errors {n_err} | elapsed {fmt_seconds(el)} | ETA all remaining "
                         f"({total_remaining_all - n_done}) {fmt_seconds(eta_all)}")

    max_chars = cfg.llm.max_report_chars
    deadline = t_start + 60 * max_minutes if max_minutes else None
    if backend.concurrent:
        conc = int(cfg.llm.concurrency)
        pending = iter(todo)
        with ThreadPoolExecutor(max_workers=conc) as ex:
            futs: dict = {}
            while True:
                while len(futs) < 2 * conc and (deadline is None or time.time() < deadline):
                    s = next(pending, None)
                    if s is None:
                        break
                    futs[ex.submit(extract_one, backend, system_prompt, reports[s], max_chars)] = s
                if not futs:
                    break
                done = next(as_completed(futs))
                _write(futs.pop(done), done.result())
        if deadline is not None and time.time() >= deadline:
            log.info(f"max_minutes={max_minutes} reached: stopped submitting new reports")
    else:
        bs = 64
        for i in range(0, len(todo), bs):
            if deadline is not None and time.time() >= deadline:
                log.info(f"max_minutes={max_minutes} reached")
                break
            chunk = todo[i:i + bs]
            batch = [[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": prompts.build_user_prompt(reports[s], max_chars)}] for s in chunk]
            t0 = time.time()
            outs = backend.chat_batch(batch)
            for s, (text, usage) in zip(chunk, outs):
                parsed, err = prompts.parse_answer(text)
                if parsed is None:  # repair path, one request at a time
                    rec = extract_one(backend, system_prompt, reports[s], max_chars)
                else:
                    rec = {"status": "ok", "labels": {k: parsed[k] for k in LABELS},
                           "postop_labels": parsed["postop_labels"], "attempts": 1, "usage": usage,
                           "latency_s": round((time.time() - t0) / len(chunk), 2)}
                _write(s, rec)
    el = time.time() - t_start
    rpm = n_done / el * 60 if el > 0 else 0
    log.info(f"done: {n_done} reports in {fmt_seconds(el)} ({rpm:.1f} reports/min), {n_err} llm_error")
    remaining = len([s for s in order if s not in load_results(out_path)])
    if remaining:
        log.info(f"ETA for the remaining {remaining} reports at this rate: {fmt_seconds(remaining / max(rpm, 1e-9) * 60)}")
    return out_path
