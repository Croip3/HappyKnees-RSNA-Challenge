"""Prompt builder for report -> per-finding state extraction.

The system prompt (definitions, rules, few-shot examples) is identical for every report, so
servers with prefix caching only prefill the report part.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from knee.constants import LABELS, LLM_STATES

RULES = """Rules:
1. "positive": the report states the finding is present and meets the threshold in its notes. The notes are binding: findings they list as negative or uncertain (e.g. low-grade injuries, small effusions) get that state.
2. "negative": the report explicitly states the finding is absent or the structure is normal/intact, e.g. "no evidence of tear", "intact ACL", "menisci normal", "no effusion". A normal description of the whole compartment or structure counts as negative for findings in it.
3. "uncertain": the finding is hedged, e.g. "possible", "suspected", "probable", "likely", "cannot be excluded", "questionable", "suggestive of", "versus".
4. "not_mentioned": the report says nothing about the finding or its structure. Never use "negative" for something that is simply not mentioned.
5. Post-operative structures: {postop}
6. The report may be in any language. Always answer in English. "evidence" is a short verbatim quote (at most 10 words) copied from the report in its original language, or "" for not_mentioned.
7. Output strict JSON only: one key per finding, in the given order, plus "postop_labels" (list of finding names, usually empty). No markdown, no explanation."""

FEW_SHOT = [
    {
        "report": "MRI right knee. The anterior cruciate ligament is intact. Possible small tear of the medial "
                  "meniscus posterior horn. Lateral meniscus normal. Moderate joint effusion. No fracture.",
        "answer": {
            "ACL": ("negative", "The anterior cruciate ligament is intact"),
            "Medial Meniscus": ("uncertain", "Possible small tear of the medial meniscus posterior horn"),
            "Lateral Meniscus": ("negative", "Lateral meniscus normal"),
            "Effusion": ("positive", "Moderate joint effusion"),
            "Fracture": ("negative", "No fracture"),
        },
    },
    {
        "report": "RM rodilla izquierda. Rotura completa del LCA. Edema óseo en cóndilo femoral lateral y meseta tibial "
                  "posterior compatible con contusión. No se puede descartar lesión del LCM. Sin quiste de Baker.",
        "answer": {
            "ACL": ("positive", "Rotura completa del LCA"),
            "MCL": ("uncertain", "No se puede descartar lesión del LCM"),
            "Contusion": ("positive", "Edema óseo en cóndilo femoral lateral y meseta tibial posterior"),
            "Baker's": ("negative", "Sin quiste de Baker"),
        },
    },
    {
        "report": "Z.n. VKB-Plastik, Transplantat intakt. Fortgeschrittene Chondropathie medial mit Osteophyten. "
                  "Femoropatellar unauffällig. Kein Erguss.",
        "answer": {
            "ACL": ("negative", "Z.n. VKB-Plastik, Transplantat intakt"),
            "Medial OA": ("positive", "Fortgeschrittene Chondropathie medial mit Osteophyten"),
            "PF OA": ("negative", "Femoropatellar unauffällig"),
            "Effusion": ("negative", "Kein Erguss"),
        },
        "postop": ["ACL"],
    },
]


def load_definitions(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        defs = yaml.safe_load(f)
    names = list(defs["labels"].keys())
    if names != LABELS:
        raise ValueError(f"label_definitions.yaml labels {names} do not match the submission columns {LABELS}")
    return defs


def _answer_json(answer: dict[str, tuple[str, str]], postop: list[str]) -> str:
    out = {}
    for label in LABELS:
        state, evidence = answer.get(label, ("not_mentioned", ""))
        out[label] = {"state": state, "evidence": evidence}
    out["postop_labels"] = postop
    return json.dumps(out, ensure_ascii=False)


def build_system_prompt(defs: dict[str, Any]) -> str:
    lines = [
        "You are an expert musculoskeletal radiologist. You read one knee MRI report and classify 12 findings.",
        "",
        "Findings (key: definition; notes):",
    ]
    for label in LABELS:
        d = defs["labels"][label]
        note = f"; {d['notes'].strip()}" if d.get("notes") else ""
        lines.append(f"- {label}: {d['definition'].strip()}{note}")
    lines += ["", "Each finding gets one state: " + ", ".join(LLM_STATES) + ".", ""]
    lines.append(RULES.format(postop=" ".join(defs.get("postoperative", {}).get("instruction", "").split())))
    lines += ["", "Examples (findings not listed in an example are not_mentioned):"]
    for ex in FEW_SHOT:
        lines += ["", f"Report: {ex['report']}", f"Answer: {_answer_json(ex['answer'], ex.get('postop', []))}"]
    return "\n".join(lines)


def build_user_prompt(report: str, max_chars: int = 6000) -> str:
    report = (report or "").strip()
    if len(report) > max_chars:
        half = max_chars // 2
        report = report[:half] + "\n[...]\n" + report[-half:]
    return f"Report:\n{report}\n\nAnswer with the JSON object only."


def build_repair_prompt(bad_output: str, error: str) -> str:
    return (f"Your previous answer was not valid ({error}). Previous answer:\n{bad_output[:3000]}\n\n"
            "Return the corrected JSON object only, with all 12 findings and \"postop_labels\".")


def json_schema() -> dict[str, Any]:
    """JSON schema of the expected answer (also used for constrained decoding)."""
    item = {
        "type": "object",
        "properties": {"state": {"type": "string", "enum": LLM_STATES}, "evidence": {"type": "string"}},
        "required": ["state", "evidence"],
        "additionalProperties": False,
    }
    props = {label: item for label in LABELS}
    props["postop_labels"] = {"type": "array", "items": {"type": "string", "enum": LABELS}}
    return {"type": "object", "properties": props, "required": LABELS + ["postop_labels"],
            "additionalProperties": False}


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    start, end = text.find("{"), text.rfind("}")
    return text[start:end + 1] if start >= 0 and end > start else text


def parse_answer(text: str) -> tuple[dict[str, Any] | None, str | None]:
    """Validate a model answer. Returns (parsed, None) or (None, error message)."""
    if text is None:
        return None, "empty answer"
    # Drop a reasoning block if the model emitted one despite think=false.
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    try:
        obj = json.loads(_strip_fences(text))
    except json.JSONDecodeError as e:
        return None, f"invalid JSON: {e}"
    if not isinstance(obj, dict):
        return None, "answer is not a JSON object"
    out: dict[str, Any] = {}
    for label in LABELS:
        item = obj.get(label)
        if not isinstance(item, dict):
            return None, f"missing or malformed entry for '{label}'"
        state = str(item.get("state", "")).strip().lower().replace(" ", "_").replace("-", "_")
        if state not in LLM_STATES:
            return None, f"invalid state '{item.get('state')}' for '{label}'"
        evidence = item.get("evidence", "")
        out[label] = {"state": state, "evidence": evidence if isinstance(evidence, str) else str(evidence)}
    postop = obj.get("postop_labels", [])
    out["postop_labels"] = [p for p in postop if p in LABELS] if isinstance(postop, list) else []
    return out, None
