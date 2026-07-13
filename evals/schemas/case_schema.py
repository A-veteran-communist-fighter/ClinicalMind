"""Helpers for loading and normalizing evaluation cases."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def ensure_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def get_case_id(case: dict[str, Any], fallback: str = "unknown_case") -> str:
    return str(case.get("case_id") or case.get("query_id") or fallback)


def get_agent_output(case: dict[str, Any]) -> dict[str, Any]:
    output = case.get("agent_output") or case.get("output") or {}
    return output if isinstance(output, dict) else {"final_answer": output}


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(data, dict):
                raise ValueError(f"Each JSONL row must be an object: {path}:{line_no}")
            records.append(data)
    return records


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def iter_cases(cases: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    for idx, case in enumerate(cases, start=1):
        if isinstance(case, dict):
            case.setdefault("case_id", get_case_id(case, f"case_{idx:03d}"))
            yield case
