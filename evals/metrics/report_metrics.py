"""Report structure metrics."""

from __future__ import annotations

from typing import Any

from evals.text_utils import contains_any, safe_divide, to_text


def report_completeness(report: dict[str, Any] | str, required_fields: list[str]) -> float:
    if isinstance(report, dict):
        hits = sum(1 for field in required_fields if report.get(field))
    else:
        text = to_text(report)
        hits = sum(1 for field in required_fields if contains_any(text, [field]))
    return safe_divide(hits, len(required_fields), default=0.0)


def has_uncertainty(text: str, uncertainty_keywords: list[str]) -> float:
    return 1.0 if contains_any(text, uncertainty_keywords) else 0.0
