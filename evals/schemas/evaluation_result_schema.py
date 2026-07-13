"""Common result schema used by all evaluators."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _clamp_score(score: float) -> float:
    try:
        value = float(score)
    except (TypeError, ValueError):
        value = 0.0
    return max(0.0, min(1.0, value))


@dataclass
class EvaluationResult:
    eval_name: str
    case_id: str
    score: float
    passed: bool
    metrics: dict[str, Any] = field(default_factory=dict)
    errors: list[dict[str, Any] | str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "eval_name": self.eval_name,
            "case_id": self.case_id,
            "score": _clamp_score(self.score),
            "passed": bool(self.passed),
            "metrics": self.metrics,
            "errors": self.errors,
            "warnings": self.warnings,
            "details": self.details,
        }

    @classmethod
    def warning(
        cls,
        eval_name: str,
        case_id: str,
        message: str,
        metrics: dict[str, Any] | None = None,
    ) -> "EvaluationResult":
        return cls(
            eval_name=eval_name,
            case_id=case_id,
            score=0.0,
            passed=False,
            metrics=metrics or {},
            warnings=[message],
            details={},
        )
