"""Base evaluator implementation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evals.llm_judge import BaseJudgeClient, make_judge_client
from evals.schemas.case_schema import get_agent_output, get_case_id
from evals.schemas.evaluation_result_schema import EvaluationResult
from evals.text_utils import collect_output_text


DEFAULT_CONFIG_PATH = Path(__file__).parents[1] / "config" / "eval_config.json"


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


class BaseEvaluator:
    eval_name = "base"

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        judge_client: BaseJudgeClient | None = None,
    ) -> None:
        self.config = config or load_config()
        self.judge_client = judge_client or make_judge_client(self.config)

    @property
    def default_threshold(self) -> float:
        return float(self.config.get("thresholds", {}).get("default_pass_score", 0.75))

    def evaluate_case(self, case: dict[str, Any]) -> EvaluationResult:
        raise NotImplementedError

    def evaluate_many(self, cases: list[dict[str, Any]]) -> list[EvaluationResult]:
        return [self.evaluate_case(case) for case in cases]

    def case_id(self, case: dict[str, Any]) -> str:
        return get_case_id(case)

    def output(self, case: dict[str, Any]) -> dict[str, Any]:
        return get_agent_output(case)

    def output_text(self, case: dict[str, Any]) -> str:
        return collect_output_text(self.output(case))

    def expected(self, case: dict[str, Any]) -> dict[str, Any]:
        expected = case.get("expected") or {}
        return expected if isinstance(expected, dict) else {}

    def patient_profile(self, case: dict[str, Any]) -> dict[str, Any]:
        profile = case.get("patient_profile") or {}
        return profile if isinstance(profile, dict) else {}

    def result(
        self,
        case: dict[str, Any],
        score: float,
        metrics: dict[str, Any],
        errors: list[dict[str, Any] | str] | None = None,
        warnings: list[str] | None = None,
        details: dict[str, Any] | None = None,
        threshold: float | None = None,
    ) -> EvaluationResult:
        threshold = self.default_threshold if threshold is None else threshold
        return EvaluationResult(
            eval_name=self.eval_name,
            case_id=self.case_id(case),
            score=max(0.0, min(1.0, float(score))),
            passed=score >= threshold and not errors,
            metrics=metrics,
            errors=errors or [],
            warnings=warnings or [],
            details=details or {},
        )
