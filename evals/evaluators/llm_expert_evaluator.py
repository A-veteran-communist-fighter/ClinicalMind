"""LLM-as-medical-expert evaluator."""

from __future__ import annotations

from typing import Any

from evals.evaluators.base import BaseEvaluator
from evals.llm_judge import JudgeUnavailable, load_prompt
from evals.metrics.scoring_metrics import normalize_1_to_5
from evals.text_utils import mean


DIMENSION_KEYS = [
    "interview_completeness",
    "differential_diagnosis_reasonableness",
    "evidence_usage_reasonableness",
    "health_plan_safety",
    "clarity",
]


class LLMExpertEvaluator(BaseEvaluator):
    eval_name = "llm_expert"

    def evaluate_case(self, case: dict[str, Any]):
        text = self.output_text(case)
        warnings: list[str] = []
        errors: list[dict[str, Any] | str] = []
        if not text:
            warnings.append("missing agent output")

        expert_cfg = self.config.get("llm_expert", {})
        prompt_files = expert_cfg.get("prompt_files") or ["expert_judge_prompt.txt"]
        num_samples = max(1, int(expert_cfg.get("num_samples", 1)))

        judge_rows: list[dict[str, Any]] = []
        for prompt_file in prompt_files:
            prompt = load_prompt(prompt_file)
            for sample_idx in range(num_samples):
                try:
                    row = self.judge_client.judge_json(
                        prompt,
                        {
                            "case_json": case,
                            "system_output": text,
                            "sample_index": sample_idx,
                        },
                    )
                    judge_rows.append(row)
                except JudgeUnavailable as exc:
                    warnings.append(str(exc))
                    break
                except Exception as exc:
                    warnings.append(f"LLM expert judge failed: {exc}")

        if not judge_rows:
            metrics = {
                "average_score_1_to_5": 0.0,
                "pass_rate": 0.0,
                "serious_error_rate": 0.0,
                "revision_required_rate": 1.0,
            }
            return self.result(
                case,
                0.0,
                metrics,
                errors,
                warnings,
                {"judge_results": []},
                threshold=normalize_1_to_5(self.config.get("thresholds", {}).get("expert_pass_score", 3.5)),
            )

        normalized_rows = [self._normalize_judge_row(row) for row in judge_rows]
        dimension_avgs = {
            key: mean([row["dimensions"].get(key, 0.0) for row in normalized_rows])
            for key in DIMENSION_KEYS
        }
        average_score = mean(dimension_avgs.values())
        pass_threshold = float(self.config.get("thresholds", {}).get("expert_pass_score", 3.5))
        pass_rate = mean([1.0 if row["average_score"] >= pass_threshold else 0.0 for row in normalized_rows])
        serious_error_rate = mean([1.0 if row["dangerous_advice"] else 0.0 for row in normalized_rows])
        revision_required_rate = mean([1.0 if row["needs_revision"] else 0.0 for row in normalized_rows])

        for row in normalized_rows:
            if row["dangerous_advice"]:
                errors.append({"type": "expert_judge_dangerous_advice", "comment": row.get("comment", "")})

        metrics = {
            **{f"{key}_avg": value for key, value in dimension_avgs.items()},
            "average_score_1_to_5": average_score,
            "pass_rate": pass_rate,
            "serious_error_rate": serious_error_rate,
            "revision_required_rate": revision_required_rate,
        }
        score = normalize_1_to_5(average_score)
        return self.result(
            case,
            score,
            metrics,
            errors,
            warnings,
            {"judge_results": normalized_rows},
            threshold=normalize_1_to_5(pass_threshold),
        )

    def _normalize_judge_row(self, row: dict[str, Any]) -> dict[str, Any]:
        dimensions = {}
        raw_dimensions = row.get("dimensions") if isinstance(row.get("dimensions"), dict) else row
        for key in DIMENSION_KEYS:
            dimensions[key] = max(1.0, min(5.0, float(raw_dimensions.get(key, row.get(key, 1.0)))))
        average_score = mean(dimensions.values(), default=1.0)
        return {
            "dimensions": dimensions,
            "average_score": average_score,
            "dangerous_advice": bool(row.get("dangerous_advice") or row.get("has_dangerous_advice")),
            "needs_revision": bool(row.get("needs_revision") or row.get("requires_revision")),
            "comment": row.get("expert_comment") or row.get("comment") or "",
            "raw": row,
        }
