"""Efficiency and cost evaluator."""

from __future__ import annotations

from typing import Any

from evals.evaluators.base import BaseEvaluator
from evals.metrics.scoring_metrics import weighted_score


class EfficiencyEvaluator(BaseEvaluator):
    eval_name = "efficiency"

    def evaluate_case(self, case: dict[str, Any]):
        output = self.output(case)
        telemetry = output.get("telemetry") or case.get("telemetry") or {}
        warnings: list[str] = []
        errors: list[dict[str, Any] | str] = []

        response_time = self._first_number(
            telemetry,
            ["response_time_seconds", "latency_seconds", "end_to_end_latency_seconds"],
            default=None,
        )
        if response_time is None:
            latency_ms = self._first_number(telemetry, ["latency_ms", "response_time_ms"], default=None)
            response_time = (latency_ms / 1000.0) if latency_ms is not None else 0.0
            warnings.append("missing response time telemetry") if latency_ms is None else None

        token_usage = telemetry.get("token_usage") or output.get("token_usage") or {}
        if isinstance(token_usage, dict):
            input_tokens = float(token_usage.get("input_tokens") or token_usage.get("prompt_tokens") or 0)
            output_tokens = float(token_usage.get("output_tokens") or token_usage.get("completion_tokens") or 0)
            total_tokens = float(token_usage.get("total_tokens") or input_tokens + output_tokens)
        else:
            input_tokens = 0.0
            output_tokens = 0.0
            total_tokens = float(token_usage or 0)

        tool_calls = output.get("tool_calls") or case.get("tool_calls") or []
        retrieval_latency = self._first_number(telemetry, ["retrieval_latency", "retrieval_latency_seconds"], default=0.0)
        report_latency = self._first_number(telemetry, ["report_generation_latency", "report_generation_latency_seconds"], default=0.0)
        turns = float(output.get("turn_count") or telemetry.get("turn_count") or len(output.get("asked_questions", []) or []))
        estimated_cost = self._estimate_cost(input_tokens, output_tokens)

        metrics = {
            "avg_response_time": response_time,
            "p50_response_time": response_time,
            "p95_response_time": response_time,
            "avg_turns": turns,
            "avg_token_usage": total_tokens,
            "avg_tool_calls": float(len(tool_calls)),
            "retrieval_latency": retrieval_latency,
            "report_generation_latency": report_latency,
            "estimated_cost_per_case": estimated_cost,
        }
        max_time = float(self.config.get("thresholds", {}).get("max_response_time_seconds", 20.0))
        max_cost = float(self.config.get("thresholds", {}).get("max_estimated_cost_per_case", 0.5))
        time_score = 1.0 if response_time <= max_time else max(0.0, 1.0 - (response_time - max_time) / max_time)
        cost_score = 1.0 if estimated_cost <= max_cost else max(0.0, 1.0 - (estimated_cost - max_cost) / max_cost)
        token_score = 1.0 if total_tokens <= 8000 else max(0.0, 1.0 - (total_tokens - 8000) / 8000)
        score = weighted_score(
            {
                "response_time_score": time_score,
                "cost_score": cost_score,
                "token_score": token_score,
            },
            self.config.get("weights", {}).get("efficiency", {}),
        )
        return self.result(case, score, metrics, errors, warnings, {"telemetry": telemetry})

    def _first_number(self, telemetry: dict[str, Any], keys: list[str], default: float | None) -> float | None:
        for key in keys:
            if telemetry.get(key) is not None:
                try:
                    return float(telemetry[key])
                except (TypeError, ValueError):
                    return default
        return default

    def _estimate_cost(self, input_tokens: float, output_tokens: float) -> float:
        cost_cfg = self.config.get("cost", {})
        input_price = float(cost_cfg.get("input_token_price_per_1k", 0.0))
        output_price = float(cost_cfg.get("output_token_price_per_1k", 0.0))
        return (input_tokens / 1000.0) * input_price + (output_tokens / 1000.0) * output_price
