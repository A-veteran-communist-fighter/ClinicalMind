"""Tool calling evaluator with structured argument matching."""

from __future__ import annotations

import json
from typing import Any

from evals.evaluators.base import BaseEvaluator
from evals.metrics.scoring_metrics import weighted_score
from evals.text_utils import contains_any, safe_divide, to_text


class ToolCallEvaluator(BaseEvaluator):
    eval_name = "tool_call"

    def evaluate_case(self, case: dict[str, Any]):
        output = self.output(case)
        expected_calls = case.get("expected_tool_calls") or self.expected(case).get("expected_tool_calls") or []
        actual_calls = output.get("tool_calls") or case.get("tool_calls") or []
        warnings: list[str] = []
        errors: list[dict[str, Any] | str] = []

        if not expected_calls and not actual_calls:
            metrics = {
                "tool_selection_accuracy": 1.0,
                "function_name_accuracy": 1.0,
                "parameter_accuracy": 1.0,
                "tool_sequence_accuracy": 1.0,
                "no_tool_needed_accuracy": 1.0,
                "tool_result_utilization_rate": 1.0,
                "tool_failure_recovery_rate": 1.0,
            }
            return self.result(case, 1.0, metrics, warnings=warnings, details={"matches": []})

        if expected_calls and not actual_calls:
            warnings.append("missing actual tool_calls")

        matches = [match_tool_call(exp, actual_calls) for exp in expected_calls]

        selection_accuracy = safe_divide(
            sum(1 for item in matches if item["tool_selected"]),
            len(expected_calls),
            default=1.0,
        )
        function_accuracy = safe_divide(
            sum(1 for item in matches if item["function_name_ok"]),
            len(expected_calls),
            default=1.0,
        )
        parameter_accuracy = safe_divide(
            sum(item["parameter_score"] for item in matches),
            len(expected_calls),
            default=1.0,
        )
        sequence_accuracy = self._sequence_accuracy(expected_calls, actual_calls)
        no_tool_needed_accuracy = 1.0 if expected_calls else float(not actual_calls)
        utilization = self._tool_result_utilization_rate(output)
        recovery = self._tool_failure_recovery_rate(output)

        for idx, match in enumerate(matches):
            if not match["matched"]:
                errors.append({"type": "tool_call_mismatch", "expected_index": idx, "detail": match})

        metrics = {
            "tool_selection_accuracy": selection_accuracy,
            "function_name_accuracy": function_accuracy,
            "parameter_accuracy": parameter_accuracy,
            "tool_sequence_accuracy": sequence_accuracy,
            "no_tool_needed_accuracy": no_tool_needed_accuracy,
            "tool_result_utilization_rate": utilization,
            "tool_failure_recovery_rate": recovery,
        }
        score = weighted_score(metrics, self.config.get("weights", {}).get("tool_call", {}))
        return self.result(case, score, metrics, errors, warnings, {"matches": matches})

    def _sequence_accuracy(self, expected_calls: list[dict[str, Any]], actual_calls: list[dict[str, Any]]) -> float:
        if not expected_calls:
            return 1.0
        ordered_expected = sorted(
            expected_calls,
            key=lambda x: x.get("call_order", expected_calls.index(x)),
        )
        expected_names = [_tool_name(x) for x in ordered_expected]
        actual_names = [_tool_name(x) for x in actual_calls[: len(expected_names)]]
        correct = sum(1 for exp, act in zip(expected_names, actual_names) if exp == act)
        return safe_divide(correct, len(expected_names))

    def _tool_result_utilization_rate(self, output: dict[str, Any]) -> float:
        results = output.get("tool_results") or []
        if not results:
            return 1.0
        answer = to_text(output.get("final_answer") or output.get("answer") or output)
        used = 0
        for result in results:
            content = to_text(result.get("content") if isinstance(result, dict) else result)
            tokens = [content[:20], content[-20:]]
            if contains_any(answer, [t for t in tokens if len(t) >= 4]):
                used += 1
        return safe_divide(used, len(results))

    def _tool_failure_recovery_rate(self, output: dict[str, Any]) -> float:
        failures = [
            r for r in output.get("tool_results", []) or []
            if isinstance(r, dict) and (r.get("error") or str(r.get("status", "")).lower() == "failed")
        ]
        if not failures:
            return 1.0
        answer = to_text(output.get("final_answer") or output.get("answer") or "")
        return 1.0 if contains_any(
            answer,
            ["工具失败", "未检索到", "无法获取", "稍后重试", "建议线下", "换用其他信息来源"],
        ) else 0.0


def _tool_name(call: dict[str, Any]) -> str:
    return str(call.get("tool_name") or call.get("name") or call.get("function") or call.get("function_name") or "")


def _args(call: dict[str, Any]) -> dict[str, Any]:
    args = call.get("arguments") or call.get("args") or call.get("parameters") or {}
    if isinstance(args, str):
        try:
            parsed = json.loads(args)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return args if isinstance(args, dict) else {}


def match_tool_call(expected: dict[str, Any], actual_calls: list[dict[str, Any]]) -> dict[str, Any]:
    expected_name = _tool_name(expected)
    candidate = None
    for call in actual_calls:
        if _tool_name(call) == expected_name:
            candidate = call
            break
    if candidate is None and actual_calls:
        candidate = actual_calls[0]

    if candidate is None:
        return {
            "matched": False,
            "tool_selected": False,
            "function_name_ok": False,
            "parameter_score": 0.0,
            "missing_args": list((expected.get("required_args") or {}).keys()),
            "unexpected_args": [],
        }

    function_ok = _tool_name(candidate) == expected_name
    expected_required = expected.get("required_args") or {}
    expected_optional = expected.get("optional_args") or {}
    candidate_args = _args(candidate)
    arg_score, missing, mismatched = match_arguments(expected_required, expected_optional, candidate_args)
    matched = function_ok and arg_score >= 0.999
    return {
        "matched": matched,
        "tool_selected": function_ok,
        "function_name_ok": function_ok,
        "parameter_score": arg_score,
        "missing_args": missing,
        "mismatched_args": mismatched,
        "actual_tool_name": _tool_name(candidate),
    }


def match_arguments(
    required_args: dict[str, Any],
    optional_args: dict[str, Any],
    actual_args: dict[str, Any],
) -> tuple[float, list[str], list[dict[str, Any]]]:
    if not required_args and not optional_args:
        return 1.0, [], []

    missing: list[str] = []
    mismatched: list[dict[str, Any]] = []
    total = len(required_args) + len(optional_args)
    correct = 0

    for key, expected_value in required_args.items():
        if key not in actual_args:
            missing.append(key)
            continue
        if _value_matches(expected_value, actual_args[key]):
            correct += 1
        else:
            mismatched.append({"arg": key, "expected": expected_value, "actual": actual_args[key]})

    for key, expected_value in optional_args.items():
        if key not in actual_args:
            total -= 1
            continue
        if _value_matches(expected_value, actual_args[key]):
            correct += 1
        else:
            mismatched.append({"arg": key, "expected": expected_value, "actual": actual_args[key]})

    return safe_divide(correct, total, default=1.0), missing, mismatched


def _value_matches(expected: Any, actual: Any) -> bool:
    if isinstance(expected, dict) and isinstance(actual, dict):
        return all(k in actual and _value_matches(v, actual[k]) for k, v in expected.items())
    if isinstance(expected, list) and isinstance(actual, list):
        return all(any(_value_matches(e, a) for a in actual) for e in expected)
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float, str)):
        try:
            return abs(float(expected) - float(actual)) <= 1e-9
        except (TypeError, ValueError):
            return False
    return str(expected).strip().lower() == str(actual).strip().lower()
