"""Interview flow quality evaluator."""

from __future__ import annotations

from typing import Any

from evals.evaluators.base import BaseEvaluator
from evals.metrics.scoring_metrics import weighted_score
from evals.text_utils import (
    collect_questions,
    contains_any,
    coverage_against_expected,
    safe_divide,
    semantic_match,
)


class InterviewQualityEvaluator(BaseEvaluator):
    eval_name = "interview_quality"

    def evaluate_case(self, case: dict[str, Any]):
        output = self.output(case)
        expected = self.expected(case)
        questions = collect_questions(output)
        warnings: list[str] = []
        errors: list[dict[str, Any] | str] = []

        if not questions:
            warnings.append("missing assistant questions or conversation log")

        must_ask = expected.get("must_ask", []) or []
        coverage, hit, missed = coverage_against_expected(questions, must_ask)

        redundant = self._redundant_question_rate(questions)
        irrelevant = self._irrelevant_question_rate(questions, case)
        turns = output.get("turn_count")
        if turns is None:
            turns = len(
                [
                    m for m in output.get("conversation", []) or []
                    if isinstance(m, dict) and m.get("role") == "assistant"
                ]
            )
        average_turns = float(turns or len(questions) or 0)

        red_flags = expected.get("red_flags", []) or []
        triage_trigger = 1.0
        if red_flags:
            triage_trigger = 1.0 if contains_any(
                self.output_text(case),
                ["急诊", "立即就医", "及时就医", "尽快就医", "高危", "分流", "呼吸困难", "胸痛"],
            ) else 0.0

        consistency = max(0.0, 1.0 - redundant - irrelevant)
        max_turns = float(self.config.get("thresholds", {}).get("max_reasonable_turns", 8))
        turn_score = (
            1.0
            if average_turns <= max_turns
            else max(0.0, 1.0 - (average_turns - max_turns) / max_turns)
        )

        metrics = {
            "must_ask_coverage": coverage,
            "key_question_hit_rate": coverage,
            "redundant_question_rate": redundant,
            "irrelevant_question_rate": irrelevant,
            "average_turns": average_turns,
            "red_flag_triage_trigger_rate": triage_trigger,
            "conversation_consistency_score": consistency,
        }
        score = weighted_score(
            {
                "must_ask_coverage": coverage,
                "redundant_question_rate": 1.0 - redundant,
                "irrelevant_question_rate": 1.0 - irrelevant,
                "turn_score": turn_score,
                "red_flag_triage_trigger_rate": triage_trigger,
                "conversation_consistency_score": consistency,
            },
            self.config.get("weights", {}).get("interview_quality", {}),
        )
        details = {"questions": questions, "hit_must_ask": hit, "missed_must_ask": missed}
        return self.result(case, score, metrics, errors, warnings, details)

    def _redundant_question_rate(self, questions: list[str]) -> float:
        if len(questions) < 2:
            return 0.0
        redundant = 0
        seen: list[str] = []
        for question in questions:
            if any(semantic_match(question, prev, threshold=0.5) for prev in seen):
                redundant += 1
            seen.append(question)
        return safe_divide(redundant, len(questions))

    def _irrelevant_question_rate(self, questions: list[str], case: dict[str, Any]) -> float:
        if not questions:
            return 0.0
        anchors = [
            case.get("chief_complaint", ""),
            "症状",
            "疼痛",
            "发热",
            "咳嗽",
            "咳痰",
            "呼吸",
            "用药",
            "过敏",
            "病史",
            "多久",
            "是否",
            "体温",
        ]
        irrelevant = sum(1 for q in questions if not contains_any(q, anchors))
        return safe_divide(irrelevant, len(questions))
