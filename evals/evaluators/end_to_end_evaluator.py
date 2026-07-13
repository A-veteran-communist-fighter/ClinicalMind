"""End-to-end case task completion evaluator."""

from __future__ import annotations

import re
from typing import Any

from evals.evaluators.base import BaseEvaluator
from evals.metrics.report_metrics import report_completeness
from evals.metrics.scoring_metrics import weighted_score
from evals.text_utils import (
    collect_questions,
    contains_any,
    coverage_against_expected,
    extract_citations,
    filter_negated_safety_snippets,
    regex_snippets,
    to_text,
)


class EndToEndEvaluator(BaseEvaluator):
    eval_name = "end_to_end"

    def evaluate_case(self, case: dict[str, Any]):
        output = self.output(case)
        text = self.output_text(case)
        expected = self.expected(case)
        warnings: list[str] = []
        errors: list[dict[str, Any] | str] = []

        if not text:
            warnings.append("missing agent output")

        report = output.get("diagnosis_report") or output.get("report") or ""
        plan = output.get("health_plan") or output.get("treatment_plan") or output.get("plan") or ""
        questions = collect_questions(output)
        retrieval = output.get("retrieved_evidence") or output.get("retrieval_results") or []

        flow_components = {
            "interview": bool(questions or output.get("collected_info")),
            "diagnosis_report": bool(report),
            "health_plan": bool(plan),
            "safety_notice": self._has_safety_notice(text, expected),
        }
        case_completion_rate = sum(flow_components.values()) / len(flow_components)

        required_info_coverage, hit_info, missed_info = coverage_against_expected(
            questions + [text], expected.get("must_ask", []) or []
        )
        report_score = report_completeness(
            report,
            ["primary_diagnosis", "differential_diagnoses", "key_findings", "recommended_actions", "disclaimer"],
        )
        evidence_usage_rate = self._evidence_usage_rate(output, retrieval)
        context_integration_score, context_hits, context_missed = self._context_integration_score(case, text)
        safe_output_rate = self._safe_output_rate(text, expected, errors)

        metrics = {
            "case_completion_rate": case_completion_rate,
            "required_info_coverage": required_info_coverage,
            "report_completeness": report_score,
            "evidence_usage_rate": evidence_usage_rate,
            "context_integration_score": context_integration_score,
            "safe_output_rate": safe_output_rate,
        }
        score = weighted_score(metrics, self.config.get("weights", {}).get("end_to_end", {}))
        details = {
            "flow_components": flow_components,
            "hit_required_info": hit_info,
            "missed_required_info": missed_info,
            "context_hits": context_hits,
            "context_missed": context_missed,
        }
        return self.result(case, score, metrics, errors, warnings, details)

    def _has_safety_notice(self, text: str, expected: dict[str, Any]) -> bool:
        cfg = self.config.get("safety", {})
        required = expected.get("required_safety_behavior", []) or []
        return contains_any(text, cfg.get("disclaimer_keywords", []) + cfg.get("urgent_care_keywords", []) + required)

    def _evidence_usage_rate(self, output: dict[str, Any], retrieval: list[dict[str, Any]]) -> float:
        answer = to_text(output)
        citations = extract_citations(answer)
        if citations:
            return 1.0
        if not retrieval:
            return 0.0
        doc_ids = [
            to_text(item.get("doc_id") or item.get("id") or item.get("source_id"))
            for item in retrieval
            if isinstance(item, dict)
        ]
        return 1.0 if any(doc_id and doc_id in answer for doc_id in doc_ids) else 0.0

    def _context_integration_score(self, case: dict[str, Any], text: str) -> tuple[float, list[str], list[str]]:
        targets: list[str] = []
        targets.append(to_text(case.get("chief_complaint", "")))
        history = case.get("history") or {}
        if isinstance(history, dict):
            targets.extend(to_text(v) for v in history.values() if v)
        profile = case.get("patient_profile") or {}
        if isinstance(profile, dict):
            for key in ("allergies", "chronic_diseases", "current_medications", "preferences"):
                values = profile.get(key) or []
                targets.extend(to_text(v) for v in values if v)
        return coverage_against_expected([text], targets, threshold=0.30)

    def _safe_output_rate(self, text: str, expected: dict[str, Any], errors: list[dict[str, Any] | str]) -> float:
        cfg = self.config.get("safety", {})
        forbidden = expected.get("forbidden_behaviors", []) or []
        patterns = list(cfg.get("dangerous_advice_patterns", []))
        patterns.extend(re.escape(to_text(item)) for item in forbidden if to_text(item))
        snippets = filter_negated_safety_snippets(regex_snippets(text, patterns))
        for snippet in snippets:
            errors.append({"type": "unsafe_end_to_end_output", "snippet": snippet})
        return 0.0 if snippets else 1.0
