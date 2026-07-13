"""Evaluator registry."""

from __future__ import annotations

from evals.evaluators.clinical_safety_evaluator import ClinicalSafetyEvaluator
from evals.evaluators.diagnosis_report_evaluator import DiagnosisReportEvaluator
from evals.evaluators.efficiency_evaluator import EfficiencyEvaluator
from evals.evaluators.end_to_end_evaluator import EndToEndEvaluator
from evals.evaluators.faithfulness_evaluator import FaithfulnessEvaluator
from evals.evaluators.health_plan_evaluator import HealthPlanEvaluator
from evals.evaluators.interview_quality_evaluator import InterviewQualityEvaluator
from evals.evaluators.llm_expert_evaluator import LLMExpertEvaluator
from evals.evaluators.rag_evaluator import RAGEvaluator
from evals.evaluators.robustness_evaluator import RobustnessEvaluator
from evals.evaluators.tool_call_evaluator import ToolCallEvaluator


EVALUATOR_REGISTRY = {
    "clinical_safety": ClinicalSafetyEvaluator,
    "end_to_end": EndToEndEvaluator,
    "llm_expert": LLMExpertEvaluator,
    "interview_quality": InterviewQualityEvaluator,
    "rag": RAGEvaluator,
    "faithfulness": FaithfulnessEvaluator,
    "tool_call": ToolCallEvaluator,
    "diagnosis_report": DiagnosisReportEvaluator,
    "health_plan": HealthPlanEvaluator,
    "robustness": RobustnessEvaluator,
    "efficiency": EfficiencyEvaluator,
}

ALL_EVAL_NAMES = list(EVALUATOR_REGISTRY.keys())
