"""Faithfulness and evidence consistency evaluator."""

from __future__ import annotations

from typing import Any

from evals.evaluators.base import BaseEvaluator
from evals.llm_judge import JudgeUnavailable, load_prompt
from evals.metrics.scoring_metrics import weighted_score
from evals.text_utils import (
    extract_citations,
    mean,
    overlap_score,
    safe_divide,
    split_claims,
    to_text,
)


class FaithfulnessEvaluator(BaseEvaluator):
    eval_name = "faithfulness"

    def evaluate_case(self, case: dict[str, Any]):
        output = self.output(case)
        text = self.output_text(case)
        warnings: list[str] = []
        errors: list[dict[str, Any] | str] = []
        evidence = self._collect_evidence(case, output)

        if not text:
            warnings.append("missing agent output")
        if not evidence:
            warnings.append("missing evidence; claim support falls back to unsupported")

        claims = split_claims(text)
        if not claims:
            warnings.append("missing extractable claims")

        claim_rows: list[dict[str, Any]] = []
        min_overlap = float(self.config.get("thresholds", {}).get("min_claim_overlap", 0.24))
        for claim in claims:
            best = self._best_evidence(claim, evidence)
            support_score = best["overlap"]
            supported = support_score >= min_overlap
            conflict = False

            if best["evidence_text"]:
                try:
                    prompt = load_prompt("faithfulness_judge_prompt.txt")
                    judge = self.judge_client.judge_json(
                        prompt,
                        {"claim": claim, "evidence": best["evidence_text"]},
                    )
                    supported = bool(judge.get("supported", supported))
                    conflict = bool(judge.get("conflict", False))
                    support_score = float(judge.get("support_score", support_score))
                except JudgeUnavailable as exc:
                    if str(exc) not in warnings:
                        warnings.append(str(exc))
                except Exception as exc:
                    warnings.append(f"LLM faithfulness judge failed: {exc}")

            row = {
                "claim": claim,
                "best_evidence_id": best["evidence_id"],
                "support_score": max(0.0, min(1.0, support_score)),
                "supported": supported,
                "conflict": conflict,
            }
            claim_rows.append(row)
            if not supported:
                errors.append({"type": "unsupported_claim", "claim": claim, "evidence_id": best["evidence_id"]})
            if conflict:
                errors.append({"type": "evidence_conflict", "claim": claim, "evidence_id": best["evidence_id"]})

        supported_rate = mean([1.0 if row["supported"] else 0.0 for row in claim_rows], default=0.0)
        conflict_rate = mean([1.0 if row["conflict"] else 0.0 for row in claim_rows], default=0.0)
        citations = extract_citations(text)
        evidence_ids = {row["id"] for row in evidence if row["id"]}
        citation_hits = sum(1 for cite in citations if cite in evidence_ids)
        citation_accuracy = safe_divide(citation_hits, len(citations), default=1.0)
        used_evidence = {row["best_evidence_id"] for row in claim_rows if row["supported"] and row["best_evidence_id"]}
        evidence_coverage = safe_divide(len(used_evidence), len(evidence), default=1.0)

        metrics = {
            "groundedness_score": supported_rate,
            "citation_accuracy": citation_accuracy,
            "evidence_coverage": evidence_coverage,
            "hallucination_rate": 1.0 - supported_rate if claim_rows else 0.0,
            "evidence_conflict_rate": conflict_rate,
        }
        score = weighted_score(
            {
                "groundedness_score": metrics["groundedness_score"],
                "citation_accuracy": metrics["citation_accuracy"],
                "evidence_coverage": metrics["evidence_coverage"],
                "hallucination_rate": 1.0 - metrics["hallucination_rate"],
                "evidence_conflict_rate": 1.0 - metrics["evidence_conflict_rate"],
            },
            self.config.get("weights", {}).get("faithfulness", {}),
        )
        return self.result(case, score, metrics, errors, warnings, {"claim_level": claim_rows})

    def _collect_evidence(self, case: dict[str, Any], output: dict[str, Any]) -> list[dict[str, str]]:
        raw = (
            output.get("retrieved_evidence")
            or output.get("retrieval_results")
            or case.get("retrieval_results")
            or case.get("evidence")
            or self.expected(case).get("evidence_snippets")
            or []
        )
        evidence: list[dict[str, str]] = []
        for idx, item in enumerate(raw):
            if isinstance(item, dict):
                evidence.append(
                    {
                        "id": to_text(item.get("doc_id") or item.get("id") or item.get("source_id") or f"evidence_{idx}"),
                        "text": to_text(item.get("text") or item.get("content") or item.get("snippet") or item),
                    }
                )
            else:
                evidence.append({"id": f"evidence_{idx}", "text": to_text(item)})
        return evidence

    def _best_evidence(self, claim: str, evidence: list[dict[str, str]]) -> dict[str, Any]:
        best = {"evidence_id": "", "evidence_text": "", "overlap": 0.0}
        for item in evidence:
            score = overlap_score(claim, item["text"])
            if score > best["overlap"]:
                best = {"evidence_id": item["id"], "evidence_text": item["text"], "overlap": score}
        return best
