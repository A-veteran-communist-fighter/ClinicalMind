"""Medical retrieval augmented generation evaluator."""

from __future__ import annotations

from typing import Any

from evals.evaluators.base import BaseEvaluator
from evals.metrics.retrieval_metrics import (
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    trusted_source_ratio,
)
from evals.metrics.scoring_metrics import weighted_score
from evals.text_utils import extract_citations, overlap_score, safe_divide, to_text


class RAGEvaluator(BaseEvaluator):
    eval_name = "rag"

    def evaluate_case(self, case: dict[str, Any]):
        output = self.output(case)
        warnings: list[str] = []
        relevant_ids = case.get("relevant_doc_ids") or self.expected(case).get("relevant_doc_ids") or []
        evidence_snippets = case.get("evidence_snippets") or self.expected(case).get("evidence_snippets") or []
        stages = output.get("retrieval_stages") or case.get("retrieval_stages") or {}
        if not stages:
            results = output.get("retrieved_evidence") or case.get("retrieval_results") or []
            if results:
                stages = {"final": results}
        if not stages:
            warnings.append("missing retrieval results")
            stages = {"final": []}

        k = int(self.config.get("thresholds", {}).get("rag_k", 5))
        trusted_domains = self.config.get("trusted_domains", [])
        stage_metrics: dict[str, dict[str, float]] = {}
        for stage_name, results in stages.items():
            results = results or []
            stage_metrics[stage_name] = {
                f"Recall@{k}": recall_at_k(results, relevant_ids, k),
                f"Precision@{k}": precision_at_k(results, relevant_ids, k),
                "MRR": mrr(results, relevant_ids),
                f"nDCG@{k}": ndcg_at_k(results, relevant_ids, k),
                "trusted_source_ratio": trusted_source_ratio(results, trusted_domains, k),
                "evidence_support_rate": self._evidence_support_rate(results, evidence_snippets),
            }

        final_stage = "final" if "final" in stage_metrics else list(stage_metrics.keys())[-1]
        final_metrics = stage_metrics[final_stage]
        answer_text = to_text(output.get("final_answer") or output.get("answer") or "")
        all_doc_ids = {
            str(item.get("doc_id") or item.get("id") or item.get("source_id"))
            for stage_results in stages.values()
            for item in (stage_results or [])
            if isinstance(item, dict)
        }
        citations = extract_citations(answer_text)
        citation_hits = sum(1 for c in citations if c in set(map(str, relevant_ids)) or c in all_doc_ids)
        citation_hit_rate = safe_divide(citation_hits, len(citations), default=1.0)
        hallucinated_citation_rate = 1.0 - citation_hit_rate if citations else 0.0

        metrics = {
            **final_metrics,
            "citation_hit_rate": citation_hit_rate,
            "hallucinated_citation_rate": hallucinated_citation_rate,
        }
        score = weighted_score(
            {
                "recall_at_k": final_metrics.get(f"Recall@{k}", 0.0),
                "precision_at_k": final_metrics.get(f"Precision@{k}", 0.0),
                "mrr": final_metrics.get("MRR", 0.0),
                "ndcg_at_k": final_metrics.get(f"nDCG@{k}", 0.0),
                "trusted_source_ratio": final_metrics.get("trusted_source_ratio", 0.0),
                "evidence_support_rate": final_metrics.get("evidence_support_rate", 0.0),
                "citation_hit_rate": citation_hit_rate,
                "hallucinated_citation_rate": 1.0 - hallucinated_citation_rate,
            },
            self.config.get("weights", {}).get("rag", {}),
        )
        return self.result(case, score, metrics, warnings=warnings, details={"stage_metrics": stage_metrics})

    def _evidence_support_rate(self, results: list[dict[str, Any]], snippets: list[str]) -> float:
        if not snippets:
            return 1.0
        if not results:
            return 0.0
        hit = 0
        for snippet in snippets:
            if any(overlap_score(snippet, to_text(r)) >= 0.25 for r in results):
                hit += 1
        return safe_divide(hit, len(snippets))
