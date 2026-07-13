"""Retrieval metrics for RAG evaluation."""

from __future__ import annotations

import math
from typing import Any, Iterable

from evals.text_utils import domain_from_result, is_trusted_domain, safe_divide


def _ids(results: Iterable[dict[str, Any] | str]) -> list[str]:
    ids: list[str] = []
    for item in results:
        if isinstance(item, dict):
            ids.append(str(item.get("doc_id") or item.get("id") or item.get("source_id") or ""))
        else:
            ids.append(str(item))
    return [x for x in ids if x]


def recall_at_k(results: Iterable[dict[str, Any] | str], relevant_ids: Iterable[str], k: int) -> float:
    rel = {str(x) for x in relevant_ids}
    if not rel:
        return 1.0
    retrieved = set(_ids(results)[:k])
    return len(retrieved & rel) / len(rel)


def precision_at_k(results: Iterable[dict[str, Any] | str], relevant_ids: Iterable[str], k: int) -> float:
    rel = {str(x) for x in relevant_ids}
    top = _ids(results)[:k]
    if k <= 0:
        return 0.0
    return len([doc_id for doc_id in top if doc_id in rel]) / k


def mrr(results: Iterable[dict[str, Any] | str], relevant_ids: Iterable[str]) -> float:
    rel = {str(x) for x in relevant_ids}
    for idx, doc_id in enumerate(_ids(results), start=1):
        if doc_id in rel:
            return 1.0 / idx
    return 0.0


def ndcg_at_k(results: Iterable[dict[str, Any] | str], relevant_ids: Iterable[str], k: int) -> float:
    rel = {str(x) for x in relevant_ids}
    top = _ids(results)[:k]
    dcg = 0.0
    for idx, doc_id in enumerate(top, start=1):
        gain = 1.0 if doc_id in rel else 0.0
        dcg += gain / math.log2(idx + 1)
    ideal_hits = min(len(rel), k)
    idcg = sum(1.0 / math.log2(idx + 1) for idx in range(1, ideal_hits + 1))
    return safe_divide(dcg, idcg)


def trusted_source_ratio(results: Iterable[dict[str, Any]], trusted_domains: Iterable[str], k: int) -> float:
    top = list(results)[:k]
    if not top:
        return 0.0
    trusted = sum(1 for item in top if is_trusted_domain(domain_from_result(item), trusted_domains))
    return trusted / len(top)
