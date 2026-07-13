from __future__ import annotations

import pytest

from evals.metrics.retrieval_metrics import mrr, ndcg_at_k, precision_at_k, recall_at_k


def test_retrieval_metrics_at_k() -> None:
    results = [{"doc_id": "doc1"}, {"doc_id": "doc2"}, {"doc_id": "doc3"}]
    relevant = ["doc2", "doc4"]

    assert recall_at_k(results, relevant, 3) == pytest.approx(0.5)
    assert precision_at_k(results, relevant, 3) == pytest.approx(1 / 3)
    assert mrr(results, relevant) == pytest.approx(0.5)
    assert ndcg_at_k(results, relevant, 3) == pytest.approx(0.3868528, rel=1e-5)
