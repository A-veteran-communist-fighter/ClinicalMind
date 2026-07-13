"""Scoring helpers."""

from __future__ import annotations

from typing import Iterable

from evals.text_utils import mean


def weighted_score(metrics: dict[str, float], weights: dict[str, float]) -> float:
    if not weights:
        return mean(metrics.values())
    total_weight = sum(max(0.0, float(w)) for w in weights.values())
    if total_weight <= 0:
        return 0.0
    score = 0.0
    for key, weight in weights.items():
        score += float(metrics.get(key, 0.0)) * max(0.0, float(weight))
    return max(0.0, min(1.0, score / total_weight))


def pass_rate(results: Iterable[dict]) -> float:
    rows = list(results)
    if not rows:
        return 0.0
    return sum(1 for row in rows if row.get("passed")) / len(rows)


def normalize_1_to_5(score: float) -> float:
    return max(0.0, min(1.0, (float(score) - 1.0) / 4.0))
