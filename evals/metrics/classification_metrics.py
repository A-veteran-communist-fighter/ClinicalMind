"""Basic classification metrics."""

from __future__ import annotations

from evals.text_utils import safe_divide


def precision(tp: int, fp: int) -> float:
    return safe_divide(tp, tp + fp)


def recall(tp: int, fn: int) -> float:
    return safe_divide(tp, tp + fn)


def f1_score(tp: int, fp: int, fn: int) -> float:
    p = precision(tp, fp)
    r = recall(tp, fn)
    return safe_divide(2 * p * r, p + r)


def accuracy(correct: int, total: int) -> float:
    return safe_divide(correct, total)
