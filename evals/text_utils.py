"""Text utilities shared by rule-based evaluators."""

from __future__ import annotations

import json
import math
import re
from difflib import SequenceMatcher
from typing import Any, Iterable
from urllib.parse import urlparse


def safe_divide(num: float, den: float, default: float = 0.0) -> float:
    return default if den == 0 else num / den


def to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return str(value)


def normalize_text(text: Any) -> str:
    raw = to_text(text).lower()
    raw = re.sub(r"\s+", "", raw)
    raw = re.sub(r"[，。！？、；：,.!?;:()\[\]{}<>《》\"'`~\-_/\\|]", "", raw)
    return raw


def contains_any(text: Any, keywords: Iterable[str]) -> bool:
    normalized = normalize_text(text)
    return any(normalize_text(k) in normalized for k in keywords if normalize_text(k))


def regex_snippets(text: str, patterns: Iterable[str], window: int = 36) -> list[str]:
    snippets: list[str] = []
    for pattern in patterns:
        if not pattern:
            continue
        try:
            matches = re.finditer(pattern, text, flags=re.IGNORECASE)
        except re.error:
            matches = re.finditer(re.escape(pattern), text, flags=re.IGNORECASE)
        for match in matches:
            start = max(0, match.start() - window)
            end = min(len(text), match.end() + window)
            snippet = text[start:end].strip()
            if snippet and snippet not in snippets:
                snippets.append(snippet)
    return snippets


def is_negated_safety_snippet(snippet: str) -> bool:
    """Return True when a risky phrase is clearly mentioned as something to avoid."""

    return contains_any(
        snippet,
        [
            "避免",
            "不要",
            "不建议",
            "不能",
            "不得",
            "请勿",
            "不应",
            "不可以",
            "禁止",
            "不能自行",
            "不自行",
            "不要自行",
            "避免自行",
        ],
    )


def filter_negated_safety_snippets(snippets: Iterable[str]) -> list[str]:
    filtered: list[str] = []
    for snippet in snippets:
        if not is_negated_safety_snippet(snippet) and snippet not in filtered:
            filtered.append(snippet)
    return filtered


def char_ngrams(text: str, n: int = 2) -> set[str]:
    normalized = normalize_text(text)
    if len(normalized) <= n:
        return {normalized} if normalized else set()
    return {normalized[i : i + n] for i in range(len(normalized) - n + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def semantic_match(text: str, target: str, threshold: float = 0.36) -> bool:
    """Offline fuzzy matcher for short Chinese clinical phrases.

    This is intentionally conservative and dependency-free. Evaluators can
    layer an LLM judge on top when semantic precision is required.
    """

    left = normalize_text(text)
    right = normalize_text(target)
    if not left or not right:
        return False
    if left in right or right in left:
        return True
    if _content_char_coverage(left, right) >= 0.85:
        return True
    if jaccard(char_ngrams(left), char_ngrams(right)) >= threshold:
        return True
    return SequenceMatcher(None, left, right).ratio() >= max(0.52, threshold)


def _content_char_coverage(text: str, target: str) -> float:
    stop_chars = set("是否有没有无吗么呢的了多少几何")
    target_chars = {ch for ch in target if ch not in stop_chars}
    if not target_chars:
        target_chars = set(target)
    if not target_chars:
        return 0.0
    text_chars = set(text)
    return len(target_chars & text_chars) / len(target_chars)


def coverage_against_expected(
    items: Iterable[str],
    expected: Iterable[str],
    threshold: float = 0.36,
) -> tuple[float, list[str], list[str]]:
    item_list = [to_text(x) for x in items if to_text(x)]
    expected_list = [to_text(x) for x in expected if to_text(x)]
    if not expected_list:
        return 1.0, [], []

    hit: list[str] = []
    missed: list[str] = []
    for target in expected_list:
        if any(semantic_match(item, target, threshold=threshold) for item in item_list):
            hit.append(target)
        else:
            missed.append(target)
    return safe_divide(len(hit), len(expected_list)), hit, missed


def collect_output_text(output: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in (
        "final_answer",
        "answer",
        "research_answer",
        "diagnosis_report",
        "health_plan",
        "treatment_plan",
        "report",
        "plan",
        "summary",
    ):
        if key in output:
            parts.append(to_text(output.get(key)))
    for msg in output.get("conversation", []) or []:
        if isinstance(msg, dict):
            parts.append(to_text(msg.get("content", "")))
        else:
            parts.append(to_text(msg))
    return "\n".join(p for p in parts if p)


def collect_questions(output: dict[str, Any]) -> list[str]:
    questions: list[str] = []
    for key in ("asked_questions", "current_questions"):
        for q in output.get(key, []) or []:
            if isinstance(q, dict):
                questions.append(to_text(q.get("text") or q.get("question") or q.get("content") or q.get("id")))
            else:
                questions.append(to_text(q))
    for msg in output.get("conversation", []) or []:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role", "")).lower()
        content = to_text(msg.get("content", ""))
        if role == "assistant" and ("?" in content or "？" in content):
            questions.extend(split_questions(content))
    seen: list[str] = []
    for question in questions:
        question = question.strip()
        if question and question not in seen:
            seen.append(question)
    return seen


def split_questions(text: str) -> list[str]:
    candidates = re.split(r"(?<=[?？])\s*", text)
    return [c.strip() for c in candidates if c.strip()]


def split_claims(text: str) -> list[str]:
    cleaned = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    pieces = re.split(r"[。；;.!?\n]+", cleaned)
    claims: list[str] = []
    for piece in pieces:
        item = piece.strip(" -*\t")
        if len(normalize_text(item)) >= 8:
            claims.append(item)
    return claims[:80]


def overlap_score(text: str, evidence: str) -> float:
    return jaccard(char_ngrams(text), char_ngrams(evidence))


def domain_from_result(result: dict[str, Any]) -> str:
    domain = result.get("domain") or result.get("source_domain")
    if domain:
        return str(domain).lower()
    url = result.get("url") or result.get("source_url") or ""
    if not url:
        return ""
    parsed = urlparse(str(url))
    return parsed.netloc.lower()


def is_trusted_domain(domain: str, trusted_domains: Iterable[str]) -> bool:
    domain = domain.lower().strip()
    if not domain:
        return False
    for trusted in trusted_domains:
        marker = str(trusted).lower().strip()
        if not marker:
            continue
        if marker.startswith("."):
            if domain.endswith(marker):
                return True
        elif domain == marker or domain.endswith("." + marker):
            return True
    return False


def extract_citations(text: str) -> list[str]:
    citations = re.findall(r"\[([A-Za-z0-9_\-:.\/]+)\]", text)
    citations.extend(re.findall(r"doc[_-]?\d+", text, flags=re.IGNORECASE))
    seen: list[str] = []
    for item in citations:
        if item not in seen:
            seen.append(item)
    return seen


def mean(values: Iterable[float], default: float = 0.0) -> float:
    vals: list[float] = []
    for value in values:
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isnan(numeric):
            vals.append(numeric)
    return default if not vals else sum(vals) / len(vals)
