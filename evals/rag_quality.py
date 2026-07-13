#!/usr/bin/env python3
"""RAG Quality Metrics: Resolution Rate, Hallucination Rate, Empty Response Rate.

Each metric evaluated on 30 test cases using LLM-as-Judge.

Metrics:
  - Resolution Rate:  Does the answer actually solve the user's problem? (0-1)
  - Hallucination Rate: Does the answer contain unsupported claims? (0-1, lower is better)
  - Empty Response Rate: Did the system fail to produce a meaningful answer? (0-1, lower is better)

Usage:
    python -m evals.rag_quality
"""

import asyncio, json, os, re, sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Load .env ────────────────────────────────────────────────────────────
_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    with open(_env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line: continue
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

from openai import AsyncOpenAI


# ── LLM Client ───────────────────────────────────────────────────────────
def _get_client() -> AsyncOpenAI:
    api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("DEEPSEEK_BASE_URL") or os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1")
    if not api_key:
        raise RuntimeError("No LLM API key found")
    return AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=120, max_retries=2)

MODEL = os.getenv("DEFAULT_LLM_MODEL", "deepseek-chat")


async def llm_judge(prompt: str) -> str:
    client = _get_client()
    resp = await client.chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": prompt}],
        max_tokens=300, temperature=0, stream=False,
    )
    return resp.choices[0].message.content or ""


def parse_bool(text: str) -> int:
    """Parse LLM boolean response. Returns 0 or 1."""
    text = text.strip().lower()
    # JSON
    try:
        d = json.loads(text)
        for k in ("resolved", "hallucinated", "empty", "score", "value", "result"):
            if k in d:
                v = d[k]
                if isinstance(v, bool): return 1 if v else 0
                if isinstance(v, (int, float)): return 1 if float(v) >= 0.5 else 0
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    # Keywords
    yes = ("yes", "true", "是", "有", "会", "能", "存在", "包含", "1")
    no = ("no", "false", "否", "无", "没有", "不会", "不能", "不存在", "不包含", "0")
    for w in yes:
        if w in text and not any(n in text for n in no):
            return 1
    for w in no:
        if w in text:
            return 0
    return 0


# ═══════════════════════════════════════════════════════════════════════════
# Knowledge base + queries (reuse from rag_eval.py)
# ═══════════════════════════════════════════════════════════════════════════

from evals.rag_eval import MEDICAL_KNOWLEDGE, TEST_CASES, retrieve, generate


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation Prompts
# ═══════════════════════════════════════════════════════════════════════════

RESOLUTION_PROMPT = """你是医疗AI评测专家。判断AI的回答是否真正解决了用户的问题。

用户问题：
{question}

参考知识（AI只能基于此回答）：
{context}

AI回答：
{answer}

判断标准：
- 如果AI的回答完整、准确地解决了用户的问题，返回 {{"resolved": true}}
- 如果回答不完整、模糊、答非所问，或没有给出用户需要的具体信息，返回 {{"resolved": false}}

注意：
- 只返回JSON：{{"resolved": true}} 或 {{"resolved": false}}
- 不要任何解释"""

HALLUCINATION_PROMPT = """你是医疗AI评测专家。判断AI的回答是否包含参考知识中不存在的信息（幻觉）。

参考知识（AI必须仅基于此回答）：
{context}

AI回答：
{answer}

判断标准：
- 如果AI回答中的任何一个事实性陈述在参考知识中找不到依据，即判定为有幻觉
- 医学常识如果不在参考知识中，也算幻觉
- 特例：如果AI明确说"参考知识不足以回答"，不算幻觉

返回JSON：
- 有幻觉：{{"hallucinated": true, "detail": "简述哪一句是幻觉"}}
- 无幻觉：{{"hallucinated": false}}

注意：只返回JSON，不要其他内容"""

EMPTY_RESPONSE_PROMPT = """你是医疗AI评测专家。判断AI的回答是否是"空答"或"无效回答"。

用户问题：
{question}

AI回答：
{answer}

空答/无效回答的定义：
- 回答内容为空、纯占位符或纯错误信息
- 回答是"I don't know""我无法回答"等拒绝回答
- 回答完全避开了问题、答非所问
- 回答毫无信息量（如纯免责声明、纯建议就医）

不是空答的情况：
- 即使回答较短，但给出了实质信息
- 回答表示"参考知识有限"但仍提供了部分有用信息

返回JSON：{{"empty": true}} 或 {{"empty": false}}
只返回JSON，不要解释"""


# ═══════════════════════════════════════════════════════════════════════════
# Rule-based empty check (no LLM needed for obvious cases)
# ═══════════════════════════════════════════════════════════════════════════

def rule_empty_check(answer: str) -> int:
    """Fast rule-based empty response detection."""
    if not answer or len(answer.strip()) < 5:
        return 1
    lower = answer.strip().lower()
    # Pure rejections
    pure_rejections = [
        "i don't know", "我无法回答", "我无法提供", "抱歉，我无法",
        "无法回答此问题", "没有足够的信息", "no information",
        "i cannot answer", "我不能回答",
    ]
    for r in pure_rejections:
        if lower == r or (len(answer) < 30 and r in lower):
            return 1
    # Pure error messages
    if "error" in lower and len(answer) < 50:
        return 1
    if lower.startswith("查询失败"):
        return 1
    return -1  # uncertain, need LLM judge


# ═══════════════════════════════════════════════════════════════════════════
# Main Evaluation
# ═══════════════════════════════════════════════════════════════════════════

async def run():
    print("=" * 65)
    print("  RAG Quality Metrics: Resolution | Hallucination | Empty")
    print("  " + "=" * 63)

    # ── Step 1: Generate answers ──────────────────────────────────────
    print(f"\n[1/2] Generating answers for {len(TEST_CASES)} queries...")
    records = []
    for i, tc in enumerate(TEST_CASES):
        ctx = retrieve(tc.question, tc.context_keys)
        ans = await generate(tc.question, ctx)
        records.append({"tc": tc, "context": ctx, "answer": ans})
        short = tc.question[:45] + "..." if len(tc.question) > 45 else tc.question
        print(f"  [{i+1:2d}/{len(TEST_CASES)}] {short}")

    # ── Step 2: Evaluate three metrics ────────────────────────────────
    print(f"\n[2/2] Evaluating 3 metrics x 30 cases = 90 LLM calls...")

    results = []
    for i, rec in enumerate(records):
        tc = rec["tc"]
        ctx = rec["context"]
        ans = rec["answer"]

        # Resolution: LLM judge
        r_raw = await llm_judge(RESOLUTION_PROMPT.format(
            question=tc.question, context=ctx, answer=ans))
        resolved = parse_bool(r_raw)

        # Hallucination: LLM judge
        h_raw = await llm_judge(HALLUCINATION_PROMPT.format(
            context=ctx, answer=ans))
        hallucinated = parse_bool(h_raw)

        # Empty: rule-first, LLM fallback
        empty_rule = rule_empty_check(ans)
        if empty_rule >= 0:
            empty = empty_rule
        else:
            e_raw = await llm_judge(EMPTY_RESPONSE_PROMPT.format(
                question=tc.question, answer=ans))
            empty = parse_bool(e_raw)

        results.append({
            "id": i + 1,
            "query": tc.question[:55],
            "answer_preview": ans[:80],
            "resolved": resolved,
            "hallucinated": hallucinated,
            "empty": empty,
        })

        label = tc.question[:30] + "..." if len(tc.question) > 30 else tc.question
        r_icon = "✓" if resolved else "✗"
        h_icon = "⚠" if hallucinated else "✓"
        e_icon = "✗" if empty else "✓"
        print(f"  [{i+1:2d}/30] {label:<33} Resolve={r_icon} Halluc={h_icon} Empty={e_icon}")

    # ── Step 3: Compute rates ─────────────────────────────────────────
    n = len(results)
    resolution_rate = sum(r["resolved"] for r in results) / n
    hallucination_rate = sum(r["hallucinated"] for r in results) / n
    empty_rate = sum(r["empty"] for r in results) / n

    print(f"\n{'='*65}")
    print("  RESULTS")
    print(f"{'='*65}")

    print(f"""
  Metric                  Score    Interpretation
  ─────────────────────────────────────────────────
  Resolution Rate         {resolution_rate:.2%}    {'绝大多数问题得到解决' if resolution_rate >= 0.8 else '存在较多未解决的问题'}
  Hallucination Rate      {hallucination_rate:.2%}    {'几乎没有幻觉' if hallucination_rate <= 0.1 else '存在一定程度的幻觉问题'}
  Empty Response Rate     {empty_rate:.2%}    {'系统稳定输出' if empty_rate <= 0.05 else '存在空答/拒答问题'}
""")

    # ── Per-case detail ───────────────────────────────────────────────
    print(f"{'─'*65}")
    print(f"  #  Query                                              R  H  E")
    print(f"{'─'*65}")
    for r in results:
        q = r["query"][:48]
        ri = "✓" if r["resolved"] else "✗"
        hi = "⚠" if r["hallucinated"] else " "
        ei = "✗" if r["empty"] else " "
        print(f"  {r['id']:2d}  {q:<48}  {ri}  {hi}  {ei}")

    # ── Problem cases ─────────────────────────────────────────────────
    unresolved = [r for r in results if not r["resolved"]]
    hallucinated = [r for r in results if r["hallucinated"]]
    empties = [r for r in results if r["empty"]]

    if unresolved:
        print(f"\n{'─'*65}")
        print(f"  UNRESOLVED CASES ({len(unresolved)}):")
        for r in unresolved:
            print(f"    [{r['id']}] {r['query']}")
            print(f"         {r['answer_preview'][:100]}...")

    if hallucinated:
        print(f"\n{'─'*65}")
        print(f"  HALLUCINATED CASES ({len(hallucinated)}):")
        for r in hallucinated:
            print(f"    [{r['id']}] {r['query']}")
            print(f"         {r['answer_preview'][:100]}...")

    if empties:
        print(f"\n{'─'*65}")
        print(f"  EMPTY RESPONSE CASES ({len(empties)}):")
        for r in empties:
            print(f"    [{r['id']}] {r['query']}")

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'═'*65}")
    print("  Summary")
    print(f"{'═'*65}")
    total_ok = sum(1 for r in results if r["resolved"] and not r["hallucinated"] and not r["empty"])
    print(f"  解决+无幻觉+非空答: {total_ok}/{n} ({total_ok/n:.1%})")
    print(f"  解决但伴随幻觉: {sum(1 for r in results if r['resolved'] and r['hallucinated'])}")
    print(f"  未解决且幻觉:   {sum(1 for r in results if not r['resolved'] and r['hallucinated'])}")
    print(f"  未解决无幻觉:   {sum(1 for r in results if not r['resolved'] and not r['hallucinated'])}")

    # ── Save ──────────────────────────────────────────────────────────
    out = Path(__file__).parent / "rag_quality_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "resolution_rate": resolution_rate,
                "hallucination_rate": hallucination_rate,
                "empty_rate": empty_rate,
                "num_cases": n,
                "fully_ok": total_ok,
            },
            "details": results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    asyncio.run(run())
