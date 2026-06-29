"""ClinicalMind LangGraph Nodes.

Uses LangGraph's `interrupt()` for human-in-the-loop (asking questions, collecting answers).
This replaces the custom orchestrator, interview engine, and all agent classes.
"""

import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import interrupt

from .llm import llm, llm_fast, extract_json
from .prompts import (
    MASTER_AGENT_PROMPT,
    SAFETY_CHECK_PROMPT,
    INTERVIEW_SYSTEM_PROMPT,
    DIAGNOSIS_SYSTEM_PROMPT,
    PLANNING_SYSTEM_PROMPT,
    RESEARCH_SYSTEM_PROMPT,
    ANSWER_PROCESSOR_PROMPT,
)
from .state import ClinicalState

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Emergency Keyword Rules (from original safety_triage.py)
# ═══════════════════════════════════════════════════════════════════════════════

_CHEST = ("胸痛", "胸闷", "心前区痛")
_DYSPNEA = ("呼吸困难", "喘不上气", "气短", "憋气")
_INSTABILITY = ("大汗", "冷汗", "晕厥", "意识模糊")
_NEURO = ("意识丧失", "昏迷", "抽搐", "口角歪斜", "一侧肢体无力")
_BLEEDING = ("大量出血", "呕血", "黑便", "咯血", "休克")
_ALLERGY = ("喉头水肿", "严重过敏", "喘鸣", "过敏性休克")


def _check_red_flags(text: str) -> list[str]:
    """Fast keyword-based emergency screening (no LLM call needed)."""
    flags = []
    t = text.lower()
    if any(w in t for w in _CHEST) and (any(w in t for w in _DYSPNEA) or any(w in t for w in _INSTABILITY)):
        flags.append("胸痛/胸闷合并呼吸困难/大汗/晕厥 - 需排除急性冠脉综合征")
    if any(w in t for w in _NEURO):
        flags.append("神经系统危险信号 - 疑似卒中或严重神经系统疾病")
    if any(w in t for w in _BLEEDING):
        flags.append("严重出血或休克风险")
    if any(w in t for w in _ALLERGY):
        flags.append("严重过敏或气道受累")
    return flags


# ═══════════════════════════════════════════════════════════════════════════════
# Node 1: Safety + Intent (combined entry point)
# ═══════════════════════════════════════════════════════════════════════════════

async def entry_node(state: ClinicalState) -> dict[str, Any]:
    """Entry node: safety check + intent classification + routing.

    Runs on every new chief complaint. Sets the route for downstream nodes.
    """
    user_input = state.get("current_user_input", state.get("chief_complaint", ""))

    # 1. Rule-based safety check
    red_flags = _check_red_flags(user_input)
    is_emergency = len(red_flags) > 0

    # 2. LLM deeper check for borderline cases
    if not is_emergency and len(user_input) > 10:
        try:
            collected = state.get("collected_info", {})
            resp = await llm_fast.ainvoke([
                SystemMessage(content=SAFETY_CHECK_PROMPT),
                HumanMessage(content=f"主诉: {user_input}\n已收集信息: {collected}"),
            ])
            data = extract_json(resp.content if hasattr(resp, "content") else str(resp))
            if data.get("has_emergency"):
                is_emergency = True
                red_flags.extend(data.get("flags", []))
        except Exception:
            pass

    # 3. Intent classification (skip for emergency)
    if is_emergency:
        return {
            "red_flags": red_flags,
            "is_emergency": True,
            "intent": "diagnosis",
            "route": "diagnosis",
        }

    try:
        resp = await llm_fast.ainvoke([
            SystemMessage(content=MASTER_AGENT_PROMPT),
            HumanMessage(content=user_input),
        ])
        data = extract_json(resp.content if hasattr(resp, "content") else str(resp))
    except Exception:
        data = {"intent": "diagnosis", "confidence": "low"}

    intent = data.get("intent", "diagnosis")
    route_map = {
        "diagnosis": "diagnosis", "planning": "planning",
        "monitoring": "research", "research": "research",
        "consultation": "diagnosis", "general": "research",
    }

    return {
        "red_flags": red_flags,
        "is_emergency": is_emergency,
        "intent": intent,
        "intent_confidence": data.get("confidence", "low"),
        "route": route_map.get(intent, "diagnosis"),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Node 2: Interview Loop (generates questions + processes answers via interrupt)
# ═══════════════════════════════════════════════════════════════════════════════

async def interview_loop_node(state: ClinicalState) -> dict[str, Any]:
    """The interview loop: generate questions → interrupt for human answer → repeat.

    Uses LangGraph's `interrupt()` to pause execution and wait for user response.
    Each iteration: generate 1-2 questions, present them, process answers, then
    decide whether to continue or synthesize.

    This single node replaces: Track1Agent, Track2Agent, InterviewOrchestrator,
    DynamicInterviewEngine, and process_answer.
    """
    chief = state.get("chief_complaint", "")
    collected = dict(state.get("collected_info", {}))
    raw_answers = dict(state.get("raw_answers", {}))
    asked = list(state.get("asked_questions", []))
    diffs = state.get("differential_diagnoses", [])
    flags = list(state.get("red_flags", []))

    MAX_ROUNDS = 12

    for round_num in range(MAX_ROUNDS):
        # --- Generate questions ---
        prompt_parts = [f"## 患者主诉\n{chief}"]

        if collected:
            prompt_parts.append("\n## 已收集信息")
            for k, v in list(collected.items())[-15:]:
                if v and v not in ("无", "没有", "不清楚", "跳过"):
                    prompt_parts.append(f"  {k}: {str(v)[:150]}")

        if diffs:
            prompt_parts.append("\n## 当前鉴别诊断")
            for d in diffs[-5:]:
                icon = {"high": "V", "medium": "?", "low": "X"}.get(d.get("confidence", ""), "?")
                prompt_parts.append(f"  [{icon}] {d.get('diagnosis','?')}: {d.get('reason','')[:80]}")

        prompt_parts.append(f"\n## 状态\n第{round_num + 1}轮，已问{len(asked)}个问题")
        if asked:
            prompt_parts.append(f"最近已问: {', '.join(asked[-8:])}")
        prompt_parts.append("\n请决定: ask(继续1-2个问题) 或 synthesize(信息充足)")

        try:
            resp = await llm.ainvoke([
                SystemMessage(content=INTERVIEW_SYSTEM_PROMPT),
                HumanMessage(content="\n".join(prompt_parts)),
            ])
            data = extract_json(resp.content if hasattr(resp, "content") else str(resp))
        except Exception as e:
            logger.warning(f"Interview LLM failed (round {round_num}): {e}")
            continue

        action = data.get("action", "ask")

        # Update differential diagnoses
        new_diffs = data.get("differential_diagnoses", [])
        existing_names = {d.get("diagnosis", "") for d in diffs}
        for nd in new_diffs:
            if nd.get("diagnosis", "") and nd["diagnosis"] not in existing_names:
                diffs.append(nd)
                existing_names.add(nd["diagnosis"])

        # --- Synthesis check ---
        if action == "synthesize":
            meaningful_count = sum(1 for v in collected.values() if v and v not in ("无", "没有", "跳过"))
            if meaningful_count >= 3 or round_num >= 3:
                logger.info(f"Interview complete: {meaningful_count} meaningful items in {round_num + 1} rounds")
                return {
                    "collected_info": collected,
                    "raw_answers": raw_answers,
                    "asked_questions": asked,
                    "differential_diagnoses": diffs,
                    "red_flags": flags,
                    "phase": "diagnosing",
                }

        questions = data.get("questions", data.get("basic_module", []))
        if not questions:
            meaningful_count = sum(1 for v in collected.values() if v and v not in ("无", "没有", "跳过"))
            if meaningful_count >= 5:
                return {
                    "collected_info": collected,
                    "raw_answers": raw_answers,
                    "asked_questions": asked,
                    "differential_diagnoses": diffs,
                    "red_flags": flags,
                    "phase": "diagnosing",
                }
            continue

        # Normalize questions
        normalized = []
        for q in questions:
            normalized.append({
                "id": q.get("id", q.get("question_id", f"q_{round_num}")),
                "text": q.get("text", q.get("question", "")),
                "type": q.get("type", "text"),
                "options": q.get("options", []),
                "hint": q.get("hint", ""),
                "phase": q.get("phase", ""),
            })

        # --- Human-in-the-loop: interrupt and wait for answers ---
        # LangGraph pauses here. Console shows questions, collects answers,
        # then resumes with Command(resume=answer_string).
        human_response: str = interrupt({
            "type": "ask_questions",
            "questions": normalized,
            "differential_diagnoses": diffs[-5:],
            "red_flags": flags,
            "round": round_num + 1,
            "collected_count": len([v for v in collected.values() if v and v not in ("无", "没有", "跳过")]),
        })

        if not human_response:
            continue

        # --- Process answers ---
        answers = human_response.split("||")
        if len(answers) != len(normalized):
            answers = [human_response] + [""] * (len(normalized) - 1)

        for i, q in enumerate(normalized):
            qid = q["id"]
            q_text = q["text"]
            ans = answers[i].strip() if i < len(answers) else ""

            if not ans or ans.lower() in ("跳过", "skip", ""):
                raw_answers[qid] = "跳过"
                collected[qid] = "跳过"
                if qid not in asked:
                    asked.append(qid)
                continue

            # Map numeric choices to text
            opts = q.get("options", [])
            if opts and q.get("type") in ("choice", "multi_choice"):
                try:
                    indices = [int(x.strip()) - 1 for x in ans.split(",") if x.strip().isdigit()]
                    mapped = ", ".join(opts[j] for j in indices if 0 <= j < len(opts))
                    if mapped:
                        ans = mapped
                except (ValueError, IndexError):
                    pass

            raw_answers[qid] = ans
            if qid not in asked:
                asked.append(qid)

            # LLM extraction for rich answers
            if ans != "跳过" and len(ans) > 3:
                try:
                    extract_prompt = ANSWER_PROCESSOR_PROMPT.format(
                        question_id=qid, question_text=q_text, answer=ans
                    )
                    resp2 = await llm_fast.ainvoke([
                        SystemMessage(content="你是医学信息提取助手。只返回JSON。"),
                        HumanMessage(content=extract_prompt),
                    ])
                    edata = extract_json(resp2.content if hasattr(resp2, "content") else str(resp2))
                    collected[qid] = edata.get("extracted", ans)
                except Exception:
                    collected[qid] = ans
            else:
                collected[qid] = ans

        # --- Update state for next iteration ---
        # (state is updated via return dict, but we need to maintain local state
        #  for the in-loop diff tracking since LangGraph merges at function return)
        meaningful = sum(1 for v in collected.values() if v and v not in ("无", "没有", "跳过"))

    # Max rounds exhausted
    return {
        "collected_info": collected,
        "raw_answers": raw_answers,
        "asked_questions": asked,
        "differential_diagnoses": diffs,
        "red_flags": flags,
        "phase": "diagnosing",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Node 3: Diagnosis
# ═══════════════════════════════════════════════════════════════════════════════

async def diagnose_node(state: ClinicalState) -> dict[str, Any]:
    """Generate structured diagnosis from all collected interview data."""
    chief = state.get("chief_complaint", "")
    collected = state.get("collected_info", {})
    raw_answers = state.get("raw_answers", {})
    diffs = state.get("differential_diagnoses", [])
    flags = state.get("red_flags", [])

    # Emergency override
    if state.get("is_emergency"):
        return {
            "diagnosis_report": {
                "primary_diagnosis": "急诊级风险 - 需立即线下评估",
                "differential_diagnoses": [
                    {"diagnosis": "需排除急危重症", "icd11_code": "",
                     "reasoning": "安全分诊识别到急诊级危险信号。"}
                ],
                "confidence": "low", "severity": "emergency",
                "key_findings": flags,
                "recommended_tests": ["由急诊医生根据临床判断决定"],
                "recommended_actions": ["立即前往急诊或呼叫急救。不要自行驾车。"],
                "red_flags": flags,
                "follow_up_required": True, "follow_up_timeline": "立即",
                "disclaimer": "本提示由安全分诊规则生成，不能替代急诊分诊。",
            },
            "phase": "completed",
        }

    # Build comprehensive summary
    lines = [f"主诉: {chief}", ""]
    for k, v in list(collected.items())[-20:]:
        if v and v not in ("无", "没有", "不清楚", "跳过"):
            lines.append(f"  [{k}]: {str(v)[:200]}")

    lines.append("")
    if diffs:
        lines.append("鉴别诊断演变:")
        for d in diffs[-8:]:
            conf = d.get("confidence", "low")
            icon = "++" if conf == "high" else "+-" if conf == "medium" else "--"
            lines.append(f"  {icon} {d.get('diagnosis','?')}: {d.get('reason','')[:120]}")

    if raw_answers:
        lines.append("\n患者原始描述:")
        for k, v in list(raw_answers.items())[-8:]:
            if v and v not in ("跳过", "", "无", "没有"):
                lines.append(f"  [{k}]: {str(v)[:200]}")

    if flags:
        lines.append(f"\n安全标记: {'; '.join(flags)}")

    summary = "\n".join(lines)

    try:
        resp = await llm.ainvoke([
            SystemMessage(content=DIAGNOSIS_SYSTEM_PROMPT),
            HumanMessage(content=f"## 患者综合信息\n{summary}\n\n请基于以上全部信息生成结构化诊断报告。只输出JSON。"),
        ])
        report = extract_json(resp.content if hasattr(resp, "content") else str(resp))
    except Exception as e:
        logger.error(f"Diagnosis failed: {e}")
        report = {
            "primary_diagnosis": "AI诊断生成失败",
            "differential_diagnoses": [],
            "confidence": "low", "severity": "unknown",
            "key_findings": ["系统暂时无法生成诊断。"],
            "recommended_tests": [], "recommended_actions": ["请稍后重试"],
            "red_flags": [], "follow_up_required": True,
            "follow_up_timeline": "如症状持续请线下就医",
            "disclaimer": "本报告由AI生成，仅供参考。",
        }

    return {"diagnosis_report": report, "phase": "diagnosed"}


# ═══════════════════════════════════════════════════════════════════════════════
# Node 4: Treatment Plan
# ═══════════════════════════════════════════════════════════════════════════════

async def treatment_plan_node(state: ClinicalState) -> dict[str, Any]:
    """Generate treatment plan from diagnosis."""
    diagnosis = state.get("diagnosis_report", {})
    if not diagnosis:
        return {"phase": "completed"}

    try:
        resp = await llm.ainvoke([
            SystemMessage(content=PLANNING_SYSTEM_PROMPT),
            HumanMessage(content=f"诊断:\n{diagnosis}\n\n请生成治疗计划。只输出JSON。"),
        ])
        plan = extract_json(resp.content if hasattr(resp, "content") else str(resp))
    except Exception as e:
        logger.error(f"Planning failed: {e}")
        plan = {"title": "生成失败", "goals": ["请咨询医生"]}

    return {"treatment_plan": plan, "phase": "completed"}


# ═══════════════════════════════════════════════════════════════════════════════
# Node 5: Research (standalone)
# ═══════════════════════════════════════════════════════════════════════════════

async def research_node(state: ClinicalState) -> dict[str, Any]:
    """Answer a medical knowledge question."""
    query = state.get("current_user_input", state.get("chief_complaint", ""))
    try:
        resp = await llm.ainvoke([
            SystemMessage(content=RESEARCH_SYSTEM_PROMPT),
            HumanMessage(content=f"问题: {query}\n请给出循证回答，引用权威来源。"),
        ])
        answer = resp.content if hasattr(resp, "content") else str(resp)
    except Exception as e:
        answer = f"查询失败: {e}"

    return {"research_answer": answer, "phase": "completed"}


# ═══════════════════════════════════════════════════════════════════════════════
# Node 6: Standalone Planning
# ═══════════════════════════════════════════════════════════════════════════════

async def standalone_planning_node(state: ClinicalState) -> dict[str, Any]:
    """Generate treatment plan from user text directly (no prior diagnosis)."""
    query = state.get("current_user_input", state.get("chief_complaint", ""))
    try:
        resp = await llm.ainvoke([
            SystemMessage(content=PLANNING_SYSTEM_PROMPT),
            HumanMessage(content=f"请基于以下信息生成治疗计划:\n{query}\n\n只输出JSON。"),
        ])
        plan = extract_json(resp.content if hasattr(resp, "content") else str(resp))
    except Exception:
        plan = {"title": "生成失败", "goals": ["请咨询医生"]}

    return {"treatment_plan": plan, "phase": "completed"}


# ═══════════════════════════════════════════════════════════════════════════════
# Routing functions
# ═══════════════════════════════════════════════════════════════════════════════

def route_after_entry(state: ClinicalState) -> str:
    """Route: where does the flow go after entry?"""
    if state.get("is_emergency"):
        return "diagnose"  # skip interview for emergencies
    route = state.get("route", "diagnosis")
    return {
        "diagnosis": "interview_loop",
        "research": "research",
        "planning": "standalone_planning",
    }.get(route, "interview_loop")


def route_after_diagnosis(state: ClinicalState) -> str:
    """Route: after diagnosis, go to treatment plan or end?"""
    if state.get("is_emergency"):
        return "__end__"
    return "treatment_plan"
