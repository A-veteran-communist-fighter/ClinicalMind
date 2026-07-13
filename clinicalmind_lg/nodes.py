"""ClinicalMind LangGraph Nodes.

Uses state-driven pauses instead of interrupt() for LangGraph >= 1.x compatibility.
Replaces the custom orchestrator, interview engine, and all agent classes.
"""

import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from .llm import get_llm, get_llm_fast, extract_json
from .prompts import (
    MASTER_AGENT_PROMPT,
    SAFETY_CHECK_PROMPT,
    INTERVIEW_SYSTEM_PROMPT,
    DIAGNOSIS_SYSTEM_PROMPT,
    PLANNING_SYSTEM_PROMPT,
    RESEARCH_SYSTEM_PROMPT,
    ANSWER_PROCESSOR_PROMPT,
    LAB_PARSER_PROMPT,
)
from .state import ClinicalState

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Emergency Keyword Rules (from original safety_triage.py)
# ═══════════════════════════════════════════════════════════════════════════════

import re as _re

# Exact keywords for classic presentations
_CHEST_EXACT = ("胸痛", "胸闷", "心前区痛", "心绞痛")
_DYSPNEA = ("呼吸困难", "喘不上气", "气短", "憋气", "喘不过气")
_INSTABILITY = ("大汗", "冷汗", "晕厥", "意识模糊", "快晕倒", "眼前发黑", "濒死感")
_NEURO = ("意识丧失", "昏迷", "抽搐", "口角歪斜", "一侧肢体无力", "说话不清",
          "突然看不见", "突然听不见", "走路不稳")
_BLEEDING = ("大量出血", "呕血", "黑便", "咯血", "休克", "便血不止",
             "血止不住", "吐血")
_ALLERGY = ("喉头水肿", "严重过敏", "喘鸣", "过敏性休克", "嘴唇肿",
            "全身风团", "全身红疹")

# Loose patterns: "胸" near "痛" — catches "胸口痛", "胸部突然很痛", etc.
_CHEST_NEAR_PAIN = _re.compile(r'胸.{0,3}痛|胸.{0,3}闷')
# "突然" near "痛" — catches "突然胸口很痛", "背上突然剧烈痛"
_SUDDEN_PAIN = _re.compile(r'突然.{0,5}痛|突然.{0,5}疼')
# Pregnancy-related
_PREGNANCY = _re.compile(r'怀孕|孕|妊娠')


def _check_red_flags(text: str) -> list[str]:
    """Fast emergency screening: exact keywords + loose patterns + urgency hints."""
    flags: list[str] = []
    t = text.lower()

    # ── 1. Exact multi-keyword combos (highest specificity) ──
    has_chest = any(w in t for w in _CHEST_EXACT)
    has_dyspnea = any(w in t for w in _DYSPNEA)
    has_instability = any(w in t for w in _INSTABILITY)

    if has_chest and (has_dyspnea or has_instability):
        flags.append("胸痛/胸闷合并呼吸困难/大汗/晕厥 - 需排除急性冠脉综合征")

    # ── 2. Loose chest pain patterns ──
    # "胸口痛" alone is concerning enough if accompanied by urgency words
    if _CHEST_NEAR_PAIN.search(t) or _SUDDEN_PAIN.search(t):
        if has_dyspnea or has_instability:
            flags.append("胸痛/胸闷合并呼吸困难/大汗/晕厥 - 需排除急性冠脉综合征")
        elif not flags:  # don't duplicate
            # Chest pain + sudden onset → urgent but not necessarily emergency
            urgency = any(w in t for w in ("突然", "剧烈", "最严重", "难以忍受", "撕裂"))
            if urgency:
                flags.append("突发胸痛/剧烈胸痛 - 建议立即急诊评估排除ACS/夹层")

    # ── 3. Neurologic ──
    if any(w in t for w in _NEURO):
        flags.append("神经系统危险信号 - 疑似卒中或严重神经系统疾病")

    # ── 4. Bleeding / shock ──
    if any(w in t for w in _BLEEDING):
        flags.append("严重出血或休克风险")

    # ── 5. Severe allergy ──
    if any(w in t for w in _ALLERGY):
        flags.append("严重过敏或气道受累")

    # ── 6. Pregnancy risk ──
    if _PREGNANCY.search(t) and any(w in t for w in ("腹痛", "阴道出血", "流血", "头痛剧烈", "看不清", "水肿严重", "头晕")):
        flags.append("妊娠合并危险信号 - 建议立即妇产科急诊评估")

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
            resp = await get_llm_fast().ainvoke([
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
        resp = await get_llm_fast().ainvoke([
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
# Node 2a: Interview Generate — create questions, then pause for human
# ═══════════════════════════════════════════════════════════════════════════════

async def interview_generate_node(state: ClinicalState) -> dict[str, Any]:
    """Generate 1-2 interview questions and return. Does NOT call interrupt().

    The graph ends here; console reads questions from state, collects answers,
    then resumes by calling the graph with human_response filled in.
    """
    chief = state.get("chief_complaint", "")
    collected = dict(state.get("collected_info", {}))
    asked = list(state.get("asked_questions", []))
    diffs = state.get("differential_diagnoses", [])
    round_num = state.get("interview_round", 0) + 1

    # Build prompt
    prompt_parts = [f"## 患者主诉\n{chief}"]
    if collected:
        prompt_parts.append("\n## 已收集信息")
        for k, v in list(collected.items())[-15:]:
            if v and v not in ("无", "没有", "不清楚", "跳过"):
                prompt_parts.append(f"  {k}: {str(v)[:150]}")
    # Lab reports — inject into prompt so LLM considers them
    lab_reports = state.get("lab_reports", [])
    if lab_reports:
        prompt_parts.append("\n## 化验检查结果")
        for rpt in lab_reports[-3:]:
            for ind in rpt.get("indicators", []):
                if ind.get("abnormal"):
                    lvl = ind.get("abnormal_level", "unknown")
                    lvl_mark = {"critical": "!!", "severe": "!", "moderate": "⚠"}.get(lvl, "")
                    prompt_parts.append(
                        f"  {lvl_mark} {ind.get('indicator_name','?')}: "
                        f"{ind.get('value','?')}{ind.get('unit','')} "
                        f"(参考:{ind.get('reference_range','N/A')}) "
                        f"[{ind.get('abnormal_direction','?')}]"
                    )
    if diffs:
        prompt_parts.append("\n## 当前鉴别诊断")
        for d in diffs[-5:]:
            icon = {"high": "V", "medium": "?", "low": "X"}.get(d.get("confidence", ""), "?")
            prompt_parts.append(f"  [{icon}] {d.get('diagnosis','?')}: {d.get('reason','')[:80]}")
    prompt_parts.append(f"\n## 状态\n第{round_num}轮，已问{len(asked)}个问题")
    if asked:
        prompt_parts.append(f"最近已问: {', '.join(asked[-8:])}")
    prompt_parts.append("\n请决定: ask(继续1-2个问题) 或 synthesize(信息充足)")

    try:
        resp = await get_llm().ainvoke([
            SystemMessage(content=INTERVIEW_SYSTEM_PROMPT),
            HumanMessage(content="\n".join(prompt_parts)),
        ])
        data = extract_json(resp.content if hasattr(resp, "content") else str(resp))
    except Exception as e:
        logger.warning(f"Interview LLM failed (round {round_num}): {e}")
        data = {"action": "synthesize"}

    action = data.get("action", "ask")

    # Merge differential diagnoses
    new_diffs = data.get("differential_diagnoses", [])
    existing_names = {d.get("diagnosis", "") for d in diffs}
    for nd in new_diffs:
        if nd.get("diagnosis", "") and nd["diagnosis"] not in existing_names:
            diffs.append(nd)
            existing_names.add(nd["diagnosis"])

    # ── Synthesis check ──
    meaningful = sum(1 for v in collected.values() if v and v not in ("无", "没有", "跳过"))
    if action == "synthesize" and (meaningful >= 3 or round_num >= 3):
        return {
            "collected_info": collected,
            "differential_diagnoses": diffs,
            "interview_round": round_num,
            "phase": "diagnosing",
            "action": "synthesize",
        }

    questions = data.get("questions", data.get("basic_module", []))
    if not questions:
        if meaningful >= 5:
            return {
                "collected_info": collected,
                "differential_diagnoses": diffs,
                "interview_round": round_num,
                "phase": "diagnosing",
                "action": "synthesize",
            }
        return {
            "interview_round": round_num,
            "differential_diagnoses": diffs,
            "current_questions": [],
            "phase": "awaiting_human",
        }

    # Normalize
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

    return {
        "collected_info": collected,
        "differential_diagnoses": diffs,
        "current_questions": normalized,
        "interview_round": round_num,
        "phase": "awaiting_human",
        "action": "ask",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Node 2b: Interview Process — extract answers, decide next step
# ═══════════════════════════════════════════════════════════════════════════════

async def interview_process_node(state: ClinicalState) -> dict[str, Any]:
    """Process user answers to the current questions.

    Called after the console fills in human_response. Decides whether to
    generate more questions or transition to diagnosis.
    """
    human_response = state.get("human_response", "")
    current_questions = state.get("current_questions", [])
    collected = dict(state.get("collected_info", {}))
    raw_answers = dict(state.get("raw_answers", {}))
    asked = list(state.get("asked_questions", []))
    round_num = state.get("interview_round", 0)

    if not human_response or not current_questions:
        # No answer provided → go back to generate more questions
        return {"phase": "interviewing", "human_response": "", "current_questions": []}

    # Parse answers
    answers = human_response.split("||")
    if len(answers) != len(current_questions):
        answers = [human_response] + [""] * (len(current_questions) - 1)

    for i, q in enumerate(current_questions):
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
                    question_id=qid, question_text=q_text, answer=ans)
                resp = await get_llm_fast().ainvoke([
                    SystemMessage(content="你是医学信息提取助手。只返回JSON。"),
                    HumanMessage(content=extract_prompt),
                ])
                edata = extract_json(resp.content if hasattr(resp, "content") else str(resp))
                collected[qid] = edata.get("extracted", ans)
            except Exception:
                collected[qid] = ans
        else:
            collected[qid] = ans

    meaningful_count = sum(1 for v in collected.values() if v and v not in ("无", "没有", "跳过"))
    max_rounds = state.get("interview_round", 0) >= 12

    if max_rounds or meaningful_count >= 10:
        return {
            "collected_info": collected,
            "raw_answers": raw_answers,
            "asked_questions": asked,
            "current_questions": [],
            "human_response": "",
            "phase": "diagnosing",
        }

    return {
        "collected_info": collected,
        "raw_answers": raw_answers,
        "asked_questions": asked,
        "current_questions": [],
        "human_response": "",
        "phase": "interviewing",
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

    # Lab reports
    lab_reports = state.get("lab_reports", [])
    if lab_reports:
        lines.append("\n化验检查异常指标:")
        for rpt in lab_reports[-5:]:
            for ind in rpt.get("indicators", []):
                if ind.get("abnormal"):
                    lvl = ind.get("abnormal_level", "unknown")
                    lvl_mark = {"critical": "!!", "severe": "!", "moderate": "⚠"}.get(lvl, "")
                    lines.append(
                        f"  {lvl_mark} {ind.get('indicator_name','?')}: "
                        f"{ind.get('value','?')}{ind.get('unit','')} "
                        f"(参考:{ind.get('reference_range','N/A')})"
                    )
                    if ind.get("notes"):
                        lines.append(f"    备注: {ind['notes'][:150]}")
        lines.append("")

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
        resp = await get_llm().ainvoke([
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
        resp = await get_llm().ainvoke([
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
        resp = await get_llm().ainvoke([
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
        resp = await get_llm().ainvoke([
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
        return "diagnose"
    # Resume case: user just answered questions, process them
    if state.get("phase") == "awaiting_human" and state.get("human_response"):
        return "interview_process"
    route = state.get("route", "diagnosis")
    return {
        "diagnosis": "interview_generate",
        "research": "research",
        "planning": "standalone_planning",
    }.get(route, "interview_generate")


def route_after_process(state: ClinicalState) -> str:
    """Route: after processing answers, generate more or diagnose?"""
    phase = state.get("phase", "interviewing")
    if phase == "diagnosing":
        return "diagnose"
    return "interview_generate"


# ═══════════════════════════════════════════════════════════════════════════════
# Lab Report Parser (standalone, called from console — not a graph node)
# ═══════════════════════════════════════════════════════════════════════════════

async def parse_lab_report_image(image_path: str) -> dict[str, Any]:
    """Parse a lab report image using multimodal LLM.

    Args:
        image_path: Path to the lab report image file (PNG/JPEG/WEBP).

    Returns:
        Dict with indicators list and metadata, or error info.
    """
    import base64
    from pathlib import Path

    path = Path(image_path)
    if not path.exists():
        return {"error": f"文件不存在: {image_path}"}
    if path.stat().st_size > 20 * 1024 * 1024:
        return {"error": "图片文件过大 (>20MB)"}

    # Read and encode
    with open(path, "rb") as f:
        image_bytes = f.read()

    # Detect MIME
    if image_bytes[:4] == b"\x89PNG":
        mime = "png"
    elif image_bytes[:2] == b"\xff\xd8":
        mime = "jpeg"
    elif image_bytes[:8:4] == b"WEBP" and image_bytes[:4] in (b"RIFF",):
        mime = "webp"
    elif image_bytes[:6] in (b"GIF87a", b"GIF89a"):
        mime = "gif"
    else:
        mime = "jpeg"  # fallback

    image_b64 = base64.b64encode(image_bytes).decode()
    image_url = f"data:image/{mime};base64,{image_b64}"

    # Call vision LLM
    try:
        from .llm import get_vision_llm
        vision = get_vision_llm()
        if vision is None:
            return {"error": "视觉模型未配置。请在 .env 中设置 VISION_API_KEY、VISION_MODEL、VISION_BASE_URL。"}
        response = await vision.ainvoke([
            SystemMessage(content=LAB_PARSER_PROMPT),
            HumanMessage(content=[
                {"type": "text", "text": "请分析这张化验单/检查报告，提取所有异常指标。"},
                {"type": "image_url", "image_url": {"url": image_url}},
            ]),
        ])
        raw = response.content if hasattr(response, "content") else str(response)
        indicators = extract_json(raw)
        if isinstance(indicators, dict):
            indicators = [indicators] if indicators.get("indicator_name") else []
        if not isinstance(indicators, list):
            indicators = []
    except Exception as e:
        return {"error": f"视觉模型调用失败: {type(e).__name__}: {str(e)[:200]}"}

    return {
        "indicators": indicators,
        "filename": path.name,
        "mime_type": mime,
        "parsed_count": len(indicators),
        "abnormal_count": sum(1 for i in indicators if i.get("abnormal")),
    }


def route_after_diagnosis(state: ClinicalState) -> str:
    """Route: after diagnosis, go to treatment plan or end?"""
    if state.get("is_emergency"):
        return "__end__"
    return "treatment_plan"
