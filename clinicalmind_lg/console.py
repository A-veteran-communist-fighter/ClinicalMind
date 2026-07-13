#!/usr/bin/env python3
"""ClinicalMind Console — LangGraph Edition (state-driven, no interrupt()).

Compatible with LangGraph >= 1.x.

Usage:
    python -m clinicalmind_lg.console
"""

import asyncio
import json
import sys
import uuid
from typing import Any

from .graph import graph
from .state import initial_state


# ── Display helpers ────────────────────────────────────────────────────────

def print_banner():
    print()
    print("=" * 60)
    print("  ClinicalMind - LangGraph Console Edition")
    print("  Multi-Agent Medical Consultation System")
    print("=" * 60)
    print()
    print("  Commands:")
    print("    quit/exit/q  - Exit")
    print("    restart      - Start new consultation")
    print("    skip         - Skip current question")
    print()
    print("  DISCLAIMER: AI-generated output is for reference only.")
    print("  If you have emergency symptoms, seek medical attention!")
    print()


def print_red_flags(flags: list[str]):
    if not flags:
        return
    print()
    print("  !! SAFETY ALERT !!")
    for f in flags:
        print(f"    {f}")


def print_differential(diffs: list[dict]):
    if not diffs:
        return
    print(f"\n  Current hypotheses:")
    for d in diffs[:5]:
        conf = d.get("confidence", "low")
        icon = {"high": "[++]", "medium": "[+-]", "low": "[--]"}.get(conf, "[??]")
        print(f"    {icon} {d.get('diagnosis', '?')}")
        if d.get("reason"):
            print(f"         {d['reason'][:100]}")


async def ask_questions(questions: list[dict]) -> str | None:
    """Present questions and collect answers.

    Returns '||'-separated answers, None to quit, or 'restart' to restart.
    Supports 'lab <path>' command to parse lab reports mid-interview.
    """
    answers = []
    for i, q in enumerate(questions):
        qtype = q.get("type", "text")
        opts = q.get("options", [])
        hint = q.get("hint", "")

        print(f"\n  Q{i + 1}: {q.get('text', '')}")
        if hint:
            print(f"         ({hint})")

        if qtype in ("choice", "multi_choice") and opts:
            for j, opt in enumerate(opts, 1):
                print(f"         [{j}] {opt}")
            prompt = "  Choice: "
        else:
            prompt = "  Answer: "

        try:
            raw = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            return None

        if raw.lower() in ("quit", "exit", "q"):
            return None
        if raw.lower() == "restart":
            return "restart"
        # Lab report mid-interview
        if raw.lower().startswith("lab "):
            image_path = raw[4:].strip().strip('"').strip("'")
            print(f"  Parsing lab report: {image_path}")
            try:
                from .nodes import parse_lab_report_image
                parsed = await parse_lab_report_image(image_path)
                if "error" in parsed:
                    print(f"  Error: {parsed['error']}")
                else:
                    inds = parsed.get("indicators", [])
                    if not inds:
                        print("  No abnormal indicators detected.")
                    else:
                        print(f"  Found {len(inds)} indicators, "
                              f"{sum(1 for x in inds if x.get('abnormal'))} abnormal.")
                    return f"LAB_PARSED:{json.dumps(parsed)}"
            except Exception as e:
                print(f"  Error: {e}")
            # Re-ask this question
            print(f"  (Please answer the question above)")
            try:
                raw = input(prompt).strip()
            except (EOFError, KeyboardInterrupt):
                return None
        if raw.lower() in ("skip", ""):
            answers.append("跳过")
            continue

        if qtype in ("choice", "multi_choice") and opts and raw.replace(",", "").replace(" ", "").isdigit():
            try:
                indices = [int(x.strip()) - 1 for x in raw.split(",")]
                mapped = ", ".join(opts[j] for j in indices if 0 <= j < len(opts))
                answers.append(mapped if mapped else raw)
            except (ValueError, IndexError):
                answers.append(raw)
        else:
            answers.append(raw)

    return "||".join(answers)


def print_diagnosis(diag: dict):
    print()
    print("=" * 60)
    print("  DIAGNOSIS REPORT")
    print("=" * 60)
    print(f"  Primary:   {diag.get('primary_diagnosis', 'N/A')}")
    print(f"  Confidence:{diag.get('confidence','N/A')} | Severity: {diag.get('severity','N/A')}")

    diffs = diag.get("differential_diagnoses", [])
    if diffs:
        print(f"\n  Differential Diagnoses ({len(diffs)}):")
        for i, d in enumerate(diffs, 1):
            print(f"    {i}. {d.get('diagnosis','?')} (ICD-11: {d.get('icd11_code','N/A')})")
            if d.get("reasoning"):
                print(f"       {d['reasoning'][:130]}")

    for section, label in [
        ("key_findings", "Key Findings"),
        ("recommended_tests", "Recommended Tests"),
        ("recommended_actions", "Recommended Actions"),
    ]:
        items = diag.get(section, [])
        if items:
            print(f"\n  {label}:")
            for item in items:
                print(f"    * {item}")

    flags = diag.get("red_flags", [])
    if flags:
        print(f"\n  !! RED FLAGS: {'; '.join(flags)}")

    if diag.get("follow_up_required"):
        print(f"\n  Follow-up: {diag.get('follow_up_timeline', 'As advised')}")

    print(f"\n  {diag.get('disclaimer', 'AI-generated, for reference only.')}")
    print("=" * 60)


def print_treatment_plan(plan: dict):
    if not plan or not plan.get("title"):
        return
    print()
    print("=" * 60)
    print(f"  TREATMENT PLAN: {plan.get('title', 'N/A')}")
    print("=" * 60)

    for section, label in [
        ("goals", "Goals"),
        ("non_pharmacological", "Non-Pharmacological"),
        ("nursing_plan", "Nursing Plan"),
        ("lifestyle_modifications", "Lifestyle"),
    ]:
        items = plan.get(section, [])
        if items:
            print(f"\n  {label}:")
            for item in items:
                print(f"    * {item}")

    meds = plan.get("medication_discussion_points", [])
    if meds:
        print("\n  Medication Discussion Points:")
        for m in meds:
            print(f"    * {m.get('category','?')}: {m.get('notes','')}")

    follow_up = plan.get("follow_up_schedule", [])
    if follow_up:
        print("\n  Follow-up Schedule:")
        for f in follow_up:
            print(f"    * {f.get('timeframe','')}: {f.get('action','')}")

    safety = plan.get("safety_notes", [])
    if safety:
        print(f"\n  Safety: {'; '.join(safety)}")

    print("=" * 60)


def print_research(answer: str):
    print()
    print("-" * 60)
    print(answer)
    print("-" * 60)
    print("DISCLAIMER: For reference only. Consult a qualified professional.")


# ── Graph runner ────────────────────────────────────────────────────────────

async def run_consultation(chief_complaint: str) -> dict[str, Any] | None:
    """Run consultation using state-driven pauses (no interrupt()).

    Compatible with LangGraph >= 1.x.

    Pattern:
    1. graph.ainvoke(initial_state) → runs entry → interview_generate → END
    2. Read phase from result: if "awaiting_human", show questions
    3. Collect answers, put in human_response, run graph again
    4. Graph runs entry → interview_process → route → generate again or diagnose
    5. Repeat until phase is not "awaiting_human"
    """
    state = dict(initial_state(chief_complaint))
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    MAX_ROUNDS = 15
    for _ in range(MAX_ROUNDS):
        result = await graph.ainvoke(state, config)

        phase = result.get("phase", "completed")
        route = result.get("route", "diagnosis")

        # Non-diagnosis paths complete in one shot
        if route in ("research", "planning"):
            return result

        # Emergency goes straight to diagnose
        if phase == "diagnosed" or phase == "completed":
            return result

        # Awaiting human input: show questions, collect answers
        if phase == "awaiting_human":
            questions = result.get("current_questions", [])
            diffs = result.get("differential_diagnoses", [])
            round_num = result.get("interview_round", "?")
            collected_info = result.get("collected_info", {})

            meaningful = sum(1 for v in collected_info.values() if v and v not in ("无", "没有", "跳过"))
            print(f"\n  [Round {round_num}] ({meaningful} items collected)")
            print_differential(diffs)
            print_red_flags(result.get("red_flags", []))

            if not questions:
                # No questions generated — retry with empty answer to force next iteration
                state = {"human_response": "跳过"}
                continue

            response = await ask_questions(questions)
            if response is None:
                return None
            if response == "restart":
                return {"_restart": True}

            # Resume: feed answer back into state, graph picks up at interview_process
            state = {"human_response": response}
            continue

        # Diagnosing or diagnosed: graph finished the diagnosis path
        if phase in ("diagnosing",):
            # The diagnose node should follow next; run once more with empty update
            state = {}
            continue

        # Fallback: graph ended unexpectedly
        return result

    return result


# ── Main loop ───────────────────────────────────────────────────────────────

async def main():
    print_banner()

    # LLM connectivity check
    try:
        from .llm import get_llm, has_vision
        get_llm()
    except RuntimeError as e:
        print(f"\n  ERROR: {e}")
        print("\n  Please edit .env and try again.")
        return

    if has_vision():
        print("  Vision model configured — lab report parsing enabled.")
    else:
        print("  Tip: Set VISION_API_KEY in .env to enable lab report parsing.")

    # Active consultation state (for lab command during interview)
    _active_config: dict = {}
    _active_state: dict = {}

    while True:
        try:
            print("-" * 60)
            print("Enter chief complaint / medical question (or 'lab <path>' to parse a lab report):")
            user_input = input("> ").strip()

            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "q"):
                print("\nGoodbye!")
                break

            # ── Lab report upload command ──
            if user_input.lower().startswith("lab "):
                image_path = user_input[4:].strip().strip('"').strip("'")
                print(f"\n  Parsing lab report: {image_path}")
                try:
                    from .nodes import parse_lab_report_image
                    parsed = await parse_lab_report_image(image_path)
                except Exception as e:
                    print(f"\n  Error: {e}")
                    continue

                if "error" in parsed:
                    print(f"\n  Error: {parsed['error']}")
                    continue

                indicators = parsed.get("indicators", [])
                if not indicators:
                    print("\n  No abnormal indicators detected in this report.")
                else:
                    print(f"\n  Parsed {parsed['parsed_count']} indicators, "
                          f"{parsed['abnormal_count']} abnormal:")
                    for ind in indicators:
                        if ind.get("abnormal"):
                            lvl = ind.get("abnormal_level", "unknown")
                            lvl_mark = {"critical": "!!", "severe": "!", "moderate": "⚠"}.get(lvl, "")
                            print(f"    {lvl_mark} {ind.get('indicator_name','?'):12s} "
                                  f"{str(ind.get('value','?')):8s} {ind.get('unit',''):6s} "
                                  f"(参考: {ind.get('reference_range','N/A')})")

                # If in active consultation, feed into graph state
                if _active_config and _active_state is not None:
                    from .graph import graph
                    _active_state = {"lab_reports": [parsed]}
                    result = await graph.ainvoke(_active_state, _active_config)
                    # Re-generate questions to incorporate lab data
                    result = await graph.ainvoke({}, _active_config)
                    _active_state = {}
                    phase = result.get("phase", "")
                    if phase == "awaiting_human":
                        questions = result.get("current_questions", [])
                        if questions:
                            print(f"\n  (Lab data incorporated. Continuing interview...)")
                            print_differential(result.get("differential_diagnoses", []))
                            response = await ask_questions(questions)
                            if response in (None, "restart"):
                                break
                            _active_state = {"human_response": response}
                            continue
                else:
                    print("\n  Tip: Start a consultation first, then use 'lab' during")
                    print("  the interview to incorporate lab results into diagnosis.")
                continue
                break

            print("\nProcessing...")
            result = await run_consultation(user_input)

            if result is None:
                print("\nConsultation cancelled.")
                continue
            if result.get("_restart"):
                continue
            if not result:
                continue

            route = result.get("route", "diagnosis")

            if route == "research":
                print_research(result.get("research_answer", "No answer."))
            else:
                print_red_flags(result.get("red_flags", []))
                diag = result.get("diagnosis_report")
                if diag:
                    print_diagnosis(diag)
                plan = result.get("treatment_plan")
                if plan:
                    print_treatment_plan(plan)

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\n[Error] {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
