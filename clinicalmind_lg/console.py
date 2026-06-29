#!/usr/bin/env python3
"""ClinicalMind Console — powered by LangGraph.

Interactive medical consultation using LangGraph's interrupt() for human-in-the-loop.

Usage:
    python -m clinicalmind_lg.console
"""

import json
import sys
import uuid
from typing import Any

from langgraph.types import Command

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


def ask_questions(questions: list[dict]) -> str | None:
    """Present questions and collect answers. Returns '||'-separated answers or None to quit."""
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
        if raw.lower() in ("skip", ""):
            answers.append("跳过")
            continue

        # Map numeric choices to option text
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
    """Run consultation using LangGraph with human-in-the-loop interrupts.

    Pattern:
    1. graph.ainvoke() runs until interrupt() is hit → returns normally
    2. Check graph.get_state(config).interrupts for interrupt data
    3. Present questions, collect answers
    4. Resume with graph.ainvoke(Command(resume=answer), config)
    5. Repeat until graph completes (next=(), no interrupts)
    """
    state = dict(initial_state(chief_complaint))
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    while True:
        result = await graph.ainvoke(state, config)
        gs = graph.get_state(config)

        if gs is None:
            return result

        next_nodes = gs.next or ()
        interrupts = gs.interrupts or ()

        # Check if graph is done
        if not next_nodes and not interrupts:
            return result

        # Handle interrupts
        if interrupts:
            for it in interrupts:
                iv = it.value if hasattr(it, "value") else it

                if isinstance(iv, dict) and iv.get("type") == "ask_questions":
                    questions = iv.get("questions", [])
                    diffs = iv.get("differential_diagnoses", [])
                    round_num = iv.get("round", "?")
                    collected = iv.get("collected_count", 0)

                    print(f"\n  [Round {round_num}] ({collected} items collected)")
                    print_differential(diffs)

                    # Check for red flags in the interrupt
                    flags = iv.get("red_flags", [])
                    print_red_flags(flags)

                    response = ask_questions(questions)
                    if response is None:
                        return None
                    if response == "restart":
                        return {"_restart": True}

                    state = Command(resume=response)

                else:
                    # Unknown interrupt — show it
                    print(f"\n[Debug] Interrupt: {str(iv)[:200]}")
                    raw = input("  > ").strip()
                    if raw.lower() in ("quit", "exit", "q"):
                        return None
                    state = Command(resume=raw)
        else:
            # No interrupt but next_nodes not empty?
            # This is a state where the graph expects input but hasn't interrupted
            if next_nodes:
                print(f"\n[Waiting at: {next_nodes}]")
                raw = input("  > ").strip()
                if raw.lower() in ("quit", "exit", "q"):
                    return None
                state = Command(resume=raw)
            else:
                return result


# ── Main loop ───────────────────────────────────────────────────────────────

async def main():
    print_banner()

    # Quick LLM connectivity check — fail early with a clear message
    try:
        from .llm import llm
        llm._get()  # force lazy init
    except RuntimeError as e:
        print(f"\n  ERROR: {e}")
        print("\n  Please edit .env and try again.")
        return

    while True:
        try:
            print("-" * 60)
            print("Enter chief complaint / medical question:")
            user_input = input("> ").strip()

            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "q"):
                print("\nGoodbye!")
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
