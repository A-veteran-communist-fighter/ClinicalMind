"""Run one ClinicalMind evaluator."""

from __future__ import annotations

import argparse
from pathlib import Path

from evals.evaluators import EVALUATOR_REGISTRY
from evals.evaluators.base import load_config
from evals.reports.report_generator import generate_reports
from evals.schemas.case_schema import iter_cases, load_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single medical agent evaluation.")
    parser.add_argument("--eval", required=True, choices=sorted(EVALUATOR_REGISTRY.keys()), help="Evaluator name.")
    parser.add_argument("--cases", required=True, help="Path to JSONL cases.")
    parser.add_argument("--output", default=None, help="Output JSON report path.")
    parser.add_argument("--config", default=None, help="Optional evaluation config JSON path.")
    args = parser.parse_args()

    config = load_config(args.config)
    cases = list(iter_cases(load_jsonl(args.cases)))
    evaluator = EVALUATOR_REGISTRY[args.eval](config=config)
    results = evaluator.evaluate_many(cases)

    output = args.output or f"evals/outputs/{args.eval}_results.json"
    paths = generate_reports(results, Path(output))
    for kind, path in paths.items():
        print(f"{kind}: {path}")


if __name__ == "__main__":
    main()
