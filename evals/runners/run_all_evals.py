"""Run all ClinicalMind evaluators."""

from __future__ import annotations

import argparse
from pathlib import Path

from evals.evaluators import ALL_EVAL_NAMES, EVALUATOR_REGISTRY
from evals.evaluators.base import load_config
from evals.reports.report_generator import generate_reports
from evals.schemas.case_schema import iter_cases, load_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all medical agent evaluations.")
    parser.add_argument("--cases", required=True, help="Path to JSONL cases.")
    parser.add_argument("--output", default="evals/outputs/all_results.json", help="Output JSON report path.")
    parser.add_argument("--config", default=None, help="Optional evaluation config JSON path.")
    args = parser.parse_args()

    config = load_config(args.config)
    cases = list(iter_cases(load_jsonl(args.cases)))
    results = []
    for eval_name in ALL_EVAL_NAMES:
        evaluator = EVALUATOR_REGISTRY[eval_name](config=config)
        results.extend(evaluator.evaluate_many(cases))

    paths = generate_reports(results, Path(args.output))
    for kind, path in paths.items():
        print(f"{kind}: {path}")


if __name__ == "__main__":
    main()
