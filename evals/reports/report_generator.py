"""Report generation for evaluation results."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from evals.schemas.evaluation_result_schema import EvaluationResult
from evals.text_utils import mean


SECTION_TITLES = [
    ("clinical_safety", "## 2. 临床安全性评价"),
    ("end_to_end", "## 3. 端到端病例任务完成评价"),
    ("llm_expert", "## 4. 大模型专家评分"),
    ("interview_quality", "## 5. 问诊流程质量评价"),
    ("rag", "## 6. 医学检索增强评价"),
    ("faithfulness", "## 7. 答案忠实性与证据一致性评价"),
    ("tool_call", "## 8. 工具调用能力评价"),
    ("diagnosis_report", "## 9. 诊断报告质量评价"),
    ("health_plan", "## 10. 个性化健康方案评价"),
    ("robustness", "## 11. 鲁棒性评价"),
    ("efficiency", "## 12. 效率与成本评价"),
]


def result_to_dict(result: EvaluationResult | dict[str, Any]) -> dict[str, Any]:
    return result.to_dict() if isinstance(result, EvaluationResult) else result


def summarize_results(results: Iterable[EvaluationResult | dict[str, Any]]) -> dict[str, Any]:
    rows = [result_to_dict(result) for result in results]
    by_eval: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_eval[row.get("eval_name", "unknown")].append(row)

    summary: dict[str, Any] = {
        "total_results": len(rows),
        "overall_pass_rate": mean([1.0 if row.get("passed") else 0.0 for row in rows]),
        "overall_average_score": mean([row.get("score", 0.0) for row in rows]),
        "by_eval": {},
    }
    for eval_name, items in by_eval.items():
        metric_values: dict[str, list[float]] = defaultdict(list)
        for item in items:
            for key, value in (item.get("metrics") or {}).items():
                if isinstance(value, (int, float)):
                    metric_values[key].append(float(value))
        summary["by_eval"][eval_name] = {
            "count": len(items),
            "pass_rate": mean([1.0 if item.get("passed") else 0.0 for item in items]),
            "average_score": mean([item.get("score", 0.0) for item in items]),
            "metrics": {key: mean(values) for key, values in sorted(metric_values.items())},
            "error_count": sum(len(item.get("errors") or []) for item in items),
            "warning_count": sum(len(item.get("warnings") or []) for item in items),
        }
    return summary


def write_json_report(results: Iterable[EvaluationResult | dict[str, Any]], path: str | Path) -> Path:
    rows = [result_to_dict(result) for result in results]
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {"summary": summarize_results(rows), "results": rows}
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def write_csv_summary(results: Iterable[EvaluationResult | dict[str, Any]], path: str | Path) -> Path:
    rows = [result_to_dict(result) for result in results]
    summary = summarize_results(rows)["by_eval"]
    metric_keys = sorted(
        {
            key
            for eval_summary in summary.values()
            for key in (eval_summary.get("metrics") or {}).keys()
        }
    )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "eval_name",
                "count",
                "pass_rate",
                "average_score",
                "error_count",
                "warning_count",
                *metric_keys,
            ],
        )
        writer.writeheader()
        for eval_name, eval_summary in sorted(summary.items()):
            row = {
                "eval_name": eval_name,
                "count": eval_summary["count"],
                "pass_rate": eval_summary["pass_rate"],
                "average_score": eval_summary["average_score"],
                "error_count": eval_summary["error_count"],
                "warning_count": eval_summary["warning_count"],
            }
            row.update(eval_summary.get("metrics") or {})
            writer.writerow(row)
    return target


def write_markdown_report(results: Iterable[EvaluationResult | dict[str, Any]], path: str | Path) -> Path:
    rows = [result_to_dict(result) for result in results]
    summary = summarize_results(rows)
    by_eval = summary["by_eval"]

    lines = [
        "# 医学智能体评价报告",
        "",
        "## 1. 总体结果",
        "",
        f"- 评价结果数：{summary['total_results']}",
        f"- 总体平均分：{summary['overall_average_score']:.3f}",
        f"- 总体通过率：{summary['overall_pass_rate']:.3f}",
        "",
    ]

    for eval_name, title in SECTION_TITLES:
        lines.extend([title, ""])
        item = by_eval.get(eval_name)
        if not item:
            lines.extend(["暂无结果。", ""])
            continue
        lines.extend(
            [
                f"- 样本数：{item['count']}",
                f"- 平均分：{item['average_score']:.3f}",
                f"- 通过率：{item['pass_rate']:.3f}",
                f"- 错误数：{item['error_count']}",
                f"- 警告数：{item['warning_count']}",
            ]
        )
        metrics = item.get("metrics") or {}
        if metrics:
            lines.append("")
            lines.append("| 指标 | 平均值 |")
            lines.append("| --- | ---: |")
            for key, value in sorted(metrics.items()):
                lines.append(f"| `{key}` | {value:.3f} |")
        lines.append("")

    lines.extend(["## 13. 主要问题与改进建议", ""])
    issues = _collect_issues(rows)
    if not issues:
        lines.append("未发现严重错误；仍建议结合真实业务日志和专家评审样本继续扩展测试集。")
    else:
        for issue in issues[:20]:
            lines.append(f"- `{issue['eval_name']}` / `{issue['case_id']}`：{issue['message']}")
    lines.append("")

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines), encoding="utf-8")
    return target


def generate_reports(results: Iterable[EvaluationResult | dict[str, Any]], output_json: str | Path) -> dict[str, Path]:
    rows = [result_to_dict(result) for result in results]
    json_path = Path(output_json)
    csv_path = json_path.with_name(json_path.stem + "_summary.csv")
    md_path = json_path.with_name(json_path.stem + "_report.md")
    return {
        "json": write_json_report(rows, json_path),
        "csv": write_csv_summary(rows, csv_path),
        "markdown": write_markdown_report(rows, md_path),
    }


def _collect_issues(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    for row in rows:
        for error in row.get("errors") or []:
            message = error if isinstance(error, str) else error.get("type") or json.dumps(error, ensure_ascii=False)
            if isinstance(error, dict) and error.get("snippet"):
                message = f"{message}: {error['snippet']}"
            issues.append(
                {
                    "eval_name": row.get("eval_name", ""),
                    "case_id": row.get("case_id", ""),
                    "message": str(message),
                }
            )
        if not row.get("errors") and row.get("warnings"):
            issues.append(
                {
                    "eval_name": row.get("eval_name", ""),
                    "case_id": row.get("case_id", ""),
                    "message": "warning: " + "; ".join(map(str, row.get("warnings", [])[:2])),
                }
            )
    return issues
