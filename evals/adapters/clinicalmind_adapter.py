"""Optional adapter boundary for running ClinicalMind from eval cases.

The current evaluators are offline-first and do not call the business graph by
default. This adapter is intentionally thin so future live evals can call the
main workflow without changing evaluator logic.
"""

from __future__ import annotations

from typing import Any


class ClinicalMindAdapter:
    async def run_case(self, case: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError(
            "Live ClinicalMind execution is not wired by default. "
            "Provide an adapter that returns agent_output-compatible fields."
        )
