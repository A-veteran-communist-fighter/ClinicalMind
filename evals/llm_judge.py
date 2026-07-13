"""Unified LLM judge interface for semantic evaluation."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from string import Template
from typing import Any


class JudgeUnavailable(RuntimeError):
    """Raised when an evaluator requests a disabled or unconfigured judge."""


def extract_json(text: str) -> dict[str, Any]:
    raw = text.strip()
    raw = re.sub(r"```(?:json)?\s*", "", raw)
    raw = re.sub(r"```\s*$", "", raw).strip()
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {"value": data}
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            raise
        data = json.loads(match.group())
        return data if isinstance(data, dict) else {"value": data}


class BaseJudgeClient:
    def judge_json(self, prompt_template: str, variables: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError


class DisabledLLMJudgeClient(BaseJudgeClient):
    def __init__(self, reason: str = "LLM judge is disabled or not configured"):
        self.reason = reason

    def judge_json(self, prompt_template: str, variables: dict[str, Any]) -> dict[str, Any]:
        raise JudgeUnavailable(self.reason)


class OpenAICompatibleJudgeClient(BaseJudgeClient):
    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 1200,
    ) -> None:
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key, base_url=base_url or None)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def judge_json(self, prompt_template: str, variables: dict[str, Any]) -> dict[str, Any]:
        rendered = Template(prompt_template).safe_substitute(
            {k: _stringify(v) for k, v in variables.items()}
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": rendered}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or "{}"
        return extract_json(content)


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, indent=2)


def load_prompt(prompt_name: str) -> str:
    path = Path(__file__).parent / "prompts" / prompt_name
    return path.read_text(encoding="utf-8")


def make_judge_client(config: dict[str, Any]) -> BaseJudgeClient:
    judge_cfg = config.get("llm_judge", {})
    if not judge_cfg.get("enabled", False):
        return DisabledLLMJudgeClient()

    key_env = judge_cfg.get("api_key_env", "OPENAI_API_KEY")
    api_key = os.getenv(key_env)
    if not api_key:
        return DisabledLLMJudgeClient(f"LLM judge enabled but {key_env} is not set")

    base_url_env = judge_cfg.get("base_url_env", "OPENAI_BASE_URL")
    return OpenAICompatibleJudgeClient(
        api_key=api_key,
        model=judge_cfg.get("model", os.getenv("DEFAULT_LLM_MODEL", "gpt-4o-mini")),
        base_url=os.getenv(base_url_env) or judge_cfg.get("base_url"),
        temperature=float(judge_cfg.get("temperature", 0.1)),
        max_tokens=int(judge_cfg.get("max_tokens", 1200)),
    )
