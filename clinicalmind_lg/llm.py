"""LLM configuration for ClinicalMind LangGraph.

Loads API key and model from environment (.env file or system env vars).
Uses lazy initialization — import never fails, only the first LLM call validates config.
"""

import json
import os
import re
from pathlib import Path
from typing import Any

# ── Load .env ──────────────────────────────────────────────────────────────

_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    with open(_env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip('"').strip("'")
            if key not in os.environ:
                os.environ[key] = value


# ── Config resolution ──────────────────────────────────────────────────────

MODEL_NAME = os.getenv("DEFAULT_LLM_MODEL", "deepseek-chat")

def _resolve_config() -> dict[str, str]:
    """Resolve LLM provider from environment variables."""
    providers = [
        ("deepseek", "DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL"),
        ("openai", "OPENAI_API_KEY", "OPENAI_BASE_URL"),
        ("glm", "GLM_API_KEY", "GLM_BASE_URL"),
        ("moonshot", "MOONSHOT_API_KEY", "MOONSHOT_BASE_URL"),
        ("dashscope", "DASHSCOPE_API_KEY", "DASHSCOPE_BASE_URL"),
    ]
    for name, key_env, url_env in providers:
        api_key = os.getenv(key_env)
        if api_key:
            return {
                "provider": name,
                "api_key": api_key,
                "base_url": os.getenv(url_env, ""),
            }
    raise RuntimeError(
        "No LLM API key found.\n"
        "Copy .env.example to .env and fill in one API key.\n"
        "Supported: DEEPSEEK_API_KEY, OPENAI_API_KEY, GLM_API_KEY, "
        "MOONSHOT_API_KEY, DASHSCOPE_API_KEY"
    )


# ── JSON helpers (available at import time, no LLM needed) ──────────────────

def extract_json(text: str) -> dict[str, Any]:
    """Robust JSON extraction from LLM output."""
    raw = text.strip()
    raw = re.sub(r"```(?:json)?\s*", "", raw)
    raw = re.sub(r"```\s*$", "", raw)
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            return json.loads(match.group())
        return {"parse_error": True, "raw": text[:500]}


def parse_json_response(response: Any) -> dict[str, Any]:
    """Extract JSON from a LangChain AIMessage response."""
    content = response.content if hasattr(response, "content") else str(response)
    return extract_json(content)


# ── Lazy LLM proxy ─────────────────────────────────────────────────────────

class _LazyLLM:
    """Proxy that creates the real ChatOpenAI on first method call.

    Importing this module never fails — only the first actual LLM
    API call triggers config validation and client creation.
    """

    def __init__(self, fast: bool = False):
        self._fast = fast
        self._instance = None
        self._initialized = False

    def _get(self):
        if not self._initialized:
            from langchain_openai import ChatOpenAI
            cfg = _resolve_config()
            if self._fast:
                self._instance = ChatOpenAI(
                    model=MODEL_NAME, api_key=cfg["api_key"],
                    base_url=cfg["base_url"] or None,
                    temperature=0.1, max_tokens=512,
                    timeout=60.0, max_retries=2,
                )
            else:
                self._instance = ChatOpenAI(
                    model=MODEL_NAME, api_key=cfg["api_key"],
                    base_url=cfg["base_url"] or None,
                    temperature=0.3, max_tokens=4096,
                    timeout=120.0, max_retries=2,
                )
                print(f"[ClinicalMind] LLM: {cfg['provider']} | Model: {MODEL_NAME}")
            self._initialized = True
        return self._instance

    def __getattr__(self, name: str):
        return getattr(self._get(), name)

    def __repr__(self):
        if self._initialized:
            return repr(self._instance)
        return f"<LazyLLM(fast={self._fast}, pending)>"


# Public lazy instances
llm = _LazyLLM(fast=False)
llm_fast = _LazyLLM(fast=True)
