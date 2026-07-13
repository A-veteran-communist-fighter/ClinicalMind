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


# ── Lazy LLM (simple getter functions — no proxy, no __getattr__) ──────────

_llm_instance = None
_llm_fast_instance = None
_vision_llm_instance = None


def get_llm():
    """Return main LLM (temperature=0.3, 4096 tokens). Creates on first call."""
    global _llm_instance
    if _llm_instance is None:
        from langchain_openai import ChatOpenAI
        cfg = _resolve_config()
        _llm_instance = ChatOpenAI(
            model=MODEL_NAME, api_key=cfg["api_key"],
            base_url=cfg["base_url"] or None,
            temperature=0.3, max_tokens=4096,
            timeout=120.0, max_retries=2,
        )
        print(f"[ClinicalMind] LLM: {cfg['provider']} | Model: {MODEL_NAME}")
    return _llm_instance


def get_llm_fast():
    """Return fast LLM (temperature=0.1, 512 tokens). Creates on first call."""
    global _llm_fast_instance
    if _llm_fast_instance is None:
        from langchain_openai import ChatOpenAI
        cfg = _resolve_config()
        _llm_fast_instance = ChatOpenAI(
            model=MODEL_NAME, api_key=cfg["api_key"],
            base_url=cfg["base_url"] or None,
            temperature=0.1, max_tokens=512,
            timeout=60.0, max_retries=2,
        )
    return _llm_fast_instance


def get_vision_llm():
    """Return vision-capable LLM for lab report parsing.

    Requires separate configuration in .env:
      VISION_API_KEY=sk-...         (required — different from main LLM)
      VISION_BASE_URL=https://...   (optional)
      VISION_MODEL=gpt-4o           (optional, defaults to gpt-4o)

    Returns None if vision is not configured, so callers can degrade gracefully.
    """
    global _vision_llm_instance
    if _vision_llm_instance is None:
        from langchain_openai import ChatOpenAI

        vision_key = os.getenv("VISION_API_KEY", "").strip()

        # No vision config → silently return None (feature not available)
        if not vision_key:
            _vision_llm_instance = None
            return None

        vision_model = os.getenv("VISION_MODEL", "gpt-4o").strip()
        vision_url = os.getenv("VISION_BASE_URL", "https://api.openai.com/v1").strip()

        _vision_llm_instance = ChatOpenAI(
            model=vision_model,
            api_key=vision_key,
            base_url=vision_url or None,
            temperature=0.1, max_tokens=2048,
            timeout=120.0, max_retries=2,
        )
        print(f"[ClinicalMind] Vision LLM: {vision_model}")
    return _vision_llm_instance


def has_vision() -> bool:
    """Check if vision model is configured (without initializing it)."""
    return bool(os.getenv("VISION_API_KEY", "").strip())
