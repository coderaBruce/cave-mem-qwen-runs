"""Environment / credential loading for the API-only harness.

Keys are read from `research1/.env` (git-ignored) or the process environment.
Nothing here ever prints a key.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]  # research1/
ENV_FILE = REPO_ROOT / ".env"

_LOADED = False


def load_env() -> None:
    """Load research1/.env into os.environ (does not override existing vars)."""
    global _LOADED
    if _LOADED:
        return
    if ENV_FILE.exists():
        for raw in ENV_FILE.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip().strip('"').strip("'")
            os.environ.setdefault(key, val)
    _LOADED = True


def require(name: str) -> str:
    load_env()
    val = os.environ.get(name)
    if not val:
        raise RuntimeError(
            f"{name} is not set. Add it to {ENV_FILE} (see r2m-api/.env.example)."
        )
    return val


def get(name: str, default: str | None = None) -> str | None:
    load_env()
    return os.environ.get(name, default)


def openai_key() -> str:
    return require("OPENAI_API_KEY")


def openai_base_url() -> str | None:
    """Optional override, e.g. an Azure or proxy endpoint."""
    return get("OPENAI_BASE_URL")


def openai_llm_key() -> str:
    """Key for chat/completion calls; falls back to OPENAI_API_KEY."""
    return get("OPENAI_LLM_API_KEY") or openai_key()


def openai_llm_base_url() -> str | None:
    """Optional chat/completion endpoint override.

    This lets a local vLLM server serve Qwen while embeddings still use the
    normal OpenAI endpoint.
    """
    return get("OPENAI_LLM_BASE_URL") or openai_base_url()


def openai_embed_key() -> str:
    """Key for embedding calls; falls back to OPENAI_API_KEY."""
    return get("OPENAI_EMBED_API_KEY") or openai_key()


def openai_embed_base_url() -> str | None:
    """Optional embedding endpoint override."""
    return get("OPENAI_EMBED_BASE_URL") or openai_base_url()
