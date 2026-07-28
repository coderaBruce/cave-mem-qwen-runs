"""Disk-cached generator for the API-only harness.

Wraps GAM's own `OpenAIGenerator` so the researcher loop is byte-for-byte the
original code, and adds three things it lacks:

1. **A disk cache.** Keyed on (model, messages, schema, temperature, top_p,
   max_tokens, n). Without this, every re-run re-pays for the whole experiment,
   and the counterfactual A/B replays we need for utility-verified experience
   would be unaffordable.
2. **Token accounting that survives caching.** A cache hit still reports the
   token count of the original call, so token-cost comparisons against the
   baseline stay honest. `live_tokens` tracks what was actually billed.
3. **A call log.** Every call is recorded with its role, so we can attribute
   cost to planning / integration / reflection rather than one opaque total.

Note on fidelity: GAM's LoCoMo driver constructs generators with
``temperature=0.3`` (not 0), so the upstream results are non-deterministic and
no seed is exposed. We keep 0.3 as the default to match, and expose
``temperature`` so we can run a deterministic variant for our own ablations.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from gam.generator.openai_generator import OpenAIGenerator

from .env import openai_llm_base_url, openai_llm_key


def _stable_hash(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class CachedOpenAIGenerator(OpenAIGenerator):
    """OpenAIGenerator + persistent response cache + per-role token accounting."""

    def __init__(self, config: Dict[str, Any]):
        cache_dir = config.pop("cache_dir", None)
        self.role = config.pop("role", "generic")
        self.enable_cache = config.pop("enable_cache", True)

        config.setdefault("api_key", openai_llm_key())
        if openai_llm_base_url():
            config.setdefault("base_url", openai_llm_base_url())

        super().__init__(config)

        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Accounting. `total_tokens` counts logical cost (cache hits included),
        # `live_tokens` counts what the provider actually billed this process.
        self.total_tokens = 0
        self.live_tokens = 0
        self.n_calls = 0
        self.n_cache_hits = 0
        self.call_log: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        # Mutated by MBR decoding so that repeated draws from an identical prompt
        # get distinct cache keys instead of collapsing to one cached completion.
        # Thread-local: the generator is shared across worker threads, and a plain
        # attribute let workers overwrite each other's salt mid-loop, silently
        # collapsing sample diversity.
        self._salt = threading.local()

    @property
    def cache_salt(self) -> Any:
        return getattr(self._salt, "v", None)

    @cache_salt.setter
    def cache_salt(self, value: Any) -> None:
        self._salt.v = value

    # ---- cache plumbing ----
    def _cache_key(self, msgs, schema, extra_params) -> str:
        return _stable_hash(
            {
                "model": self.model_name,
                "messages": msgs,
                "schema": schema if self.use_schema else None,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "max_tokens": self.max_tokens,
                "n": self.n,
                "extra": extra_params,
                "salt": getattr(self._salt, "v", None),
            }
        )

    def _cache_path(self, key: str) -> Optional[Path]:
        if not self.cache_dir:
            return None
        # Shard by first two hex chars to keep directories small.
        sub = self.cache_dir / key[:2]
        sub.mkdir(parents=True, exist_ok=True)
        return sub / f"{key}.json"

    def _cache_read(self, key: str) -> Optional[Dict[str, Any]]:
        path = self._cache_path(key)
        if not path or not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _cache_write(self, key: str, value: Dict[str, Any]) -> None:
        path = self._cache_path(key)
        if not path:
            return
        try:
            tmp = path.with_suffix(".tmp")
            tmp.write_text(
                json.dumps(value, ensure_ascii=False), encoding="utf-8"
            )
            os.replace(tmp, path)
        except Exception:
            pass

    # ---- main entry point ----
    def generate_single(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        schema: Optional[Dict[str, Any]] = None,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        msgs = self._build_messages(prompt, messages)
        key = self._cache_key(msgs, schema, extra_params)

        cached = self._cache_read(key) if self.enable_cache else None
        if cached is not None:
            tokens = self._tokens_of(cached)
            with self._lock:
                self.total_tokens += tokens
                self.n_calls += 1
                self.n_cache_hits += 1
                self.call_log.append(
                    {"role": self.role, "tokens": tokens, "cached": True}
                )
            return cached

        out = super().generate_single(
            messages=msgs, schema=schema, extra_params=extra_params
        )
        tokens = self._tokens_of(out)
        with self._lock:
            self.total_tokens += tokens
            self.live_tokens += tokens
            self.n_calls += 1
            self.call_log.append(
                {"role": self.role, "tokens": tokens, "cached": False}
            )
        if self.enable_cache:
            self._cache_write(key, out)
        return out

    @staticmethod
    def _tokens_of(out: Dict[str, Any]) -> int:
        try:
            return int(out["response"]["usage"]["total_tokens"])
        except Exception:
            return 0

    def stats(self) -> Dict[str, Any]:
        by_role: Dict[str, int] = {}
        for rec in self.call_log:
            by_role[rec["role"]] = by_role.get(rec["role"], 0) + rec["tokens"]
        return {
            "model": self.model_name,
            "n_calls": self.n_calls,
            "n_cache_hits": self.n_cache_hits,
            "total_tokens": self.total_tokens,
            "live_tokens": self.live_tokens,
            "tokens_by_role": by_role,
        }
