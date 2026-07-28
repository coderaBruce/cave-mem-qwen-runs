"""Path 2: an LLM bottleneck probe, to fix the zero-information-features problem.

The learned per-query router failed (E29) because every cheap feature we tried —
retrieval score geometry, question form, groundedness — carried no signal about
which strategy would win. Label accuracy was at or below the majority-class rate
on all three corpora, while oracle routing was worth +2.5 to +6.0 F1. The
headroom is real; the features are blind.

So instead of inferring the bottleneck from surface statistics, we ask. After the
base pass the model has a question, the evidence it retrieved, and its own draft
answer. It is in a good position to say what is limiting it:

  MISSING_EVIDENCE   the answer is not in what was retrieved -> retrieve wider
  WORDING            the answer is present but the draft paraphrases it -> hand
                     the verbatim source to the answer step
  SYNTHESIS          no single passage states it; it must be composed across the
                     narrative -> calibrate answer style instead

This is one extra cheap call per query on top of the base pass. That cost is real
and we report it rather than hiding it: routed inference is ~2x baseline for
queries the router moves off the default.

The probe is deliberately asked about *the bottleneck*, not about *which strategy
to use* — the model has no knowledge of our strategy inventory, so asking it to
pick one would be asking it to guess.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

BOTTLENECKS = ["MISSING_EVIDENCE", "WORDING", "SYNTHESIS"]

# bottleneck -> strategy
PROBE_TO_STRATEGY = {
    "MISSING_EVIDENCE": "widen",
    "WORDING": "evidence",
    "SYNTHESIS": "answer_style",
}

_SCHEMA = {
    "type": "object",
    "properties": {
        "reason": {"type": "string"},
        "bottleneck": {"type": "string", "enum": BOTTLENECKS},
        "confidence": {"type": "number"},
    },
    "required": ["reason", "bottleneck", "confidence"],
    "additionalProperties": False,
}

_SYSTEM = """You diagnose why a retrieval-based QA system may have answered imperfectly.
You are NOT asked to answer the question or to judge correctness. You are asked to
identify the single most limiting factor, choosing exactly one of:

- MISSING_EVIDENCE: the retrieved passages do not appear to contain the information the
  question asks for. The system would need to search more widely.
- WORDING: the retrieved passages do contain the answer, but the draft answer restates
  or generalises it instead of using the specific name, date, title or phrase present in
  the source.
- SYNTHESIS: no single passage states the answer. Answering requires combining or
  summarising across the narrative, so there is no exact phrase to copy.

Return ONLY valid JSON."""

_USER = """QUESTION:
{question}

RETRIEVED PASSAGES (truncated):
{evidence}

THE SYSTEM'S DRAFT ANSWER:
{draft}

Which single factor most limits answering this question well?
Return JSON: {{"reason": "<one sentence>", "bottleneck": "MISSING_EVIDENCE" | "WORDING" | "SYNTHESIS", "confidence": <0..1>}}"""


def probe_bottleneck(
    generator,
    question: str,
    evidence: str,
    draft_answer: str,
    max_evidence_chars: int = 3500,
) -> Tuple[Optional[str], float, int]:
    """Return (bottleneck, confidence, tokens). Bottleneck is None on failure."""
    prompt = _USER.format(
        question=question,
        evidence=(evidence or "")[:max_evidence_chars] or "(nothing retrieved)",
        draft=(draft_answer or "(no answer produced)"),
    )
    try:
        resp = generator.generate_single(
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": prompt},
            ],
            schema=_SCHEMA,
        )
    except Exception:
        return None, 0.0, 0

    tokens = 0
    try:
        tokens = int(resp["response"]["usage"]["total_tokens"])
    except Exception:
        pass

    data = resp.get("json")
    if not data:
        from .r2mem import _loose_json

        data = _loose_json(resp.get("text", "")) or {}
    b = str(data.get("bottleneck", "")).strip().upper()
    if b not in BOTTLENECKS:
        return None, 0.0, tokens
    try:
        conf = float(data.get("confidence", 0.5))
    except Exception:
        conf = 0.5
    return b, conf, tokens


def strategy_for(bottleneck: Optional[str], default: str = "base") -> str:
    if bottleneck is None:
        return default
    return PROBE_TO_STRATEGY.get(bottleneck, default)
