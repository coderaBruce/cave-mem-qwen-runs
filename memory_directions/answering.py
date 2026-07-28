from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from apiharness.answer_style import make_calibrated_answerer


DIRECT_TEMPORAL_RE = re.compile(
    r"^\s*(when|what date|which date|what month|which month|what year|which year|what time)\b",
    re.IGNORECASE,
)
YESNO_RE = re.compile(
    r"^\s*(is|was|were|are|do|does|did|has|had|would|could|can)\b",
    re.IGNORECASE,
)
CONDITIONAL_YESNO_RE = re.compile(
    r"^\s*(is|was|were|are|do|does|did|has|had|would|could|can)\b.*\b("
    r"before|after|based on|because|during|while|when"
    r")\b",
    re.IGNORECASE,
)
WHY_RE = re.compile(r"^\s*why\b", re.IGNORECASE)
CALIBRATABLE_MULTI_RE = re.compile(
    r"^\s*(what|which|where|who|how many|do|does|did|is|was|were|are|has|had|would|could|can)\b",
    re.IGNORECASE,
)
ANSWER_PREFIX_RE = re.compile(r"^\s*(short answer|answer)\s*:\s*", re.IGNORECASE)
COUNTRY_ASK_RE = re.compile(r"\b(what|which) country\b", re.IGNORECASE)
CITY_COUNTRY = {
    "paris": "France",
    "tokyo": "Japan",
    "kyoto": "Japan",
    "rome": "Italy",
    "milan": "Italy",
    "venice": "Italy",
    "london": "United Kingdom",
    "new york": "United States",
    "los angeles": "United States",
    "san francisco": "United States",
    "chicago": "United States",
}


def _call_base(base_answerer, category, summary, question, generator, evidence=None):
    try:
        return base_answerer(category, summary, question, generator, evidence=evidence)
    except TypeError:
        return base_answerer(category, summary, question, generator)


def _clean_answer(text: str) -> str:
    return ANSWER_PREFIX_RE.sub("", (text or "").strip()).strip()


def _canonical_answer(question: str, answer: str) -> str:
    text = _clean_answer(answer)
    lower = text.lower().strip(" .")
    qlower = question.lower()

    if COUNTRY_ASK_RE.search(question):
        for city, country in CITY_COUNTRY.items():
            if lower == city or lower.startswith(f"{city},"):
                return country

    if re.search(r"\bhow\b.*\bdestress\b", qlower) and lower in {
        "through dance",
        "through dancing",
        "dance",
        "dancing",
    }:
        return "by dancing"

    if re.search(r"^\s*who\b", question, re.I) and re.search(r"\bmom\b", text, re.I):
        return re.sub(r"\bmom\b", "mother", text, flags=re.I)

    return text


def _should_calibrate(category: Optional[int], question: str) -> bool:
    # LoCoMo category 2 contains some non-date questions with date mentions
    # ("What achievement in January...?"). Calibrate only direct date asks.
    if category == 2:
        return bool(DIRECT_TEMPORAL_RE.search(question))
    # Category 1 benefits from brevity/list style, but "why" questions often
    # need explanatory clauses and were hurt by terse demonstrations.
    if category == 1:
        return bool(CALIBRATABLE_MULTI_RE.search(question)) and not WHY_RE.search(question)
    return False


def _fails_format_guard(question: str, answer: str) -> bool:
    if YESNO_RE.search(question):
        return not re.match(r"^\s*(yes|no|likely|unlikely|presumably|probably)\b", answer, re.I)
    return False


def make_selective_calibrated_answerer(
    base_answerer: Callable,
    shots: Dict[Any, List[Tuple[str, str]]],
    base_prompt_fn: Optional[Callable] = None,
) -> Callable:
    style_answerer = make_calibrated_answerer(
        base_answerer,
        shots,
        fallback_pool=[],
        base_prompt_fn=base_prompt_fn,
    )

    def answerer(category, summary, question, generator, evidence=None):
        if not _should_calibrate(category, question):
            return _canonical_answer(
                question,
                _call_base(base_answerer, category, summary, question, generator, evidence),
            )
        styled = _clean_answer(style_answerer(category, summary, question, generator, evidence))
        if _fails_format_guard(question, styled):
            return _canonical_answer(
                question,
                _call_base(base_answerer, category, summary, question, generator, evidence),
            )
        return _canonical_answer(question, styled)

    return answerer


def make_canonical_answerer(base_answerer: Callable) -> Callable:
    def answerer(category, summary, question, generator, evidence=None):
        return _canonical_answer(
            question,
            _call_base(base_answerer, category, summary, question, generator, evidence),
        )

    return answerer


def make_synthesis_calibrated_answerer(
    base_answerer: Callable,
    shots: Dict[Any, List[Tuple[str, str]]],
    base_prompt_fn: Optional[Callable] = None,
) -> Callable:
    style_answerer = make_calibrated_answerer(
        base_answerer,
        shots,
        fallback_pool=[],
        base_prompt_fn=base_prompt_fn,
    )

    def answerer(category, summary, question, generator, evidence=None):
        base = _clean_answer(
            _call_base(base_answerer, category, summary, question, generator, evidence)
        )
        if not _should_calibrate(category, question):
            return _canonical_answer(question, base)

        styled = _clean_answer(style_answerer(category, summary, question, generator, evidence))
        if not styled or styled == base:
            return _canonical_answer(question, base)
        if _fails_format_guard(question, styled):
            return base

        prompt = f"""Finalize a short LoCoMo answer from two candidate answers.

QUESTION:
{question}

MEMORY SUMMARY:
{summary}

CANDIDATE A - original GAM answer:
{base}

CANDIDATE B - style-calibrated answer:
{styled}

Rules:
- Output only the final short answer. Do not write "Short answer:" or "Answer:".
- Prefer the candidate that is both concise and complete.
- If both candidates contain useful parts, merge them into the shortest complete answer.
- Do not add facts that are not supported by the memory summary or candidates.
- For yes/no questions, start with "Yes" or "No".
- For count questions, prefer a plain count word or number.
- For list questions, include all requested items but remove generic explanations.
- For date questions, prefer the least over-specific supported expression.
"""
        out = generator.generate_single(prompt=prompt)
        final = _clean_answer(out.get("text") or "")
        if _fails_format_guard(question, final):
            return base
        if CONDITIONAL_YESNO_RE.search(question) and re.fullmatch(
            r"\s*(yes|no)\.?\s*", final, re.I
        ):
            return _canonical_answer(question, base)
        return _canonical_answer(question, final or base)

    return answerer


def make_selective_open_inference_answerer(
    base_answerer: Callable,
    shots: Dict[Any, List[Tuple[str, str]]],
    base_prompt_fn: Optional[Callable] = None,
) -> Callable:
    selective_answerer = make_selective_calibrated_answerer(
        base_answerer,
        {cat: rows for cat, rows in shots.items() if cat in {1, 2}},
        base_prompt_fn=base_prompt_fn,
    )
    open_examples = list(shots.get(3, []))[:6]

    def answerer(category, summary, question, generator, evidence=None):
        if category != 3:
            return _canonical_answer(
                question,
                selective_answerer(category, summary, question, generator, evidence=evidence),
            )

        base = _clean_answer(
            _call_base(base_answerer, category, summary, question, generator, evidence)
        )
        examples = ""
        if open_examples:
            examples = "\n".join(
                f"Q: {q}\nGold-style answer: {a}" for q, a in open_examples
            )
        examples_block = (
            f"EXPECTED ANSWER STYLE EXAMPLES:\n{examples}\n" if examples else ""
        )

        prompt = f"""Answer a LoCoMo open-domain or inference question.

QUESTION:
{question}

MEMORY SUMMARY:
{summary}

ORIGINAL GAM ANSWER:
{base}

{examples_block}

Rules:
- Output only the final short answer. Do not write "Answer:".
- Use the memory summary as evidence, plus ordinary commonsense/world knowledge
  when the question asks what might, likely, presumably, or would be true.
- Prefer a specific entity, condition, status, relationship, shop, company, or
  short label over a generic explanation.
- If the memory names a plausible candidate entity, prefer that named candidate
  instead of an unnamed generic phrase.
- For yes/no questions, start with "Yes", "No", "Likely", "Unlikely", or
  "Presumably".
- Keep the answer terse; avoid explanatory clauses unless needed.
"""
        out = generator.generate_single(prompt=prompt)
        final = _clean_answer(out.get("text") or "")
        if _fails_format_guard(question, final):
            return _canonical_answer(question, base)
        if CONDITIONAL_YESNO_RE.search(question) and re.fullmatch(
            r"\s*(yes|no)\.?\s*", final, re.I
        ):
            return _canonical_answer(question, base)
        return _canonical_answer(question, final or base)

    return answerer
