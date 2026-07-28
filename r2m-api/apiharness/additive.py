"""OUR METHOD: additive experience injection.

Motivated by a control R²-Mem never ran (notes/02-experiment-log.md E6). Running
their online agent with an **empty** bank — their `*_PROMPT_exp` templates, zero
experiences — scores 54.40 F1 against GAM's 60.59 on the same questions. So the
template rewrite alone costs **-6.19 F1**, and the experience content then buys
back **+5.27**. Their mechanism works; it is funding a template that costs more
than it earns.

Diffing the templates shows what was lost. GAM's `InfoCheck_PROMPT` specifies:

    1. Decompose REQUEST: identify the key pieces required to answer completely.
    2. Check RESULT: for each required piece, check RESULT provides it clearly
       and specifically. RESULT must be specific enough that someone could write
       a final answer directly from it without further retrieval.
    3. "enough" = true ONLY IF RESULT covers all required pieces with sufficient
       clarity and specificity.

`InfoCheck_PROMPT_exp` deletes all three and substitutes "STEP 1: Analyze
Experience Applicability". The strict sufficiency bar is simply gone — which is
precisely the premature-stopping failure we measured (iterations 1.48 -> 1.32
*with an empty bank*).

The fix is conservative: keep every GAM instruction verbatim and *append* an
experience block, rather than rewriting the procedure around the experience.
Experience becomes advisory context instead of replacing the agent's method.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from gam.prompts import (
    GenerateRequests_PROMPT,
    InfoCheck_PROMPT,
    Planning_PROMPT,
)
from gam.schemas import (
    GENERATE_REQUESTS_SCHEMA,
    INFO_CHECK_SCHEMA,
    PLANNING_SCHEMA,
    ReflectionDecision,
    SearchPlan,
)

from .r2mem import (
    ResearchAgentExp,
    _abstract_situation_planning,
    _abstract_situation_reflection,
    format_experiences,
)

_PLANNING_BLOCK = """

RELEVANT EXPERIENCE FROM PAST SEARCHES:
{experience}

HOW TO USE THE EXPERIENCE:
- These are heuristics distilled from earlier searches on similar situations.
- Treat them as advisory only. If one does not fit this QUESTION, ignore it.
- Let them influence HOW you decompose info_needs and choose tools.
- Do NOT quote or reference them in your output.
- They do NOT relax any requirement stated above.
"""

_REFLECTION_BLOCK = """

RELEVANT EXPERIENCE FROM PAST SEARCHES:
{experience}

HOW TO USE THE EXPERIENCE:
- These are heuristics distilled from earlier sufficiency judgements.
- Treat them as advisory only. If one does not fit, ignore it.
- They do NOT lower the bar set above: "enough" is still true ONLY IF every
  required piece of information is present with sufficient clarity and
  specificity.
"""


def _splice(prompt: str, block: str) -> str:
    """Insert `block` before the output spec so the procedure stays intact.

    Appending after "return ONLY the JSON object" tends to get treated as
    trailing noise, and prepending displaces the task framing, so we cut in
    just ahead of the output contract.
    """
    for marker in ("OUTPUT JSON SPEC", "OUTPUT REQUIREMENTS", "OUTPUT FORMAT"):
        idx = prompt.find(marker)
        if idx != -1:
            return prompt[:idx] + block.rstrip() + "\n\n" + prompt[idx:]
    return prompt + block


class ResearchAgentAdditive(ResearchAgentExp):
    """GAM's prompts, verbatim, plus an appended experience block."""

    def _planning(self, request, memory_state, planning_prompt=None) -> SearchPlan:
        if not memory_state.abstracts:
            memory_context = "No memory currently."
        else:
            memory_context = "\n".join(
                f"Page {i}: {a}" for i, a in enumerate(memory_state.abstracts)
            )

        situation, tok = _abstract_situation_planning(self.generator, request)
        self.total_tokens += tok
        hits = self._retrieve("Planning", request, situation)

        base = Planning_PROMPT.format(request=request, memory=memory_context)
        # No usable experience -> fall back to GAM exactly, not to a degraded variant.
        prompt = (
            _splice(base, _PLANNING_BLOCK.format(experience=format_experiences(hits)))
            if hits
            else base
        )

        try:
            resp = self.generator.generate_single(prompt=prompt, schema=PLANNING_SCHEMA)
            self.total_tokens += int(resp["response"]["usage"]["total_tokens"])
            data = resp.get("json") or json.loads(resp["text"])
            return SearchPlan(
                info_needs=data.get("info_needs", []),
                tools=data.get("tools", []),
                keyword_collection=data.get("keyword_collection", []),
                vector_queries=data.get("vector_queries", []),
                page_index=data.get("page_index", []),
            )
        except Exception as exc:
            print(f"Error in planning(additive): {exc}")
            return SearchPlan()

    def _reflection(self, request, result, reflection_prompt=None) -> ReflectionDecision:
        situation, tok = _abstract_situation_reflection(
            self.generator, request, result.content
        )
        self.total_tokens += tok
        condition = f"question:{request} temp_memory:{result.content}"
        hits = self._retrieve("Reflection", condition, situation)
        block = (
            _REFLECTION_BLOCK.format(experience=format_experiences(hits))
            if hits
            else None
        )

        try:
            base_check = InfoCheck_PROMPT.format(request=request, result=result.content)
            check_prompt = _splice(base_check, block) if block else base_check
            resp = self.generator.generate_single(
                prompt=check_prompt, schema=INFO_CHECK_SCHEMA
            )
            self.total_tokens += int(resp["response"]["usage"]["total_tokens"])
            data = resp.get("json") or json.loads(resp["text"])
            if data.get("enough", False):
                return ReflectionDecision(enough=True, new_request=None)

            base_gen = GenerateRequests_PROMPT.format(
                request=request, result=result.content
            )
            gen_prompt = _splice(base_gen, block) if block else base_gen
            resp2 = self.generator.generate_single(
                prompt=gen_prompt, schema=GENERATE_REQUESTS_SCHEMA
            )
            self.total_tokens += int(resp2["response"]["usage"]["total_tokens"])
            data2 = resp2.get("json") or json.loads(resp2["text"])
            reqs = data2.get("new_requests", [])
            return ReflectionDecision(
                enough=False,
                new_request=" ".join(reqs) if isinstance(reqs, list) and reqs else None,
            )
        except Exception as exc:
            print(f"Error in reflection(additive): {exc}")
            return ReflectionDecision(enough=False, new_request=None)
