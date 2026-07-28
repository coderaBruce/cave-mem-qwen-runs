"""Utility-verified experience — our main proposed departure from R²-Mem.

Motivation (see notes/01-baseline-critique.md §3.3). R²-Mem's pipeline is a
triple indirection with no link verified:

    judge's opinion of a step -> text rule -> cosine similarity -> hoped-for behaviour

An experience enters the bank because a rubric liked the *step it came from*,
and it gets retrieved because its situation embedding is *near the query*.
Neither fact implies the experience actually improves the step it is injected
into. Nothing in the pipeline ever checks.

This module measures that directly. For a probe set of real planning steps we
run the step twice — once with the retrieved experience injected, once without —
and ask a judge which plan is better under the paper's own four Planning rubric
dimensions. Each retrieved experience is credited with the outcome. An
experience's utility is then

    u(e) = P(win | e injected) - P(loss | e injected)      in [-1, 1]

estimated over the probe steps where `e` was actually retrieved.

Why pairwise rather than absolute rubric scores: scoring two plans on a 0-12
scale and subtracting is dominated by judge noise at this granularity, whereas
a forced A/B choice with randomised presentation order is a far lower-variance
estimator of the same quantity. Order is randomised per comparison to cancel
position bias.

What this buys, if it works:
  * K_low/K_high stop being needed — utility filters, so the discarded middle
    (44% of scored steps in our pilot) can be distilled and kept when useful.
  * Pruning becomes principled: drop u(e) <= 0.
  * Retrieval can rank by similarity x utility rather than similarity alone.
  * The Evaluator becomes optional, which removes R²-Mem's GPT-4o dependency
    and with it their weakest cost claim.
"""
from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from gam.schemas import PLANNING_SCHEMA

from .r2mem import Experience, ExperienceBank, _loose_json, format_experiences

_PAIRWISE_SCHEMA = {
    "type": "object",
    "properties": {
        "reason": {"type": "string"},
        "better": {"type": "string", "enum": ["A", "B", "tie"]},
    },
    "required": ["reason", "better"],
    "additionalProperties": False,
}

# The four dimensions are lifted from R²-Mem's own Planning rubric so that our
# utility signal is measured on their yardstick, not one we invented to win on.
_PAIRWISE_SYSTEM = """You are an expert evaluator of retrieval planning for an AI memory deep-search system.
You will see a QUESTION, the available MEMORY abstracts, and two candidate retrieval PLANS.
Judge which plan is better using exactly these criteria:
- Info Needs Coverage: does it cover all key information required to answer the question?
- Info Needs Non-Redundancy: are the info_needs distinct rather than overlapping?
- Tool-Info Alignment: are keyword / vector / page_index chosen appropriately for each need?
- Planning Efficiency: does it avoid unnecessary needs, tools and retrieval fields?

Answer "tie" only if the plans are genuinely equivalent in quality.
Return ONLY valid JSON."""

_PAIRWISE_USER = """QUESTION: {question}

MEMORY:
{memory}

PLAN A:
{plan_a}

PLAN B:
{plan_b}

Which plan is better under the four criteria? Return JSON:
{{"reason": "<one sentence>", "better": "A" | "B" | "tie"}}"""


@dataclass
class ProbeStep:
    """One real planning step to replay with and without experience."""
    question: str
    memory_context: str
    source: str = ""


def collect_probe_steps(
    run_dir: Path, samples: List[str], limit: Optional[int] = None
) -> List[ProbeStep]:
    """Pull planning steps out of a finished baseline run.

    Uses the initial question of each trace (step 0), which is the planning
    condition R²-Mem itself uses (`c_i = q_i`).
    """
    probes: List[ProbeStep] = []
    for sid in samples:
        state_file = run_dir / sid / "memory_state.json"
        memory_context = "No memory currently."
        if state_file.exists():
            try:
                abstracts = json.loads(state_file.read_text(encoding="utf-8")).get(
                    "abstracts", []
                )
                if abstracts:
                    memory_context = "\n".join(
                        f"Page {i}: {a}" for i, a in enumerate(abstracts)
                    )
            except Exception:
                pass

        qa_file = run_dir / sid / "qa_results.json"
        if not qa_file.exists():
            continue
        for item in json.loads(qa_file.read_text(encoding="utf-8")):
            if "error" in item or not item.get("question"):
                continue
            probes.append(
                ProbeStep(
                    question=item["question"],
                    memory_context=memory_context,
                    source=sid,
                )
            )
    if limit:
        probes = probes[:limit]
    return probes


def _plan_text(data: Dict[str, Any]) -> str:
    if not data:
        return "(planning failed)"
    return json.dumps(
        {
            k: data.get(k)
            for k in (
                "info_needs",
                "tools",
                "keyword_collection",
                "vector_queries",
                "page_index",
            )
        },
        ensure_ascii=False,
        indent=2,
    )


def _make_plan(generator, prompt: str) -> Optional[Dict[str, Any]]:
    try:
        resp = generator.generate_single(prompt=prompt, schema=PLANNING_SCHEMA)
        return resp.get("json") or _loose_json(resp.get("text", ""))
    except Exception:
        return None


def judge_pair(
    generator,
    question: str,
    memory: str,
    plan_no_exp: Dict[str, Any],
    plan_with_exp: Dict[str, Any],
    rng: random.Random,
) -> str:
    """Return 'with', 'without' or 'tie'. Presentation order is randomised."""
    with_is_a = rng.random() < 0.5
    a, b = (
        (plan_with_exp, plan_no_exp) if with_is_a else (plan_no_exp, plan_with_exp)
    )
    prompt = _PAIRWISE_USER.format(
        question=question,
        memory=memory[:4000],
        plan_a=_plan_text(a),
        plan_b=_plan_text(b),
    )
    try:
        resp = generator.generate_single(
            messages=[
                {"role": "system", "content": _PAIRWISE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            schema=_PAIRWISE_SCHEMA,
        )
        data = resp.get("json") or _loose_json(resp.get("text", "")) or {}
        choice = str(data.get("better", "tie")).strip().upper()
    except Exception:
        return "tie"

    if choice not in ("A", "B"):
        return "tie"
    chose_a = choice == "A"
    return "with" if chose_a == with_is_a else "without"


def measure_utility(
    bank: ExperienceBank,
    probes: List[ProbeStep],
    planner_generator,
    judge_generator,
    top_k: int = 3,
    seed: int = 42,
    on_progress=None,
) -> Dict[str, Any]:
    """Estimate per-experience utility by A/B replay over `probes`.

    Mutates `bank`: each entry gains `extra["utility"]`, `extra["n_trials"]`,
    `extra["wins"]`, `extra["losses"]`.
    """
    import exp.prompts.research_prompts_exp as rp
    from gam.prompts import Planning_PROMPT

    rng = random.Random(seed)
    tally: Dict[int, Dict[str, int]] = defaultdict(
        lambda: {"win": 0, "loss": 0, "tie": 0}
    )
    id_of = {id(e): i for i, e in enumerate(bank.entries)}

    outcomes = {"with": 0, "without": 0, "tie": 0}
    n_no_retrieval = 0

    for n, probe in enumerate(probes, 1):
        hits = bank.search(f"({probe.question})" + probe.question, "Planning", top_k)
        if not hits:
            n_no_retrieval += 1
            continue

        plan_0 = _make_plan(
            planner_generator,
            Planning_PROMPT.format(request=probe.question, memory=probe.memory_context),
        )
        plan_1 = _make_plan(
            planner_generator,
            rp.Planning_PROMPT_exp.format(
                request=probe.question,
                memory=probe.memory_context,
                experience=format_experiences(hits),
            ),
        )
        if plan_0 is None or plan_1 is None:
            continue

        verdict = judge_pair(
            judge_generator, probe.question, probe.memory_context, plan_0, plan_1, rng
        )
        outcomes[verdict] += 1

        # Every retrieved experience shares the credit for this step. Coarse,
        # but unbiased in expectation across many probes, and far cheaper than
        # ablating each experience individually.
        for exp_obj, _sim in hits:
            idx = id_of.get(id(exp_obj))
            if idx is None:
                continue
            key = {"with": "win", "without": "loss", "tie": "tie"}[verdict]
            tally[idx][key] += 1

        if on_progress:
            on_progress(n, len(probes), outcomes)

    for idx, rec in tally.items():
        trials = rec["win"] + rec["loss"] + rec["tie"]
        util = (rec["win"] - rec["loss"]) / trials if trials else 0.0
        bank.entries[idx].extra.update(
            {
                "utility": round(util, 4),
                "n_trials": trials,
                "wins": rec["win"],
                "losses": rec["loss"],
                "ties": rec["tie"],
            }
        )
    for e in bank.entries:
        e.extra.setdefault("utility", None)
        e.extra.setdefault("n_trials", 0)

    measured = [e for e in bank.entries if e.extra.get("n_trials", 0) > 0]
    positive = [e for e in measured if (e.extra.get("utility") or 0) > 0]
    return {
        "n_probes": len(probes),
        "n_no_retrieval": n_no_retrieval,
        "step_outcomes": outcomes,
        "win_rate": (
            outcomes["with"] / max(1, sum(outcomes.values()))
        ),
        "n_measured": len(measured),
        "n_positive_utility": len(positive),
        "n_nonpositive_utility": len(measured) - len(positive),
    }


def prune_by_utility(
    bank: ExperienceBank, min_utility: float = 0.0, keep_unmeasured: bool = True
) -> ExperienceBank:
    """Return a new bank keeping only entries whose measured utility clears the bar."""
    out = ExperienceBank(bank.embedder)
    for e in bank.entries:
        u = e.extra.get("utility")
        if u is None:
            if keep_unmeasured:
                out.add(e)
            continue
        if u > min_utility:
            out.add(e)
    return out


def utility_weighted_selector(alpha: float = 1.0, floor: float = 0.0):
    """A `select` hook for `ResearchAgentExp` that re-ranks by similarity x utility.

    Unmeasured entries keep their similarity ranking (utility treated as 0).
    Entries whose measured utility is at or below `floor` are dropped outright.
    """

    def select(hits, module, condition):
        kept = []
        for e, sim in hits:
            u = e.extra.get("utility")
            if u is not None and u <= floor:
                continue
            boost = 1.0 + alpha * (u or 0.0)
            kept.append((e, sim * boost))
        kept.sort(key=lambda kv: kv[1], reverse=True)
        return kept

    return select
