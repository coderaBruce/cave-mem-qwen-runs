"""OUR METHOD v3: answer-granularity calibration.

Error analysis on the GAM baseline (notes/02-experiment-log.md E8) found that
**15% of answers score exactly 0 token-F1, and 24/34 of those are single-hop** —
the largest category. Inspecting them, the search is not what failed:

    gold "Glad"                  -> pred "positive attitude"
    gold "They look graceful"    -> pred "a means of self-expression and happiness"
    gold "\"Finding Freedom\""     -> pred "contemporary dance piece"
    gold "February, 2023"        -> pred "next month"

The evidence was retrieved. The answer step then states it at the wrong
granularity — elaborating where the gold is terse, describing where the gold
names, leaving relative time unresolved. Token-F1 scores these as total misses.

GAM and R²-Mem both leave answer generation untouched; R²-Mem's experience only
reaches planning and reflection. So this axis is unexploited by both.

We spend the *same* data budget R²-Mem spends on its experience bank — the
accumulation conversation, held out from evaluation — on calibrating answer
style instead: a handful of (question, gold answer) pairs per category, shown as
demonstrations. No gold answer from any evaluation conversation is ever used.

The honest framing for the paper: this is partly a statement about the *metric*.
Token-F1 rewards matching the annotator's phrasing, and a large share of the
headroom that memory-search papers attribute to retrieval is in fact answer
formatting. That is worth reporting either way, and it is why we report the
retrieval-side and answer-side contributions separately rather than as one
number.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


def build_style_shots(
    run_dir: Path,
    samples: List[str],
    per_category: int = 6,
    seed: int = 42,
    max_gold_chars: int = 60,
) -> Dict[Any, List[Tuple[str, str]]]:
    """Collect (question, gold answer) demonstrations per category.

    Only reads `samples`, which must be the accumulation split.
    """
    pools: Dict[Any, List[Tuple[str, str]]] = defaultdict(list)
    for sid in samples:
        f = Path(run_dir) / sid / "qa_results.json"
        if not f.exists():
            continue
        for r in json.loads(f.read_text(encoding="utf-8")):
            if "error" in r:
                continue
            gold = r.get("gold_answer")
            if isinstance(gold, list) or gold is None:
                ga = r.get("gold_answers") or ([] if gold is None else [gold])
                gold = ga[0] if ga else None
            q = r.get("question")
            if not q or gold is None:
                continue
            gold = str(gold).strip()
            # Demonstrations should show terseness; long golds teach the wrong lesson.
            if not gold or len(gold) > max_gold_chars:
                continue
            pools[r.get("category")].append((q, gold))

    rng = random.Random(seed)
    shots: Dict[Any, List[Tuple[str, str]]] = {}
    for cat, items in pools.items():
        rng.shuffle(items)
        shots[cat] = items[:per_category]
    return shots


def format_shots(shots: List[Tuple[str, str]]) -> str:
    return "\n".join(f"QUESTION: {q}\nShort answer: {a}" for q, a in shots)


def make_calibrated_answerer(
    base_answerer: Callable,
    shots: Dict[Any, List[Tuple[str, str]]],
    fallback_pool: Optional[List[Tuple[str, str]]] = None,
    base_prompt_fn: Optional[Callable] = None,
) -> Callable:
    """Wrap a DatasetSpec answerer so it shows style demonstrations first.

    Falls straight through to the original answerer when no demonstrations
    exist for the category, so the baseline path is untouched.
    """
    if fallback_pool is None:
        fallback_pool = [s for v in shots.values() for s in v][:6]

    def answerer(category, summary, question, generator, evidence=None):  # noqa: C901
        examples = shots.get(category) or fallback_pool
        if not examples and not evidence:
            return base_answerer(category, summary, question, generator)

        # Augment, do not rewrite. Our first attempt replaced the upstream prompt
        # wholesale and silently dropped its date-format spec ("strictly follow
        # the format '15 July 2023'"), costing 3.6 F1 on single-hop — the same
        # error we diagnosed in R2-Mem's templates. So: take the original prompt
        # verbatim and splice demonstrations in just before its answer cue.
        if base_prompt_fn is not None:
            base = base_prompt_fn(summary, question)
        else:
            from eval.locomo_test import (
                make_summary_prompt,
                make_summary_prompt_category3,
            )

            base = (
                make_summary_prompt_category3(summary, question)
                if category == 3
                else make_summary_prompt(summary, question)
            )
        block = ""
        if evidence:
            # GAM's own Table 4: integration-only 53.18 -> +extraction 54.47 ->
            # +page 55.71. The shipped default is integration-only, so the answer
            # step never sees the source text and cannot recover exact surface
            # forms that `_integrate` paraphrased away ("Finding Freedom" ->
            # "contemporary dance piece"). We pass the cited pages back in.
            block += (
                "\nSOURCE EXCERPTS (verbatim; prefer exact wording from here):\n"
                f"{evidence}\n"
            )
        if examples:
            block += (
                "\nEXAMPLES OF THE EXPECTED ANSWER STYLE "
                "(real answers from this dataset — match their brevity and wording):\n"
                f"{format_shots(examples)}\n"
            )
        idx = -1
        for cue in ("Short answer:", "Answer:"):
            idx = base.rfind(cue)
            if idx != -1:
                break
        prompt = base[:idx] + block + "\n" + base[idx:] if idx != -1 else base + block

        out = generator.generate_single(prompt=prompt)
        return (out.get("text") or "").strip()

    return answerer
