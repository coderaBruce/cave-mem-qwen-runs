#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
R2M_API = ROOT / "r2m-api"
UPSTREAM = ROOT / "R2M-official"
for p in (str(R2M_API), str(UPSTREAM), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from apiharness.cached_generator import CachedOpenAIGenerator
from apiharness.scoring import score

from memory_directions.offline_elaa import (
    CACHE,
    CATEGORY_NAMES,
    RESULTS,
    _answer,
    _clip,
    _load_rows,
    _postprocess_answer,
    _resolve_path,
)


GUARD_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "final_answer": {"type": "string"},
        "preferred_source": {
            "type": "string",
            "enum": ["current", "elaa", "revised"],
        },
        "risk_type": {
            "type": "string",
            "enum": [
                "none",
                "off_slot",
                "over_specific",
                "under_specific",
                "partial_list",
                "wrong_temporal_anchor",
                "wrong_entity",
                "style_mismatch",
                "unsupported",
                "other",
            ],
        },
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "reason": {"type": "string"},
    },
    "required": [
        "final_answer",
        "preferred_source",
        "risk_type",
        "confidence",
        "reason",
    ],
    "additionalProperties": False,
}


def _norm(text: str) -> str:
    return " ".join(str(text or "").strip().lower().split())


def _guard_prompt(
    *,
    question: str,
    category: Any,
    current_answer: str,
    current_summary: str,
    gam_answer: str,
    gam_summary: str,
    elaa_answer: str,
    elaa_decision: Dict[str, Any],
    max_summary_chars: int,
    prompt_version: str,
) -> str:
    category_name = CATEGORY_NAMES.get(category, str(category))
    elaa_reason = str(elaa_decision.get("reason") or "")
    elaa_choice = str(elaa_decision.get("chosen_source") or "")
    elaa_type = str(elaa_decision.get("answer_type") or "")
    gam_extract = str(elaa_decision.get("gam_summary_extract") or "")
    current_extract = str(elaa_decision.get("enhanced_summary_extract") or "")

    extra = ""
    if prompt_version == "v2":
        extra = """
Additional validity checks:
- If CURRENT and ELAA are both supported, do not automatically choose the
  answer with more detail. Choose the phrase whose semantic type best fills
  the question slot.
- If the question asks for a type/kind/style, prefer a type phrase over a named
  instance, unless the instance is itself the requested answer.
- If the question asks how long, prefer the duration wording used by the
  evidence over a derived exact day count.
- If the question asks when and one candidate is an anchored relative phrase
  from the evidence, do not replace it with a derived calendar date unless the
  summary explicitly states that date.
- If the question asks what two people have in common, prefer the shared event
  or property over a narrower hobby or single-person detail.
- If the question asks about multiple entities, keep all required entities; do
  not drop one because a shorter answer is supported.
- If the question asks what a creative work is about, themes/topics may be the
  expected answer rather than a plot synopsis.
"""
    elif prompt_version == "v3":
        extra = """
Veto policy:
- Treat ELAA_FINAL as the default replacement. Reject it only when there is a
  clear validity failure: off-slot answer, wrong entity, partial list, wrong
  temporal anchor, unsupported fact, or duration/type/style mismatch.
- Do not keep CURRENT_ANSWER merely because it is also supported. If ELAA_FINAL
  is a plausible exact date/entity/title/count that fills the question slot,
  keep ELAA_FINAL.
- For when/date questions, exact dates often score better than vague relative
  phrases. Veto an exact date only if CURRENT_ANSWER preserves an essential
  before/after/duration relation that ELAA_FINAL loses.
- For how-long/duration questions, veto over-precise derived day counts when
  CURRENT_ANSWER gives the natural duration phrase.
- For title/name questions, or when the evidence says a work is called/named,
  prefer the proper title even if the question contains words like kind or
  piece.
- For type/kind/style questions without a named work, prefer the type/category
  phrase over a named instance.
- For lists or multiple entities, veto answers that drop a required entity or
  event.
- For commonality questions, veto answers that name only one narrow shared
  hobby when CURRENT_ANSWER gives the shared event/property requested.
"""

    return f"""You are an answer-validity guard for LoCoMo memory QA.

An upstream arbitrator proposed ELAA_FINAL as a replacement for CURRENT_ANSWER.
Your task is to decide whether this replacement is safe under token-F1
benchmark scoring. You do not see the gold answer.

Choose the final short answer that best matches the question's requested
semantic slot and likely benchmark answer style. A fact can be supported and
still be the wrong answer if it answers a nearby but different slot.

QUESTION CATEGORY:
{category_name}

QUESTION:
{question}

CURRENT_ANSWER:
{current_answer}

CURRENT_MEMORY_SUMMARY:
{_clip(current_summary, max_summary_chars)}

GAM_ANSWER:
{gam_answer}

GAM_MEMORY_SUMMARY:
{_clip(gam_summary, max_summary_chars)}

ELAA_FINAL:
{elaa_answer}

ELAA_METADATA:
- chosen_source: {elaa_choice}
- answer_type: {elaa_type}
- reason: {elaa_reason}

ELAA_EXTRACT_FROM_GAM:
{_clip(gam_extract, max_summary_chars // 3)}

ELAA_EXTRACT_FROM_CURRENT:
{_clip(current_extract, max_summary_chars // 3)}

General rules:
- Prefer a short answer, but not if the short answer drops required content.
- Prefer exact entities, dates, counts, and lists only when they directly fill
  the question slot.
- Do not reward specificity by itself. Check whether the question asks for a
  category/type, duration, reason, activity, title, location, or list.
- Keep CURRENT_ANSWER when ELAA_FINAL is merely a supported neighboring fact.
- Keep ELAA_FINAL when it corrects CURRENT_ANSWER with a directly supported
  answer that better fills the slot.
- You may output a revised short answer if both candidates are flawed but the
  summaries contain a clearly supported better span.
{extra}

Return JSON. final_answer must be only the answer text, with no prefix.
"""


def _fallback(
    question: str,
    current_answer: str,
    elaa_answer: str,
    use_current: bool,
) -> Dict[str, Any]:
    answer = current_answer if use_current else elaa_answer
    return {
        "final_answer": _postprocess_answer(question, answer),
        "preferred_source": "current" if use_current else "elaa",
        "risk_type": "other",
        "confidence": "low",
        "reason": "fallback after guard failure",
    }


def _run_guard_one(
    qid: str,
    gam: Dict[str, Any],
    current: Dict[str, Any],
    elaa: Dict[str, Any],
    generator: CachedOpenAIGenerator,
    max_summary_chars: int,
    prompt_version: str,
    default_to_current: bool,
) -> Dict[str, Any]:
    question = str(current.get("question") or gam.get("question") or "")
    current_answer = _answer(current)
    elaa_answer = _answer(elaa)
    elaa_decision = elaa.get("elaa", {}).get("decision", {})
    prompt = _guard_prompt(
        question=question,
        category=current.get("category", gam.get("category")),
        current_answer=current_answer,
        current_summary=str(current.get("research_summary") or ""),
        gam_answer=_answer(gam),
        gam_summary=str(gam.get("research_summary") or ""),
        elaa_answer=elaa_answer,
        elaa_decision=elaa_decision if isinstance(elaa_decision, dict) else {},
        max_summary_chars=max_summary_chars,
        prompt_version=prompt_version,
    )
    try:
        response = generator.generate_single(prompt=prompt, schema=GUARD_SCHEMA)
        data = response.get("json")
        if not isinstance(data, dict):
            data = _fallback(question, current_answer, elaa_answer, default_to_current)
    except Exception as exc:
        data = _fallback(question, current_answer, elaa_answer, default_to_current)
        data["reason"] = f"guard failed: {type(exc).__name__}: {exc}"

    final = str(data.get("final_answer") or "").strip()
    if not final:
        final = current_answer if default_to_current else elaa_answer
    final = _postprocess_answer(question, final)
    data["final_answer_postprocessed"] = final

    row = dict(elaa)
    row["summary_answer"] = final
    row["pred"] = final
    row["elaa_guard"] = {
        "qid": qid,
        "current_answer": current_answer,
        "elaa_answer": elaa_answer,
        "decision": data,
    }
    return row


def _should_guard(
    current: Dict[str, Any],
    elaa: Dict[str, Any],
    mode: str,
) -> bool:
    current_answer = _answer(current)
    elaa_answer = _answer(elaa)
    if _norm(current_answer) == _norm(elaa_answer):
        return False
    if mode == "multi_hop":
        return current.get("category") == 1
    if mode == "disagreement":
        return True
    decision = elaa.get("elaa", {}).get("decision", {})
    if not isinstance(decision, dict):
        return True
    if mode == "high_risk":
        answer_type = decision.get("answer_type")
        chosen = decision.get("chosen_source")
        question = str(current.get("question") or "").lower()
        return (
            answer_type in {"attribute", "quote", "reason", "place"}
            or (answer_type == "date" and question.startswith(("when", "how long")))
            or (chosen == "gam_answer" and answer_type in {"attribute", "count", "quote"})
            or "what type" in question
            or "what kind" in question
            or "how long" in question
            or " in common" in question
            or " about" in question
        )
    raise ValueError(f"unknown guard mode: {mode}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="locomo", choices=["locomo"])
    parser.add_argument("--gam-run", default="r2m-api/results/gam-locomo-4omini")
    parser.add_argument(
        "--current-run",
        default="memory_directions/results/conv30-50-no26-full-datecount-selective-open-v5-fast",
    )
    parser.add_argument(
        "--elaa-run",
        default="memory_directions/results/locomo-full-elaa-v3-post-yn-recalc",
    )
    parser.add_argument("--tag", default="locomo-full-elaa-guard-v1")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--max-summary-chars", type=int, default=5000)
    parser.add_argument("--prompt-version", choices=["v1", "v2", "v3"], default="v2")
    parser.add_argument(
        "--guard-mode",
        choices=["disagreement", "high_risk", "multi_hop"],
        default="disagreement",
    )
    parser.add_argument("--default-to-current", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--exclude-prefix", nargs="*", default=["conv-26"])
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    gam_rows = _load_rows(_resolve_path(args.gam_run) / "all_qa_results.json")
    current_rows = _load_rows(_resolve_path(args.current_run) / "all_qa_results.json")
    elaa_rows = _load_rows(_resolve_path(args.elaa_run) / "all_qa_results.json")

    ids = sorted(set(gam_rows) & set(current_rows) & set(elaa_rows))
    for prefix in args.exclude_prefix or []:
        ids = [qid for qid in ids if not qid.startswith(prefix)]
    if args.only:
        allowed = set(args.only)
        ids = [qid for qid in ids if qid in allowed or qid.rsplit("-q", 1)[0] in allowed]
    if args.limit:
        ids = ids[: args.limit]

    guard_ids = [
        qid
        for qid in ids
        if _should_guard(current_rows[qid], elaa_rows[qid], args.guard_mode)
    ]
    guard_set = set(guard_ids)

    outdir = RESULTS / args.tag
    outdir.mkdir(parents=True, exist_ok=True)

    generator = CachedOpenAIGenerator(
        {
            "model_name": args.model,
            "cache_dir": str(CACHE / "llm"),
            "temperature": args.temperature,
            "max_tokens": 512,
            "role": "elaa_guard",
            "use_schema": True,
        }
    )
    if args.no_cache:
        generator.enable_cache = False

    print(
        f"dataset={args.dataset} ids={len(ids)} guard_ids={len(guard_ids)} "
        f"mode={args.guard_mode} model={args.model} workers={args.workers} "
        f"outdir={outdir}"
    )

    t0 = time.time()
    guarded: Dict[str, Dict[str, Any]] = {}
    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {
                pool.submit(
                    _run_guard_one,
                    qid,
                    gam_rows[qid],
                    current_rows[qid],
                    elaa_rows[qid],
                    generator,
                    args.max_summary_chars,
                    args.prompt_version,
                    args.default_to_current,
                ): qid
                for qid in guard_ids
            }
            for done, fut in enumerate(as_completed(futs), 1):
                qid = futs[fut]
                guarded[qid] = fut.result()
                if done % 25 == 0 or done == len(futs):
                    print(f"[{done}/{len(futs)}] elapsed={round(time.time() - t0)}s")
    else:
        for idx, qid in enumerate(guard_ids, 1):
            guarded[qid] = _run_guard_one(
                qid,
                gam_rows[qid],
                current_rows[qid],
                elaa_rows[qid],
                generator,
                args.max_summary_chars,
                args.prompt_version,
                args.default_to_current,
            )
            if idx % 25 == 0 or idx == len(guard_ids):
                print(f"[{idx}/{len(guard_ids)}] elapsed={round(time.time() - t0)}s")

    all_qa: List[Dict[str, Any]] = []
    for qid in ids:
        if qid in guard_set:
            all_qa.append(guarded[qid])
        else:
            row = dict(elaa_rows[qid])
            row["elaa_guard"] = {
                "qid": qid,
                "skipped": True,
                "reason": "current and elaa agree"
                if _norm(_answer(current_rows[qid])) == _norm(_answer(elaa_rows[qid]))
                else f"not selected by {args.guard_mode}",
            }
            all_qa.append(row)

    metrics = score(all_qa, args.dataset)
    summary = {
        "tag": args.tag,
        "dataset": args.dataset,
        "method": "offline_elaa_guard",
        "gam_run": str(_resolve_path(args.gam_run)),
        "current_run": str(_resolve_path(args.current_run)),
        "elaa_run": str(_resolve_path(args.elaa_run)),
        "model": args.model,
        "temperature": args.temperature,
        "workers": args.workers,
        "max_summary_chars": args.max_summary_chars,
        "prompt_version": args.prompt_version,
        "guard_mode": args.guard_mode,
        "default_to_current": args.default_to_current,
        "n_questions": len(all_qa),
        "n_guarded": len(guard_ids),
        "metrics": metrics,
        "generator_stats": generator.stats(),
        "wall_secs": round(time.time() - t0, 1),
    }
    (outdir / "all_qa_results.json").write_text(
        json.dumps(all_qa, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (outdir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n=== metrics ===")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"summary -> {outdir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
