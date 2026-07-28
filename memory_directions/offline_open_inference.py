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

from memory_directions.answering import _canonical_answer
from memory_directions.offline_elaa import CACHE, RESULTS, _answer, _clip, _load_rows, _resolve_path


OPEN_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "final_answer": {"type": "string"},
        "answer_kind": {
            "type": "string",
            "enum": [
                "entity",
                "yes_no_with_reason",
                "relationship",
                "inference",
                "commonsense",
                "count",
                "place",
                "other",
            ],
        },
        "uses_world_knowledge": {"type": "boolean"},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "reason": {"type": "string"},
    },
    "required": [
        "final_answer",
        "answer_kind",
        "uses_world_knowledge",
        "confidence",
        "reason",
    ],
    "additionalProperties": False,
}


def _postprocess_open(question: str, answer: str) -> str:
    # For open-domain LoCoMo, yes/no gold answers often include a short reason.
    # Keep that reason instead of using offline_elaa's global yes/no minimizer.
    return _canonical_answer(question, answer)


def _prompt(
    *,
    question: str,
    gam_answer: str,
    gam_summary: str,
    current_answer: str,
    current_summary: str,
    elaa_answer: str,
    elaa_decision: Dict[str, Any],
    prompt_version: str,
) -> str:
    elaa_reason = str(elaa_decision.get("reason") or "")
    elaa_gam_extract = str(elaa_decision.get("gam_summary_extract") or "")
    elaa_current_extract = str(elaa_decision.get("enhanced_summary_extract") or "")
    extra = ""
    if prompt_version == "v2":
        extra = """
Open-domain heuristics:
- If the question asks "what might", "likely", "presumably", "would", or
  "could", make the most specific plausible inference from the memory facts.
- Use ordinary world knowledge to name a specific shop, game, company, medical
  condition, career, country, state, hobby, or object when the memory provides
  enough clues.
- If the question asks yes/no and the likely gold answer includes a reason,
  answer with "Yes/No, because ..." in one concise clause.
- If the question asks beach or mountains, distinguish living close to a place
  from taking trips to a place.
- Prefer a specific named entity over a generic phrase such as "renowned
  company", "cat-themed card game", or "social work" when the clues support it.
- Do not hallucinate beyond the clues; if multiple answers are plausible,
  give the concise set of plausible answers.
"""
    return f"""Answer this LoCoMo open-domain / inference question.

These questions are not always answered by copying one span. Use the memory
facts plus ordinary commonsense or world knowledge when the question asks for a
likely, possible, or inferred answer.

QUESTION:
{question}

CANDIDATE ANSWERS:
- GAM: {gam_answer}
- CURRENT: {current_answer}
- ELAA: {elaa_answer}

GAM SUMMARY:
{_clip(gam_summary, 1800)}

CURRENT SUMMARY:
{_clip(current_summary, 2200)}

ELAA DECISION:
- reason: {elaa_reason}
- extract from GAM: {_clip(elaa_gam_extract, 700)}
- extract from CURRENT: {_clip(elaa_current_extract, 700)}

Rules:
- Output only the final short answer in final_answer.
- The answer may use commonsense/world knowledge, but it must be anchored in
  memory facts from the summaries.
- Prefer the most specific plausible answer over a generic category.
- For yes/no questions, include a short reason if the question asks "based on"
  or the answer would otherwise be too underspecified for token-F1.
- For "likely yes/no" questions, use "Likely yes/no" or "Presumably yes/no" if
  uncertainty is part of the answer.
{extra}

Return JSON.
"""


def _fallback(question: str, elaa: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "final_answer": _postprocess_open(question, _answer(elaa)),
        "answer_kind": "other",
        "uses_world_knowledge": False,
        "confidence": "low",
        "reason": "fallback after open inference failure",
    }


def _run_one(
    qid: str,
    gam: Dict[str, Any],
    current: Dict[str, Any],
    elaa: Dict[str, Any],
    generator: CachedOpenAIGenerator,
    prompt_version: str,
) -> Dict[str, Any]:
    question = str(current.get("question") or gam.get("question") or "")
    decision = elaa.get("elaa", {}).get("decision", {})
    if not isinstance(decision, dict):
        decision = {}
    prompt = _prompt(
        question=question,
        gam_answer=_answer(gam),
        gam_summary=str(gam.get("research_summary") or ""),
        current_answer=_answer(current),
        current_summary=str(current.get("research_summary") or ""),
        elaa_answer=_answer(elaa),
        elaa_decision=decision,
        prompt_version=prompt_version,
    )
    try:
        response = generator.generate_single(prompt=prompt, schema=OPEN_SCHEMA)
        data = response.get("json")
        if not isinstance(data, dict):
            data = _fallback(question, elaa)
    except Exception as exc:
        data = _fallback(question, elaa)
        data["reason"] = f"open inference failed: {type(exc).__name__}: {exc}"

    final = str(data.get("final_answer") or "").strip()
    if not final:
        final = _answer(elaa)
    final = _postprocess_open(question, final)
    data["final_answer_postprocessed"] = final

    row = dict(elaa)
    row["summary_answer"] = final
    row["pred"] = final
    row["open_inference"] = {
        "qid": qid,
        "gam_answer": _answer(gam),
        "current_answer": _answer(current),
        "elaa_answer": _answer(elaa),
        "decision": data,
    }
    return row


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
    parser.add_argument("--tag", default="locomo-full-open-inference-v1")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--prompt-version", choices=["v1", "v2"], default="v2")
    parser.add_argument("--only-open", action="store_true", default=True)
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

    open_ids = [qid for qid in ids if current_rows[qid].get("category") == 3]
    outdir = RESULTS / args.tag
    outdir.mkdir(parents=True, exist_ok=True)

    generator = CachedOpenAIGenerator(
        {
            "model_name": args.model,
            "cache_dir": str(CACHE / "llm"),
            "temperature": args.temperature,
            "max_tokens": 512,
            "role": "open_inference",
            "use_schema": True,
        }
    )
    if args.no_cache:
        generator.enable_cache = False

    print(
        f"dataset={args.dataset} ids={len(ids)} open_ids={len(open_ids)} "
        f"model={args.model} workers={args.workers} outdir={outdir}",
        flush=True,
    )
    t0 = time.time()
    open_rows: Dict[str, Dict[str, Any]] = {}
    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {
                pool.submit(
                    _run_one,
                    qid,
                    gam_rows[qid],
                    current_rows[qid],
                    elaa_rows[qid],
                    generator,
                    args.prompt_version,
                ): qid
                for qid in open_ids
            }
            for done, fut in enumerate(as_completed(futs), 1):
                qid = futs[fut]
                open_rows[qid] = fut.result()
                if done % 25 == 0 or done == len(futs):
                    print(f"[{done}/{len(futs)}] elapsed={round(time.time() - t0)}s", flush=True)
    else:
        for idx, qid in enumerate(open_ids, 1):
            open_rows[qid] = _run_one(
                qid,
                gam_rows[qid],
                current_rows[qid],
                elaa_rows[qid],
                generator,
                args.prompt_version,
            )
            if idx % 25 == 0 or idx == len(open_ids):
                print(f"[{idx}/{len(open_ids)}] elapsed={round(time.time() - t0)}s", flush=True)

    all_qa: List[Dict[str, Any]] = []
    for qid in ids:
        if qid in open_rows:
            all_qa.append(open_rows[qid])
        else:
            row = dict(elaa_rows[qid])
            row["open_inference"] = {"qid": qid, "skipped": True}
            all_qa.append(row)

    metrics = score(all_qa, args.dataset)
    summary = {
        "tag": args.tag,
        "dataset": args.dataset,
        "method": "offline_open_inference",
        "gam_run": str(_resolve_path(args.gam_run)),
        "current_run": str(_resolve_path(args.current_run)),
        "elaa_run": str(_resolve_path(args.elaa_run)),
        "model": args.model,
        "temperature": args.temperature,
        "workers": args.workers,
        "prompt_version": args.prompt_version,
        "n_questions": len(all_qa),
        "n_open": len(open_ids),
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

