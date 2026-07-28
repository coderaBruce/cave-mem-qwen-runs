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


FUSER_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "final_answer": {"type": "string"},
        "chosen_candidate": {
            "type": "string",
            "enum": ["gam", "current", "elaa", "raw", "revised"],
        },
        "risk_assessment": {
            "type": "string",
            "enum": [
                "none",
                "candidate_off_slot",
                "raw_wrong_anchor",
                "summary_too_generic",
                "partial_list",
                "over_specific",
                "under_specific",
                "unsupported",
                "other",
            ],
        },
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "reason": {"type": "string"},
    },
    "required": [
        "final_answer",
        "chosen_candidate",
        "risk_assessment",
        "confidence",
        "reason",
    ],
    "additionalProperties": False,
}


def _norm(text: str) -> str:
    return " ".join(str(text or "").strip().lower().split())


def _prompt(
    *,
    question: str,
    category: Any,
    gam_answer: str,
    gam_summary: str,
    current_answer: str,
    current_summary: str,
    elaa_answer: str,
    elaa_decision: Dict[str, Any],
    raw_answer: str,
    raw_decision: Dict[str, Any],
    prompt_version: str,
) -> str:
    category_name = CATEGORY_NAMES.get(category, str(category))
    raw_quote = str(raw_decision.get("evidence_quote") or "")
    raw_reason = str(raw_decision.get("reason") or "")
    elaa_reason = str(elaa_decision.get("reason") or "")
    elaa_gam_extract = str(elaa_decision.get("gam_summary_extract") or "")
    elaa_current_extract = str(elaa_decision.get("enhanced_summary_extract") or "")
    extra = ""
    if prompt_version == "v2":
        extra = """
Selection policy:
- ELAA is the default strong candidate because it already arbitrated between
  GAM and CURRENT. Do not abandon it unless another candidate clearly fills the
  question slot better.
- RAW can be very useful for exact quotes, attributes, short reasons, and
  missing list items, but it can also select the wrong temporal anchor. Use RAW
  only when its evidence quote directly answers the question.
- CURRENT is useful when ELAA over-specializes, drops a relation such as
  before/after/how-long, or selects a nearby but wrong fact.
- GAM is useful when CURRENT/raw over-expand or when GAM gives the concise
  benchmark-style answer.
- Prefer the answer that maximizes likely token-F1: short, exact, complete,
  and aligned to the requested slot.
- For lists, include all and only requested items. For yes/no, answer only Yes
  or No. For what someone said, prefer a quoted or near-quoted phrase.
"""
    return f"""You are a candidate-level answer fuser for LoCoMo memory QA.

You are given four independently produced answer candidates. Choose the final
short answer most likely to match the benchmark token-F1 answer. You do not see
the gold answer.

QUESTION CATEGORY:
{category_name}

QUESTION:
{question}

CANDIDATES:
1. GAM: {gam_answer}
2. CURRENT: {current_answer}
3. ELAA: {elaa_answer}
4. RAW: {raw_answer}

GAM SUMMARY:
{_clip(gam_summary, 1200)}

CURRENT SUMMARY:
{_clip(current_summary, 1600)}

ELAA DECISION:
- reason: {elaa_reason}
- extract from GAM: {_clip(elaa_gam_extract, 700)}
- extract from CURRENT: {_clip(elaa_current_extract, 700)}

RAW DECISION:
- evidence quote: {_clip(raw_quote, 900)}
- reason: {raw_reason}

Rules:
- Output only the short answer text in final_answer.
- A candidate can be supported but still wrong if it answers a nearby slot.
- Prefer exact spans from evidence when they directly answer the question.
- Do not add explanatory context unless the question asks why/how.
{extra}

Return JSON.
"""


def _fallback(question: str, elaa: Dict[str, Any]) -> Dict[str, Any]:
    answer = _answer(elaa)
    return {
        "final_answer": _postprocess_answer(question, answer),
        "chosen_candidate": "elaa",
        "risk_assessment": "other",
        "confidence": "low",
        "reason": "fallback after fuser failure",
    }


def _run_one(
    qid: str,
    gam: Dict[str, Any],
    current: Dict[str, Any],
    elaa: Dict[str, Any],
    raw: Dict[str, Any],
    generator: CachedOpenAIGenerator,
    prompt_version: str,
) -> Dict[str, Any]:
    question = str(current.get("question") or gam.get("question") or "")
    elaa_decision = elaa.get("elaa", {}).get("decision", {})
    raw_decision = raw.get("raw_localizer", {}).get("decision", {})
    if not isinstance(elaa_decision, dict):
        elaa_decision = {}
    if not isinstance(raw_decision, dict):
        raw_decision = {}
    prompt = _prompt(
        question=question,
        category=current.get("category", gam.get("category")),
        gam_answer=_answer(gam),
        gam_summary=str(gam.get("research_summary") or ""),
        current_answer=_answer(current),
        current_summary=str(current.get("research_summary") or ""),
        elaa_answer=_answer(elaa),
        elaa_decision=elaa_decision,
        raw_answer=_answer(raw),
        raw_decision=raw_decision,
        prompt_version=prompt_version,
    )
    try:
        response = generator.generate_single(prompt=prompt, schema=FUSER_SCHEMA)
        data = response.get("json")
        if not isinstance(data, dict):
            data = _fallback(question, elaa)
    except Exception as exc:
        data = _fallback(question, elaa)
        data["reason"] = f"fuser failed: {type(exc).__name__}: {exc}"

    final = str(data.get("final_answer") or "").strip()
    if not final:
        final = _answer(elaa)
    final = _postprocess_answer(question, final)
    data["final_answer_postprocessed"] = final

    row = dict(elaa)
    row["summary_answer"] = final
    row["pred"] = final
    row["candidate_fuser"] = {
        "qid": qid,
        "gam_answer": _answer(gam),
        "current_answer": _answer(current),
        "elaa_answer": _answer(elaa),
        "raw_answer": _answer(raw),
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
    parser.add_argument("--raw-run", required=True)
    parser.add_argument("--tag", default="locomo-full-candidate-fuser-v1")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--prompt-version", choices=["v1", "v2"], default="v2")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--exclude-prefix", nargs="*", default=["conv-26"])
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    gam_rows = _load_rows(_resolve_path(args.gam_run) / "all_qa_results.json")
    current_rows = _load_rows(_resolve_path(args.current_run) / "all_qa_results.json")
    elaa_rows = _load_rows(_resolve_path(args.elaa_run) / "all_qa_results.json")
    raw_rows = _load_rows(_resolve_path(args.raw_run) / "all_qa_results.json")

    ids = sorted(set(gam_rows) & set(current_rows) & set(elaa_rows) & set(raw_rows))
    for prefix in args.exclude_prefix or []:
        ids = [qid for qid in ids if not qid.startswith(prefix)]
    if args.only:
        allowed = set(args.only)
        ids = [qid for qid in ids if qid in allowed or qid.rsplit("-q", 1)[0] in allowed]
    if args.limit:
        ids = ids[: args.limit]

    outdir = RESULTS / args.tag
    outdir.mkdir(parents=True, exist_ok=True)

    generator = CachedOpenAIGenerator(
        {
            "model_name": args.model,
            "cache_dir": str(CACHE / "llm"),
            "temperature": args.temperature,
            "max_tokens": 512,
            "role": "candidate_fuser",
            "use_schema": True,
        }
    )
    if args.no_cache:
        generator.enable_cache = False

    print(
        f"dataset={args.dataset} ids={len(ids)} model={args.model} "
        f"workers={args.workers} outdir={outdir}",
        flush=True,
    )
    t0 = time.time()
    rows: List[Optional[Dict[str, Any]]] = [None] * len(ids)
    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {
                pool.submit(
                    _run_one,
                    qid,
                    gam_rows[qid],
                    current_rows[qid],
                    elaa_rows[qid],
                    raw_rows[qid],
                    generator,
                    args.prompt_version,
                ): idx
                for idx, qid in enumerate(ids)
            }
            for done, fut in enumerate(as_completed(futs), 1):
                rows[futs[fut]] = fut.result()
                if done % 50 == 0 or done == len(futs):
                    print(f"[{done}/{len(futs)}] elapsed={round(time.time() - t0)}s", flush=True)
    else:
        for idx, qid in enumerate(ids):
            rows[idx] = _run_one(
                qid,
                gam_rows[qid],
                current_rows[qid],
                elaa_rows[qid],
                raw_rows[qid],
                generator,
                args.prompt_version,
            )
            if (idx + 1) % 50 == 0 or idx + 1 == len(ids):
                print(f"[{idx + 1}/{len(ids)}] elapsed={round(time.time() - t0)}s", flush=True)

    all_qa = [r for r in rows if r is not None]
    metrics = score(all_qa, args.dataset)
    summary = {
        "tag": args.tag,
        "dataset": args.dataset,
        "method": "offline_candidate_fuser",
        "gam_run": str(_resolve_path(args.gam_run)),
        "current_run": str(_resolve_path(args.current_run)),
        "elaa_run": str(_resolve_path(args.elaa_run)),
        "raw_run": str(_resolve_path(args.raw_run)),
        "model": args.model,
        "temperature": args.temperature,
        "workers": args.workers,
        "prompt_version": args.prompt_version,
        "n_questions": len(all_qa),
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

