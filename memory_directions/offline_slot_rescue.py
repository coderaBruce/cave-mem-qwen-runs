#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
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

from memory_directions.offline_elaa import CACHE, RESULTS, _answer, _clip, _load_rows, _postprocess_answer, _resolve_path
from memory_directions.offline_raw_localizer import (
    _build_raw_block,
    _collect_source_ids,
    _load_json,
    _pages_path,
    _trace_path,
)


SLOT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "should_replace": {"type": "boolean"},
        "final_answer": {"type": "string"},
        "evidence_phrase": {"type": "string"},
        "slot_type": {
            "type": "string",
            "enum": [
                "quote",
                "description",
                "feeling",
                "reason",
                "action",
                "offer",
                "advice",
                "attribute",
                "list",
                "other",
            ],
        },
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "reason": {"type": "string"},
    },
    "required": [
        "should_replace",
        "final_answer",
        "evidence_phrase",
        "slot_type",
        "confidence",
        "reason",
    ],
    "additionalProperties": False,
}


TARGET_PATTERNS = [
    re.compile(
        r"\bwhat (did|does|do|was|is) .{0,80}\b"
        r"(say|describe|feel|think|appreciate|symbolize|represent|mean|offer|"
        r"advice|take away|learn|happen|happened)\b",
        re.I,
    ),
    re.compile(r"\bhow (did|does|do|was|is) .{0,80}\b(describe|feel|adjust|impact|react)\b", re.I),
    re.compile(r"\bwhat (emotion|emotions|attitude|sentiment|offer|advice|impact|positive impact)\b", re.I),
    re.compile(r"\bwhat inspired\b", re.I),
    re.compile(r"\bwhy did .{0,80}\b(apologize|feel inspired|start volunteering|join|decide)\b", re.I),
]


def _is_target(question: str) -> bool:
    return any(pattern.search(question or "") for pattern in TARGET_PATTERNS)


def _prompt(
    *,
    question: str,
    base_answer: str,
    gam_answer: str,
    current_answer: str,
    elaa_answer: str,
    summary: str,
    raw_block: str,
    prompt_version: str,
) -> str:
    extra = ""
    if prompt_version == "v2":
        extra = """
Replacement policy:
- Default to should_replace=false. Replace only when the evidence contains a
  clearly better phrase that directly fills the question slot.
- A better phrase is usually an exact or near-exact wording after verbs such as
  said, described, felt, called, learned, offered, advised, inspired, or
  happened.
- Do not replace with a broader paraphrase, a neighboring fact, or a longer
  explanation.
- If the current/base answer is already the exact adjective, title, quote,
  event, or list item, keep it.
- For "what did X say" questions, prefer what X said, not what the narrator
  inferred from it.
- For "how did/does X describe/feel" questions, prefer the adjective or short
  phrase used in the evidence.
- For "what advice/offer" questions, extract the concrete advice or offer.
"""
    return f"""You are a conservative exact-slot rescue module for LoCoMo QA.

The base answer is already produced by a strong answer arbitrator. Your job is
only to rescue cases where the evidence has a clearer exact phrase for the
question slot. If you are not sure, keep the base answer.

QUESTION:
{question}

BASE ANSWER:
{base_answer}

OTHER CANDIDATES:
- GAM: {gam_answer}
- CURRENT: {current_answer}
- ELAA: {elaa_answer}

MEMORY SUMMARY:
{_clip(summary, 2200)}

RAW EVIDENCE SNIPPETS:
{_clip(raw_block, 3500)}

Rules:
- Return should_replace=false unless the new final_answer is clearly better
  supported and more slot-aligned than BASE ANSWER.
- final_answer must be only the short answer text.
- Prefer exact phrases from MEMORY SUMMARY or RAW EVIDENCE.
- Do not use the gold answer; it is not available.
{extra}

Return JSON.
"""


def _fallback(question: str, base: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "should_replace": False,
        "final_answer": _postprocess_answer(question, _answer(base)),
        "evidence_phrase": "",
        "slot_type": "other",
        "confidence": "low",
        "reason": "fallback after slot rescue failure",
    }


def _run_one(
    qid: str,
    gam: Dict[str, Any],
    current: Dict[str, Any],
    elaa: Dict[str, Any],
    base: Dict[str, Any],
    current_run_dir: Path,
    generator: CachedOpenAIGenerator,
    max_raw_chars: int,
    prompt_version: str,
) -> Dict[str, Any]:
    question = str(base.get("question") or current.get("question") or gam.get("question") or "")
    try:
        trace_file = _trace_path(current_run_dir, qid)
        pages_file = _pages_path(current_run_dir, qid)
        trace = _load_json(trace_file) if trace_file.exists() else {}
        pages = _load_json(pages_file) if pages_file.exists() else []
        source_ids = _collect_source_ids(trace)
        raw_block = _build_raw_block(pages, source_ids, question, max_raw_chars)
        prompt = _prompt(
            question=question,
            base_answer=_answer(base),
            gam_answer=_answer(gam),
            current_answer=_answer(current),
            elaa_answer=_answer(elaa),
            summary=str(base.get("research_summary") or current.get("research_summary") or ""),
            raw_block=raw_block,
            prompt_version=prompt_version,
        )
        response = generator.generate_single(prompt=prompt, schema=SLOT_SCHEMA)
        data = response.get("json")
        if not isinstance(data, dict):
            data = _fallback(question, base)
    except Exception as exc:
        data = _fallback(question, base)
        data["reason"] = f"slot rescue failed: {type(exc).__name__}: {exc}"
        source_ids = []
        raw_block = ""

    final = str(data.get("final_answer") or "").strip()
    if not final:
        final = _answer(base)
    final = _postprocess_answer(question, final)
    data["final_answer_postprocessed"] = final

    row = dict(base)
    if data.get("should_replace"):
        row["summary_answer"] = final
        row["pred"] = final
    row["slot_rescue"] = {
        "qid": qid,
        "source_ids": source_ids,
        "raw_chars": len(raw_block),
        "base_answer": _answer(base),
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
    parser.add_argument(
        "--base-run",
        default="memory_directions/results/locomo-full-elaa-vbr-refined",
    )
    parser.add_argument("--tag", default="locomo-full-slot-rescue-v1")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--max-raw-chars", type=int, default=4500)
    parser.add_argument("--prompt-version", choices=["v1", "v2"], default="v2")
    parser.add_argument(
        "--accept-slot-types",
        nargs="*",
        default=None,
        help="If set, apply replacements only for these predicted slot_type values.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--exclude-prefix", nargs="*", default=["conv-26"])
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    gam_rows = _load_rows(_resolve_path(args.gam_run) / "all_qa_results.json")
    current_run_dir = _resolve_path(args.current_run)
    current_rows = _load_rows(current_run_dir / "all_qa_results.json")
    elaa_rows = _load_rows(_resolve_path(args.elaa_run) / "all_qa_results.json")
    base_rows = _load_rows(_resolve_path(args.base_run) / "all_qa_results.json")

    ids = sorted(set(gam_rows) & set(current_rows) & set(elaa_rows) & set(base_rows))
    for prefix in args.exclude_prefix or []:
        ids = [qid for qid in ids if not qid.startswith(prefix)]
    if args.only:
        allowed = set(args.only)
        ids = [qid for qid in ids if qid in allowed or qid.rsplit("-q", 1)[0] in allowed]
    if args.limit:
        ids = ids[: args.limit]

    target_ids = [qid for qid in ids if _is_target(str(base_rows[qid].get("question") or ""))]

    outdir = RESULTS / args.tag
    outdir.mkdir(parents=True, exist_ok=True)

    generator = CachedOpenAIGenerator(
        {
            "model_name": args.model,
            "cache_dir": str(CACHE / "llm"),
            "temperature": args.temperature,
            "max_tokens": 512,
            "role": "slot_rescue",
            "use_schema": True,
        }
    )
    if args.no_cache:
        generator.enable_cache = False

    print(
        f"dataset={args.dataset} ids={len(ids)} target_ids={len(target_ids)} "
        f"model={args.model} workers={args.workers} outdir={outdir}",
        flush=True,
    )
    t0 = time.time()
    rescued: Dict[str, Dict[str, Any]] = {}
    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {
                pool.submit(
                    _run_one,
                    qid,
                    gam_rows[qid],
                    current_rows[qid],
                    elaa_rows[qid],
                    base_rows[qid],
                    current_run_dir,
                    generator,
                    args.max_raw_chars,
                    args.prompt_version,
                ): qid
                for qid in target_ids
            }
            for done, fut in enumerate(as_completed(futs), 1):
                qid = futs[fut]
                rescued[qid] = fut.result()
                if done % 25 == 0 or done == len(futs):
                    print(f"[{done}/{len(futs)}] elapsed={round(time.time() - t0)}s", flush=True)
    else:
        for idx, qid in enumerate(target_ids, 1):
            rescued[qid] = _run_one(
                qid,
                gam_rows[qid],
                current_rows[qid],
                elaa_rows[qid],
                base_rows[qid],
                current_run_dir,
                generator,
                args.max_raw_chars,
                args.prompt_version,
            )
            if idx % 25 == 0 or idx == len(target_ids):
                print(f"[{idx}/{len(target_ids)}] elapsed={round(time.time() - t0)}s", flush=True)

    rows: List[Dict[str, Any]] = []
    n_replaced = 0
    accept_slot_types = set(args.accept_slot_types or [])
    for qid in ids:
        if qid in rescued:
            row = rescued[qid]
            decision = row.get("slot_rescue", {}).get("decision", {})
            should_replace = bool(decision.get("should_replace"))
            if should_replace and accept_slot_types and decision.get("slot_type") not in accept_slot_types:
                row["summary_answer"] = _answer(base_rows[qid])
                row["pred"] = _answer(base_rows[qid])
                decision["filtered_out"] = True
                should_replace = False
            n_replaced += int(should_replace)
            rows.append(row)
        else:
            row = dict(base_rows[qid])
            row["slot_rescue"] = {"qid": qid, "skipped": True}
            rows.append(row)

    metrics = score(rows, args.dataset)
    summary = {
        "tag": args.tag,
        "dataset": args.dataset,
        "method": "offline_slot_rescue",
        "gam_run": str(_resolve_path(args.gam_run)),
        "current_run": str(current_run_dir),
        "elaa_run": str(_resolve_path(args.elaa_run)),
        "base_run": str(_resolve_path(args.base_run)),
        "model": args.model,
        "temperature": args.temperature,
        "workers": args.workers,
        "prompt_version": args.prompt_version,
        "accept_slot_types": sorted(accept_slot_types),
        "n_questions": len(rows),
        "n_targets": len(target_ids),
        "n_replaced": n_replaced,
        "metrics": metrics,
        "generator_stats": generator.stats(),
        "wall_secs": round(time.time() - t0, 1),
    }
    (outdir / "all_qa_results.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
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
