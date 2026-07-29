#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
R2M_API = ROOT / "r2m-api"
UPSTREAM = ROOT / "R2M-official"
for p in (str(R2M_API), str(UPSTREAM), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from apiharness.cached_generator import CachedOpenAIGenerator
from apiharness.scoring import score
from memory_directions.answering import _canonical_answer
from memory_directions.offline_elaa import RESULTS, _answer, _load_rows, _resolve_path


VALIDATION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "final_answer": {"type": "string"},
        "source": {"type": "string"},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "expected_effect": {
            "type": "string",
            "enum": ["positive", "neutral", "negative"],
        },
        "reason": {"type": "string"},
    },
    "required": ["final_answer", "source", "confidence", "expected_effect", "reason"],
    "additionalProperties": False,
}


CATEGORY_NAMES = {
    1: "multi-hop",
    2: "temporal",
    3: "open-domain",
    4: "single-hop",
}


def _parse_candidate(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        path = _resolve_path(value)
        return path.name, path
    label, raw_path = value.split("=", 1)
    return label.strip(), _resolve_path(raw_path.strip())


def _clip(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    head = max_chars * 2 // 3
    tail = max_chars - head
    return text[:head].rstrip() + "\n...[middle omitted]...\n" + text[-tail:].lstrip()


def _candidate_block(label: str, row: Dict[str, Any], max_summary_chars: int) -> str:
    return f"""CANDIDATE: {label}
FINAL ANSWER:
{_answer(row)}

MEMORY SUMMARY:
{_clip(str(row.get("research_summary") or ""), max_summary_chars)}
"""


def _prompt(
    *,
    question: str,
    category: Any,
    base_label: str,
    base_row: Dict[str, Any],
    candidates: List[Tuple[str, Dict[str, Any]]],
    max_summary_chars: int,
    policy: str,
) -> str:
    category_name = CATEGORY_NAMES.get(category, str(category))
    candidate_text = "\n\n".join(
        _candidate_block(label, row, max_summary_chars) for label, row in candidates
    )
    if policy == "direct_safe":
        policy_rules = """
- Be especially conservative for direct factual, multi-hop, and single-hop
  questions. If GAM gives a short answer and the base/candidate answer is longer
  or more abstract, choose GAM only when the GAM memory summary supports it.
- For open-domain questions, do not reward extra explanation by itself. Prefer a
  concise supported status, entity, reason, or yes/no answer.
- For temporal questions, keep the base answer unless another candidate clearly
  provides a more exact supported date or count. Do not replace a supported
  temporal answer with a vague conversational phrase.
"""
    else:
        policy_rules = """
- Choose a non-base candidate when it is more directly responsive to the
  question slot, better supported, or less verbose while preserving all required
  facts.
- Keep the base when candidates merely rephrase it, add unsupported detail, or
  trade a specific supported fact for a broader phrase.
"""
    return f"""Validate candidate answers for a memory QA system.

QUESTION CATEGORY:
{category_name}

QUESTION:
{question}

BASE SYSTEM: {base_label}
BASE FINAL ANSWER:
{_answer(base_row)}

BASE MEMORY SUMMARY:
{_clip(str(base_row.get("research_summary") or ""), max_summary_chars)}

OTHER CANDIDATES:
{candidate_text}

Decision principle:
Select the answer expected to improve short-answer token F1 relative to the
base system. This is an abstaining validator: if no candidate is clearly better
than the base, keep the base.

General rules:
- Use only facts supported by the shown memory summaries and candidate answers.
- Prefer the shortest complete answer that directly fills the question slot.
- Do not choose a candidate only because it is more detailed; detail helps only
  when it supplies a required answer component.
- If the question asks for a type/kind/category, choose the type phrase rather
  than an over-specific instance unless the question requests the instance.
- If the question asks why/how, include the concrete cause/action when present.
- For yes/no questions, start with Yes or No unless the question asks for a
  likely/presumed status.
- For dates/counts, prefer exact supported answers over relative/vague wording.
{policy_rules}

Return JSON with:
- final_answer: answer text only, no prefix.
- source: one of {base_label} or a candidate label.
- confidence: high/medium/low.
- expected_effect: positive only if switching or keeping base is expected to
  improve/maintain F1; neutral if uncertain; negative if all candidates are bad.
- reason: concise evidence-based reason.
"""


def _fallback(base_label: str, base: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "final_answer": _answer(base),
        "source": base_label,
        "confidence": "low",
        "expected_effect": "neutral",
        "reason": "fallback to base after validation failure",
    }


def _run_one(
    qid: str,
    base_label: str,
    base: Dict[str, Any],
    candidates: List[Tuple[str, Dict[str, Any]]],
    generator: CachedOpenAIGenerator,
    max_summary_chars: int,
    policy: str,
) -> Dict[str, Any]:
    prompt = _prompt(
        question=str(base.get("question") or ""),
        category=base.get("category"),
        base_label=base_label,
        base_row=base,
        candidates=candidates,
        max_summary_chars=max_summary_chars,
        policy=policy,
    )
    try:
        response = generator.generate_single(prompt=prompt, schema=VALIDATION_SCHEMA)
        data = response.get("json")
        if not isinstance(data, dict):
            data = _fallback(base_label, base)
    except Exception as exc:
        data = _fallback(base_label, base)
        data["reason"] = f"validation failed: {type(exc).__name__}: {exc}"

    source = str(data.get("source") or base_label).strip()
    by_label = {base_label: base, **{label: row for label, row in candidates}}
    if source not in by_label:
        source = base_label
        data["source"] = base_label
    final = str(data.get("final_answer") or "").strip()
    if not final:
        final = _answer(by_label[source]) or _answer(base)
    final = _canonical_answer(str(base.get("question") or ""), final)

    row = dict(base)
    row["summary_answer"] = final
    row["pred"] = final
    row["candidate_validator"] = {
        "qid": qid,
        "base_label": base_label,
        "base_answer": _answer(base),
        "candidate_answers": {label: _answer(crow) for label, crow in candidates},
        "decision": data,
        "final_source": source,
    }
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="locomo", choices=["locomo"])
    parser.add_argument("--base-run", required=True)
    parser.add_argument("--base-label", default="base")
    parser.add_argument("--candidate-run", action="append", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--max-summary-chars", type=int, default=4200)
    parser.add_argument("--policy", choices=["balanced", "direct_safe"], default="balanced")
    parser.add_argument("--exclude-prefix", nargs="*", default=["conv-26"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    base_path = _resolve_path(args.base_run)
    base_rows = _load_rows(base_path / "all_qa_results.json")
    candidate_items = [_parse_candidate(item) for item in args.candidate_run]
    candidate_rows = [
        (label, _load_rows(path / "all_qa_results.json"), path)
        for label, path in candidate_items
    ]

    ids = set(base_rows)
    for _, rows, _ in candidate_rows:
        ids &= set(rows)
    ordered_ids = sorted(ids)
    for prefix in args.exclude_prefix or []:
        ordered_ids = [qid for qid in ordered_ids if not qid.startswith(prefix)]
    if args.limit:
        ordered_ids = ordered_ids[: args.limit]

    outdir = RESULTS / args.tag
    outdir.mkdir(parents=True, exist_ok=True)
    generator = CachedOpenAIGenerator(
        {
            "model_name": args.model,
            "cache_dir": str(ROOT / "memory_directions" / ".cache" / "llm"),
            "temperature": args.temperature,
            "max_tokens": 384,
            "role": f"candidate_validator_{args.policy}",
            "use_schema": True,
        }
    )
    if args.no_cache:
        generator.enable_cache = False

    print(
        f"dataset={args.dataset} ids={len(ordered_ids)} model={args.model} "
        f"policy={args.policy} workers={args.workers} outdir={outdir}"
    )
    t0 = time.time()
    rows: List[Optional[Dict[str, Any]]] = [None] * len(ordered_ids)
    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    _run_one,
                    qid,
                    args.base_label,
                    base_rows[qid],
                    [(label, crows[qid]) for label, crows, _ in candidate_rows],
                    generator,
                    args.max_summary_chars,
                    args.policy,
                ): idx
                for idx, qid in enumerate(ordered_ids)
            }
            for done, fut in enumerate(as_completed(futures), 1):
                rows[futures[fut]] = fut.result()
                if done % 50 == 0 or done == len(futures):
                    print(f"[{done}/{len(futures)}] elapsed={round(time.time() - t0)}s")
    else:
        for idx, qid in enumerate(ordered_ids):
            rows[idx] = _run_one(
                qid,
                args.base_label,
                base_rows[qid],
                [(label, crows[qid]) for label, crows, _ in candidate_rows],
                generator,
                args.max_summary_chars,
                args.policy,
            )

    all_qa = [row for row in rows if row is not None]
    metrics = score(all_qa, args.dataset)
    summary = {
        "tag": args.tag,
        "dataset": args.dataset,
        "method": "offline_candidate_validator",
        "base_run": str(base_path),
        "base_label": args.base_label,
        "candidate_runs": [
            {"label": label, "path": str(path)} for label, _, path in candidate_rows
        ],
        "model": args.model,
        "temperature": args.temperature,
        "workers": args.workers,
        "max_summary_chars": args.max_summary_chars,
        "policy": args.policy,
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
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
