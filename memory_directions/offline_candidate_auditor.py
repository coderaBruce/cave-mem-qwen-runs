#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
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
from eval.locomo_test import normalize_text

from memory_directions.offline_elaa import CACHE, RESULTS, _answer, _clip, _load_rows, _postprocess_answer, _resolve_path


AUDITOR_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "should_switch": {"type": "boolean"},
        "chosen_candidate": {
            "type": "string",
            "enum": ["base", "gam", "current", "elaa", "vbr", "open"],
        },
        "answer_slot": {
            "type": "string",
            "enum": [
                "date",
                "count",
                "person",
                "place",
                "activity",
                "object",
                "title",
                "type_or_kind",
                "reason",
                "feeling",
                "quote",
                "list",
                "yes_no",
                "yes_no_with_reason",
                "description",
                "other",
            ],
        },
        "failure_boundary": {
            "type": "string",
            "enum": [
                "none_keep_base",
                "base_off_slot",
                "base_wrong_temporal_anchor",
                "base_over_specific",
                "base_under_specific",
                "base_overlong",
                "base_missing_reason",
                "base_missing_list_item",
                "candidate_more_exact_phrase",
                "candidate_unsupported",
                "semantic_equivalent",
                "uncertain_keep_base",
            ],
        },
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "reason": {"type": "string"},
    },
    "required": [
        "should_switch",
        "chosen_candidate",
        "answer_slot",
        "failure_boundary",
        "confidence",
        "reason",
    ],
    "additionalProperties": False,
}


QUESTION_SLOT_RE = re.compile(
    r"^(what|which|who|where|when|how many|how long|how old|did|does|do|was|were|is|are|would|could|has|have|had)\b",
    re.I,
)
YES_NO_RE = re.compile(r"^(did|does|do|was|were|is|are|would|could|has|have|had)\b", re.I)


def _norm(text: str) -> str:
    return normalize_text(str(text or ""))


def _tokens(text: str) -> List[str]:
    return _norm(text).split()


def _jaccard(a: str, b: str) -> float:
    at = set(_tokens(a))
    bt = set(_tokens(b))
    if not at and not bt:
        return 1.0
    if not at or not bt:
        return 0.0
    return len(at & bt) / len(at | bt)


def _contains(a: str, b: str) -> bool:
    na = _norm(a)
    nb = _norm(b)
    return bool(na and nb and (na in nb or nb in na))


def _shortish(text: str, max_tokens: int = 18) -> bool:
    return len(_tokens(text)) <= max_tokens


def _yes_no_only(text: str) -> bool:
    return _norm(text) in {"yes", "no", "likely yes", "likely no", "presumably yes", "presumably no"}


def _candidate_lines(candidates: Dict[str, str]) -> str:
    return "\n".join(f"- {name.upper()}: {answer}" for name, answer in candidates.items())


def _is_target(
    qid: str,
    row: Dict[str, Any],
    candidates: Dict[str, str],
    *,
    target_mode: str,
) -> Tuple[bool, str]:
    question = str(row.get("question") or "")
    q = question.lower()
    base = candidates.get("base", "")
    unique = {_norm(v) for v in candidates.values() if _norm(v)}
    if len(unique) <= 1:
        return False, "all_candidates_same"
    if target_mode == "all":
        return True, "all_disagreement"

    boundary = row.get("validity_boundary") or {}
    if boundary.get("used_current"):
        reason = str(boundary.get("reason") or "unknown")
        if target_mode == "vbr":
            return True, f"vbr_{reason}"
        if reason in {
            "temporal_relation_lost",
            "reason_component_dropped",
            "enhanced_list_recalibration",
            "quote_paraphrase",
            "gam_extract_boundary",
        }:
            return True, f"high_risk_vbr_{reason}"
    elif target_mode == "vbr":
        return False, "not_vbr_changed"

    if row.get("category") == 3:
        if YES_NO_RE.search(question) and _yes_no_only(base):
            return True, "open_yes_no_bare"
        return True, "open_domain_disagreement"

    if YES_NO_RE.search(question) and _yes_no_only(base):
        for name, ans in candidates.items():
            if name != "base" and len(_tokens(ans)) >= 4 and _norm(base) in _norm(ans):
                return True, "yes_no_candidate_has_reason"

    if not QUESTION_SLOT_RE.search(question):
        return False, "not_slot_question"

    base_len = len(_tokens(base))
    for name, ans in candidates.items():
        if name == "base" or not ans:
            continue
        if _norm(ans) == _norm(base):
            continue
        if _shortish(base) and _shortish(ans):
            jac = _jaccard(base, ans)
            if jac <= 0.35 and not _contains(base, ans):
                return True, "short_slot_disagreement"
        if base_len >= 8 and len(_tokens(ans)) <= max(5, base_len // 2) and _contains(ans, base):
            return True, "candidate_shorter_contained"
        if len(_tokens(ans)) >= 4 and base_len <= 2 and _contains(base, ans):
            return True, "candidate_expands_short_base"

    if any(x in q for x in ["type", "kind", "activity", "hobby", "event", "issue", "outlook"]):
        return True, "semantic_slot_disagreement"
    return False, "low_risk"


def _prompt(
    *,
    question: str,
    category: Any,
    target_reason: str,
    candidates: Dict[str, str],
    gam_summary: str,
    current_summary: str,
    elaa_decision: Dict[str, Any],
    validity_boundary: Dict[str, Any],
    prompt_version: str,
) -> str:
    elaa_reason = str(elaa_decision.get("reason") or "")
    gam_extract = str(elaa_decision.get("gam_summary_extract") or "")
    current_extract = str(elaa_decision.get("enhanced_summary_extract") or "")
    boundary_reason = str(validity_boundary.get("reason") or "")
    boundary_current = str(validity_boundary.get("current_answer") or "")
    boundary_elaa = str(validity_boundary.get("elaa_answer") or "")
    extra = ""
    if prompt_version == "v2":
        extra = """
Candidate-boundary policy:
- BASE is the default. Switch only when another candidate is clearly more
  aligned to the requested answer slot.
- Prefer the candidate that answers the exact slot asked by the question, not a
  neighboring event, reason, date, object, or broader paraphrase.
- For "when" questions, do not prefer a relative phrase such as "the week
  before ..." unless the question itself asks for a relative relation or the
  relative phrase is the only supported answer.
- For "what type/kind/activity/hobby/event/issue/outlook" questions, prefer the
  requested abstraction level: a type over a named instance, or a named item
  over a generic type only when the question asks for the named item.
- For yes/no questions, keep bare Yes/No unless the question asks for a basis,
  reason, or the candidate reason is directly supported by memory.
- Do not switch just because two candidates are paraphrases. Switch only for a
  boundary violation: wrong slot, wrong temporal anchor, missing required
  component, or over/under-specific answer.
"""
    return f"""You are a candidate-boundary auditor for LoCoMo memory QA.

The BASE answer is the current best answer. Other candidates come from earlier
memory-search variants. Your task is not to invent a new answer. Choose whether
BASE crossed a validity boundary, and if so choose the better existing
candidate.

QUESTION CATEGORY:
{category}

TARGETING REASON:
{target_reason}

QUESTION:
{question}

CANDIDATE ANSWERS:
{_candidate_lines(candidates)}

GAM SUMMARY:
{_clip(gam_summary, 1600)}

CURRENT SUMMARY:
{_clip(current_summary, 2100)}

ELAA DECISION:
- reason: {elaa_reason}
- extract from GAM: {_clip(gam_extract, 700)}
- extract from CURRENT: {_clip(current_extract, 700)}

VALIDITY BOUNDARY RECORD:
- reason: {boundary_reason}
- current answer: {boundary_current}
- elaa answer: {boundary_elaa}

Rules:
- Output chosen_candidate as one of the listed candidates.
- If uncertain, choose BASE and should_switch=false.
- If should_switch=true, chosen_candidate must not be BASE.
- Do not use the gold answer; it is not available.
- Optimize for a concise benchmark-style short answer, but only using support
  from the provided summaries and candidate provenance.
{extra}

Return JSON.
"""


def _fallback(question: str, base: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "should_switch": False,
        "chosen_candidate": "base",
        "answer_slot": "other",
        "failure_boundary": "uncertain_keep_base",
        "confidence": "low",
        "reason": "fallback after candidate auditor failure",
    }


def _run_one(
    qid: str,
    rows_by_name: Dict[str, Dict[str, Any]],
    generator: CachedOpenAIGenerator,
    prompt_version: str,
    target_reason: str,
) -> Dict[str, Any]:
    base = rows_by_name["base"]
    gam = rows_by_name["gam"]
    current = rows_by_name["current"]
    elaa = rows_by_name["elaa"]
    candidates = {name: _answer(row) for name, row in rows_by_name.items()}
    question = str(base.get("question") or current.get("question") or gam.get("question") or "")
    elaa_decision = elaa.get("elaa", {}).get("decision", {})
    if not isinstance(elaa_decision, dict):
        elaa_decision = {}
    validity_boundary = base.get("validity_boundary") or {}
    if not isinstance(validity_boundary, dict):
        validity_boundary = {}

    try:
        prompt = _prompt(
            question=question,
            category=base.get("category"),
            target_reason=target_reason,
            candidates=candidates,
            gam_summary=str(gam.get("research_summary") or ""),
            current_summary=str(current.get("research_summary") or ""),
            elaa_decision=elaa_decision,
            validity_boundary=validity_boundary,
            prompt_version=prompt_version,
        )
        response = generator.generate_single(prompt=prompt, schema=AUDITOR_SCHEMA)
        data = response.get("json")
        if not isinstance(data, dict):
            data = _fallback(question, base)
    except Exception as exc:
        data = _fallback(question, base)
        data["reason"] = f"candidate auditor failed: {type(exc).__name__}: {exc}"

    chosen = str(data.get("chosen_candidate") or "base").strip().lower()
    if chosen not in candidates:
        chosen = "base"
        data["chosen_candidate"] = "base"
        data["should_switch"] = False
        data["reason"] = f"invalid chosen candidate; {data.get('reason') or ''}".strip()
    final = candidates[chosen]
    final = _postprocess_answer(question, final)
    data["final_answer_postprocessed"] = final

    row = dict(base)
    if data.get("should_switch") and chosen != "base":
        row["summary_answer"] = final
        row["pred"] = final
    row["candidate_auditor"] = {
        "qid": qid,
        "target_reason": target_reason,
        "candidates": candidates,
        "decision": data,
    }
    return row


def _parse_extra(item: str) -> Tuple[str, str]:
    if "=" in item:
        name, run = item.split("=", 1)
        return name.strip(), run.strip()
    path = Path(item)
    return path.name.replace("-", "_"), item


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
        "--vbr-run",
        default="memory_directions/results/locomo-full-elaa-vbr-refined",
    )
    parser.add_argument(
        "--base-run",
        default="memory_directions/results/locomo-full-vbr-slot-rfq",
    )
    parser.add_argument("--extra-run", action="append", default=[])
    parser.add_argument("--tag", default="locomo-full-candidate-auditor-v1")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--prompt-version", choices=["v1", "v2"], default="v2")
    parser.add_argument("--target-mode", choices=["risk", "vbr", "all"], default="risk")
    parser.add_argument(
        "--min-confidence",
        choices=["high", "medium", "low"],
        default="high",
        help="Minimum confidence required to apply a switch.",
    )
    parser.add_argument(
        "--accept-boundaries",
        nargs="*",
        default=None,
        help="If set, apply switches only for these predicted failure_boundary labels.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--exclude-prefix", nargs="*", default=["conv-26"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    run_rows: Dict[str, Dict[str, Dict[str, Any]]] = {
        "gam": _load_rows(_resolve_path(args.gam_run) / "all_qa_results.json"),
        "current": _load_rows(_resolve_path(args.current_run) / "all_qa_results.json"),
        "elaa": _load_rows(_resolve_path(args.elaa_run) / "all_qa_results.json"),
        "vbr": _load_rows(_resolve_path(args.vbr_run) / "all_qa_results.json"),
        "base": _load_rows(_resolve_path(args.base_run) / "all_qa_results.json"),
    }
    for item in args.extra_run:
        name, run = _parse_extra(item)
        if name not in AUDITOR_SCHEMA["properties"]["chosen_candidate"]["enum"]:
            raise ValueError(f"extra-run name must be one of schema candidates, got {name!r}")
        run_rows[name] = _load_rows(_resolve_path(run) / "all_qa_results.json")

    ids = sorted(set.intersection(*(set(rows) for rows in run_rows.values())))
    for prefix in args.exclude_prefix or []:
        ids = [qid for qid in ids if not qid.startswith(prefix)]
    if args.only:
        allowed = set(args.only)
        ids = [qid for qid in ids if qid in allowed or qid.rsplit("-q", 1)[0] in allowed]
    if args.limit:
        ids = ids[: args.limit]

    target_reasons: Dict[str, str] = {}
    for qid in ids:
        rows_by_name = {name: rows[qid] for name, rows in run_rows.items()}
        candidates = {name: _answer(row) for name, row in rows_by_name.items()}
        target, reason = _is_target(
            qid,
            rows_by_name["base"],
            candidates,
            target_mode=args.target_mode,
        )
        if target:
            target_reasons[qid] = reason

    outdir = RESULTS / args.tag
    outdir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        summary = {
            "tag": args.tag,
            "dataset": args.dataset,
            "method": "offline_candidate_auditor",
            "target_mode": args.target_mode,
            "n_questions": len(ids),
            "n_targets": len(target_reasons),
            "target_reasons": dict(sorted({k: list(target_reasons.values()).count(k) for k in set(target_reasons.values())}.items())),
        }
        (outdir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    generator = CachedOpenAIGenerator(
        {
            "model_name": args.model,
            "cache_dir": str(CACHE / "llm"),
            "temperature": args.temperature,
            "max_tokens": 512,
            "role": "candidate_auditor",
            "use_schema": True,
        }
    )
    if args.no_cache:
        generator.enable_cache = False

    print(
        f"dataset={args.dataset} ids={len(ids)} target_ids={len(target_reasons)} "
        f"target_mode={args.target_mode} model={args.model} workers={args.workers} outdir={outdir}",
        flush=True,
    )
    t0 = time.time()
    audited: Dict[str, Dict[str, Any]] = {}
    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {
                pool.submit(
                    _run_one,
                    qid,
                    {name: rows[qid] for name, rows in run_rows.items()},
                    generator,
                    args.prompt_version,
                    target_reasons[qid],
                ): qid
                for qid in target_reasons
            }
            for done, fut in enumerate(as_completed(futs), 1):
                qid = futs[fut]
                audited[qid] = fut.result()
                if done % 25 == 0 or done == len(futs):
                    print(f"[{done}/{len(futs)}] elapsed={round(time.time() - t0)}s", flush=True)
    else:
        for idx, qid in enumerate(target_reasons, 1):
            audited[qid] = _run_one(
                qid,
                {name: rows[qid] for name, rows in run_rows.items()},
                generator,
                args.prompt_version,
                target_reasons[qid],
            )
            if idx % 25 == 0 or idx == len(target_reasons):
                print(f"[{idx}/{len(target_reasons)}] elapsed={round(time.time() - t0)}s", flush=True)

    confidence_rank = {"low": 0, "medium": 1, "high": 2}
    min_conf = confidence_rank[args.min_confidence]
    accept_boundaries = set(args.accept_boundaries or [])
    rows: List[Dict[str, Any]] = []
    n_switched = 0
    switch_counts: Dict[str, int] = {}
    boundary_counts: Dict[str, int] = {}
    for qid in ids:
        if qid in audited:
            row = audited[qid]
            auditor = row.get("candidate_auditor", {})
            decision = auditor.get("decision", {})
            chosen = str(decision.get("chosen_candidate") or "base").lower()
            boundary = str(decision.get("failure_boundary") or "")
            conf = str(decision.get("confidence") or "low").lower()
            should_switch = bool(decision.get("should_switch")) and chosen != "base"
            if confidence_rank.get(conf, 0) < min_conf:
                should_switch = False
                decision["filtered_out"] = "confidence"
            if should_switch and accept_boundaries and boundary not in accept_boundaries:
                should_switch = False
                decision["filtered_out"] = "boundary"
            if not should_switch:
                base_answer = _answer(run_rows["base"][qid])
                row["summary_answer"] = base_answer
                row["pred"] = base_answer
            else:
                n_switched += 1
                switch_counts[chosen] = switch_counts.get(chosen, 0) + 1
                boundary_counts[boundary] = boundary_counts.get(boundary, 0) + 1
            rows.append(row)
        else:
            row = dict(run_rows["base"][qid])
            row["candidate_auditor"] = {"qid": qid, "skipped": True}
            rows.append(row)

    metrics = score(rows, args.dataset)
    summary = {
        "tag": args.tag,
        "dataset": args.dataset,
        "method": "offline_candidate_auditor",
        "base_run": str(_resolve_path(args.base_run)),
        "gam_run": str(_resolve_path(args.gam_run)),
        "current_run": str(_resolve_path(args.current_run)),
        "elaa_run": str(_resolve_path(args.elaa_run)),
        "vbr_run": str(_resolve_path(args.vbr_run)),
        "extra_run": args.extra_run,
        "model": args.model,
        "temperature": args.temperature,
        "workers": args.workers,
        "prompt_version": args.prompt_version,
        "target_mode": args.target_mode,
        "min_confidence": args.min_confidence,
        "accept_boundaries": sorted(accept_boundaries),
        "n_questions": len(rows),
        "n_targets": len(target_reasons),
        "n_switched": n_switched,
        "switch_counts": dict(sorted(switch_counts.items())),
        "boundary_counts": dict(sorted(boundary_counts.items())),
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
