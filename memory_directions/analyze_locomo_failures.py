#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
R2M_API = ROOT / "r2m-api"
UPSTREAM = ROOT / "R2M-official"
for p in (str(R2M_API), str(UPSTREAM), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from eval.locomo_test import bleu1_score, f1_score, normalize_text

from memory_directions.offline_elaa import RESULTS, _answer, _load_rows, _resolve_path


CATEGORY_NAMES = {
    1: "multi_hop",
    2: "temporal",
    3: "open_domain",
    4: "single_hop",
}


def _score(pred: str, gold: str) -> Dict[str, float]:
    return {
        "f1": 100 * f1_score(str(pred or ""), str(gold or "")),
        "bleu1": 100 * bleu1_score(str(pred or ""), str(gold or "")),
    }


def _tokens(text: str) -> List[str]:
    return normalize_text(text).split()


def _first_word(question: str) -> str:
    toks = re.findall(r"[A-Za-z]+", str(question or "").lower())
    if not toks:
        return ""
    if len(toks) >= 2 and toks[0] in {"what", "which", "how"}:
        return f"{toks[0]} {toks[1]}"
    return toks[0]


def _pattern(question: str, pred: str, gold: str, category: Any) -> str:
    q = str(question or "").lower()
    p_norm = normalize_text(pred)
    g_norm = normalize_text(gold)
    p_toks = set(_tokens(pred))
    g_toks = set(_tokens(gold))
    if not p_norm:
        return "empty_pred"
    if g_norm in p_norm and len(_tokens(pred)) > len(_tokens(gold)) + 2:
        return "overlong_contains_gold"
    if p_norm in g_norm and len(_tokens(gold)) > len(_tokens(pred)) + 1:
        return "under_specific"
    if q.startswith(("did ", "does ", "do ", "was ", "were ", "is ", "are ", "has ", "have ", "had ", "would ", "could ")):
        if p_norm in {"yes", "no", "likely yes", "likely no", "presumably yes", "presumably no"} and len(_tokens(gold)) > 1:
            return "bare_yes_no_missing_reason"
        if g_norm in {"yes", "no"} and len(_tokens(pred)) > 1:
            return "yes_no_overexplained"
    if any(w in q for w in ["feel", "feeling", "emotion", "react"]):
        return "feeling_phrase"
    if q.startswith("why") or " why " in f" {q} ":
        return "reason_phrase"
    if "say" in q or "quote" in q or "told" in q:
        return "quote_phrase"
    if "where" in q or "which place" in q or "what location" in q:
        return "place_slot"
    if category == 3:
        return "open_domain"
    overlap = len(p_toks & g_toks)
    if overlap == 0:
        return "no_token_overlap"
    if len(_tokens(pred)) > len(_tokens(gold)) + 3:
        return "overlong_partial"
    if len(_tokens(pred)) + 2 < len(_tokens(gold)):
        return "short_partial"
    return "partial_overlap"


def _summarize(scores: Iterable[float]) -> float:
    values = list(scores)
    return sum(values) / len(values) if values else 0.0


def _effect_summary(values: List[float]) -> Dict[str, Any]:
    return {
        "n": len(values),
        "mean_delta": _summarize(values),
        "positive": sum(1 for x in values if x > 1e-9),
        "negative": sum(1 for x in values if x < -1e-9),
        "tie": sum(1 for x in values if abs(x) <= 1e-9),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--best-run", default="memory_directions/results/locomo-full-vbr-slot-rfq")
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
    parser.add_argument("--extra-run", action="append", default=[])
    parser.add_argument("--low-threshold", type=float, default=50.0)
    parser.add_argument("--out", default="memory_directions/reports/locomo_slot_rfq_failure_analysis.json")
    args = parser.parse_args()

    best_rows = _load_rows(_resolve_path(args.best_run) / "all_qa_results.json")
    candidate_runs = {
        "gam": _load_rows(_resolve_path(args.gam_run) / "all_qa_results.json"),
        "current": _load_rows(_resolve_path(args.current_run) / "all_qa_results.json"),
        "elaa": _load_rows(_resolve_path(args.elaa_run) / "all_qa_results.json"),
        "vbr": _load_rows(_resolve_path(args.vbr_run) / "all_qa_results.json"),
        "best": best_rows,
    }
    for item in args.extra_run:
        if "=" in item:
            name, run = item.split("=", 1)
        else:
            run = item
            name = Path(item).name
        candidate_runs[name] = _load_rows(_resolve_path(run) / "all_qa_results.json")

    ids = sorted(set(best_rows))
    candidate_ids = set.intersection(*(set(rows) for rows in candidate_runs.values()))
    ids = [qid for qid in ids if qid in candidate_ids]

    rows: List[Dict[str, Any]] = []
    by_cat: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    pattern_counts: Counter[str] = Counter()
    first_word_counts: Counter[str] = Counter()
    oracle_winner: Counter[str] = Counter()

    for qid in ids:
        row = best_rows[qid]
        question = str(row.get("question") or "")
        gold = str(row.get("gold_answer") or "")
        category = row.get("category")
        cat_name = CATEGORY_NAMES.get(category, str(category))
        pred = _answer(row)
        base_score = _score(pred, gold)
        candidate_scores: Dict[str, Dict[str, Any]] = {}
        for name, run_rows in candidate_runs.items():
            ans = _answer(run_rows[qid])
            candidate_scores[name] = {"answer": ans, **_score(ans, gold)}
        best_name, best_cand = max(candidate_scores.items(), key=lambda kv: kv[1]["f1"])
        oracle_winner[best_name] += 1
        pattern = _pattern(question, pred, gold, category)
        pattern_counts[pattern] += 1
        first_word_counts[_first_word(question)] += 1

        rec = {
            "qid": qid,
            "category": category,
            "category_name": cat_name,
            "question": question,
            "gold": gold,
            "pred": pred,
            "f1": base_score["f1"],
            "bleu1": base_score["bleu1"],
            "pattern": pattern,
            "oracle_best": best_name,
            "oracle_f1": best_cand["f1"],
            "oracle_gain": best_cand["f1"] - base_score["f1"],
            "validity_boundary": row.get("validity_boundary") or {},
            "slot_rescue": row.get("slot_rescue") or {},
            "candidates": candidate_scores,
        }
        rows.append(rec)
        by_cat[cat_name].append(rec)

    low_rows = [r for r in rows if r["f1"] < args.low_threshold]
    low_by_cat = {cat: [r for r in recs if r["f1"] < args.low_threshold] for cat, recs in by_cat.items()}
    low_patterns = Counter(r["pattern"] for r in low_rows)
    low_first_words = Counter(_first_word(r["question"]) for r in low_rows)
    recoverable = [r for r in low_rows if r["oracle_gain"] >= 30.0 and r["oracle_best"] != "best"]
    oracle_f1 = _summarize(r["oracle_f1"] for r in rows)
    oracle_gain = oracle_f1 - _summarize(r["f1"] for r in rows)

    vbr_effects: Dict[str, List[float]] = defaultdict(list)
    vbr_bad_examples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    slot_effects: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        boundary = r.get("validity_boundary") or {}
        if boundary.get("used_current"):
            reason = str(boundary.get("reason") or "unknown")
            delta = r["candidates"]["vbr"]["f1"] - r["candidates"]["elaa"]["f1"]
            vbr_effects[reason].append(delta)
            if delta < -20:
                vbr_bad_examples[reason].append(r)
        slot = r.get("slot_rescue") or {}
        if slot.get("used_rescue"):
            decision = slot.get("decision") or {}
            slot_type = str(decision.get("slot_type") or "unknown")
            delta = r["candidates"]["best"]["f1"] - r["candidates"]["vbr"]["f1"]
            slot_effects[slot_type].append(delta)

    report: Dict[str, Any] = {
        "best_run": str(_resolve_path(args.best_run)),
        "n": len(rows),
        "low_threshold": args.low_threshold,
        "overall_f1": _summarize(r["f1"] for r in rows),
        "candidate_oracle_f1": oracle_f1,
        "candidate_oracle_gain": oracle_gain,
        "overall_low_count": len(low_rows),
        "by_category": {
            cat: {
                "n": len(recs),
                "f1": _summarize(r["f1"] for r in recs),
                "low_count": len(low_by_cat[cat]),
                "low_patterns": Counter(r["pattern"] for r in low_by_cat[cat]).most_common(),
            }
            for cat, recs in sorted(by_cat.items())
        },
        "pattern_counts": pattern_counts.most_common(),
        "low_pattern_counts": low_patterns.most_common(),
        "question_prefix_counts": first_word_counts.most_common(25),
        "low_question_prefix_counts": low_first_words.most_common(25),
        "oracle_winner_counts": oracle_winner.most_common(),
        "vbr_effects_by_reason": {
            reason: _effect_summary(values)
            for reason, values in sorted(vbr_effects.items())
        },
        "vbr_bad_examples": {
            reason: sorted(examples, key=lambda r: r["candidates"]["vbr"]["f1"] - r["candidates"]["elaa"]["f1"])[:25]
            for reason, examples in sorted(vbr_bad_examples.items())
        },
        "slot_effects_by_type": {
            slot_type: _effect_summary(values)
            for slot_type, values in sorted(slot_effects.items())
        },
        "recoverable_low_count": len(recoverable),
        "top_recoverable_low": sorted(recoverable, key=lambda r: r["oracle_gain"], reverse=True)[:80],
        "lowest_rows": sorted(rows, key=lambda r: (r["f1"], -r["oracle_gain"]))[:120],
        "open_domain_low": sorted(
            [r for r in low_rows if r["category_name"] == "open_domain"],
            key=lambda r: (r["f1"], -r["oracle_gain"]),
        ),
    }

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({
        "n": report["n"],
        "overall_f1": report["overall_f1"],
        "candidate_oracle_f1": report["candidate_oracle_f1"],
        "candidate_oracle_gain": report["candidate_oracle_gain"],
        "overall_low_count": report["overall_low_count"],
        "by_category": report["by_category"],
        "low_pattern_counts": report["low_pattern_counts"][:12],
        "oracle_winner_counts": report["oracle_winner_counts"],
        "vbr_effects_by_reason": report["vbr_effects_by_reason"],
        "slot_effects_by_type": report["slot_effects_by_type"],
        "recoverable_low_count": report["recoverable_low_count"],
        "out": str(out),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
