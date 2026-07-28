"""Scoring, delegating to upstream metric implementations.

LoCoMo is scored per question category with token-F1 and BLEU-1; HotpotQA and
NarrativeQA take the max F1 over the gold answer list. All three use the
upstream functions so our numbers sit on the same scale as the papers'.

Reminder when reading these against the literature: GAM and R²-Mem report
*token-overlap F1*, while most industry LoCoMo numbers (Mem0, ByteRover, ~92%)
are *LLM-judge accuracy*. The two are not comparable and must never share a
table.
"""
from __future__ import annotations

from typing import Any, Dict, List

CATEGORY_NAMES = {
    1: "multi_hop",
    2: "temporal",
    3: "open_domain",
    4: "single_hop",
}


def score_locomo(qa_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    from eval.locomo_test import bleu1_score, f1_score

    buckets: Dict[Any, List[Dict[str, float]]] = {}
    for r in qa_results:
        if "error" in r:
            continue
        pred = r.get("summary_answer") or ""
        gold = r.get("gold_answer")
        gold = "" if gold is None else str(gold)
        rec = {"f1": f1_score(pred, gold), "bleu1": bleu1_score(pred, gold)}
        buckets.setdefault(r.get("category"), []).append(rec)

    out: Dict[str, Any] = {"by_category": {}}
    all_recs: List[Dict[str, float]] = []
    for cat, recs in sorted(buckets.items(), key=lambda kv: (kv[0] is None, kv[0])):
        all_recs.extend(recs)
        out["by_category"][CATEGORY_NAMES.get(cat, str(cat))] = {
            "n": len(recs),
            "f1": 100 * sum(x["f1"] for x in recs) / len(recs),
            "bleu1": 100 * sum(x["bleu1"] for x in recs) / len(recs),
        }
    if all_recs:
        out["overall"] = {
            "n": len(all_recs),
            "f1": 100 * sum(x["f1"] for x in all_recs) / len(all_recs),
            "bleu1": 100 * sum(x["bleu1"] for x in all_recs) / len(all_recs),
        }
    return out


def score_max_f1(qa_results: List[Dict[str, Any]], dataset: str) -> Dict[str, Any]:
    """HotpotQA / NarrativeQA: max token-F1 over the gold answer list."""
    if dataset.startswith("hotpotqa"):
        from eval.hotpotqa_test import _calculate_f1
    else:
        from eval.narrativeqa_test import _calculate_f1

    scores = [
        _calculate_f1(r.get("pred") or "", r.get("gold_answers") or [])
        for r in qa_results
        if "error" not in r
    ]
    if not scores:
        return {"overall": {"n": 0, "f1": 0.0}}
    return {
        "overall": {"n": len(scores), "f1": 100 * sum(scores) / len(scores)},
        "f1_scores": scores,
    }


def score(qa_results: List[Dict[str, Any]], dataset: str) -> Dict[str, Any]:
    if dataset == "locomo":
        return score_locomo(qa_results)
    return score_max_f1(qa_results, dataset)
