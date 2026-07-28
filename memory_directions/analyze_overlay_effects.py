#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
R2M_API = ROOT / "r2m-api"
UPSTREAM = ROOT / "R2M-official"
for p in (str(R2M_API), str(UPSTREAM), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from eval.locomo_test import f1_score

from memory_directions.offline_elaa import _answer, _load_rows, _resolve_path


def _f1(answer: str, row: Dict[str, Any]) -> float:
    return 100 * f1_score(str(answer or ""), str(row.get("gold_answer") or ""))


def _summary(values: List[float]) -> Dict[str, Any]:
    return {
        "n": len(values),
        "mean_delta": sum(values) / len(values) if values else 0.0,
        "positive": sum(1 for x in values if x > 1e-9),
        "negative": sum(1 for x in values if x < -1e-9),
        "tie": sum(1 for x in values if abs(x) <= 1e-9),
    }


def _decision(row: Dict[str, Any], key: str) -> Dict[str, Any]:
    block = row.get(key) or {}
    if not isinstance(block, dict):
        return {}
    decision = block.get("decision") or {}
    return decision if isinstance(decision, dict) else {}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-run", default="memory_directions/results/locomo-full-vbr-slot-rfq")
    parser.add_argument("--overlay-run", required=True)
    parser.add_argument("--decision-key", default="raw_localizer")
    parser.add_argument("--category", type=int, default=None)
    parser.add_argument("--out", default="memory_directions/reports/overlay_effects.json")
    args = parser.parse_args()

    base_rows = _load_rows(_resolve_path(args.base_run) / "all_qa_results.json")
    overlay_rows = _load_rows(_resolve_path(args.overlay_run) / "all_qa_results.json")
    ids = sorted(set(base_rows) & set(overlay_rows))

    groups: Dict[str, List[float]] = defaultdict(list)
    examples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    all_values: List[float] = []
    rows: List[Dict[str, Any]] = []
    for qid in ids:
        base = base_rows[qid]
        if args.category is not None and base.get("category") != args.category:
            continue
        overlay = overlay_rows[qid]
        decision = _decision(overlay, args.decision_key)
        base_score = _f1(_answer(base), base)
        overlay_score = _f1(_answer(overlay), base)
        delta = overlay_score - base_score
        all_values.append(delta)
        keys = {
            "answer_type": str(
                decision.get("answer_type")
                or decision.get("answer_kind")
                or decision.get("slot_type")
                or "unknown"
            ),
            "chosen_source": str(decision.get("chosen_source") or "unknown"),
            "confidence": str(decision.get("confidence") or "unknown"),
        }
        combo_keys = [
            f"answer_type={keys['answer_type']}",
            f"chosen_source={keys['chosen_source']}",
            f"confidence={keys['confidence']}",
            f"answer_type={keys['answer_type']}|confidence={keys['confidence']}",
            f"answer_type={keys['answer_type']}|chosen_source={keys['chosen_source']}",
            f"chosen_source={keys['chosen_source']}|confidence={keys['confidence']}",
        ]
        block = overlay.get(args.decision_key) or {}
        if isinstance(block, dict):
            for item in block.get("applied") or []:
                if isinstance(item, dict):
                    source = str(item.get("source") or "unknown")
                    reason = str(item.get("reason") or "unknown")
                    combo_keys.append(f"applied_source={source}")
                    combo_keys.append(f"applied_reason={reason}")
        rec = {
            "qid": qid,
            "category": base.get("category"),
            "question": base.get("question"),
            "gold": base.get("gold_answer"),
            "base": _answer(base),
            "overlay": _answer(overlay),
            "base_f1": base_score,
            "overlay_f1": overlay_score,
            "delta": delta,
            "decision": decision,
        }
        rows.append(rec)
        for key in combo_keys:
            groups[key].append(delta)
            if abs(delta) > 20:
                examples[key].append(rec)

    report = {
        "base_run": str(_resolve_path(args.base_run)),
        "overlay_run": str(_resolve_path(args.overlay_run)),
        "decision_key": args.decision_key,
        "category": args.category,
        "n": len(ids),
        "overall": _summary(all_values),
        "groups": {
            key: _summary(values)
            for key, values in sorted(groups.items())
        },
        "top_positive": sorted(rows, key=lambda r: r["delta"], reverse=True)[:60],
        "top_negative": sorted(rows, key=lambda r: r["delta"])[:60],
        "examples": {
            key: sorted(value, key=lambda r: abs(r["delta"]), reverse=True)[:25]
            for key, value in sorted(examples.items())
        },
    }
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    compact = {
        "n": report["n"],
        "overall": report["overall"],
        "groups": {
            key: value
            for key, value in report["groups"].items()
            if value["n"] >= 3 and value["mean_delta"] > 0
        },
        "out": str(out),
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
