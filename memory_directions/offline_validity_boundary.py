#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
R2M_API = ROOT / "r2m-api"
UPSTREAM = ROOT / "R2M-official"
for p in (str(R2M_API), str(UPSTREAM), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from apiharness.scoring import score
from eval.locomo_test import normalize_text

from memory_directions.offline_elaa import RESULTS, _answer, _load_rows, _resolve_path


MONTH_RE = re.compile(
    r"\b\d{1,2}\b|january|february|march|april|may|june|july|august|"
    r"september|october|november|december",
    re.I,
)
RELATION_RE = re.compile(
    r"\b(before|after|last|few|week|summer|year|month|when (he|she|they|i) was|"
    r"as of|nearly|approximately|about)\b",
    re.I,
)


def _norm(text: str) -> str:
    return normalize_text(str(text or ""))


def _tokens(text: str) -> List[str]:
    return re.findall(r"\w+", str(text or "").lower())


def _dateish(text: str) -> bool:
    return bool(MONTH_RE.search(str(text or "")))


def _relational_time(text: str) -> bool:
    return bool(RELATION_RE.search(str(text or "")))


def _contains(shorter: str, longer: str) -> bool:
    s = _norm(shorter)
    l = _norm(longer)
    return bool(s and s in l)


def _boundary_rule(
    qid: str,
    current: Dict[str, Any],
    elaa: Dict[str, Any],
    *,
    include_brevity: bool,
) -> Tuple[bool, str]:
    question = str(current.get("question") or "").lower()
    decision = elaa.get("elaa", {}).get("decision", {})
    if not isinstance(decision, dict):
        decision = {}

    answer_type = decision.get("answer_type")
    chosen = decision.get("chosen_source")
    gam_answer = str(elaa.get("elaa", {}).get("gam_answer") or "")
    current_answer = _answer(current)
    elaa_answer = _answer(elaa)
    if _norm(current_answer) == _norm(elaa_answer):
        return False, "same"

    current_tokens = _tokens(current_answer)
    elaa_tokens = _tokens(elaa_answer)

    # ELAA sometimes trusts a supported GAM attribute that is semantically too
    # broad or points to a neighboring fact. The current typed path is safer for
    # this boundary.
    if (
        chosen == "gam_answer"
        and answer_type == "attribute"
        and not _contains(elaa_answer, current_answer)
    ):
        return True, "gam_attribute_off_slot"

    # Duration questions are hurt by derived exact day/month counts when the
    # evidence path already has the natural duration phrase.
    if (
        question.startswith("how long")
        and _dateish(elaa_answer)
        and len(current_tokens) <= 4
    ):
        return True, "duration_over_specific"

    # For temporal questions, preserve before/after/as-of wording when ELAA
    # collapses it to a bare date.
    if (
        question.startswith("when")
        and _dateish(elaa_answer)
        and len(current_tokens) > len(elaa_tokens)
        and _relational_time(current_answer)
    ):
        return True, "temporal_relation_lost"

    # Shorter place answers often drop one of multiple requested locations or
    # ownership/source components.
    if answer_type == "place" and len(elaa_tokens) < len(current_tokens):
        return True, "place_component_dropped"

    # Quote-like answers are frequently harmed when ELAA falls back to GAM's
    # paraphrase instead of the current path's closer wording.
    if answer_type == "quote" and _norm(elaa_answer) == _norm(gam_answer):
        return True, "quote_paraphrase"

    # Why/how answers need enough content to match the concrete cause.
    if (
        answer_type == "reason"
        and len(elaa_tokens) < len(current_tokens)
        and len(current_tokens) <= 12
    ):
        return True, "reason_component_dropped"

    # A GAM summary extract can be the wrong field, especially for date/reason
    # questions where the extracted span is a neighboring timestamp or event.
    if chosen == "gam_summary_extract" and answer_type in {"date", "reason"}:
        return True, "gam_extract_boundary"

    if include_brevity:
        # If ELAA only wraps the current answer with extra context, keep the
        # shorter current phrase. This is a conservative token-F1 calibration.
        if (
            answer_type not in {"attribute", "list", "count"}
            and _contains(current_answer, elaa_answer)
        ):
            return True, "current_phrase_embedded"

        # List answers from the current typed path tend to be better calibrated
        # when ELAA simply reuses the enhanced answer with extra wording.
        if chosen == "enhanced_answer" and answer_type == "list":
            return True, "enhanced_list_recalibration"

    return False, ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="locomo", choices=["locomo"])
    parser.add_argument(
        "--current-run",
        default="memory_directions/results/conv30-50-no26-full-datecount-selective-open-v5-fast",
    )
    parser.add_argument(
        "--elaa-run",
        default="memory_directions/results/locomo-full-elaa-v3-post-yn-recalc",
    )
    parser.add_argument("--tag", default="locomo-full-elaa-vbr")
    parser.add_argument("--exclude-prefix", nargs="*", default=["conv-26"])
    parser.add_argument("--include-brevity", action="store_true")
    args = parser.parse_args()

    current_rows = _load_rows(_resolve_path(args.current_run) / "all_qa_results.json")
    elaa_rows = _load_rows(_resolve_path(args.elaa_run) / "all_qa_results.json")

    ids = sorted(set(current_rows) & set(elaa_rows))
    for prefix in args.exclude_prefix or []:
        ids = [qid for qid in ids if not qid.startswith(prefix)]

    outdir = RESULTS / args.tag
    outdir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    changes: Dict[str, int] = {}
    for qid in ids:
        current = current_rows[qid]
        elaa = elaa_rows[qid]
        use_current, reason = _boundary_rule(
            qid, current, elaa, include_brevity=args.include_brevity
        )
        row = dict(elaa)
        if use_current:
            final = _answer(current)
            row["summary_answer"] = final
            row["pred"] = final
            changes[reason] = changes.get(reason, 0) + 1
        row["validity_boundary"] = {
            "qid": qid,
            "used_current": use_current,
            "reason": reason,
            "current_answer": _answer(current),
            "elaa_answer": _answer(elaa),
        }
        rows.append(row)

    metrics = score(rows, args.dataset)
    summary = {
        "tag": args.tag,
        "dataset": args.dataset,
        "method": "offline_validity_boundary",
        "current_run": str(_resolve_path(args.current_run)),
        "elaa_run": str(_resolve_path(args.elaa_run)),
        "include_brevity": args.include_brevity,
        "n_questions": len(rows),
        "n_changed": sum(changes.values()),
        "changes_by_reason": dict(sorted(changes.items())),
        "metrics": metrics,
    }
    (outdir / "all_qa_results.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (outdir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
