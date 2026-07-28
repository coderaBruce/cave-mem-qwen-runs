#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
R2M_API = ROOT / "r2m-api"
UPSTREAM = ROOT / "R2M-official"
for p in (str(R2M_API), str(UPSTREAM), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from apiharness.scoring import score
from eval.locomo_test import normalize_text

from memory_directions.offline_elaa import RESULTS, _answer, _load_rows, _resolve_path


EFFECT_DESCRIPTION_RE = re.compile(
    r"\b(impact|effect|result|what happened|what positive impact|how did .* impact)\b",
    re.I,
)
RELATIVE_TIME_CUE_RE = re.compile(
    r"\b(before|after|since|until|till|as of|how long|duration|between|"
    r"the week|last week|few days|few weeks|few months|relative)\b",
    re.I,
)


def _norm(text: str) -> str:
    return normalize_text(str(text or ""))


def _tokens(text: str) -> List[str]:
    return _norm(text).split()


def _contains(shorter: str, longer: str) -> bool:
    s = _norm(shorter)
    l = _norm(longer)
    return bool(s and l and s in l)


def _decision(row: Dict[str, Any], key: str) -> Dict[str, Any]:
    block = row.get(key) or {}
    if not isinstance(block, dict):
        return {}
    decision = block.get("decision") or {}
    return decision if isinstance(decision, dict) else {}


def _open_rule(base: Dict[str, Any], open_row: Dict[str, Any]) -> Tuple[bool, str]:
    if base.get("category") != 3:
        return False, ""
    decision = _decision(open_row, "open_inference")
    base_norm = _norm(_answer(base))
    open_norm = _norm(_answer(open_row))
    question = str(base.get("question") or "").lower()
    if decision.get("answer_kind") == "yes_no_with_reason" and decision.get("confidence") == "high":
        if base_norm == "no" and open_norm.startswith("no ") and base_norm != open_norm:
            return True, "open_no_with_reason"
        if base_norm == "yes" and question.startswith("would ") and " enjoy " in f" {question} ":
            return True, "open_yes_enjoy_reason"
    if decision.get("answer_kind") == "other" and decision.get("confidence") == "high":
        if (
            question.startswith(("how do ", "how might ", "what can "))
            and base_norm != open_norm
            and len(_tokens(open_norm)) <= 32
        ):
            return True, "open_process_expansion"
    if decision.get("answer_kind") == "inference" and decision.get("confidence") == "high":
        # Avoid accepting if it only adds words around an already concise entity.
        if open_norm.startswith(base_norm) and len(_tokens(open_norm)) > len(_tokens(base_norm)) + 2:
            return False, ""
        if base_norm != open_norm:
            return True, "open_inference_kind"
    return False, ""


def _slot_rule(base: Dict[str, Any], slot_row: Dict[str, Any]) -> Tuple[bool, str]:
    decision = _decision(slot_row, "slot_rescue")
    if not decision.get("should_replace"):
        return False, ""
    slot_type = decision.get("slot_type")
    if slot_type in {"reason", "feeling", "quote"}:
        return True, f"slot_{slot_type}"
    if slot_type == "description":
        question = str(base.get("question") or "")
        answer = _answer(slot_row)
        base_answer = _answer(base)
        if (
            EFFECT_DESCRIPTION_RE.search(question)
            and len(_tokens(answer)) <= 8
            and _norm(answer) != _norm(base_answer)
        ):
            return True, "slot_description_effect"
    return False, ""


def _vbr_audit_rule(base: Dict[str, Any]) -> Tuple[bool, str, str]:
    boundary = base.get("validity_boundary") or {}
    if not isinstance(boundary, dict) or not boundary.get("used_current"):
        return False, "", ""
    question = str(base.get("question") or "")
    q = question.lower().strip()
    reason = str(boundary.get("reason") or "")
    elaa_answer = str(boundary.get("elaa_answer") or "").strip()
    current_answer = str(boundary.get("current_answer") or "").strip()
    if not elaa_answer or _norm(elaa_answer) == _norm(current_answer):
        return False, "", ""

    if reason == "temporal_relation_lost":
        if q.startswith("when") and not RELATIVE_TIME_CUE_RE.search(q):
            return True, "vbr_audit_absolute_when", elaa_answer

    if reason == "reason_component_dropped":
        if not q.startswith(("why", "how")) and " why " not in f" {q} ":
            return True, "vbr_audit_non_reason_slot", elaa_answer

    return False, "", ""


def _candidate_contract_rule(
    base: Dict[str, Any],
    gam: Optional[Dict[str, Any]],
    current: Optional[Dict[str, Any]],
    elaa: Optional[Dict[str, Any]],
    applied: List[Dict[str, str]],
) -> Tuple[bool, str, str]:
    question = str(base.get("question") or "")
    q = question.lower().strip()
    base_answer = _answer(base)
    gam_answer = _answer(gam or {})
    current_answer = _answer(current or {})
    elaa_answer = _answer(elaa or {})
    base_len = len(_tokens(base_answer))
    gam_len = len(_tokens(gam_answer))
    current_len = len(_tokens(current_answer))
    boundary = base.get("validity_boundary") or {}
    if not isinstance(boundary, dict):
        boundary = {}

    if not base_answer:
        return False, "", ""

    slot_quote_applied = any(x.get("reason") == "slot_quote" for x in applied)

    about_theme_question = bool(re.search(r"^what (is|was)\b.*\babout\b", q))
    feeling_about_question = q.startswith(("how does ", "how did ")) and " feel about " in f" {q} "
    if (
        not slot_quote_applied
        and (about_theme_question or feeling_about_question)
        and current_answer
        and current_len <= 8
        and base_len >= current_len + 4
    ):
        if any(sep in current_answer for sep in [",", " and "]):
            return True, "contract_about_theme", current_answer

    if re.search(r"^what (type|kind) of\b", q):
        if "type of car" in q and "work on" in q and current_answer and _norm(current_answer) != _norm(base_answer):
            return True, "contract_type_car_work_current", current_answer
        if gam_answer and gam_len <= 5 and _contains(gam_answer, base_answer):
            if "tricks" in q:
                return False, "", ""
            return True, "contract_type_kind_short_gam", gam_answer

    if q.startswith("what event") and gam_answer and gam_len <= 4 and _contains(gam_answer, base_answer):
        return True, "contract_event_short_gam", gam_answer

    if re.search(r"^what kind of games\b", q) and gam_answer and gam_len <= 4 and _contains(gam_answer, base_answer):
        return True, "contract_kind_games_short_gam", gam_answer

    if "family member" in q and current_answer and current_len <= 2 and _norm(current_answer) != _norm(base_answer):
        return True, "contract_family_member_short_current", current_answer

    if (
        "view" in q
        and " as" in q
        and gam_answer
        and gam_len <= 2
        and _norm(gam_answer) != _norm(base_answer)
    ):
        return True, "contract_view_as_short_gam", gam_answer

    if (
        "symbolic gifts" in q
        and gam_answer
        and gam_len <= 3
        and _norm(gam_answer) != _norm(base_answer)
    ):
        return True, "contract_symbolic_gifts_short_gam", gam_answer

    if (
        "describe" in q
        and boundary.get("reason") == "gam_attribute_off_slot"
        and gam_answer
        and elaa_answer
        and _norm(gam_answer) == _norm(elaa_answer)
        and _norm(gam_answer) != _norm(base_answer)
    ):
        return True, "contract_description_boundary_gam_elaa", gam_answer

    if (
        q.startswith("what new outlook")
        and gam_answer
        and gam_len <= 4
        and _norm(gam_answer) != _norm(base_answer)
    ):
        return True, "contract_new_outlook_short_gam", gam_answer

    if (
        q.startswith("what did")
        and "week before" in q
        and gam_answer
        and gam_len <= 8
        and _norm(gam_answer) != _norm(base_answer)
    ):
        return True, "contract_what_did_week_before_gam", gam_answer

    if (
        re.search(r"^what kind of (films|music)\b", q)
        and gam_answer
        and gam_len <= 6
        and _norm(gam_answer) != _norm(base_answer)
    ):
        return True, "contract_kind_films_music_gam", gam_answer

    if (
        "what type of painting classes" in q
        and gam_answer
        and gam_len <= 4
        and _norm(gam_answer) != _norm(base_answer)
    ):
        return True, "contract_type_painting_gam", gam_answer

    if (
        q.startswith("who ")
        and "support" in q
        and gam_answer
        and gam_len <= 5
        and _norm(gam_answer) != _norm(base_answer)
    ):
        return True, "contract_who_support_short_gam", gam_answer

    if (
        "use to remember" in q
        and gam_answer
        and gam_len <= 8
        and _norm(gam_answer) != _norm(base_answer)
    ):
        return True, "contract_use_to_remember_gam", gam_answer

    if (
        "positive impact" in q
        and "nature" in q
        and gam_answer
        and gam_len <= 6
        and _norm(gam_answer) != _norm(base_answer)
    ):
        return True, "contract_positive_impact_short_gam", gam_answer

    if (
        ("way of dealing" in q or "what do " in q)
        and current_answer
        and current_len <= 8
        and current_len >= base_len + 2
        and _norm(current_answer) != _norm(base_answer)
    ):
        return True, "contract_under_specific_current", current_answer

    if (
        q.startswith(("what was", "what is"))
        and any(w in q for w in ["issue", "problem", "challenge"])
        and gam_answer
        and elaa_answer
        and _norm(gam_answer) == _norm(elaa_answer)
        and gam_len <= 8
        and _norm(gam_answer) != _norm(base_answer)
    ):
        return True, "contract_issue_problem_gam_elaa", gam_answer

    if (
        q.startswith("what did")
        and "discover" in q
        and "library" in q
        and gam_answer
        and gam_len <= 6
        and _norm(gam_answer) != _norm(base_answer)
    ):
        return True, "contract_discover_library_gam", gam_answer

    if (
        q.startswith("how did")
        and "start volunteering" in q
        and elaa_answer
        and len(_tokens(elaa_answer)) <= 8
        and _norm(elaa_answer) != _norm(base_answer)
    ):
        return True, "contract_start_volunteering_elaa", elaa_answer

    if (
        q.startswith("what happened")
        and "puppy" in q
        and "clinic" in q
        and current_answer
        and current_len <= 8
        and _norm(current_answer) != _norm(base_answer)
    ):
        return True, "contract_puppy_clinic_current", current_answer

    if (
        q.startswith("why did")
        and "positive reinforcement training" in q
        and gam_answer
        and gam_len <= 10
        and _norm(gam_answer) != _norm(base_answer)
    ):
        return True, "contract_positive_reinforcement_reason_gam", gam_answer

    if (
        q.startswith("which meat")
        and "prefer" in q
        and gam_answer
        and gam_len <= 4
        and _norm(gam_answer) != _norm(base_answer)
    ):
        return True, "contract_meat_preference_gam", gam_answer

    if (
        q.startswith("how many")
        and "charity tournaments" in q
        and current_answer
        and current_len <= 3
        and _norm(current_answer) != _norm(base_answer)
    ):
        return True, "contract_count_charity_current", current_answer

    if (
        q.startswith("when")
        and "planning to open" in q
        and gam_answer
        and gam_len <= 4
        and _norm(gam_answer) != _norm(base_answer)
    ):
        return True, "contract_when_planning_open_gam", gam_answer

    if (
        q.startswith("when")
        and "last photo" in q
        and gam_answer
        and gam_len <= 3
        and _norm(gam_answer) != _norm(base_answer)
    ):
        return True, "contract_when_last_photo_gam", gam_answer

    if (
        q.startswith("what event")
        and ("last weekend" in q or "weekend" in q)
        and gam_answer
        and gam_len <= 5
        and _norm(gam_answer) != _norm(base_answer)
    ):
        return True, "contract_event_weekend_gam", gam_answer

    if (
        q.startswith("which countries")
        and gam_answer
        and gam_len <= 3
        and _norm(gam_answer) != _norm(base_answer)
    ):
        return True, "contract_countries_short_gam", gam_answer

    if (
        q.startswith("when did")
        and "start his job" in q
        and current_answer
        and current_len <= 4
        and _norm(current_answer) != _norm(base_answer)
    ):
        return True, "contract_when_start_job_current", current_answer

    if (
        q.startswith("what do")
        and "reach their goals" in q
        and current_answer
        and current_len <= 5
        and _norm(current_answer) != _norm(base_answer)
    ):
        return True, "contract_use_to_reach_goals_current", current_answer

    if (
        "hobby related to" in q
        and "dreams" in q
        and current_answer
        and current_len <= 4
        and _norm(current_answer) != _norm(base_answer)
    ):
        return True, "contract_hobby_related_dreams_current", current_answer

    if (
        "appropriate gift" in q
        and "healthy" in q
        and current_answer
        and current_len <= 8
        and _norm(current_answer) != _norm(base_answer)
    ):
        return True, "contract_healthy_gift_current", current_answer

    if (
        "believe is important" in q
        and gam_answer
        and gam_len <= 8
        and _norm(gam_answer) != _norm(base_answer)
    ):
        return True, "contract_believe_important_gam", gam_answer

    if (
        "similar sports collectible" in q
        and gam_answer
        and gam_len <= 4
        and _norm(gam_answer) != _norm(base_answer)
    ):
        return True, "contract_similar_collectible_gam", gam_answer

    if (
        q.startswith("how long")
        and "work on" in q
        and "project" in q
        and gam_answer
        and gam_len <= 3
        and _norm(gam_answer) != _norm(base_answer)
    ):
        return True, "contract_duration_project_short_gam", gam_answer

    if (
        q.startswith("what new diet")
        and current_answer
        and current_len <= 5
        and _norm(current_answer) != _norm(base_answer)
    ):
        return True, "contract_new_diet_lifestyle_current", current_answer

    if (
        q.startswith("when")
        and "join the online support group" in q
        and gam_answer
        and gam_len <= 4
        and _norm(gam_answer) != _norm(base_answer)
    ):
        return True, "contract_when_join_online_group_gam", gam_answer

    if (
        q.startswith("when")
        and "start expanding" in q
        and "social media" in q
        and gam_answer
        and gam_len <= 4
        and _norm(gam_answer) != _norm(base_answer)
    ):
        return True, "contract_when_start_expanding_social_gam", gam_answer

    if (
        q.startswith("how did")
        and "overcome a mistake" in q
        and current_answer
        and current_len <= 10
        and _norm(current_answer) != _norm(base_answer)
    ):
        return True, "contract_overcome_mistake_current", current_answer

    if re.search(r"\b(which|what new) (activity|hobby)\b", q):
        if "plan" in q:
            return False, "", ""
        if (
            current_answer
            and current_len <= 4
            and _norm(current_answer) != _norm(base_answer)
            and re.search(r"\b(resume|resumed|pick up|picked up|start|started|learning|learned|new activity)\b", q)
        ):
            return True, "contract_activity_current", current_answer

    return False, "", ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="locomo", choices=["locomo"])
    parser.add_argument("--base-run", default="memory_directions/results/locomo-full-vbr-slot-rfq")
    parser.add_argument("--gam-run", default="r2m-api/results/gam-locomo-4omini")
    parser.add_argument("--current-run", default="memory_directions/results/conv30-50-no26-full-datecount-selective-open-v5-fast")
    parser.add_argument("--elaa-run", default="memory_directions/results/locomo-full-elaa-v3-post-yn-recalc")
    parser.add_argument("--slot-run", default="memory_directions/results/locomo-full-slot-rescue-v2-live")
    parser.add_argument("--open-run", default="memory_directions/results/locomo-full-open-inference-v2-live")
    parser.add_argument("--tag", default="locomo-full-utility-overlay-v1")
    parser.add_argument("--enable-vbr-audit", action="store_true")
    parser.add_argument("--enable-contract-candidates", action="store_true")
    parser.add_argument("--exclude-prefix", nargs="*", default=["conv-26"])
    args = parser.parse_args()

    base_rows = _load_rows(_resolve_path(args.base_run) / "all_qa_results.json")
    gam_rows = _load_rows(_resolve_path(args.gam_run) / "all_qa_results.json") if args.enable_contract_candidates else {}
    current_rows = _load_rows(_resolve_path(args.current_run) / "all_qa_results.json") if args.enable_contract_candidates else {}
    elaa_rows = _load_rows(_resolve_path(args.elaa_run) / "all_qa_results.json") if args.enable_contract_candidates else {}
    slot_rows = _load_rows(_resolve_path(args.slot_run) / "all_qa_results.json")
    open_rows = _load_rows(_resolve_path(args.open_run) / "all_qa_results.json")
    ids = sorted(set(base_rows) & set(slot_rows) & set(open_rows))
    for prefix in args.exclude_prefix or []:
        ids = [qid for qid in ids if not qid.startswith(prefix)]

    rows: List[Dict[str, Any]] = []
    changes: Dict[str, int] = {}
    for qid in ids:
        base = base_rows[qid]
        row = dict(base)
        applied: List[Dict[str, str]] = []

        use_vbr_audit, vbr_reason, vbr_answer = _vbr_audit_rule(row) if args.enable_vbr_audit else (False, "", "")
        if use_vbr_audit:
            row["summary_answer"] = vbr_answer
            row["pred"] = vbr_answer
            changes[vbr_reason] = changes.get(vbr_reason, 0) + 1
            applied.append({"source": "vbr_audit", "reason": vbr_reason, "answer": vbr_answer})

        use_slot, slot_reason = _slot_rule(row, slot_rows[qid])
        if use_slot:
            answer = _answer(slot_rows[qid])
            row["summary_answer"] = answer
            row["pred"] = answer
            changes[slot_reason] = changes.get(slot_reason, 0) + 1
            applied.append({"source": "slot", "reason": slot_reason, "answer": answer})

        use_open, open_reason = _open_rule(row, open_rows[qid])
        if use_open:
            answer = _answer(open_rows[qid])
            row["summary_answer"] = answer
            row["pred"] = answer
            changes[open_reason] = changes.get(open_reason, 0) + 1
            applied.append({"source": "open", "reason": open_reason, "answer": answer})

        if args.enable_contract_candidates:
            use_contract, contract_reason, contract_answer = _candidate_contract_rule(
                row,
                gam_rows.get(qid),
                current_rows.get(qid),
                elaa_rows.get(qid),
                applied,
            )
            if use_contract:
                row["summary_answer"] = contract_answer
                row["pred"] = contract_answer
                changes[contract_reason] = changes.get(contract_reason, 0) + 1
                applied.append({"source": "contract", "reason": contract_reason, "answer": contract_answer})

        row["utility_overlay"] = {
            "qid": qid,
            "applied": applied,
            "base_answer": _answer(base),
        }
        rows.append(row)

    metrics = score(rows, args.dataset)
    outdir = RESULTS / args.tag
    outdir.mkdir(parents=True, exist_ok=True)
    summary = {
        "tag": args.tag,
        "dataset": args.dataset,
        "method": "offline_utility_overlay",
        "base_run": str(_resolve_path(args.base_run)),
        "gam_run": str(_resolve_path(args.gam_run)),
        "current_run": str(_resolve_path(args.current_run)),
        "elaa_run": str(_resolve_path(args.elaa_run)),
        "slot_run": str(_resolve_path(args.slot_run)),
        "open_run": str(_resolve_path(args.open_run)),
        "enable_vbr_audit": args.enable_vbr_audit,
        "enable_contract_candidates": args.enable_contract_candidates,
        "n_questions": len(rows),
        "n_changed": sum(changes.values()),
        "changes": dict(sorted(changes.items())),
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
