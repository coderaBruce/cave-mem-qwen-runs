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
from apiharness.answer_style import build_style_shots, format_shots
from apiharness.scoring import score
from memory_directions.answering import _canonical_answer


RESULTS = ROOT / "memory_directions" / "results"
CACHE = ROOT / "memory_directions" / ".cache"


ELAA_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "final_answer": {"type": "string"},
        "gam_summary_extract": {"type": "string"},
        "enhanced_summary_extract": {"type": "string"},
        "chosen_source": {
            "type": "string",
            "enum": [
                "gam_answer",
                "enhanced_answer",
                "gam_summary_extract",
                "enhanced_summary_extract",
                "merged_extract",
            ],
        },
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "answer_type": {
            "type": "string",
            "enum": [
                "person",
                "place",
                "date",
                "count",
                "title",
                "quote",
                "reason",
                "boolean",
                "list",
                "attribute",
                "other",
            ],
        },
        "gam_supported": {"type": "boolean"},
        "enhanced_supported": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": [
        "final_answer",
        "gam_summary_extract",
        "enhanced_summary_extract",
        "chosen_source",
        "confidence",
        "answer_type",
        "gam_supported",
        "enhanced_supported",
        "reason",
    ],
    "additionalProperties": False,
}


CATEGORY_NAMES = {
    1: "multi-hop",
    2: "temporal",
    3: "open-domain",
    4: "single-hop",
}


def _load_rows(path: Path) -> Dict[str, Dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {r["_id"]: r for r in rows if "error" not in r and "_id" in r}


def _resolve_path(value: str) -> Path:
    path = Path(value)
    if path.exists():
        return path
    path = RESULTS / value
    if path.exists():
        return path
    raise FileNotFoundError(value)


def _clip(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    head = max_chars * 3 // 4
    tail = max_chars - head
    return text[:head].rstrip() + "\n...[middle omitted]...\n" + text[-tail:].lstrip()


def _answer(row: Dict[str, Any]) -> str:
    return str(row.get("pred") or row.get("summary_answer") or "").strip()


def _gold_list(row: Dict[str, Any]) -> List[str]:
    gold = row.get("gold_answers")
    if isinstance(gold, list):
        return [str(x) for x in gold]
    one = row.get("gold_answer")
    return [] if one is None else [str(one)]


def _prompt(
    *,
    question: str,
    category: Any,
    gam_answer: str,
    gam_summary: str,
    enhanced_answer: str,
    enhanced_summary: str,
    max_summary_chars: int,
    style_examples: Optional[List[Any]] = None,
    prompt_version: str = "v3",
) -> str:
    category_name = CATEGORY_NAMES.get(category, str(category))
    examples = ""
    if style_examples:
        examples = (
            "\nEXAMPLES OF EXPECTED SHORT ANSWER STYLE FOR THIS CATEGORY:\n"
            f"{format_shots(style_examples)}\n"
        )
    extra_rules = ""
    extra_examples = ""
    if prompt_version == "v5":
        extra_rules = """
- Do not assume that the most specific answer is best. If the question asks for
  a `type`, `kind`, `style`, or `category`, prefer the supported type/category
  phrase over a named instance.
- If the question asks `how long`, prefer a short rounded duration that matches
  conversational wording over an over-precise day count, unless the question
  explicitly asks for an exact date span.
- If the question asks what a creative work is `about`, prefer the supported
  themes/topics over a long plot description.
- If two answers are both supported, choose the one whose wording most directly
  fills the question slot, even if the other has more surrounding detail.
"""
        extra_examples = """
- If the question asks what type of car someone worked on and the evidence has
  both `classic muscle car` and `vintage Mustang`, extract `classic muscle car`
  because the question asks for a type.
- If the question asks what a screenplay is about and the evidence lists
  `loss, identity, and connection`, extract those themes rather than a long
  character plot.
- If the question asks how long something took and the evidence has both
  `nearly four months` and an exact day-count approximation, prefer
  `nearly four months`.
"""
    return f"""You are an evidence-localized answer arbitrator for LoCoMo memory QA.

You are given two candidate inference paths:
- GAM: the plain memory-search baseline.
- ENHANCED: a typed-control memory path.

Your job is to output the final short answer that is most likely to match the
benchmark's token-F1 answer style. You do NOT see the gold answer.

QUESTION CATEGORY:
{category_name}

QUESTION:
{question}

{examples}

GAM FINAL ANSWER:
{gam_answer}

GAM MEMORY SUMMARY:
{_clip(gam_summary, max_summary_chars)}

ENHANCED FINAL ANSWER:
{enhanced_answer}

ENHANCED MEMORY SUMMARY:
{_clip(enhanced_summary, max_summary_chars)}

Rules:
- Use only facts supported by the two memory summaries and candidate answers.
- Prefer the shortest complete answer, not the most explanatory sentence.
- Even when GAM and ENHANCED final answers are identical, scan both summaries
  for a more exact answer span before choosing.
- If a concise GAM answer is directly supported, keep it unless the enhanced
  path clearly supplies a more exact or more complete answer.
- If a summary contains a precise entity, title, nickname, quote, date, count,
  or list that directly answers the question, extract that span instead of a
  generic paraphrase.
- Do not choose "Insufficient evidence" when either path provides a supported
  answer.
- For date questions, prefer absolute dates over relative phrases.
- For count questions, prefer exact counts such as "two times" or "three".
- For yes/no questions, start with Yes or No and keep the answer minimal.
- For "what/which" questions, match the requested type: title, object, hobby,
  animal, company, food, activity, or attribute. Avoid answering with a broader
  category when a specific supported value is present.
- For "why/how" questions, include the concrete cause/action if it is present;
  avoid vague abstractions such as "significant hardships" when concrete items
  are listed.
{extra_rules}

Extraction examples:
- If the summary says `a contemporary dance piece called "Finding Freedom"` and
  the question asks what dance piece, extract `"Finding Freedom"`, not
  `contemporary dance piece`.
- If the summary says `hardships including a divorce, losing her job, and
  becoming homeless` and the question asks what hardships, extract
  `divorce, job loss, homelessness`, not `significant hardships`.
- If the summary says `adjusting well ... learning commands and house training`
  and the question asks how the puppy is adjusting, extract both the status and
  the concrete progress.
- If the summary says `started volunteering after witnessing a family
  struggling and reaching out to the shelter`, extract that cause/action, not a
  time expression such as `about a year ago`.
{extra_examples}

Return JSON. The final_answer must be only the answer text, with no prefix.
"""


def _fallback_choice(gam: Dict[str, Any], enhanced: Dict[str, Any]) -> Dict[str, Any]:
    ans = _answer(enhanced) or _answer(gam)
    return {
        "final_answer": ans,
        "gam_summary_extract": "",
        "enhanced_summary_extract": "",
        "chosen_source": "enhanced_answer" if _answer(enhanced) else "gam_answer",
        "confidence": "low",
        "answer_type": "other",
        "gam_supported": False,
        "enhanced_supported": False,
        "reason": "fallback after arbitration failure",
    }


def _postprocess_answer(question: str, answer: str) -> str:
    text = _canonical_answer(question, answer)
    qlower = question.lower()
    lower = text.lower().strip(" .")
    if qlower.strip().startswith(
        (
            "is ",
            "was ",
            "were ",
            "are ",
            "do ",
            "does ",
            "did ",
            "has ",
            "had ",
            "can ",
            "could ",
            "would ",
        )
    ):
        if lower.startswith("yes"):
            return "Yes"
        if lower.startswith("no"):
            return "No"
    if "style of dance" in qlower and lower.endswith(" dance"):
        return text[: -len(" dance")].strip()
    return text


def _run_one(
    qid: str,
    gam: Dict[str, Any],
    enhanced: Dict[str, Any],
    generator: CachedOpenAIGenerator,
    max_summary_chars: int,
    shots: Dict[Any, List[Any]],
    prompt_version: str,
) -> Dict[str, Any]:
    prompt = _prompt(
        question=str(enhanced.get("question") or gam.get("question") or ""),
        category=enhanced.get("category", gam.get("category")),
        gam_answer=_answer(gam),
        gam_summary=str(gam.get("research_summary") or ""),
        enhanced_answer=_answer(enhanced),
        enhanced_summary=str(enhanced.get("research_summary") or ""),
        max_summary_chars=max_summary_chars,
        style_examples=shots.get(enhanced.get("category", gam.get("category"))) or [],
        prompt_version=prompt_version,
    )
    try:
        response = generator.generate_single(prompt=prompt, schema=ELAA_SCHEMA)
        data = response.get("json")
        if not isinstance(data, dict):
            data = _fallback_choice(gam, enhanced)
    except Exception as exc:
        data = _fallback_choice(gam, enhanced)
        data["reason"] = f"arbitration failed: {type(exc).__name__}: {exc}"

    question = str(enhanced.get("question") or gam.get("question") or "")
    final = str(data.get("final_answer") or "").strip()
    if not final:
        final = _answer(enhanced) or _answer(gam)
        data["final_answer"] = final
        data["confidence"] = "low"
    final = _postprocess_answer(question, final)
    data["final_answer_postprocessed"] = final

    row = dict(enhanced)
    row["summary_answer"] = final
    row["pred"] = final
    row["elaa"] = {
        "qid": qid,
        "gam_answer": _answer(gam),
        "enhanced_answer": _answer(enhanced),
        "decision": data,
    }
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="locomo", choices=["locomo"])
    parser.add_argument("--gam-run", default="r2m-api/results/gam-locomo-4omini")
    parser.add_argument(
        "--enhanced-run",
        default="memory_directions/results/conv30-50-no26-full-datecount-selective-open-v5-fast",
    )
    parser.add_argument("--tag", default="locomo-full-elaa-offline-v1")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--max-summary-chars", type=int, default=6000)
    parser.add_argument("--answer-shots", type=int, default=0)
    parser.add_argument("--prompt-version", choices=["v3", "v5"], default="v3")
    parser.add_argument("--shots-run", default=None)
    parser.add_argument("--shots-samples", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--exclude-prefix", nargs="*", default=["conv-26"])
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    gam_path = _resolve_path(args.gam_run) / "all_qa_results.json"
    enhanced_path = _resolve_path(args.enhanced_run) / "all_qa_results.json"
    gam_rows = _load_rows(gam_path)
    enhanced_rows = _load_rows(enhanced_path)

    ids = sorted(set(gam_rows) & set(enhanced_rows))
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
            "role": "elaa",
            "use_schema": True,
        }
    )
    if args.no_cache:
        generator.enable_cache = False

    shots: Dict[Any, List[Any]] = {}
    if args.answer_shots > 0:
        if not args.shots_run or not args.shots_samples:
            print("--answer-shots requires --shots-run and --shots-samples")
            return 2
        shots = build_style_shots(
            _resolve_path(args.shots_run),
            args.shots_samples,
            per_category=args.answer_shots,
            seed=42,
        )
        print(
            "ELAA style shots per category: "
            + json.dumps({str(k): len(v) for k, v in shots.items()})
        )

    print(
        f"dataset={args.dataset} ids={len(ids)} model={args.model} "
        f"workers={args.workers} outdir={outdir}"
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
                    enhanced_rows[qid],
                    generator,
                    args.max_summary_chars,
                    shots,
                    args.prompt_version,
                ): idx
                for idx, qid in enumerate(ids)
            }
            for done, fut in enumerate(as_completed(futs), 1):
                rows[futs[fut]] = fut.result()
                if done % 50 == 0 or done == len(futs):
                    print(f"[{done}/{len(futs)}] elapsed={round(time.time() - t0)}s")
    else:
        for idx, qid in enumerate(ids):
            rows[idx] = _run_one(
                qid,
                gam_rows[qid],
                enhanced_rows[qid],
                generator,
                args.max_summary_chars,
                shots,
                args.prompt_version,
            )
            if (idx + 1) % 50 == 0 or idx + 1 == len(ids):
                print(f"[{idx + 1}/{len(ids)}] elapsed={round(time.time() - t0)}s")

    all_qa = [r for r in rows if r is not None]
    metrics = score(all_qa, args.dataset)
    summary = {
        "tag": args.tag,
        "dataset": args.dataset,
        "method": "offline_elaa",
        "gam_run": str(_resolve_path(args.gam_run)),
        "enhanced_run": str(_resolve_path(args.enhanced_run)),
        "model": args.model,
        "temperature": args.temperature,
        "workers": args.workers,
        "max_summary_chars": args.max_summary_chars,
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
