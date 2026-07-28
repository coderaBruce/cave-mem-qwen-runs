#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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


RAW_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "final_answer": {"type": "string"},
        "evidence_quote": {"type": "string"},
        "chosen_source": {
            "type": "string",
            "enum": ["raw_evidence", "gam_answer", "current_answer", "elaa_answer"],
        },
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
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "reason": {"type": "string"},
    },
    "required": [
        "final_answer",
        "evidence_quote",
        "chosen_source",
        "answer_type",
        "confidence",
        "reason",
    ],
    "additionalProperties": False,
}


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "both",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "his",
    "how",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "she",
    "that",
    "the",
    "their",
    "they",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


def _norm(text: str) -> str:
    return " ".join(str(text or "").strip().lower().split())


def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9']+", str(text or "").lower())


def _query_terms(question: str) -> List[str]:
    terms = []
    for term in _tokens(question):
        if len(term) <= 2:
            continue
        if term in STOPWORDS:
            continue
        terms.append(term)
    return terms


def _qid_parts(qid: str) -> Tuple[str, str]:
    conv, qnum = qid.rsplit("-q", 1)
    return conv, qnum


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _trace_path(run_dir: Path, qid: str) -> Path:
    conv, qnum = _qid_parts(qid)
    return run_dir / conv / f"research_trace_q{qnum}.json"


def _pages_path(run_dir: Path, qid: str) -> Path:
    conv, _ = _qid_parts(qid)
    return run_dir / conv / "pages.json"


def _collect_source_ids(trace: Dict[str, Any]) -> List[int]:
    out: List[int] = []

    def add(value: Any) -> None:
        if value is None:
            return
        if isinstance(value, (list, tuple)):
            for x in value:
                add(x)
            return
        try:
            out.append(int(str(value)))
        except Exception:
            return

    raw = trace.get("raw_memory") if isinstance(trace, dict) else {}
    if not isinstance(raw, dict):
        raw = {}
    add((raw.get("temp_memory") or {}).get("sources"))
    add((raw.get("date_anchor") or {}).get("sources"))
    add((raw.get("count_anchor") or {}).get("sources"))
    for it in raw.get("iterations") or []:
        if not isinstance(it, dict):
            continue
        add((it.get("temp_memory") or {}).get("sources"))
        plan = it.get("plan") or {}
        add(plan.get("page_index"))
    seen = set()
    uniq = []
    for x in out:
        if x < 0 or x in seen:
            continue
        seen.add(x)
        uniq.append(x)
    return uniq[:10]


def _sentence_windows(text: str, terms: Sequence[str], max_windows: int) -> List[str]:
    chunks = re.split(r"(?<=[.!?])\s+|\n{2,}", text)
    scored: List[Tuple[float, int, str]] = []
    term_set = set(terms)
    for idx, chunk in enumerate(chunks):
        clean = " ".join(chunk.split())
        if not clean:
            continue
        toks = _tokens(clean)
        if not toks:
            continue
        overlap = sum(1 for t in toks if t in term_set)
        score_value = float(overlap)
        # Keep dialogue turns with named speakers and dates slightly favored.
        if re.search(r"\bD\d+:\d+\b", clean):
            score_value += 0.5
        if re.search(
            r"\b(20\d{2}|january|february|march|april|may|june|july|august|"
            r"september|october|november|december)\b",
            clean.lower(),
        ):
            score_value += 0.25
        if score_value <= 0:
            continue
        start = max(0, idx - 1)
        end = min(len(chunks), idx + 2)
        window = " ".join(" ".join(chunks[j].split()) for j in range(start, end))
        scored.append((score_value, idx, window))
    scored.sort(key=lambda x: (-x[0], x[1]))
    windows: List[str] = []
    seen = set()
    for _, _, window in scored:
        key = window[:160]
        if key in seen:
            continue
        seen.add(key)
        windows.append(window)
        if len(windows) >= max_windows:
            break
    return windows


def _build_raw_block(
    pages: List[Dict[str, Any]],
    source_ids: Iterable[int],
    question: str,
    max_chars: int,
) -> str:
    terms = _query_terms(question)
    blocks: List[str] = []
    for pid in source_ids:
        if pid >= len(pages):
            continue
        page = pages[pid]
        header = str(page.get("header") or "")
        content = str(page.get("content") or "")
        windows = _sentence_windows(header + "\n" + content, terms, max_windows=2)
        if not windows:
            windows = [_clip(header + "\n" + content, 900)]
        text = "\n".join(f"- {w}" for w in windows)
        blocks.append(f"[page {pid}] {header}\n{text}")
    if not blocks:
        return ""
    return _clip("\n\n".join(blocks), max_chars)


def _prompt(
    *,
    question: str,
    category: Any,
    gam_answer: str,
    current_answer: str,
    elaa_answer: str,
    current_summary: str,
    raw_block: str,
    prompt_version: str,
) -> str:
    category_name = CATEGORY_NAMES.get(category, str(category))
    extra = ""
    if prompt_version == "v2":
        extra = """
Answer-style checks:
- If the raw evidence contains an exact phrase that directly answers the
  question, extract that phrase rather than a paraphrase.
- A supported nearby fact is not enough. The answer must fill the question
  slot: activity, title, type, reason, place, duration, count, quote, or list.
- For questions asking what someone said, prefer the quoted or near-quoted
  phrase from the raw evidence.
- For questions asking how someone felt or described something, prefer the
  adjective or phrase used in the evidence.
- For list questions, include only the requested items and do not add
  neighboring items.
- For yes/no questions, answer only Yes or No.
"""
    return f"""You are a raw-evidence answer localizer for LoCoMo memory QA.

Use the raw evidence snippets as the strongest source. Candidate answers can
help identify the right area, but the final answer should be the shortest
complete phrase supported by the raw snippets.

QUESTION CATEGORY:
{category_name}

QUESTION:
{question}

CANDIDATE ANSWERS:
- GAM: {gam_answer}
- CURRENT: {current_answer}
- ELAA: {elaa_answer}

CURRENT MEMORY SUMMARY:
{_clip(current_summary, 1500)}

RAW EVIDENCE SNIPPETS:
{raw_block}

Rules:
- Output only the short answer text in final_answer.
- Prefer exact wording from RAW EVIDENCE SNIPPETS whenever possible.
- Do not choose a candidate answer if the raw snippets support a better exact
  phrase.
- Do not answer with an explanation unless the question asks why/how.
- If raw evidence is insufficient, choose the best supported candidate answer.
{extra}

Return JSON.
"""


def _fallback(question: str, elaa: Dict[str, Any], current: Dict[str, Any]) -> Dict[str, Any]:
    answer = _answer(elaa) or _answer(current)
    return {
        "final_answer": _postprocess_answer(question, answer),
        "evidence_quote": "",
        "chosen_source": "elaa_answer" if _answer(elaa) else "current_answer",
        "answer_type": "other",
        "confidence": "low",
        "reason": "fallback after raw localizer failure",
    }


def _run_one(
    qid: str,
    gam: Dict[str, Any],
    current: Dict[str, Any],
    elaa: Dict[str, Any],
    current_run_dir: Path,
    generator: CachedOpenAIGenerator,
    max_raw_chars: int,
    prompt_version: str,
) -> Dict[str, Any]:
    question = str(current.get("question") or gam.get("question") or "")
    try:
        trace_file = _trace_path(current_run_dir, qid)
        pages_file = _pages_path(current_run_dir, qid)
        trace = _load_json(trace_file) if trace_file.exists() else {}
        pages = _load_json(pages_file) if pages_file.exists() else []
        source_ids = _collect_source_ids(trace)
        raw_block = _build_raw_block(pages, source_ids, question, max_raw_chars)
        prompt = _prompt(
            question=question,
            category=current.get("category", gam.get("category")),
            gam_answer=_answer(gam),
            current_answer=_answer(current),
            elaa_answer=_answer(elaa),
            current_summary=str(current.get("research_summary") or ""),
            raw_block=raw_block,
            prompt_version=prompt_version,
        )
        response = generator.generate_single(prompt=prompt, schema=RAW_SCHEMA)
        data = response.get("json")
        if not isinstance(data, dict):
            data = _fallback(question, elaa, current)
    except Exception as exc:
        data = _fallback(question, elaa, current)
        data["reason"] = f"raw localizer failed: {type(exc).__name__}: {exc}"
        source_ids = []
        raw_block = ""

    final = str(data.get("final_answer") or "").strip()
    if not final:
        final = _answer(elaa) or _answer(current)
    final = _postprocess_answer(question, final)
    data["final_answer_postprocessed"] = final

    row = dict(elaa)
    row["summary_answer"] = final
    row["pred"] = final
    row["raw_localizer"] = {
        "qid": qid,
        "source_ids": source_ids,
        "raw_chars": len(raw_block),
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
    parser.add_argument("--tag", default="locomo-full-raw-localizer-v1")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--max-raw-chars", type=int, default=4500)
    parser.add_argument("--prompt-version", choices=["v1", "v2"], default="v2")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--exclude-prefix", nargs="*", default=["conv-26"])
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    gam_rows = _load_rows(_resolve_path(args.gam_run) / "all_qa_results.json")
    current_run_dir = _resolve_path(args.current_run)
    current_rows = _load_rows(current_run_dir / "all_qa_results.json")
    elaa_rows = _load_rows(_resolve_path(args.elaa_run) / "all_qa_results.json")

    ids = sorted(set(gam_rows) & set(current_rows) & set(elaa_rows))
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
            "role": "raw_localizer",
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
                    current_run_dir,
                    generator,
                    args.max_raw_chars,
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
                current_run_dir,
                generator,
                args.max_raw_chars,
                args.prompt_version,
            )
            if (idx + 1) % 50 == 0 or idx + 1 == len(ids):
                print(f"[{idx + 1}/{len(ids)}] elapsed={round(time.time() - t0)}s", flush=True)

    all_qa = [r for r in rows if r is not None]
    metrics = score(all_qa, args.dataset)
    summary = {
        "tag": args.tag,
        "dataset": args.dataset,
        "method": "offline_raw_localizer",
        "gam_run": str(_resolve_path(args.gam_run)),
        "current_run": str(current_run_dir),
        "elaa_run": str(_resolve_path(args.elaa_run)),
        "model": args.model,
        "temperature": args.temperature,
        "workers": args.workers,
        "max_raw_chars": args.max_raw_chars,
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

