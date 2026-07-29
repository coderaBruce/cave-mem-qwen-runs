#!/usr/bin/env python
"""Re-answer a completed LoCoMo run with a different answer-style policy.

This keeps the existing research summaries fixed and only re-runs the final
answerer. It is intended for cheap Qwen tuning of answer-style policies before
spending time on a full CAVE rerun.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List


ROOT = Path(__file__).resolve().parents[2]
R2M_API = ROOT / "r2m-api"
UPSTREAM = ROOT / "R2M-official"
for p in (str(R2M_API), str(UPSTREAM), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from apiharness.answer_style import build_style_shots, make_calibrated_answerer
from apiharness.datasets import LOADERS
from apiharness.pipeline import make_generators
from apiharness.scoring import score


RESULTS = ROOT / "memory_directions" / "results"
CACHE = ROOT / "memory_directions" / ".cache"


def resolve_run(value: str) -> Path:
    path = Path(value)
    if path.exists():
        return path
    for root in (ROOT, RESULTS, R2M_API / "results"):
        path = root / value
        if path.exists():
            return path
    raise FileNotFoundError(value)


def load_rows(run_dir: Path) -> List[Dict[str, Any]]:
    rows = json.loads((run_dir / "all_qa_results.json").read_text(encoding="utf-8"))
    return [row for row in rows if isinstance(row, dict) and "error" not in row]


def make_answerer(args: argparse.Namespace) -> Callable:
    spec = LOADERS["locomo"]()
    answerer = spec.answerer
    shots: Dict[Any, List[Any]] = {}
    if args.answer_shots > 0:
        if not (args.shots_run and args.shots_samples):
            raise SystemExit("--answer-shots requires --shots-run and --shots-samples")
        shots_path = resolve_run(args.shots_run)
        shots = build_style_shots(
            shots_path,
            args.shots_samples,
            per_category=args.answer_shots,
            seed=args.seed,
        )
        if args.answer_shot_categories:
            keep = set(args.answer_shot_categories)
            shots = {cat: values for cat, values in shots.items() if cat in keep}
    print(
        "answer-style shots per category: "
        + json.dumps({str(k): len(v) for k, v in shots.items()})
    )

    if args.answer_style_policy == "plain":
        return answerer
    if args.answer_style_policy == "canonical":
        from memory_directions.answering import make_canonical_answerer

        return make_canonical_answerer(answerer)
    if args.answer_style_policy == "selective":
        from memory_directions.answering import make_selective_calibrated_answerer

        return make_selective_calibrated_answerer(answerer, shots)
    if args.answer_style_policy == "selective_open":
        from memory_directions.answering import make_selective_open_inference_answerer

        return make_selective_open_inference_answerer(answerer, shots)
    if args.answer_style_policy == "synthesize":
        from memory_directions.answering import make_synthesis_calibrated_answerer

        return make_synthesis_calibrated_answerer(answerer, shots)
    if args.answer_style_policy == "all":
        return make_calibrated_answerer(answerer, shots, fallback_pool=[])
    raise SystemExit(f"unknown answer_style_policy={args.answer_style_policy}")


def answer_one(answerer: Callable, generator: Any, row: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(row)
    question = str(row.get("question") or "")
    summary = str(row.get("research_summary") or "")
    category = row.get("category")
    try:
        answer = answerer(category, summary, question, generator, evidence=None)
    except TypeError:
        answer = answerer(category, summary, question, generator)
    out["summary_answer"] = answer
    out["pred"] = answer
    out["reanswer_source_answer"] = row.get("pred") or row.get("summary_answer")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument(
        "--answer-style-policy",
        choices=["plain", "canonical", "all", "selective", "selective_open", "synthesize"],
        default="plain",
    )
    parser.add_argument("--answer-shots", type=int, default=0)
    parser.add_argument("--answer-shot-categories", nargs="*", type=int, default=None)
    parser.add_argument("--shots-run", default=None)
    parser.add_argument("--shots-samples", nargs="*", default=None)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()

    source_run = resolve_run(args.source_run)
    rows = load_rows(source_run)
    rows.sort(key=lambda row: str(row.get("_id") or row.get("question") or ""))

    answerer = make_answerer(args)
    generators = make_generators(args.model, CACHE / "llm")
    generator = generators["working"]

    t0 = time.time()
    out_rows: List[Dict[str, Any]] = []
    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(answer_one, answerer, generator, row) for row in rows]
            for done, fut in enumerate(as_completed(futures), 1):
                out_rows.append(fut.result())
                if done % 100 == 0 or done == len(futures):
                    print(f"[{done}/{len(futures)}] elapsed={round(time.time() - t0)}s", flush=True)
    else:
        for row in rows:
            out_rows.append(answer_one(answerer, generator, row))
    out_rows.sort(key=lambda row: str(row.get("_id") or row.get("question") or ""))

    metrics = score(out_rows, "locomo")
    outdir = RESULTS / args.tag
    outdir.mkdir(parents=True, exist_ok=True)
    summary = {
        "tag": args.tag,
        "dataset": "locomo",
        "method": "reanswer_locomo",
        "source_run": str(source_run),
        "model": args.model,
        "answer_style_policy": args.answer_style_policy,
        "answer_shots": args.answer_shots,
        "answer_shot_categories": args.answer_shot_categories or [],
        "shots_run": args.shots_run,
        "shots_samples": args.shots_samples or [],
        "n_questions": len(out_rows),
        "generator_stats": generator.stats(),
        "metrics": metrics,
    }
    (outdir / "all_qa_results.json").write_text(
        json.dumps(out_rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (outdir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
