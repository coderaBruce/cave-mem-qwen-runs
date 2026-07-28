#!/usr/bin/env python
"""R²-Mem offline stage: Evaluator -> Learner -> experience bank.

Consumes a GAM baseline run produced by `run.py` and emits an experience bank.

Protocol follows the paper (App. B.1): LoCoMo accumulates from **Conv-26 only**
(chosen upstream because its category mix is close to the dataset average) and
evaluates on the remaining nine; HotpotQA and NarrativeQA accumulate from 20%
of the pool.

Examples
--------
    python build_bank.py --run gam-locomo-4omini --source conv-26 --out banks/locomo-faithful.json
    python build_bank.py --run gam-locomo-4omini --source conv-26 --keep-middle \
        --out banks/locomo-keepmiddle.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import apiharness  # noqa: F401
from apiharness.cached_generator import CachedOpenAIGenerator
from apiharness.env import get, openai_base_url, openai_key
from apiharness.r2mem import ExperienceBank, build_bank_from_traces
from apiharness.retrievers import EmbeddingClient

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
CACHE = ROOT / ".cache"


def collect_records(run_dir: Path, sources: list[str], limit: int | None):
    """Gather {question, gold, pred, trace} from a finished baseline run."""
    records = []
    for src in sources:
        qa_file = run_dir / src / "qa_results.json"
        if not qa_file.exists():
            print(f"  ! {qa_file} missing, skipping")
            continue
        for item in json.loads(qa_file.read_text(encoding="utf-8")):
            if "error" in item:
                continue
            trace_path = item.get("research_trace_file")
            if not trace_path or not Path(trace_path).exists():
                continue
            trace = json.loads(Path(trace_path).read_text(encoding="utf-8"))
            gold = item.get("gold_answer")
            if isinstance(gold, list):
                gold = "; ".join(str(g) for g in gold)
            records.append(
                {
                    "question": item.get("question", ""),
                    "gold": gold,
                    "pred": item.get("summary_answer") or item.get("pred") or "",
                    "trace": trace,
                    "source": src,
                }
            )
    if limit:
        records = records[:limit]
    return records


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="Tag of a finished run under results/")
    p.add_argument("--source", nargs="+", required=True,
                   help="Sample ids to accumulate from, e.g. conv-26")
    p.add_argument("--out", required=True, help="Output bank JSON path")
    p.add_argument("--evaluator-model", default="gpt-4o",
                   help="Upstream uses gpt-4o. Use the backbone for the Self-Evo variant.")
    p.add_argument("--learner-model", default="gpt-4o-mini")
    p.add_argument("--embed-model", default="text-embedding-3-small")
    p.add_argument("--k-low", type=int, default=5)
    p.add_argument("--k-high", type=int, default=10)
    p.add_argument("--keep-middle", action="store_true",
                   help="OUR VARIANT: also distil mid-scoring steps, which upstream discards.")
    p.add_argument("--limit", type=int, default=None, help="Cap traces (for cheap tests).")
    args = p.parse_args()

    run_dir = RESULTS / args.run
    if not run_dir.exists():
        print(f"No such run: {run_dir}")
        return 1

    records = collect_records(run_dir, args.source, args.limit)
    print(f"collected {len(records)} traces from {args.source}")
    if not records:
        return 1

    embedder = EmbeddingClient(args.embed_model, str(CACHE / "embed"))
    learner = CachedOpenAIGenerator(
        {
            "model_name": args.learner_model,
            "cache_dir": str(CACHE / "llm"),
            "temperature": 0.5,      # upstream self_reflection.py uses 0.5
            "max_tokens": 1024,
            "role": "learner",
        }
    )
    bank = ExperienceBank(embedder)

    t0 = time.time()

    def progress(done, total, st):
        if done % 5 == 0 or done == total:
            print(
                f"  [{done}/{total}] good={st['n_good']} bad={st['n_bad']} "
                f"mid_skipped={st['n_middle_skipped']} "
                f"eval_tok={st['evaluator_tokens']:,} "
                f"elapsed={round(time.time()-t0)}s",
                flush=True,
            )

    stats = build_bank_from_traces(
        records,
        bank,
        learner,
        evaluator_api_key=openai_key(),
        evaluator_base_url=openai_base_url(),
        k_low=args.k_low,
        k_high=args.k_high,
        keep_middle=args.keep_middle,
        on_progress=progress,
    )

    out = Path(args.out)
    bank.save(out)
    meta = {
        "run": args.run,
        "source": args.source,
        "evaluator_model": args.evaluator_model,
        "learner_model": args.learner_model,
        "k_low": args.k_low,
        "k_high": args.k_high,
        "keep_middle": args.keep_middle,
        "bank_stats": bank.stats(),
        "build_stats": stats,
        "wall_secs": round(time.time() - t0, 1),
    }
    out.with_suffix(".meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\nbank -> {out}  ({len(bank)} entries)")
    print(json.dumps(bank.stats(), indent=2))
    print("score histogram:", json.dumps(stats["score_hist"], sort_keys=True))
    print(f"evaluator tokens {stats['evaluator_tokens']:,} | "
          f"learner tokens {stats['learner_tokens']:,}")
    if stats["errors"]:
        print(f"errors: {len(stats['errors'])} (first: {stats['errors'][0]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
