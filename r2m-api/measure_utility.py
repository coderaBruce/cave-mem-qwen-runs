#!/usr/bin/env python
"""Measure per-experience utility by A/B replay, then emit a pruned bank.

    python measure_utility.py --bank banks/locomo-faithful.json \
        --run gam-locomo-4omini --probe-samples conv-26 --limit 60 \
        --out banks/locomo-utility.json

See `apiharness/utility.py` for why this exists and how the estimator works.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import apiharness  # noqa: F401
from apiharness.cached_generator import CachedOpenAIGenerator
from apiharness.r2mem import ExperienceBank
from apiharness.retrievers import EmbeddingClient
from apiharness.utility import collect_probe_steps, measure_utility, prune_by_utility

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
CACHE = ROOT / ".cache"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bank", required=True)
    p.add_argument("--run", required=True, help="Baseline run to draw probe steps from.")
    p.add_argument("--probe-samples", nargs="+", required=True)
    p.add_argument("--limit", type=int, default=60, help="Probe steps to replay.")
    p.add_argument("--out", required=True, help="Where to write the annotated bank.")
    p.add_argument("--pruned-out", default=None,
                   help="Also write a bank with non-positive-utility entries removed.")
    p.add_argument("--model", default="gpt-4o-mini", help="Planner used for the replay.")
    p.add_argument("--judge-model", default="gpt-4o-mini",
                   help="Pairwise judge. Deliberately NOT gpt-4o: part of the claim is "
                        "that utility can be measured without a frontier judge.")
    p.add_argument("--embed-model", default="text-embedding-3-small")
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    embedder = EmbeddingClient(args.embed_model, str(CACHE / "embed"))
    bank = ExperienceBank.load(Path(args.bank), embedder)
    print(f"bank: {len(bank)} entries {json.dumps(bank.stats())}")

    probes = collect_probe_steps(RESULTS / args.run, args.probe_samples, args.limit)
    print(f"probe steps: {len(probes)} from {args.probe_samples}")
    if not probes:
        return 1

    planner = CachedOpenAIGenerator(
        {"model_name": args.model, "cache_dir": str(CACHE / "llm"),
         "temperature": 0.3, "max_tokens": 2048, "role": "utility-planner"}
    )
    judge = CachedOpenAIGenerator(
        {"model_name": args.judge_model, "cache_dir": str(CACHE / "llm"),
         "temperature": 0.0, "max_tokens": 512, "role": "utility-judge"}
    )

    t0 = time.time()

    def progress(n, total, outcomes):
        if n % 10 == 0 or n == total:
            tot = max(1, sum(outcomes.values()))
            print(f"  [{n}/{total}] with={outcomes['with']} without={outcomes['without']} "
                  f"tie={outcomes['tie']}  winrate={outcomes['with']/tot:.3f} "
                  f"elapsed={round(time.time()-t0)}s", flush=True)

    stats = measure_utility(
        bank, probes, planner, judge,
        top_k=args.top_k, seed=args.seed, on_progress=progress,
    )

    bank.save(Path(args.out))
    Path(args.out).with_suffix(".meta.json").write_text(
        json.dumps({"source_bank": args.bank, "stats": stats,
                    "judge_model": args.judge_model, "n_probes": len(probes)},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n=== utility summary ===")
    print(json.dumps(stats, indent=2))

    utils = [e.extra["utility"] for e in bank.entries
             if e.extra.get("n_trials", 0) > 0 and e.extra.get("utility") is not None]
    if utils:
        utils_sorted = sorted(utils)
        print(f"utility: min={min(utils):.3f} median={utils_sorted[len(utils)//2]:.3f} "
              f"max={max(utils):.3f} mean={sum(utils)/len(utils):.3f}")

    if args.pruned_out:
        pruned = prune_by_utility(bank, 0.0, keep_unmeasured=False)
        pruned.save(Path(args.pruned_out))
        print(f"pruned bank: {len(pruned)}/{len(bank)} entries -> {args.pruned_out}")

    print(f"annotated bank -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
