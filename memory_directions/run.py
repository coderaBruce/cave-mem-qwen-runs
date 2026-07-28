#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R2M_API = ROOT / "r2m-api"
UPSTREAM = ROOT / "R2M-official"
for p in (str(R2M_API), str(UPSTREAM), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import apiharness  # noqa: F401
from apiharness.datasets import LOADERS
from apiharness.pipeline import make_generators, run_sample
from apiharness.retrievers import EmbeddingClient
from apiharness.scoring import score

from memory_directions.agents import (
    AdaptiveContractMemoryAgent,
    ContractMemoryAgent,
    CausalValidatedDateCountAnchoredAddendumMemoryAgent,
    DateAnchoredAddendumMemoryAgent,
    DateCountAnchoredAddendumMemoryAgent,
    GuardedAddendumMemoryAgent,
    GuardedHybridMemoryAgent,
    HeuristicRoutedAddendumMemoryAgent,
    HybridMemoryAgent,
    HypothesisMemoryAgent,
    LLMRoutedAddendumMemoryAgent,
    RoutedAddendumMemoryAgent,
    TemporalRoutedAddendumMemoryAgent,
)

RESULTS = ROOT / "memory_directions" / "results"
CACHE = ROOT / "memory_directions" / ".cache"


def _factory_for(method: str, max_iters: int, base_k: int, wide_k: int):
    if method == "gam":
        return None
    cls = {
        "contract": ContractMemoryAgent,
        "hypothesis": HypothesisMemoryAgent,
        "adaptive": AdaptiveContractMemoryAgent,
        "hybrid": HybridMemoryAgent,
        "guarded_addendum": GuardedAddendumMemoryAgent,
        "guarded_hybrid": GuardedHybridMemoryAgent,
        "routed_addendum": RoutedAddendumMemoryAgent,
        "heuristic_routed_addendum": HeuristicRoutedAddendumMemoryAgent,
        "llm_routed_addendum": LLMRoutedAddendumMemoryAgent,
        "temporal_routed_addendum": TemporalRoutedAddendumMemoryAgent,
        "date_anchor_addendum": DateAnchoredAddendumMemoryAgent,
        "date_count_anchor_addendum": DateCountAnchoredAddendumMemoryAgent,
        "cave_date_count_anchor": CausalValidatedDateCountAnchoredAddendumMemoryAgent,
    }[method]

    def factory(page_store, memory_store, retrievers, generator, **_):
        return cls(
            page_store=page_store,
            memory_store=memory_store,
            retrievers=retrievers,
            generator=generator,
            max_iters=max_iters,
            base_k=base_k,
            wide_k=wide_k,
        )

    return factory


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=sorted(LOADERS))
    parser.add_argument("--split", default="eval_400")
    parser.add_argument(
        "--method",
        required=True,
        choices=[
            "gam",
            "contract",
            "hypothesis",
            "adaptive",
            "hybrid",
            "guarded_addendum",
            "guarded_hybrid",
            "routed_addendum",
            "heuristic_routed_addendum",
            "llm_routed_addendum",
            "temporal_routed_addendum",
            "date_anchor_addendum",
            "date_count_anchor_addendum",
            "cave_date_count_anchor",
        ],
    )
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--embed-model", default="text-embedding-3-small")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--max-iters", type=int, default=3)
    parser.add_argument("--base-k", type=int, default=5)
    parser.add_argument("--wide-k", type=int, default=12)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--limit-samples", type=int, default=None)
    parser.add_argument("--limit-questions", type=int, default=None)
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--answer-shots", type=int, default=0)
    parser.add_argument("--answer-shot-categories", nargs="*", type=int, default=None)
    parser.add_argument(
        "--answer-style-policy",
        choices=["all", "selective", "synthesize", "selective_open", "canonical"],
        default="all",
    )
    parser.add_argument("--shots-run", default=None)
    parser.add_argument("--shots-samples", nargs="*", default=None)
    parser.add_argument("--allow-shot-overlap", action="store_true")
    parser.add_argument("--evidence-pages", type=int, default=0)
    parser.add_argument("--evidence-chars", type=int, default=1200)
    parser.add_argument("--evidence-header", action="store_true")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    loader = LOADERS[args.dataset]
    spec = (
        loader(split=args.split, max_tokens=args.max_tokens)
        if args.dataset == "hotpotqa"
        else loader(max_tokens=args.max_tokens)
        if args.dataset == "narrativeqa"
        else loader()
    )

    samples = spec.samples
    if args.only:
        allowed = set(args.only)
        samples = [s for s in samples if s.sample_id in allowed]
    if args.dataset in {"hotpotqa", "narrativeqa"} and args.limit_samples:
        rng = random.Random(args.seed)
        samples = rng.sample(samples, min(args.limit_samples, len(samples)))
    elif args.limit_samples:
        samples = samples[: args.limit_samples]

    if args.limit_questions:
        for sample in samples:
            sample.qas = sample.qas[: args.limit_questions]

    if (
        args.answer_shots > 0
        or args.evidence_pages > 0
        or args.answer_style_policy == "canonical"
    ):
        from apiharness.answer_style import build_style_shots, make_calibrated_answerer

        if (
            args.answer_shots > 0
            and args.answer_style_policy != "canonical"
            and not (args.shots_run and args.shots_samples)
        ):
            print("--answer-shots requires --shots-run and --shots-samples")
            return 2
        shot_samples = args.shots_samples or []
        overlap = set(shot_samples) & {s.sample_id for s in samples}
        if overlap and not args.allow_shot_overlap:
            print(f"REFUSING: demonstration samples overlap the evaluation set: {overlap}")
            return 2
        shots_path = Path(args.shots_run) if args.shots_run else None
        if shots_path and not shots_path.exists():
            shots_path = RESULTS / args.shots_run
        shots = (
            build_style_shots(
                shots_path,
                shot_samples,
                per_category=args.answer_shots,
                seed=args.seed,
            )
            if args.answer_shots > 0
            else {}
        )
        if args.answer_shot_categories:
            keep = set(args.answer_shot_categories)
            shots = {cat: rows for cat, rows in shots.items() if cat in keep}
        print(
            "answer-style shots per category: "
            + json.dumps({str(k): len(v) for k, v in shots.items()})
        )
        base_fn = None
        if args.dataset == "hotpotqa":
            from eval.hotpotqa_test import make_prompt as base_fn
        elif args.dataset == "narrativeqa":
            from eval.narrativeqa_test import make_prompt as base_fn
        if args.answer_style_policy == "canonical":
            from memory_directions.answering import make_canonical_answerer

            spec.answerer = make_canonical_answerer(spec.answerer)
        elif args.answer_style_policy == "selective":
            from memory_directions.answering import make_selective_calibrated_answerer

            spec.answerer = make_selective_calibrated_answerer(
                spec.answerer, shots, base_prompt_fn=base_fn
            )
        elif args.answer_style_policy == "selective_open":
            from memory_directions.answering import make_selective_open_inference_answerer

            spec.answerer = make_selective_open_inference_answerer(
                spec.answerer, shots, base_prompt_fn=base_fn
            )
        elif args.answer_style_policy == "synthesize":
            from memory_directions.answering import make_synthesis_calibrated_answerer

            spec.answerer = make_synthesis_calibrated_answerer(
                spec.answerer, shots, base_prompt_fn=base_fn
            )
        else:
            spec.answerer = make_calibrated_answerer(
                spec.answerer, shots, fallback_pool=[], base_prompt_fn=base_fn
            )

    tag = args.tag or f"{args.dataset}-{args.method}-{args.model}"
    outdir = RESULTS / tag
    outdir.mkdir(parents=True, exist_ok=True)

    n_questions = sum(len(s.qas) for s in samples)
    print(
        f"dataset={spec.name} method={args.method} model={args.model} "
        f"samples={len(samples)} questions={n_questions} workers={args.workers}"
    )
    print(f"outdir={outdir}")

    if args.dry_run:
        from apiharness.datasets import materialise_chunks

        for sample in samples[:3]:
            chunks = sample.chunks or materialise_chunks(sample)
            print(f"{sample.sample_id}: chunks={len(chunks)} questions={len(sample.qas)}")
        return 0

    generators = make_generators(args.model, CACHE / "llm", args.temperature)
    if args.no_cache:
        for generator in generators.values():
            generator.enable_cache = False
    embedder = EmbeddingClient(args.embed_model, str(CACHE / "embed"))
    researcher_factory = _factory_for(args.method, args.max_iters, args.base_k, args.wide_k)

    per_sample_workers = args.workers if spec.multi_qa_per_sample else 1
    sample_workers = 1 if spec.multi_qa_per_sample else args.workers
    t0 = time.time()

    def do_sample(indexed_sample):
        idx, sample = indexed_sample
        try:
            kwargs = {}
            if researcher_factory is not None:
                kwargs["researcher_factory"] = researcher_factory
            result = run_sample(
                sample,
                spec,
                outdir,
                generators,
                embedder,
                rebuild=args.rebuild,
                max_workers=per_sample_workers,
                evidence_pages=args.evidence_pages,
                evidence_chars=args.evidence_chars,
                include_header=args.evidence_header,
                **kwargs,
            )
            return idx, sample, result, None
        except Exception as exc:
            return idx, sample, None, f"{type(exc).__name__}: {exc}"

    indexed = list(enumerate(samples, 1))
    if sample_workers > 1 and len(indexed) > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        rows = []
        with ThreadPoolExecutor(max_workers=sample_workers) as pool:
            futures = [pool.submit(do_sample, x) for x in indexed]
            for done, fut in enumerate(as_completed(futures), 1):
                rows.append(fut.result())
                if done % 5 == 0 or done == len(indexed):
                    print(f"[{done}/{len(indexed)}] elapsed={round(time.time()-t0)}s")
        rows.sort(key=lambda x: x[0])
    else:
        rows = []
        for x in indexed:
            row = do_sample(x)
            rows.append(row)
            print(f"[{len(rows)}/{len(indexed)}] {row[1].sample_id} elapsed={round(time.time()-t0)}s")

    all_qa = []
    all_stats = []
    for _, sample, result, err in rows:
        if err:
            print(f"FAILED {sample.sample_id}: {err}", file=sys.stderr)
            all_stats.append({"sample_id": sample.sample_id, "fatal": err})
            continue
        all_qa.extend(result["qa_results"])
        all_stats.append(result["stats"])

    metrics = score([r for r in all_qa if "error" not in r], spec.name)
    summary = {
        "tag": tag,
        "dataset": spec.name,
        "method": args.method,
        "model": args.model,
        "temperature": args.temperature,
        "workers": args.workers,
        "max_iters": args.max_iters,
        "base_k": args.base_k,
        "wide_k": args.wide_k,
        "answer_shots": args.answer_shots,
        "answer_shot_categories": args.answer_shot_categories or [],
        "answer_style_policy": args.answer_style_policy,
        "shots_run": args.shots_run,
        "shots_samples": args.shots_samples or [],
        "evidence_pages": args.evidence_pages,
        "evidence_chars": args.evidence_chars,
        "evidence_header": args.evidence_header,
        "n_samples": len(samples),
        "n_questions": len(all_qa),
        "n_errors": sum(1 for r in all_qa if "error" in r),
        "wall_secs": round(time.time() - t0, 1),
        "metrics": metrics,
        "generator_stats": {k: g.stats() for k, g in generators.items()},
        "embed_calls": embedder.n_embedded,
        "per_sample": all_stats,
    }
    (outdir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (outdir / "all_qa_results.json").write_text(
        json.dumps(all_qa, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n=== metrics ===")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"summary -> {outdir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
