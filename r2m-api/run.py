#!/usr/bin/env python
"""CLI for the API-only GAM / R²-Mem harness.

Examples
--------
    # smoke test: one LoCoMo conversation, 5 questions
    python run.py --dataset locomo --method gam --limit-samples 1 --limit-questions 5

    # GAM baseline on HotpotQA 56K, matching the paper's 128-question pool
    python run.py --dataset hotpotqa --split eval_400 --method gam

    # NarrativeQA, 300-question subset with the paper's seed
    python run.py --dataset narrativeqa --method gam --limit-samples 300 --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import apiharness  # noqa: F401  (puts R2M-official on sys.path)
from apiharness.datasets import LOADERS
from apiharness.cached_generator import CachedOpenAIGenerator
from apiharness.pipeline import make_generators, run_sample
from apiharness.retrievers import EmbeddingClient
from apiharness.scoring import score

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
CACHE = ROOT / ".cache"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=sorted(LOADERS))
    p.add_argument("--split", default="eval_400",
                   help="HotpotQA split: eval_400 (56K) | eval_1600 (224K) | eval_3200 (448K)")
    p.add_argument("--method", default="gam", choices=["gam", "r2mem", "additive", "rrf", "wide", "routed"],
                   help="gam = baseline; r2mem = their templates; "
                        "additive = OURS, GAM prompts + appended experience; "
                        "rrf = OURS, rank-based hybrid retrieval fusion.")
    p.add_argument("--bank", default=None, help="Experience bank JSON (--method r2mem).")
    p.add_argument("--top-k", type=int, default=3, help="Experiences retrieved per step.")
    p.add_argument("--top-pages", type=int, default=0,
                   help="Truncate fused hits before _integrate. 0 = no truncation, "
                        "matching GAM (which passes all deduped hits).")
    p.add_argument("--rrf-k", type=int, default=60)
    p.add_argument("--boundary-chunks", action="store_true",
                   help="OURS: pack whole documents/paragraphs into pages instead of "
                        "GAM's fixed 2048-token slicing, which severs 23-25 of 26 "
                        "HotpotQA pages mid-document.")
    p.add_argument("--index-header", default="none",
                   choices=["none", "dense", "bm25", "both"],
                   help="OURS: index page.header alongside content, per retriever. GAM "
                        "computes the header then discards it in BOTH retrievers.")
    p.add_argument("--mbr", type=int, default=0,
                   help="OURS: MBR/consensus decoding of the answer step with this many "
                        "samples (0 = off, greedy single sample as in GAM).")
    p.add_argument("--mbr-temp", type=float, default=0.7)
    p.add_argument("--allow-shot-overlap", action="store_true",
                   help="Permit shot source == eval set. ONLY for labelling the "
                        "accumulation split when training the router; the shot "
                        "questions must then be excluded from the labelled set.")
    p.add_argument("--profile-run", default=None,
                   help="--method routed: baseline run to profile the corpus from.")
    p.add_argument("--profile-samples", nargs="*", default=None,
                   help="--method routed: accumulation samples for profiling.")
    p.add_argument("--per-retriever-k", type=int, default=5,
                   help="Hits per retriever. GAM ships 5; its Fig.2 sweeps 3-20.")
    p.add_argument("--evidence-pages", type=int, default=0,
                   help="OURS: pass this many cited source pages to the answer step. "
                        "GAM records source ids but never passes their text.")
    p.add_argument("--evidence-header", action="store_true",
                   help="Include each page's Memorizer abstract (has absolute dates).")
    p.add_argument("--evidence-chars", type=int, default=1200,
                   help="Truncate each source page to this many characters.")
    p.add_argument("--answer-shots", type=int, default=0,
                   help="OURS: few-shot answer-style demonstrations drawn from "
                        "--shots-run/--shots-samples (the accumulation split).")
    p.add_argument("--shots-run", default=None, help="Run tag to draw demonstrations from.")
    p.add_argument("--shots-samples", nargs="*", default=None,
                   help="Accumulation samples for demonstrations (must be held out).")
    p.add_argument("--max-iters", type=int, default=3,
                   help="Reflection depth cap. GAM's default is 3; its Fig.2 shows F1 "
                        "rising monotonically to 5.")
    p.add_argument("--min-sim", type=float, default=None,
                   help="OURS: inject only experiences above this cosine similarity. "
                        "Below it the agent falls back to plain GAM behaviour. "
                        "R2-Mem injects Top-K unconditionally (no threshold).")
    p.add_argument("--model", default="gpt-4o-mini",
                   help="Backbone. gpt-4o-mini reproduces GAM's own reported setting.")
    p.add_argument("--embed-model", default="text-embedding-3-small")
    p.add_argument("--temperature", type=float, default=0.3,
                   help="Upstream uses 0.3. Set 0.0 for a deterministic variant.")
    p.add_argument("--max-tokens", type=int, default=2048, help="Page size in tokens.")
    p.add_argument("--limit-samples", type=int, default=None)
    p.add_argument("--only", nargs="*", default=None,
                   help="Restrict to these sample ids (e.g. the held-out conversations).")
    p.add_argument("--limit-questions", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tag", default=None, help="Run name; defaults to dataset-method-model.")
    p.add_argument("--workers", type=int, default=8,
                   help="Parallel questions per sample. Upstream is serial (1).")
    p.add_argument("--rebuild", action="store_true", help="Rebuild memory even if cached.")
    p.add_argument("--no-cache", action="store_true", help="Bypass the LLM response cache.")
    p.add_argument("--dry-run", action="store_true",
                   help="Load and chunk data, report sizes, make no API calls.")
    args = p.parse_args()

    loader = LOADERS[args.dataset]
    spec = (
        loader(split=args.split, max_tokens=args.max_tokens)
        if args.dataset == "hotpotqa"
        else loader(max_tokens=args.max_tokens)
        if args.dataset == "narrativeqa"
        else loader()
    )

    if args.boundary_chunks:
        from apiharness.datasets import use_boundary_chunkers

        spec = use_boundary_chunkers(spec)
        print("boundary-respecting chunking enabled")

    samples = spec.samples
    if args.only:
        samples = [s for s in samples if s.sample_id in set(args.only)]
    if args.dataset in ("hotpotqa", "narrativeqa") and args.limit_samples:
        # Match the papers: random subset under a fixed seed, not a prefix.
        rng = random.Random(args.seed)
        samples = rng.sample(samples, min(args.limit_samples, len(samples)))
    elif args.limit_samples:
        samples = samples[: args.limit_samples]

    if args.limit_questions:
        for s in samples:
            s.qas = s.qas[: args.limit_questions]

    tag = args.tag or f"{spec.name}-{args.method}-{args.model}"
    outdir = RESULTS / tag
    outdir.mkdir(parents=True, exist_ok=True)

    n_q = sum(len(s.qas) for s in samples)
    print(f"dataset={spec.name}  method={args.method}  model={args.model}")
    print(f"samples={len(samples)}  questions={n_q}  outdir={outdir}")

    if args.dry_run:
        from apiharness.datasets import materialise_chunks

        probe = samples[: min(3, len(samples))]
        for s in probe:
            chunks = materialise_chunks(s) if not s.chunks else s.chunks
            print(f"  {s.sample_id}: {len(chunks)} chunks, {len(s.qas)} questions")
        print("dry run: no API calls made")
        return 0

    researcher_factory = None
    if args.method == "routed":
        from apiharness.router import (ANSWER_STYLE, EVIDENCE, WIDEN,
                                       BottleneckRouter, profile_corpus)

        if not (args.profile_run and args.profile_samples):
            print("--method routed requires --profile-run and --profile-samples")
            return 2
        overlap = set(args.profile_samples) & {s.sample_id for s in samples}
        if overlap:
            print(f"REFUSING: profiling samples overlap evaluation set: {overlap}")
            return 2
        prof = profile_corpus(RESULTS / args.profile_run, args.profile_samples)
        decision = BottleneckRouter(prof).route()
        print(f"corpus profile: {json.dumps(prof.as_dict())}")
        print(f"ROUTE -> {decision.strategy}  ({decision.reason})")

        if decision.strategy == WIDEN:
            args.method, args.per_retriever_k = "wide", decision.width
        elif decision.strategy == EVIDENCE:
            args.method, args.evidence_pages = "gam", 4
            args.evidence_header = True
        elif decision.strategy == ANSWER_STYLE:
            args.method, args.answer_shots = "gam", 8
            args.shots_run = args.shots_run or args.profile_run
            args.shots_samples = args.shots_samples or args.profile_samples
        routed_strategy = decision.strategy
    else:
        routed_strategy = None

    if args.method == "wide":
        from apiharness.wide import WideResearchAgent

        def researcher_factory(page_store, memory_store, retrievers, generator, **_):
            return WideResearchAgent(
                page_store=page_store, memory_store=memory_store,
                retrievers=retrievers, generator=generator,
                max_iters=args.max_iters, per_retriever_k=args.per_retriever_k)

    if args.method == "rrf":
        from apiharness.rrf import RRFResearchAgent

        def researcher_factory(page_store, memory_store, retrievers, generator, **_):
            return RRFResearchAgent(
                page_store=page_store, memory_store=memory_store,
                retrievers=retrievers, generator=generator,
                max_iters=args.max_iters, rrf_k=args.rrf_k,
                top_pages=args.top_pages,
                per_retriever_k=max(5, args.top_pages),
            )

    if args.method == "gam" and args.max_iters != 3:
        from apiharness.pipeline import ResearchAgent as _RA

        def researcher_factory(page_store, memory_store, retrievers, generator, **_):
            return _RA(page_store=page_store, memory_store=memory_store,
                       retrievers=retrievers, generator=generator,
                       max_iters=args.max_iters)

    if args.method in ("r2mem", "additive"):
        if not args.bank:
            print(f"--method {args.method} requires --bank")
            return 2
        from apiharness.r2mem import ExperienceBank, ResearchAgentExp
        from apiharness.additive import ResearchAgentAdditive

        AgentCls = ResearchAgentAdditive if args.method == "additive" else ResearchAgentExp

        bank_embedder = EmbeddingClient(args.embed_model, str(CACHE / "embed"))
        bank = ExperienceBank.load(Path(args.bank), bank_embedder)
        print(f"experience bank: {len(bank)} entries {json.dumps(bank.stats())}")

        select = None
        if args.min_sim is not None:
            def select(hits, module, condition):
                return [(e, s) for e, s in hits if s >= args.min_sim]

        def researcher_factory(page_store, memory_store, retrievers, generator, **_):
            return AgentCls(
                page_store=page_store,
                memory_store=memory_store,
                retrievers=retrievers,
                generator=generator,
                max_iters=args.max_iters,
                bank=bank,
                top_k=args.top_k,
                select=select,
            )

    if args.answer_shots > 0 or args.evidence_pages > 0:
        from apiharness.answer_style import build_style_shots, make_calibrated_answerer

        if args.answer_shots > 0 and not (args.shots_run and args.shots_samples):
            print("--answer-shots requires --shots-run and --shots-samples")
            return 2
        args.shots_samples = args.shots_samples or []
        overlap = set(args.shots_samples) & {s.sample_id for s in samples}
        if overlap and not args.allow_shot_overlap:
            print(f"REFUSING: demonstration samples overlap the evaluation set: {overlap}")
            return 2
        if overlap:
            print(f"WARNING: shot source overlaps eval ({overlap}) -- labelling mode")
        shots = (build_style_shots(RESULTS / args.shots_run, args.shots_samples,
                                   per_category=args.answer_shots, seed=args.seed)
                 if args.answer_shots > 0 else {})
        print("answer-style shots per category: "
              + json.dumps({str(k): len(v) for k, v in shots.items()}))
        base_fn = None
        if args.dataset == "hotpotqa":
            from eval.hotpotqa_test import make_prompt as base_fn
        elif args.dataset == "narrativeqa":
            from eval.narrativeqa_test import make_prompt as base_fn
        spec.answerer = make_calibrated_answerer(spec.answerer, shots,
                                                 base_prompt_fn=base_fn)

    generators = make_generators(args.model, CACHE / "llm", args.temperature)
    if args.mbr > 0:
        from apiharness.mbr import make_mbr_answerer

        sampler = CachedOpenAIGenerator({
            "model_name": args.model, "cache_dir": str(CACHE / "llm"),
            "temperature": args.mbr_temp, "max_tokens": 256, "role": "mbr"})
        generators["mbr"] = sampler
        spec.answerer = make_mbr_answerer(spec.answerer, n_samples=args.mbr,
                                          temperature=args.mbr_temp,
                                          sampler_generator=sampler)
        print(f"MBR decoding: {args.mbr} samples @ T={args.mbr_temp}")
    if args.no_cache:
        for g in generators.values():
            g.enable_cache = False
    embedder = EmbeddingClient(args.embed_model, str(CACHE / "embed"))

    all_qa, all_stats = [], []
    t0 = time.time()

    # LoCoMo samples hold many questions each, so parallelism belongs inside a
    # sample. HotpotQA/NarrativeQA samples hold exactly one question, so
    # within-sample parallelism does nothing and the work must be spread across
    # samples instead — otherwise those runs are fully serial.
    per_sample_workers = args.workers if spec.multi_qa_per_sample else 1
    sample_workers = 1 if spec.multi_qa_per_sample else args.workers

    def do_sample(idx_s):
        i, s = idx_s
        kw = {"rebuild": args.rebuild, "max_workers": per_sample_workers,
              "evidence_pages": args.evidence_pages,
              "evidence_chars": args.evidence_chars if args.evidence_pages else 0,
              "include_header": args.evidence_header,
              "index_header": args.index_header}
        if researcher_factory is not None:
            kw["researcher_factory"] = researcher_factory
        try:
            return i, s, run_sample(s, spec, outdir, generators, embedder, **kw), None
        except Exception as exc:
            return i, s, None, f"{type(exc).__name__}: {exc}"

    indexed = list(enumerate(samples, 1))
    done = 0
    if sample_workers > 1 and len(indexed) > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=sample_workers) as pool:
            futs = [pool.submit(do_sample, x) for x in indexed]
            results = []
            for fut in as_completed(futs):
                results.append(fut.result())
                done += 1
                if done % 10 == 0 or done == len(indexed):
                    print(f"  [{done}/{len(indexed)}] elapsed={round(time.time()-t0)}s",
                          flush=True, file=sys.stderr)
        results.sort(key=lambda r: r[0])
    else:
        results = []
        for x in indexed:
            r = do_sample(x)
            results.append(r)
            done += 1
            print(f"[{done}/{len(indexed)}] {r[1].sample_id} "
                  f"elapsed={round(time.time()-t0)}s", flush=True, file=sys.stderr)

    for i, s, res, err in results:
        if err is not None:
            print(f"  FAILED {s.sample_id}: {err}", file=sys.stderr)
            all_stats.append({"sample_id": s.sample_id, "fatal": err})
            continue
        all_qa.extend(res["qa_results"])
        all_stats.append(res["stats"])

    metrics = score(all_qa, spec.name)
    summary = {
        "tag": tag,
        "dataset": spec.name,
        "method": args.method,
        "model": args.model,
        "temperature": args.temperature,
        "workers": args.workers,
        "bank": args.bank,
        "top_k": args.top_k,
        "max_iters": args.max_iters,
        "answer_shots": args.answer_shots,
        "mbr": args.mbr,
        "index_header": args.index_header,
        "boundary_chunks": args.boundary_chunks,
        "top_pages": args.top_pages,
        "rrf_k": args.rrf_k,
        "per_retriever_k": args.per_retriever_k,
        "routed_strategy": routed_strategy,
        "evidence_pages": args.evidence_pages,
        "evidence_header": args.evidence_header,
        "min_sim": args.min_sim,
        "embed_model": args.embed_model,
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
    print(json.dumps(metrics.get("by_category", metrics.get("overall")), indent=2))
    if "overall" in metrics:
        print("overall:", json.dumps(metrics["overall"]))
    live = sum(g.stats()["live_tokens"] for g in generators.values())
    print(f"live tokens billed this run: {live:,}")
    print(f"summary -> {outdir/'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
