#!/usr/bin/env python
"""Vanilla RAG baseline for the API-only GAM / R2-Mem harness.

This implements the memory-free baseline described in R2-Mem: split the input
into 2048-token chunks, retrieve the top-5 chunks with embedding similarity,
concatenate them, and answer in one generation step. It intentionally does not
run GAM's Memorizer/Researcher loop and does not use any experience bank.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

import apiharness  # noqa: F401  (puts R2M-official on sys.path)
from apiharness.cached_generator import CachedOpenAIGenerator
from apiharness.datasets import LOADERS, DatasetSpec, QAItem, Sample, materialise_chunks
from apiharness.retrievers import EmbeddingClient, _tokenize
from apiharness.scoring import score

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
CACHE = ROOT / ".cache"


def split_fixed_tokens(text: str, max_tokens: int) -> List[str]:
    """Split arbitrary text into fixed token windows."""
    if not text.strip():
        return []
    try:
        import tiktoken

        enc = tiktoken.encoding_for_model("gpt-4o-2024-08-06")
        toks = enc.encode(text, disallowed_special=())
        if len(toks) <= max_tokens:
            return [text]
        chunks: List[str] = []
        for start in range(0, len(toks), max_tokens):
            piece = enc.decode(toks[start : start + max_tokens]).strip()
            if piece:
                chunks.append(piece)
        return chunks
    except Exception:
        width = max(1, max_tokens * 4)
        return [text[i : i + width].strip() for i in range(0, len(text), width) if text[i : i + width].strip()]


def rag_chunks(sample: Sample, spec: DatasetSpec, max_tokens: int) -> List[str]:
    """Return RAG chunks without memorization.

    HotpotQA and NarrativeQA already expose fixed token chunks via the upstream
    chunkers. LoCoMo naturally exposes one chunk per dialogue session, which is
    a memory-specific organization; for the memory-free baseline we concatenate
    those sessions and re-split into fixed token windows.
    """
    chunks = materialise_chunks(sample) if not sample.chunks else sample.chunks
    if spec.name == "locomo":
        joined = "\n\n".join(chunks)
        return [
            f"[Chunk {i + 1}]\n{chunk}"
            for i, chunk in enumerate(split_fixed_tokens(joined, max_tokens))
        ]
    return [f"[Chunk {i + 1}]\n{chunk}" for i, chunk in enumerate(chunks)]


def bm25_scores(query: str, docs: Sequence[str]) -> np.ndarray:
    """Small Okapi BM25 scorer using the same tokenizer as the API harness."""
    tokenized = [_tokenize(d) for d in docs]
    n = len(tokenized)
    if n == 0:
        return np.zeros(0, dtype="float32")
    q_terms = set(_tokenize(query))
    if not q_terms:
        return np.zeros(n, dtype="float32")
    avgdl = sum(len(d) for d in tokenized) / n if n else 1.0
    dfs: Dict[str, int] = {}
    tfs: List[Dict[str, int]] = []
    for doc in tokenized:
        counts: Dict[str, int] = {}
        for term in doc:
            counts[term] = counts.get(term, 0) + 1
        tfs.append(counts)
        for term in counts:
            dfs[term] = dfs.get(term, 0) + 1
    k1, b = 0.9, 0.4
    scores = np.zeros(n, dtype="float32")
    for term in q_terms:
        df = dfs.get(term, 0)
        idf = np.log(1.0 + (n - df + 0.5) / (df + 0.5))
        for i, counts in enumerate(tfs):
            freq = counts.get(term, 0)
            if not freq:
                continue
            dl = len(tokenized[i]) or 1
            denom = freq + k1 * (1 - b + b * dl / (avgdl or 1.0))
            scores[i] += idf * (freq * (k1 + 1)) / denom
    return scores


def retrieve(
    query: str,
    chunks: Sequence[str],
    embedder: EmbeddingClient,
    top_k: int,
    mode: str,
    matrix: np.ndarray | None,
) -> List[Tuple[int, float]]:
    if not chunks:
        return []
    k = min(top_k, len(chunks))
    if mode == "bm25":
        scores = bm25_scores(query, chunks)
    else:
        if matrix is None:
            raise ValueError("dense retrieval requires a precomputed matrix")
        q = embedder.embed([query])
        dense_scores = (q @ matrix.T).reshape(-1)
        if mode == "dense":
            scores = dense_scores
        elif mode == "hybrid":
            sparse = bm25_scores(query, chunks)
            if float(np.max(sparse, initial=0.0)) > 0:
                sparse = sparse / float(np.max(sparse))
            scores = dense_scores + sparse
        else:
            raise ValueError(f"unknown retriever mode: {mode}")
    top = np.argsort(-scores)[:k]
    return [(int(i), float(scores[int(i)])) for i in top]


def answer_with_context(
    spec: DatasetSpec,
    qa: QAItem,
    context: str,
    generator: CachedOpenAIGenerator,
) -> str:
    try:
        return spec.answerer(qa.category, context, qa.question, generator)
    except TypeError:
        return spec.answerer(qa.category, context, qa.question, generator)


def run_sample(
    sample: Sample,
    spec: DatasetSpec,
    outdir: Path,
    generator: CachedOpenAIGenerator,
    embedder: EmbeddingClient,
    top_k: int,
    max_tokens: int,
    retriever_mode: str,
    question_workers: int,
) -> Dict[str, Any]:
    sample_dir = outdir / sample.sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)

    chunks = rag_chunks(sample, spec, max_tokens)
    matrix = None
    if retriever_mode in ("dense", "hybrid"):
        matrix = embedder.embed(list(chunks)) if chunks else np.zeros((0, 1), dtype="float32")

    def do_qa(idx_qa: Tuple[int, QAItem]) -> Tuple[int, Dict[str, Any]]:
        i, qa = idx_qa
        hits = retrieve(qa.question, chunks, embedder, top_k, retriever_mode, matrix)
        context_parts = []
        for rank, (chunk_idx, sim) in enumerate(hits, 1):
            context_parts.append(
                f"[Retrieved chunk {rank}; score={sim:.4f}; id={chunk_idx}]\n{chunks[chunk_idx]}"
            )
        context = "\n\n".join(context_parts)
        answer = answer_with_context(spec, qa, context, generator)
        row = {
            "question": qa.question,
            "gold_answer": qa.gold,
            "gold_answers": qa.gold if isinstance(qa.gold, list) else [qa.gold],
            "category": qa.category,
            "research_summary": context,
            "summary_answer": answer,
            "pred": answer,
            "rag_hits": [
                {"rank": rank, "chunk_id": idx, "score": sim}
                for rank, (idx, sim) in enumerate(hits, 1)
            ],
        }
        return i, row

    indexed = list(enumerate(sample.qas))
    if question_workers > 1 and len(indexed) > 1:
        rows: List[Tuple[int, Dict[str, Any]]] = []
        with ThreadPoolExecutor(max_workers=question_workers) as pool:
            futs = [pool.submit(do_qa, x) for x in indexed]
            for fut in as_completed(futs):
                rows.append(fut.result())
        rows.sort(key=lambda x: x[0])
    else:
        rows = [do_qa(x) for x in indexed]

    qa_results = [row for _, row in rows]
    (sample_dir / "qa_results.json").write_text(
        json.dumps(qa_results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "qa_results": qa_results,
        "stats": {
            "sample_id": sample.sample_id,
            "n_chunks": len(chunks),
            "n_questions": len(qa_results),
        },
    }


def load_spec(args: argparse.Namespace) -> DatasetSpec:
    loader = LOADERS[args.dataset]
    if args.dataset == "hotpotqa":
        return loader(split=args.split, max_tokens=args.max_tokens)
    if args.dataset == "narrativeqa":
        return loader(max_tokens=args.max_tokens)
    return loader()


def select_samples(args: argparse.Namespace, spec: DatasetSpec) -> List[Sample]:
    samples = spec.samples
    if args.only:
        keep = set(args.only)
        samples = [s for s in samples if s.sample_id in keep]
    if args.dataset in ("hotpotqa", "narrativeqa") and args.limit_samples:
        rng = random.Random(args.seed)
        samples = rng.sample(samples, min(args.limit_samples, len(samples)))
    elif args.limit_samples:
        samples = samples[: args.limit_samples]
    if args.limit_questions:
        for sample in samples:
            sample.qas = sample.qas[: args.limit_questions]
    return samples


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, choices=sorted(LOADERS))
    p.add_argument("--split", default="eval_400")
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--embed-model", default="text-embedding-3-small")
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--retriever", choices=["dense", "bm25", "hybrid"], default="dense")
    p.add_argument("--limit-samples", type=int, default=None)
    p.add_argument("--only", nargs="*", default=None)
    p.add_argument("--limit-questions", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--tag", default=None)
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    spec = load_spec(args)
    samples = select_samples(args, spec)
    tag = args.tag or f"{spec.name}-static-rag-{args.retriever}-{args.model}"
    outdir = RESULTS / tag
    outdir.mkdir(parents=True, exist_ok=True)

    n_q = sum(len(s.qas) for s in samples)
    print(f"dataset={spec.name}  method=static_rag  model={args.model}")
    print(f"samples={len(samples)}  questions={n_q}  outdir={outdir}")
    if args.dry_run:
        for s in samples[: min(3, len(samples))]:
            chunks = rag_chunks(s, spec, args.max_tokens)
            print(f"  {s.sample_id}: {len(chunks)} chunks, {len(s.qas)} questions")
        print("dry run: no API calls made")
        return 0

    generator = CachedOpenAIGenerator(
        {
            "model_name": args.model,
            "cache_dir": str(CACHE / "llm"),
            "temperature": args.temperature,
            "max_tokens": 256,
            "role": "static_rag_answer",
        }
    )
    generator.enable_cache = not args.no_cache
    embedder = EmbeddingClient(args.embed_model, str(CACHE / "embed"))

    t0 = time.time()
    all_qa: List[Dict[str, Any]] = []
    all_stats: List[Dict[str, Any]] = []

    per_sample_workers = args.workers if spec.multi_qa_per_sample else 1
    sample_workers = 1 if spec.multi_qa_per_sample else args.workers

    def do_sample(idx_s: Tuple[int, Sample]):
        i, sample = idx_s
        try:
            result = run_sample(
                sample,
                spec,
                outdir,
                generator,
                embedder,
                args.top_k,
                args.max_tokens,
                args.retriever,
                per_sample_workers,
            )
            return i, sample, result, None
        except Exception as exc:
            return i, sample, None, f"{type(exc).__name__}: {exc}"

    indexed = list(enumerate(samples, 1))
    if sample_workers > 1 and len(indexed) > 1:
        results = []
        done = 0
        with ThreadPoolExecutor(max_workers=sample_workers) as pool:
            futs = [pool.submit(do_sample, item) for item in indexed]
            for fut in as_completed(futs):
                results.append(fut.result())
                done += 1
                if done % 10 == 0 or done == len(indexed):
                    print(f"  [{done}/{len(indexed)}] elapsed={round(time.time()-t0)}s", flush=True, file=sys.stderr)
        results.sort(key=lambda x: x[0])
    else:
        results = []
        for item in indexed:
            result = do_sample(item)
            results.append(result)
            print(f"[{len(results)}/{len(indexed)}] {result[1].sample_id} elapsed={round(time.time()-t0)}s", flush=True, file=sys.stderr)

    for _, sample, result, err in results:
        if err is not None:
            print(f"FAILED {sample.sample_id}: {err}", file=sys.stderr)
            all_stats.append({"sample_id": sample.sample_id, "fatal": err})
            continue
        all_qa.extend(result["qa_results"])
        all_stats.append(result["stats"])

    metrics = score(all_qa, spec.name)
    summary = {
        "tag": tag,
        "dataset": spec.name,
        "method": "static_rag",
        "retriever": args.retriever,
        "model": args.model,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "max_tokens": args.max_tokens,
        "embed_model": args.embed_model,
        "workers": args.workers,
        "n_samples": len(samples),
        "n_questions": len(all_qa),
        "n_errors": sum(1 for r in all_qa if "error" in r),
        "wall_secs": round(time.time() - t0, 1),
        "metrics": metrics,
        "generator_stats": generator.stats(),
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
    print(f"live tokens billed this run: {generator.stats()['live_tokens']:,}")
    print(f"summary -> {outdir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
