#!/usr/bin/env python
"""API-only structured-memory baselines on the shared GAM/R2Mem scorer.

These are lightweight reproductions of the external baselines reported by
R2Mem. They keep the evaluation protocol fixed: same backbone, same benchmark
split, same answer prompts, and same token-F1/BLEU scorer. The goal is not to
claim official reproduction of systems whose released code/training recipes are
external; it is to provide matched-backbone structured-memory baseline numbers.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

import apiharness  # noqa: F401
from apiharness.cached_generator import CachedOpenAIGenerator
from apiharness.datasets import LOADERS, DatasetSpec, QAItem, Sample, materialise_chunks
from apiharness.retrievers import EmbeddingClient, _tokenize
from apiharness.scoring import score

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
CACHE = ROOT / ".cache"


def _strip_session_header(chunk: str) -> Tuple[str, str]:
    lines = chunk.splitlines()
    header = lines[0] if lines else ""
    return header, "\n".join(lines[1:]).strip()


def _clean_lines(text: str) -> List[str]:
    lines: List[str] = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        line = re.sub(r"^\s*[-*•]\s*", "", line)
        line = re.sub(r"^\s*\d+[\.)]\s*", "", line)
        if not line or line in ("[]", "None", "N/A"):
            continue
        if len(line) > 500:
            parts = [p.strip() for p in re.split(r"(?<=[.;])\s+", line) if p.strip()]
            lines.extend(parts[:4])
        else:
            lines.append(line)
    # Stable de-duplication.
    out, seen = [], set()
    for line in lines:
        key = re.sub(r"\s+", " ", line.lower())
        if key not in seen:
            out.append(line)
            seen.add(key)
    return out


def _memory_prompt(method: str, dataset: str, chunk_header: str, chunk_text: str) -> str:
    if dataset == "locomo":
        source_name = "LoCoMo dialogue QA"
        source_label = "DIALOGUE"
        guidance = "Preserve names, dates, places, counts, reasons, feelings, preferences, events, and relationships."
    elif dataset.startswith("hotpotqa"):
        source_name = "HotpotQA multi-hop QA"
        source_label = "CONTEXT CHUNK"
        guidance = "Preserve entities, aliases, dates, locations, relations, comparison facts, bridge facts, and evidence-bearing sentences."
    elif dataset == "narrativeqa":
        source_name = "NarrativeQA long-document QA"
        source_label = "DOCUMENT CHUNK"
        guidance = "Preserve characters, events, motivations, causal links, temporal order, locations, and quoted or near-quoted evidence."
    else:
        source_name = f"{dataset} QA"
        source_label = "SOURCE CHUNK"
        guidance = "Preserve facts that may help answer future questions."

    common = f"""You are constructing a long-term memory for {source_name}.
Keep facts that may help answer future questions. {guidance}
Return concise bullet lines only. Do not answer any question.

SOURCE HEADER:
{chunk_header}

{source_label}:
{chunk_text}
"""
    if method == "mem0":
        return common + "\nExtract atomic user/assistant memories, one fact per bullet."
    if method == "amem":
        return common + """\nCreate agentic memory notes. Each bullet should include:
fact; keywords=[comma-separated keywords]; context=<why this matters>."""
    if method == "memoryos":
        return common + """\nCreate hierarchical memory entries:
STM: concrete event facts;
MTM: session-level summaries;
LTM: stable preferences/relationships/profile facts."""
    if method == "lightmem":
        return common + """\nCreate lightweight topic memories. Keep only high-value
topic-grounded facts. Prefix each bullet with topic=<short topic>; fact=<fact>."""
    if method == "memoryr1":
        return common + """\nSimulate a memory-manager policy. Emit memory operations as
plain bullet lines with ADD:, UPDATE:, or DELETE:. Prefer ADD/UPDATE facts
that are useful for future QA. Do not invent facts."""
    raise ValueError(method)


def build_memories(
    method: str,
    spec: DatasetSpec,
    sample: Sample,
    outdir: Path,
    generator: CachedOpenAIGenerator,
    rebuild: bool,
) -> List[Dict[str, Any]]:
    sample_dir = outdir / sample.sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    path = sample_dir / "memories.json"
    if path.exists() and not rebuild:
        return json.loads(path.read_text(encoding="utf-8"))

    memories: List[Dict[str, Any]] = []
    for idx, chunk in enumerate(materialise_chunks(sample)):
        header, body = _strip_session_header(chunk)
        prompt = _memory_prompt(method, spec.name, header, body)
        out = generator.generate_single(prompt=prompt)
        lines = _clean_lines(out.get("text") or "")
        for j, line in enumerate(lines):
            op = None
            content = line
            m = re.match(r"^(ADD|UPDATE|DELETE)\s*:\s*(.*)$", line, flags=re.I)
            if m:
                op = m.group(1).upper()
                content = m.group(2).strip()
            if method == "memoryr1" and op == "DELETE":
                # A no-RL reproduction cannot reliably match deletes to memory ids;
                # record the decision but do not remove possibly useful evidence.
                continue
            if not content:
                continue
            memories.append(
                {
                    "id": f"{sample.sample_id}-s{idx}-m{j}",
                    "content": content,
                    "session": idx,
                    "session_header": header,
                    "dataset": spec.name,
                    "method": method,
                    "op": op,
                }
            )
    path.write_text(json.dumps(memories, ensure_ascii=False, indent=2), encoding="utf-8")
    return memories


def _bm25_scores(query: str, docs: Iterable[str]) -> np.ndarray:
    docs = list(docs)
    toks = [_tokenize(d) for d in docs]
    n = len(toks)
    if n == 0:
        return np.zeros(0, dtype="float32")
    q_terms = set(_tokenize(query))
    avgdl = sum(len(d) for d in toks) / n if n else 1.0
    dfs: Dict[str, int] = {}
    tfs: List[Dict[str, int]] = []
    for doc in toks:
        counts: Dict[str, int] = {}
        for term in doc:
            counts[term] = counts.get(term, 0) + 1
        tfs.append(counts)
        for term in counts:
            dfs[term] = dfs.get(term, 0) + 1
    scores = np.zeros(n, dtype="float32")
    k1, b = 0.9, 0.4
    for term in q_terms:
        df = dfs.get(term, 0)
        idf = np.log(1.0 + (n - df + 0.5) / (df + 0.5))
        for i, counts in enumerate(tfs):
            freq = counts.get(term, 0)
            if not freq:
                continue
            dl = len(toks[i]) or 1
            scores[i] += idf * (freq * (k1 + 1)) / (
                freq + k1 * (1 - b + b * dl / (avgdl or 1.0))
            )
    return scores


def retrieve_memories(
    question: str,
    memories: List[Dict[str, Any]],
    matrix: np.ndarray | None,
    embedder: EmbeddingClient,
    top_k: int,
    mode: str,
) -> List[Tuple[Dict[str, Any], float]]:
    if not memories:
        return []
    texts = [m["content"] for m in memories]
    if mode == "bm25":
        scores = _bm25_scores(question, texts)
    else:
        if matrix is None:
            raise ValueError("dense retrieval requires matrix")
        q = embedder.embed([question])
        dense = (q @ matrix.T).reshape(-1)
        if mode == "dense":
            scores = dense
        elif mode == "hybrid":
            sparse = _bm25_scores(question, texts)
            if len(sparse) and float(np.max(sparse)) > 0:
                sparse = sparse / float(np.max(sparse))
            scores = dense + sparse
        else:
            raise ValueError(mode)
    k = min(top_k, len(memories))
    top = np.argsort(-scores)[:k]
    return [(memories[int(i)], float(scores[int(i)])) for i in top]


def context_from_hits(method: str, hits: List[Tuple[Dict[str, Any], float]]) -> str:
    parts = []
    for rank, (mem, sim) in enumerate(hits, 1):
        head = mem.get("session_header") or f"session {mem.get('session')}"
        parts.append(
            f"[{method} memory {rank}; score={sim:.4f}; {head}]\n{mem['content']}"
        )
    return "\n\n".join(parts) if parts else "[NO RETRIEVED MEMORIES]"


def run_sample(
    method: str,
    sample: Sample,
    spec: DatasetSpec,
    outdir: Path,
    memory_generator: CachedOpenAIGenerator,
    answer_generator: CachedOpenAIGenerator,
    embedder: EmbeddingClient,
    top_k: int,
    retrieval: str,
    rebuild: bool,
    workers: int,
) -> Dict[str, Any]:
    memories = build_memories(method, spec, sample, outdir, memory_generator, rebuild)
    texts = [m["content"] for m in memories]
    matrix = embedder.embed(texts) if texts and retrieval in ("dense", "hybrid") else None

    sample_dir = outdir / sample.sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)

    def answer_one(idx_qa: Tuple[int, QAItem]):
        i, qa = idx_qa
        hits = retrieve_memories(qa.question, memories, matrix, embedder, top_k, retrieval)
        context = context_from_hits(method, hits)
        answer = spec.answerer(qa.category, context, qa.question, answer_generator)
        return i, {
            "question": qa.question,
            "gold_answer": qa.gold,
            "gold_answers": qa.gold if isinstance(qa.gold, list) else [qa.gold],
            "category": qa.category,
            "research_summary": context,
            "summary_answer": answer,
            "pred": answer,
            "retrieved_memories": [
                {"rank": r, "score": s, **m} for r, (m, s) in enumerate(hits, 1)
            ],
        }

    indexed = list(enumerate(sample.qas))
    if workers > 1 and len(indexed) > 1:
        rows = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(answer_one, item) for item in indexed]
            for fut in as_completed(futs):
                rows.append(fut.result())
        rows.sort(key=lambda x: x[0])
    else:
        rows = [answer_one(item) for item in indexed]

    qa_results = [row for _, row in rows]
    (sample_dir / "qa_results.json").write_text(
        json.dumps(qa_results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "qa_results": qa_results,
        "stats": {
            "sample_id": sample.sample_id,
            "n_memories": len(memories),
            "n_questions": len(qa_results),
        },
    }


def load_spec(args: argparse.Namespace) -> DatasetSpec:
    loader = LOADERS[args.dataset]
    if args.dataset == "hotpotqa":
        return loader(split=args.split, max_tokens=args.max_tokens)
    if args.dataset == "narrativeqa":
        return loader(max_tokens=args.max_tokens)
    return loader(max_tokens=args.max_tokens)


def select_samples(args: argparse.Namespace, spec: DatasetSpec) -> List[Sample]:
    samples = spec.samples
    if args.only:
        keep = set(args.only)
        samples = [s for s in samples if s.sample_id in keep]
    if args.dataset in ("hotpotqa", "narrativeqa") and args.limit_samples:
        rng = random.Random(args.seed)
        samples = rng.sample(samples, min(args.limit_samples, len(samples)))
    elif args.limit_samples:
        rng = random.Random(args.seed)
        samples = rng.sample(samples, min(args.limit_samples, len(samples)))
    if args.limit_questions:
        for sample in samples:
            sample.qas = sample.qas[: args.limit_questions]
    return samples


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="locomo", choices=sorted(LOADERS))
    p.add_argument("--split", default="eval_400")
    p.add_argument("--method", required=True, choices=["mem0", "amem", "memoryos", "lightmem", "memoryr1"])
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--embed-model", default="text-embedding-3-small")
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--retrieval", choices=["dense", "bm25", "hybrid"], default="hybrid")
    p.add_argument("--only", nargs="*", default=None)
    p.add_argument("--limit-samples", type=int, default=None)
    p.add_argument("--limit-questions", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--tag", default=None)
    p.add_argument("--rebuild", action="store_true")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    spec = load_spec(args)
    samples = select_samples(args, spec)

    tag = args.tag or f"{args.method}-{spec.name}-{args.model}"
    outdir = RESULTS / tag
    outdir.mkdir(parents=True, exist_ok=True)
    n_q = sum(len(s.qas) for s in samples)
    print(f"dataset={spec.name} method={args.method} model={args.model}")
    print(f"samples={len(samples)} questions={n_q} outdir={outdir}")
    if args.dry_run:
        for s in samples[:3]:
            chunks = materialise_chunks(s)
            print(f"  {s.sample_id}: {len(chunks)} chunks, {len(s.qas)} questions")
        return 0

    memory_generator = CachedOpenAIGenerator(
        {
            "model_name": args.model,
            "cache_dir": str(CACHE / "llm"),
            "temperature": args.temperature,
            "max_tokens": 1024,
            "role": f"{args.method}_memory",
        }
    )
    answer_generator = CachedOpenAIGenerator(
        {
            "model_name": args.model,
            "cache_dir": str(CACHE / "llm"),
            "temperature": args.temperature,
            "max_tokens": 256,
            "role": f"{args.method}_answer",
        }
    )
    if args.no_cache:
        memory_generator.enable_cache = False
        answer_generator.enable_cache = False
    embedder = EmbeddingClient(args.embed_model, str(CACHE / "embed"))

    all_qa: List[Dict[str, Any]] = []
    all_stats: List[Dict[str, Any]] = []
    t0 = time.time()
    for idx, sample in enumerate(samples, 1):
        try:
            res = run_sample(
                args.method,
                sample,
                spec,
                outdir,
                memory_generator,
                answer_generator,
                embedder,
                args.top_k,
                args.retrieval,
                args.rebuild,
                args.workers,
            )
            all_qa.extend(res["qa_results"])
            all_stats.append(res["stats"])
        except Exception as exc:
            print(f"FAILED {sample.sample_id}: {type(exc).__name__}: {exc}", file=sys.stderr)
            all_stats.append({"sample_id": sample.sample_id, "fatal": f"{type(exc).__name__}: {exc}"})
        print(f"[{idx}/{len(samples)}] {sample.sample_id} elapsed={round(time.time()-t0)}s", file=sys.stderr, flush=True)

    metrics = score(all_qa, spec.name)
    summary = {
        "tag": tag,
        "dataset": spec.name,
        "method": args.method,
        "protocol": "api_only_structured_memory_reproduction",
        "note": "Matched-backbone reproduction of external structured-memory baselines; not the official external training/runtime stack.",
        "model": args.model,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "retrieval": args.retrieval,
        "embed_model": args.embed_model,
        "n_samples": len(samples),
        "n_questions": len(all_qa),
        "n_errors": sum(1 for r in all_qa if "error" in r),
        "wall_secs": round(time.time() - t0, 1),
        "metrics": metrics,
        "generator_stats": {
            "memory": memory_generator.stats(),
            "answer": answer_generator.stats(),
        },
        "embed_calls": embedder.n_embedded,
        "per_sample": all_stats,
    }
    (outdir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (outdir / "all_qa_results.json").write_text(json.dumps(all_qa, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== metrics ===")
    print(json.dumps(metrics.get("by_category", metrics.get("overall")), indent=2))
    if "overall" in metrics:
        print("overall:", json.dumps(metrics["overall"]))
    live = memory_generator.stats()["live_tokens"] + answer_generator.stats()["live_tokens"]
    print(f"live tokens billed this run: {live:,}")
    print(f"summary -> {outdir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
