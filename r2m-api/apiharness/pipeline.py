"""The GAM pipeline, API-backed.

Memory construction, retriever assembly and the question loop, reusing GAM's
own `MemoryAgent`, `ResearchAgent`, prompts and schemas unchanged. The only
substitutions are the two retrievers that need GPUs/JVMs (see `retrievers.py`)
and a caching layer on the generator (see `cached_generator.py`).

The researcher is injected via `researcher_factory`, so R²-Mem and our own
variants can be dropped in without touching this file. Anything with a
`.research(request) -> ResearchOutput` method and a `.total_tokens` attribute
works; if it accepts a `category` kwarg (as R²-Mem's `ResearchAgent_exp` does)
that is passed through automatically.

Traces are written in exactly the upstream layout — `research_trace_q{i}.json`
with `question` / `iterations` / `integrated_memory`, and `qa_results.json` —
so R²-Mem's offline reflection stage can consume our runs unmodified.
"""
from __future__ import annotations

import contextlib
import inspect
import io
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from gam.agents.memory_agent import MemoryAgent
from gam.agents.research_agent import ResearchAgent
from gam.retriever.index_retriever import IndexRetriever
from gam.schemas import InMemoryMemoryStore, InMemoryPageStore

from .cached_generator import CachedOpenAIGenerator
from .datasets import DatasetSpec, QAItem, Sample, materialise_chunks
from .retrievers import APIDenseRetriever, EmbeddingClient, LiteBM25Retriever


# --------------------------------------------------------------------------
def make_generators(
    model: str,
    cache_dir: Path,
    temperature: float = 0.3,
) -> Dict[str, CachedOpenAIGenerator]:
    """Three generators with upstream's roles and token limits.

    `temperature=0.3` and these `max_tokens` mirror `eval/locomo_test.py`.
    """
    common = dict(model_name=model, cache_dir=str(cache_dir), temperature=temperature)
    memory_max_tokens = int(os.environ.get("MEMORY_MAX_TOKENS", "256"))
    research_max_tokens = int(os.environ.get("RESEARCH_MAX_TOKENS", "2048"))
    working_max_tokens = int(os.environ.get("WORKING_MAX_TOKENS", "256"))
    return {
        "memory": CachedOpenAIGenerator(
            {**common, "max_tokens": memory_max_tokens, "role": "memory"}
        ),
        "research": CachedOpenAIGenerator(
            {**common, "max_tokens": research_max_tokens, "role": "research"}
        ),
        "working": CachedOpenAIGenerator(
            {**common, "max_tokens": working_max_tokens, "role": "working"}
        ),
    }


def build_memory_and_retrievers(
    chunks: List[str],
    workdir: Path,
    memory_generator,
    embedder: EmbeddingClient,
    rebuild: bool = False,
    index_header: str = "none",
):
    """Run the Memorizer over `chunks`, then build the three retrievers."""
    workdir.mkdir(parents=True, exist_ok=True)
    memory_store = InMemoryMemoryStore(dir_path=str(workdir))
    page_store = InMemoryPageStore(dir_path=str(workdir))

    state_file = workdir / "memory_state.json"
    if rebuild or not state_file.exists():
        agent = MemoryAgent(
            memory_store=memory_store, page_store=page_store, generator=memory_generator
        )
        for chunk in chunks:
            agent.memorize(chunk)
        state = memory_store.load()
        state_file.write_text(
            json.dumps(state.model_dump(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    retrievers: Dict[str, Any] = {}

    # Headers are topical abstracts. Putting them in the *lexical* index dilutes
    # exact-term matching (single-hop -2.13 F1); putting them in the *semantic*
    # index is what contextual retrieval actually intends. So the two are
    # controlled separately.
    bm25 = LiteBM25Retriever({"index_header": index_header in ("both", "bm25")})
    bm25.build(page_store)
    retrievers["keyword"] = bm25

    dense = APIDenseRetriever({"embedder": embedder,
                               "index_header": index_header in ("both", "dense")})
    dense.build(page_store)
    retrievers["vector"] = dense

    index = IndexRetriever({"index_dir": str(workdir / "page_index")})
    index.build(page_store)
    retrievers["page_index"] = index

    return memory_store, page_store, retrievers


def default_researcher_factory(page_store, memory_store, retrievers, generator, **_):
    return ResearchAgent(
        page_store=page_store,
        memory_store=memory_store,
        retrievers=retrievers,
        generator=generator,
        max_iters=3,
    )


# --------------------------------------------------------------------------
def run_sample(
    sample: Sample,
    spec: DatasetSpec,
    outdir: Path,
    generators: Dict[str, CachedOpenAIGenerator],
    embedder: EmbeddingClient,
    researcher_factory: Callable = default_researcher_factory,
    rebuild: bool = False,
    max_workers: int = 1,
    evidence_pages: int = 0,
    evidence_chars: int = 0,
    include_header: bool = False,
    index_header: str = "none",   # none | dense | bm25 | both
) -> Dict[str, Any]:
    """Build memory for one sample, then answer each of its questions."""
    sample_dir = outdir / sample.sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)

    chunks = materialise_chunks(sample) if not sample.chunks else sample.chunks

    t0 = time.time()
    memory_store, page_store, retrievers = build_memory_and_retrievers(
        chunks, sample_dir, generators["memory"], embedder, rebuild=rebuild,
        index_header=index_header,
    )
    build_secs = time.time() - t0

    # One researcher per worker thread. The stores and retrievers are read-only
    # once memory is built, so they are safe to share; the ResearchAgent itself
    # is not (it mutates `total_tokens` and `_last_page_count`), hence one each.
    local = threading.local()

    def get_researcher():
        if not hasattr(local, "researcher"):
            local.researcher = researcher_factory(
                page_store=page_store,
                memory_store=memory_store,
                retrievers=retrievers,
                generator=generators["research"],
            )
        return local.researcher

    def answer_one(idx_qa):
        i, qa = idx_qa
        researcher = get_researcher()
        accepts_category = (
            "category" in inspect.signature(researcher.research).parameters
        )
        try:
            if accepts_category:
                out = researcher.research(qa.question, qa.category or 0)
            else:
                out = researcher.research(qa.question)

            iterations = out.raw_memory.get("iterations", [])

            evidence = None
            if evidence_chars > 0:
                srcs = (out.raw_memory.get("temp_memory") or {}).get("sources") or []
                pages, seen = [], set()
                for sid_ in srcs[:evidence_pages]:
                    try:
                        pid = int(str(sid_))
                    except (TypeError, ValueError):
                        continue
                    if pid in seen:
                        continue
                    seen.add(pid)
                    pg = page_store.get(pid)
                    if pg is not None:
                        # Include the header: GAM's Memorizer writes an abstract there
                        # with absolute dates ("On December 17, 2022, ..."), which the
                        # raw dialogue only expresses relatively. Passing content alone
                        # cost 3.5 F1 on temporal questions.
                        head = (pg.header or "").strip()
                        body = pg.content[:evidence_chars]
                        pages.append(
                            f"[page {pid}] {head}\n{body}" if include_header
                            else f"[page {pid}] {body}"
                        )
                evidence = "\n\n".join(pages) if pages else None

            try:
                answer = spec.answerer(
                    qa.category, out.integrated_memory, qa.question,
                    generators["working"], evidence=evidence,
                )
            except TypeError:
                answer = spec.answerer(
                    qa.category, out.integrated_memory, qa.question,
                    generators["working"],
                )

            trace_file = sample_dir / f"research_trace_q{i}.json"
            trace_file.write_text(
                json.dumps(
                    {
                        "question": qa.question,
                        "raw_memory": out.raw_memory,
                        "integrated_memory": out.integrated_memory,
                        "iterations": iterations,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            return i, {
                "question": qa.question,
                # Upstream uses "gold_answer" for LoCoMo and "gold_answers" for
                # the other two; emit both so either downstream reader works.
                "gold_answer": qa.gold,
                "gold_answers": qa.gold if isinstance(qa.gold, list) else [qa.gold],
                "category": qa.category,
                "research_summary": out.integrated_memory,
                "summary_answer": answer,
                "pred": answer,
                "iterations": len(iterations),
                "research_trace_file": str(trace_file),
                "_id": qa.qid or f"{sample.sample_id}-q{i}",
            }
        except Exception as exc:  # a dead question shouldn't kill the run
            return i, {
                "question": qa.question,
                "gold_answer": qa.gold,
                "gold_answers": qa.gold if isinstance(qa.gold, list) else [qa.gold],
                "category": qa.category,
                "error": f"{type(exc).__name__}: {exc}",
            }

    indexed = list(enumerate(sample.qas, 1))
    collected: List[tuple] = []

    # Upstream's agents print heavily. Swap stdout ONCE around the whole parallel
    # section, from the main thread only, and drop the noise; per-thread swaps
    # would race. Progress reporting goes to stderr, which is untouched.
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink):
        if max_workers > 1 and len(indexed) > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                collected = list(pool.map(answer_one, indexed))
        else:
            collected = [answer_one(x) for x in indexed]

    qa_results = [r for _, r in sorted(collected, key=lambda kv: kv[0])]

    (sample_dir / "qa_results.json").write_text(
        json.dumps(qa_results, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    stats = {
        "sample_id": sample.sample_id,
        "n_chunks": len(chunks),
        "n_pages": len(page_store.load()),
        "n_questions": len(sample.qas),
        "n_errors": sum(1 for r in qa_results if "error" in r),
        "memory_build_secs": round(build_secs, 1),
        "iterations_mean": (
            sum(r.get("iterations", 0) for r in qa_results if "error" not in r)
            / max(1, sum(1 for r in qa_results if "error" not in r))
        ),
    }
    (sample_dir / "run_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"qa_results": qa_results, "stats": stats}
