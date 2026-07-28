# r2m-api — API-only harness for GAM / R²-Mem

Runs the full GAM pipeline against hosted APIs, with no GPU, no vLLM, no torch,
no FAISS and no JVM. Built because we have API credit now and one H100 later.

`R2M-official/` stays a pristine upstream checkout. This directory imports from
it; it never edits it.

## What is reused vs. replaced

**Reused verbatim from `R2M-official/`** — this is where fidelity comes from:

- `MemoryAgent`, `ResearchAgent` (the whole Planning→Search→Integrate→Reflect loop)
- All researcher prompts and JSON schemas
- `IndexRetriever` (pure Python already)
- Dataset loaders, context chunkers, answer prompts, and the F1 / BLEU-1 metrics

**Replaced, because the originals need hardware we don't have:**

| Upstream | Needs | Our replacement |
|---|---|---|
| `DenseRetriever` (BGE-M3) | torch + GPU | OpenAI `text-embedding-3-small`, exact inner-product search over numpy |
| `BM25Retriever` (pyserini) | JVM + Lucene | pure-Python Okapi BM25, Lucene defaults k1=0.9, b=0.4 |
| `OpenAIGenerator` | — | subclassed to add a disk cache + per-role token accounting |

## Deviations to report in the paper

These are real and must be disclosed, not buried:

1. **Embedding space differs.** `text-embedding-3-small` (1536-d) instead of
   BGE-M3 (1024-d). Dense retrieval quality will not match the papers exactly.
2. **BM25 implementation differs.** Pure-Python Okapi vs. Lucene's analyzer
   chain. Same parameters, near-identical but not identical rankings.
3. **Backbones differ.** R²-Mem used Qwen2.5-3B/7B/14B and Llama3.1-8B locally.
   We cannot serve those. GAM, however, *does* report GPT-4o-mini, so that is
   our anchor: we reproduce GAM's own GPT-4o-mini numbers to validate the
   harness, then measure the GAM → R²-Mem delta on the same backbone.
   **We are reproducing R²-Mem's mechanism and its delta, not its absolute numbers.**
4. **NarrativeQA chunking is patched.** Upstream's
   `narrativeqa_test._smart_split_by_tokens` calls
   `tokenizer.decode(..., skip_special_tokens=True)`, a HuggingFace kwarg that
   tiktoken rejects — so its tiktoken fallback crashes and NarrativeQA silently
   requires a local BGE-M3. The identical function in `hotpotqa_test.py` wraps
   the call in try/except (their comment: "🔥 关键修复点"); the fix was never
   ported. We apply their own fix. See `datasets._narrativeqa_chunker`.
5. **Temperature is 0.3**, matching upstream's `eval/locomo_test.py`. Runs are
   therefore non-deterministic and no seed is exposed — which is consistent
   with the same config appearing with three different Overall F1 values across
   the R²-Mem tables. `--temperature 0.0` gives us a deterministic variant they
   never ran.

## Setup

```bash
cd research1
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python \
    openai anthropic numpy tqdm tiktoken rank_bm25 python-dotenv google-genai pyarrow pydantic
cp r2m-api/.env.example .env      # then add your OPENAI_API_KEY
```

For local OpenAI-compatible LLM servers, such as vLLM serving Qwen, chat and
embedding endpoints can be separated:

```bash
export OPENAI_LLM_BASE_URL=http://127.0.0.1:8001/v1
export OPENAI_LLM_API_KEY=EMPTY
export OPENAI_EMBED_API_KEY=$OPENAI_API_KEY
# optional, only if your embedding provider is not the default OpenAI endpoint
export OPENAI_EMBED_BASE_URL=https://api.openai.com/v1
```

For fully local Qwen runs with no OpenAI API usage, use the local embedding
backend instead:

```bash
export OPENAI_LLM_BASE_URL=http://127.0.0.1:8001/v1
export OPENAI_LLM_API_KEY=EMPTY
export EMBED_BACKEND=local
export EMBED_MODEL=local:BAAI/bge-m3
```

The convenience wrapper `memory_directions/scripts/run_qwen_local_full_eval.sh`
sets these defaults automatically.

If the split variables are unset, the harness falls back to the original
`OPENAI_API_KEY` and optional `OPENAI_BASE_URL` behavior.

Data (already downloaded into `research1/data/`):

```bash
# LoCoMo ships with the upstream repo; symlinked into data/locomo/
# HotpotQA: eval_400 (56K) / eval_1600 (224K) / eval_3200 (448K)
curl -L -o data/hotpotqa/eval_400.json \
  https://huggingface.co/datasets/BytedTsinghua-SIA/hotpotqa/resolve/main/eval_400.json
# NarrativeQA: 8 test shards from deepmind/narrativeqa
curl -L -o data/narrativeqa/test-00000-of-00008.parquet \
  https://huggingface.co/datasets/deepmind/narrativeqa/resolve/main/data/test-00000-of-00008.parquet
```

Note `scripts/download_data.sh` upstream is broken: it calls a
`download_data/download_narrativeqa.py` that does not exist, and fetches
HotpotQA `eval_6400` although every eval script iterates `eval_400/1600/3200`.

On an H100 server with an NVIDIA driver reporting CUDA 12.8, the Qwen remote
setup script installs PyTorch from the cu128 wheel index before installing
vLLM:

```bash
RECREATE_VENV=1 bash memory_directions/scripts/prepare_qwen_remote.sh
```

This avoids accidentally installing a newer CUDA build of PyTorch than the
server driver can load.

## Running

```bash
cd r2m-api

# verify wiring, zero API calls
../.venv/bin/python run.py --dataset locomo --dry-run --limit-samples 2

# smoke test
../.venv/bin/python run.py --dataset locomo --method gam --limit-samples 1 --limit-questions 5

# full baselines
../.venv/bin/python run.py --dataset locomo      --method gam
../.venv/bin/python run.py --dataset hotpotqa    --method gam --split eval_400
../.venv/bin/python run.py --dataset narrativeqa --method gam --limit-samples 300 --seed 42

# R2M memory-free vanilla RAG baseline:
# fixed 2048-token chunks, top-5 dense retrieval, one-pass answer generation.
../.venv/bin/python run_static_rag.py \
  --dataset locomo \
  --only conv-30 conv-41 conv-42 conv-43 conv-44 conv-47 conv-48 conv-49 conv-50 \
  --tag rag-locomo-4omini-full-no26 \
  --workers 5

../.venv/bin/python run_static_rag.py \
  --dataset hotpotqa \
  --split eval_400 \
  --limit-samples 128 \
  --seed 42 \
  --tag rag-hotpot400 \
  --workers 5

../.venv/bin/python run_static_rag.py \
  --dataset narrativeqa \
  --limit-samples 300 \
  --seed 42 \
  --tag rag-nqa300 \
  --workers 5
```

API-only structured-memory baseline reproductions for the LoCoMo common no26
split:

```bash
for method in mem0 amem memoryos lightmem memoryr1; do
  ../.venv/bin/python run_memory_baselines.py \
    --method "$method" \
    --only conv-30 conv-41 conv-42 conv-43 conv-44 conv-47 conv-48 conv-49 conv-50 \
    --tag "${method}-locomo-4omini-full-no26" \
    --workers 5
done
```

These are matched-backbone adapter reproductions, not official implementations.
Memory-R1 is a no-RL memory-operation simulation.

Outputs land in `results/<tag>/`:

- `summary.json` — metrics, token accounting, per-sample stats
- `<sample_id>/research_trace_q*.json` — traces in the **exact upstream format**,
  so R²-Mem's offline reflection stage consumes our runs unmodified
- `<sample_id>/qa_results.json` — emits both `gold_answer` (LoCoMo flavour) and
  `gold_answers`/`pred` (HotpotQA flavour) so either downstream reader works

## Caching

Every LLM call and every embedding is cached to disk under `.cache/`, keyed by
content. This matters for three reasons: re-runs are free, the counterfactual
A/B replays needed for utility-verified experience become affordable, and
`summary.json` separates `total_tokens` (logical cost, cache hits included, the
number to report) from `live_tokens` (what was actually billed).

## Status

GAM/R2Mem GPT-4o-mini full/common comparisons have been executed on LoCoMo,
HotpotQA, and NarrativeQA. `run_static_rag.py` adds the R2M memory-free RAG
baseline in the same loader/scorer/cache harness. `run_memory_baselines.py`
adds API-only matched-backbone reproductions for Mem0, A-Mem, MemoryOS,
LightMem, and Memory-R1 on LoCoMo common no26. These rows are diagnostic
controls, not official memory-system implementations; Memory-R1 is no-RL.
