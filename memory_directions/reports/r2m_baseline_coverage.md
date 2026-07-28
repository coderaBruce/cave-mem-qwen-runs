# R2M Baseline Coverage

Date: 2026-07-28

Source inspected:

- `/Users/xinyu.li/Downloads/2605.13486v1.pdf`
- local text extraction at `/private/tmp/r2m_paper.txt`
- local upstream checkout `R2M-official/`

## Baselines in R2M

R2M groups baselines into four families:

| Family | Methods | Current workspace status |
|---|---|---|
| Memory-free retrieval | Vanilla RAG | Implemented in `r2m-api/run_static_rag.py` |
| Structured memory management | Mem0, A-Mem, MemoryOS, LightMem | Not implemented in `R2M-official/`; requires external repos or separate reproduction |
| RL-based memory | Memory-R1 | Not implemented in `R2M-official/`; requires RL memory-management training/evaluation code |
| Deep search memory | GAM | Implemented and reproduced in `r2m-api/run.py` |
| Reflective experience memory | R2Mem | Implemented and reproduced in `r2m-api/run.py` with local API-compatible substitutions |

## Static RAG Reproduction

R2M describes vanilla RAG as fixed 2048-token chunks, top-5 embedding retrieval,
and direct single-pass answer generation. The new script follows that setup:

```bash
./.venv/bin/python r2m-api/run_static_rag.py \
  --dataset locomo \
  --only conv-30 conv-41 conv-42 conv-43 conv-44 conv-47 conv-48 conv-49 conv-50 \
  --tag rag-locomo-4omini-full-no26 \
  --workers 5

./.venv/bin/python r2m-api/run_static_rag.py \
  --dataset hotpotqa \
  --split eval_400 \
  --limit-samples 128 \
  --seed 42 \
  --tag rag-hotpot400 \
  --workers 5

./.venv/bin/python r2m-api/run_static_rag.py \
  --dataset narrativeqa \
  --limit-samples 300 \
  --seed 42 \
  --tag rag-nqa300 \
  --workers 5
```

Design choices:

- LoCoMo sessions are concatenated and re-split into fixed token windows so the
  RAG baseline does not inherit session-level memory organization.
- HotpotQA and NarrativeQA use the same R2M/GAM dataset loader, seed sampling,
  answer prompt, and F1 scorer as the GAM/R2Mem runs.
- Dense retrieval uses `text-embedding-3-small`, matching the rest of the
  API-only harness. This is a disclosed deviation from R2M's BGE-M3 setup.

Completed GPT-4o-mini results:

| Dataset | N | Static RAG F1 |
|---|---:|---:|
| LoCoMo common no26 | 1388 | 39.92 |
| HotpotQA eval_400 | 128 | 46.86 |
| NarrativeQA 300 | 300 | 32.54 |

Result files:

- `r2m-api/results/rag-locomo-4omini-full-no26/summary.json`
- `r2m-api/results/rag-hotpot400/summary.json`
- `r2m-api/results/rag-nqa300/summary.json`

## Structured-Memory Baseline Reproductions

The local `R2M-official/` checkout does not contain runnable implementations
for Mem0, A-Mem, MemoryOS, LightMem, or Memory-R1. To avoid mixing published
numbers from other backbones with our GPT-4o-mini table, we added API-only
matched-backbone reproductions in `r2m-api/run_memory_baselines.py`.

These are not official implementations. Each method writes structured memories
from LoCoMo dialogue chunks, retrieves with the shared hybrid dense/BM25
interface, answers with the same GPT-4o-mini answerer, and scores with the same
LoCoMo token-F1/BLEU scorer. Memory-R1 is a no-RL simulation of memory
operations, not the trained RL policy.

Completed LoCoMo common no26 results:

| Method | Protocol | N | F1 | BLEU-1 | Result file |
|---|---|---:|---:|---:|---|
| Mem0-style | API-only structured memory | 1388 | 48.27 | 42.10 | `r2m-api/results/mem0-locomo-4omini-full-no26/summary.json` |
| A-Mem-style | API-only structured memory | 1388 | 46.59 | 40.55 | `r2m-api/results/amem-locomo-4omini-full-no26/summary.json` |
| MemoryOS-style | API-only structured memory | 1388 | 45.26 | 39.28 | `r2m-api/results/memoryos-locomo-4omini-full-no26/summary.json` |
| LightMem-style | API-only structured memory | 1388 | 45.80 | 39.90 | `r2m-api/results/lightmem-locomo-4omini-full-no26/summary.json` |
| Memory-R1 no-RL style | API-only memory-operation simulation | 1388 | 48.53 | 42.60 | `r2m-api/results/memoryr1-locomo-4omini-full-no26/summary.json` |

Current paper treatment:

- Main LoCoMo GPT-4o-mini table includes these five rows with a dagger marker.
- The caption and text state clearly that they are matched-backbone
  reproductions, not official implementations.
- GAM, R2Mem, and CAVE-Mem remain the primary end-to-end baselines because
  those are fully implemented in the local harness.
- A faithful future comparison should import official memory-system code behind
  the same dataset/answerer/scorer adapter.
