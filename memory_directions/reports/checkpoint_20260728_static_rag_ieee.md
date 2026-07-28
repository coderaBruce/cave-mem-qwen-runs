# Checkpoint: IEEE Draft + R2M Baselines

Date: 2026-07-28

Current workspace:

- `/Users/xinyu.li/Documents/research1`

Completed:

- IEEE conference draft created and compiled:
  - `memory_directions/paper_ieee_cavemem/main.tex`
  - `memory_directions/paper_ieee_cavemem/main.pdf`
  - `memory_directions/paper_ieee_cavemem/README.md`
- R2M baseline coverage report:
  - `memory_directions/reports/r2m_baseline_coverage.md`
- Static RAG baseline script:
  - `r2m-api/run_static_rag.py`
- `r2m-api/README.md` updated with static RAG commands and baseline scope.
- Static RAG full results completed:
  - HotpotQA: `r2m-api/results/rag-hotpot400/summary.json`, F1 = 46.8603
  - NarrativeQA: `r2m-api/results/rag-nqa300/summary.json`, F1 = 32.5368

Paused/stopped:

- LoCoMo static RAG was stopped before expected network interruption.
- It had printed completion through `[2/9] conv-41 elapsed=196s`.
- Completed per-sample files should exist for `conv-30` and `conv-41` under:
  - `r2m-api/results/rag-locomo-4omini-full-no26/`
- No final LoCoMo static RAG `summary.json` is expected yet.

Resume command:

```bash
./.venv/bin/python r2m-api/run_static_rag.py \
  --dataset locomo \
  --only conv-30 conv-41 conv-42 conv-43 conv-44 conv-47 conv-48 conv-49 conv-50 \
  --tag rag-locomo-4omini-full-no26 \
  --workers 5
```

Notes:

- Re-running with the same tag is OK. LLM and embedding calls are cached under
  `r2m-api/.cache/`, so completed prompt calls should be reused.
- After LoCoMo finishes, update:
  - `memory_directions/RESULTS.md`
  - `memory_directions/paper_ieee_cavemem/main.tex`
  - `memory_directions/paper_ieee_cavemem/README.md`
- Then recompile the IEEE paper twice with `pdflatex`.
