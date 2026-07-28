# IEEE Paper Draft: CAVE-Mem

This directory contains the IEEE conference-format draft.

Files:

- `main.tex`: IEEEtran conference paper source.
- `main.pdf`: compiled PDF.
- `IEEEtran.cls`: copied from `/Users/xinyu.li/Downloads/IEEE_Conference_Template__1_`.

Build:

```bash
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Current GPT-4o-mini main table:

| Dataset | N | Static RAG F1 | GAM F1 | R2Mem F1 | Ours F1 | vs GAM | vs R2Mem |
|---|---:|---:|---:|---:|---:|---:|---:|
| LoCoMo common no26 | 1388 | 39.92 | 54.50 | 53.88 | 60.35 | +5.85 | +6.47 |
| HotpotQA eval_400 | 128 | 46.86 | 64.07 | 62.74 | 66.24 | +2.16 | +3.49 |
| NarrativeQA 300 | 300 | 32.54 | 38.66 | 37.24 | 39.37 | +0.71 | +2.12 |

R2M-style structure currently mirrored:

- Introduction: limitation of flat experience reuse.
- Related Work: deep memory search, experience learning, negative transfer.
- Preliminary: iterative memory-search trajectory and experience reuse.
- Methodology: substrate profiling, answer-contract validation, boundary suppression.
- Experiments: RQ layout following R2M, datasets, baselines, implementation, main results.
- Analysis: intervention frequency, failure patterns, baseline coverage.
- Limitations and Conclusion.

Figures currently included:

- Fig. 1: flat experience reuse vs validated experience reuse.
- Fig. 2: deep memory search loop.
- Fig. 3: CAVE-Mem framework.

Pending before a submission draft:

- Replace placeholder citations with verified BibTeX entries.
- Add Qwen 3B/7B/14B tables after server runs.
- Decide whether Mem0/MemoryOS/LightMem/Memory-R1 are external-baseline citations only
  or separately reproduced with their official implementations.
