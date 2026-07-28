# Paper Draft: CAVE-Mem

Draft files:

- `main.tex`: 8-page LaTeX draft.
- `main.pdf`: compiled PDF generated from `main.tex`.
- Qwen run script: `../scripts/run_qwen_full_eval.sh`.

Current empirical claim, GPT-4o-mini backbone:

| Dataset | N | GAM F1 | R2Mem F1 | Ours F1 | vs GAM | vs R2Mem |
|---|---:|---:|---:|---:|---:|---:|
| LoCoMo common no26 | 1388 | 54.50 | 53.88 | 60.35 | +5.85 | +6.47 |
| HotpotQA eval_400 | 128 | 64.07 | 62.74 | 66.24 | +2.16 | +3.49 |
| NarrativeQA 300 | 300 | 38.66 | 37.24 | 39.37 | +0.71 | +2.12 |

Principle:

Experience reuse should not be flat Top-K prompt injection. A memory-search
agent should reuse experience only when the current question, memory substrate,
answer contract, and failure-boundary checks indicate positive utility.

Before submission:

- Fill verified BibTeX citations for GAM, R2Mem, LoCoMo, HotpotQA,
  NarrativeQA, and related agent-memory work.
- Add bootstrap confidence intervals or paired significance tests.
- Add Qwen 3B/7B/14B runs when server results are available.
- Convert LoCoMo contract rules into cleaner rule families or validate on an
  additional split.
- Add exact command appendix from `memory_directions/RESULTS.md`.
