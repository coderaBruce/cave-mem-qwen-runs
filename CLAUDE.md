# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

`research1/` is a **paper project**, not a software product. The goal is to write and submit a paper that
**uses R²-Mem as the primary baseline and beats it**. Target venue: **EACL** (ACL-family, so submission
goes through ARR — the concrete cycle/deadline is not recorded here; ask the user and fill it in).

Everything in this directory serves that goal. When asked to "improve the method" or "run an experiment,"
the success criterion is *a number in a table that clears the bar below*, reproduced under the baseline's
own protocol.

## Layout

```
research1/
├── CLAUDE.md                # this file — project-level intent
├── papers/
│   ├── R2-Mem-2605.13486v1.pdf   # primary baseline (25pp)
│   └── GAM-2511.18423v1.pdf      # R²-Mem's own baseline, which it builds on (17pp)
├── notes/
│   └── 01-baseline-critique.md   # full critique of both papers + competitive landscape
└── R2M-official/            # R²-Mem's official implementation (git repo, separate)
    └── CLAUDE.md            # code-level guide: architecture, pipeline, config, gotchas
```

**Both PDFs are local — read them with the Read tool, never re-download.** Read
`notes/01-baseline-critique.md` first; it already contains the section-by-section analysis, the numbers to
beat, and the list of neighbouring work, with every claim tagged `[paper]` / `[code]` / `[ours]`.
Read `R2M-official/CLAUDE.md` before touching any code.

Treat `R2M-official/` as **reference/baseline code that should stay reproducible**: prefer building our
method in a sibling directory over editing the baseline in place, so that "GAM vs. R²-Mem vs. ours" stays a
clean three-way comparison.

Note that GAM is the *substrate*, not just a competitor: R²-Mem is GAM plus an offline experience loop. Any
method we propose sits on the same GAM researcher loop, so GAM's own findings constrain us — in particular
that the Researcher (not the Memorizer) is the capacity bottleneck, and that GAM's F1 rises monotonically
with reflection depth and retrieved-page count, i.e. the baseline is compute-starved at its defaults.

## The baseline in one paragraph

R²-Mem (arXiv:2605.13486v1, Wang et al., USTC) sits on top of **GAM**, a deep memory-search agent that loops
Planning → Searching → Reflection over raw memory. R²-Mem adds an *offline* stage: a **Rubric-guided
Evaluator** (GPT-4o) scores every step of historical trajectories on 4 dimensions × 0–3, and a
**self-Reflection Learner** distills the extreme steps (total ≤5 = low-quality, ≥10 = high-quality; the
middle is discarded) into two FAISS-indexed **experience banks**, one for Planning and one for Reflection.
At *online* inference the agent abstracts the current step into a "situation," retrieves Top-3 experiences,
and injects them into the planning/reflection prompts. The pitch is **RL-free, low-cost self-improvement**:
better F1 *and* fewer tokens/iterations.

## The bar to clear

All numbers below are from the paper's tables. These are what we have to beat — and note we should beat
them **on both axes**, since R²-Mem's headline claim is quality *and* efficiency simultaneously.

**LoCoMo overall (F1 / BLEU-1)** — Table 1:

| Backbone | GAM | R²-Mem | ← must exceed |
|---|---|---|---|
| Qwen2.5-7B | 45.77 / 39.42 | **51.35 / 44.91** | |
| Llama3.1-8B | 44.15 / 37.48 | **47.13 / 40.66** | |

Per-category on Qwen2.5-7B (R²-Mem): Multi-hop 41.62/32.34, Temporal 46.76/40.67, Open-domain 25.06/20.78,
Single-hop 59.03/53.13. **Temporal is where R²-Mem gains most over GAM (+50% F1); Open-domain is its
weakest category** and the one where the RL baseline Memory-R1 (28.36 F1) still beats it — that gap is an
obvious target.

**Scaling + efficiency on LoCoMo (F1 / tokens in M / iterations)** — Table 3:

| Backbone | GAM | R²-Mem |
|---|---|---|
| Qwen2.5-3B | 32.00 / 44.87 / 2.47 | **39.24 / 39.07 / 1.97** |
| Qwen2.5-7B | 45.77 / 34.01 / 1.84 | **51.35 / 28.93 / 1.52** |
| Qwen2.5-14B | 52.99 / 28.85 / 1.55 | **56.21 / 27.73 / 1.47** |

Gains shrink as the backbone grows (+22.6% at 3B → +6.1% at 14B). **A method whose advantage holds at 14B+
is a strong story.**

**NarrativeQA / HotpotQA F1** — Table 2 (HotpotQA at 56K / 224K / 448K context = the eval_400/1600/3200 splits):

| Backbone | NarrativeQA | HotpotQA 56K / 224K / 448K |
|---|---|---|
| Qwen2.5-3B GAM | 20.19 | 36.54 / 31.59 / 28.05 |
| Qwen2.5-3B R²-Mem | **23.33** | **41.05 / 33.98 / 29.94** |
| Qwen2.5-7B GAM | 28.56 | 44.47 / 46.63 / 44.61 |
| Qwen2.5-7B R²-Mem | **31.52** | **51.53 / 47.70 / 44.69** |

Note the 7B/448K column: 44.61 → 44.69 is **+0.08 F1, i.e. noise**. R²-Mem essentially fails at the longest
context. Also worth knowing: the paper reports **single runs, no seeds or variance anywhere** — so we
should report std over seeds, which is both better science and a fair way to contextualize their margins.

## Attack surface

Observations from reading the paper *and* the released code — these are our own analysis, not the paper's
admissions. Not yet validated; verify before building a paper on any one of them.

1. **Half the supervision is thrown away.** Steps scoring between `K_low=5` and `K_high=10` are `continue`d
   over in `exp/self_reflection.py`. Nothing learns from the middle of the distribution.
2. **The bank is write-once and never maintained.** Built offline, then read-only. No dedup, no
   consolidation of near-identical experiences, no aging/forgetting, no update from online outcomes. It
   grows monotonically and retrieval quality must degrade with size — the paper only tests small banks
   (10–20% of a dataset).
3. **Retrieval is a flat `IndexFlatIP` with a fixed Top-K=3** over `(situation)condition` BGE-M3 embeddings.
   No adaptive K, no relevance thresholding, no applicability check — and since experiences are IF-THEN
   rules, two retrieved rules can carry contradictory THEN-branches and both get injected. Note that Top-K
   is *effectively unablated*: §5.5/Fig. 5 call the swept axis "retrieval size k" while App. C.3/Table 9
   call the same axis an "exponential weighting factor," so it is unclear what was actually varied.
4. **Transfer is barely tested.** Banks are keyed by dataset *and* model size (`Planning_{model_size}_*`).
   App. B.1 does check robustness to the *source conversation* within LoCoMo (Conv-26/41/47/49, comparable
   trends), but cross-dataset and cross-backbone reuse are untested — the natural reviewer question.
5. **The "low-cost" claim leans on GPT-4o.** The Evaluator is an external stronger model; the self-evolving
   variant (Learner judges itself) loses points, most on larger backbones (§5.6). An approach needing *no*
   external judge and matching these numbers is a clean contribution.
6. **Token savings come from fewer iterations, not cheaper steps** — injected experience makes each prompt
   *longer*. At 14B the iteration count barely drops (1.55 → 1.47) and so the token win nearly vanishes.
7. **Ablation says low-quality experience > high-quality experience** (Table 4: only-low 39.33 vs. only-high
   37.36 multi-hop F1). The paper notes this but doesn't exploit it. There may be a much better use of
   failure signal than symmetric distillation into two banks.
8. **The LoCoMo protocol is a replay.** `exp/eval/locomo.py` reads questions out of the GAM run's
   `qa_results.json` and re-runs the search loop — memory construction is shared with the baseline. Good
   for isolating the search-loop contribution, but it means R²-Mem never demonstrates end-to-end gains
   including memory building. Match this protocol when comparing, and say so explicitly in our paper.

## Working agreements

- **Reproduce before improving.** No claim of beating R²-Mem is meaningful until their numbers are
  reproduced locally on at least one backbone. Expect this to be the long pole: the released code needs
  local Qwen2.5 + BGE-M3 weights, a vLLM server, GPT-4o API access, and ~a dozen hardcoded paths edited
  (see `R2M-official/CLAUDE.md`).
- **Report what actually ran.** Failed runs, skipped configs, and OOMs go in the notes, not the memory hole.
  The baseline's own single-run numbers are the reason we should be more careful, not less.
- **Keep the comparison honest**: same backbone, same retriever, same `max_iters`, same experience budget
  (10% LoCoMo / 20% HotpotQA+NarrativeQA), same eval script. Efficiency numbers (tokens, iterations) must
  be reported alongside F1 — dropping them would be conceding their headline claim.
