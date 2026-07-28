# Baseline critique: GAM and R²-Mem

Written 2026-07-26 after reading both papers end to end (`papers/GAM-2511.18423v1.pdf`,
`papers/R2-Mem-2605.13486v1.pdf`) and the R²-Mem released code (`R2M-official/`).

Claims here are marked as **[paper]** (stated by the authors), **[code]** (verified in the released
implementation), or **[ours]** (our own analysis, not yet validated).

---

## 1. What each paper actually does

### GAM (Yan et al., BAAI, arXiv:2511.18423, Nov 2025)

Thesis: memory systems built **Ahead-of-Time** (compress history into a structure, then serve queries from
it) suffer irreversible information loss. GAM instead does **Just-in-Time compilation** — keep the complete
history in a page-store, build only a light memory offline, and spend real compute *at query time*.

- **Memorizer** (offline): per session, `memorize` produces a memo appended to a running memory, and `page`
  writes a header-decorated page into the page-store. Same idea as BGE landmark / Anthropic contextual
  retrieval.
- **Researcher** (online): iterative Planning → Searching → Integration → Reflection. Three tools: BM25,
  dense (BGE-M3), page-id. Defaults: reflection depth 3, top-5 pages, 2048-token pages.
- Evaluated on LoCoMo, HotpotQA (56K/224K/448K), **RULER (128K)**, NarrativeQA. Backbones GPT-4o-mini and
  Qwen2.5-14B.

Three GAM findings that matter for us:

1. **The Researcher is the bottleneck, the Memorizer is nearly free.** [paper, Table 2] Shrinking the
   memorizer 14B → 0.5B costs ~4 avg F1 (53.18 → 48.83). Shrinking the *researcher* 14B → 0.5B costs ~44
   (53.18 → 9.08). Anything that improves researcher behavior is attacking the right module — which is what
   R²-Mem does.
2. **GAM at default settings is compute-starved.** [paper, Fig. 2] F1 rises *monotonically* with reflection
   depth 1→5 and with retrieved pages 3→20, on every dataset. Defaults are depth 3 / 5 pages. So there is
   headroom on the GAM side that nobody has spent.
3. **§2.2.3 proposes end-to-end RL for GAM (Eq. 6–7) but no RL experiment is ever run.** [ours, by
   omission] The abstract sells "facilitating end-to-end performance optimization through reinforcement
   learning"; §3 evaluates only off-the-shelf backbones. GAM's optimization story is unimplemented.

### R²-Mem (Wang et al., USTC, arXiv:2605.13486, May 2026)

Sits on top of GAM. Adds an offline loop that mines GAM's own trajectories:

- **Rubric-guided Evaluator** (GPT-4o): scores every step, 4 dimensions × 0–3, separately for Planning
  (Info Needs Coverage / Non-Redundancy / Tool–Info Alignment / Planning Efficiency) and Reflection
  (Sufficiency Judgment Accuracy / Minimal Sufficiency Recognition / Follow-up Query Quality / Answer
  Completeness Awareness). Emits `(score, reason, advice)`.
- **self-Reflection Learner**: for steps with total ≥ K_high=10 or ≤ K_low=5, distils an
  **`IF <abstract situation> THEN <strategy>`** rule [paper, App. D.2] with fields
  `{thinking, summary, situation, experience}`. Mid-range steps are discarded.
- **Two banks** (Planning 𝒟^P, Reflection 𝒟^R), FAISS `IndexFlatIP` over BGE-M3 embeddings of
  `"(situation)" + condition`. Condition is `q_i` for planning, `[Q + m_i]` for reflection.
- **Online**: abstract the current step into a situation, retrieve Top-3, inject into the prompt.
- Note the framing in App. F.1: R²-Mem positions itself *against* skill libraries — its experiences are
  "not executable routines" but step-level reflective suggestions.

Headline: +22.6% F1, −12.9% tokens, −20.2% iterations. Datasets LoCoMo / HotpotQA / NarrativeQA.
The PDF's App. B links a `NeurIPS-2026-code` GitHub org — **this was a NeurIPS 2026 submission**, so a
camera-ready or a revised version may appear. Worth monitoring.

---

## 2. Strengths we should not pretend away

Reviewers will know these. Our paper has to concede them and still win.

- **Right target.** Per GAM Table 2, the researcher's planning/reflection is where capacity matters, and
  that is exactly what R²-Mem improves. Not a coincidence — a well-chosen intervention point.
- **Genuinely Pareto-better than GAM**, not a quality/cost trade: at Qwen2.5-7B on LoCoMo it is
  +5.6 F1 *and* −15% tokens *and* −17% iterations simultaneously. [paper, Table 3]
- **Self-evolution works.** [paper, Table 10] Replacing GPT-4o with the 7B Learner itself still beats GAM
  at every scale (3B 32.00→38.09, 7B 45.77→49.76, 14B 52.99→54.93), landing within 0.4–1.6 F1 of the
  GPT-4o-judged version. The method is not parasitic on a frontier model.
- **The low-quality > high-quality finding is real and interesting.** [paper, Table 4] Learning only from
  failures (39.33 multi-hop F1) beats learning only from successes (37.36). Under-exploited by them.
- **The cost accounting is honest.** [paper, App. C.1] They report accumulation cost explicitly
  (Conv-26: 0.29–0.36M Evaluator + 0.24–0.25M Learner tokens) and exclude Conv-26 from the averages.
- **Source-conversation robustness was checked.** [paper, App. B.1] They repeated accumulation on Conv-26,
  41, 47, 49 and saw comparable trends. (Not full transfer — see below — but not zero either.)

---

## 3. Weaknesses — the attack surface

### 3.1 Reporting problems (verifiable now, no GPUs needed)

**The default configuration is reported with three different Overall F1 values.** [ours, from paper tables]
Qwen2.5-7B, `(K_low,K_high)=(5,10)`, `exp=3` is the default everywhere, yet:

| Source | Multi-hop F1 | Temporal F1 | Open F1 | Single F1 | **Overall F1 / BLEU** | Tokens |
|---|---|---|---|---|---|---|
| Table 1 / Table 3 | 41.62 | 46.76 | 25.06 | 59.03 | **51.35 / 44.91** | 28.93 |
| Table 8, row (5,10) | 41.57 | 46.76 | 25.26 | 59.34 | **51.51 / 44.49** | 28.93 |
| Table 9, row exp=3 | 41.57 | 46.76 | 25.26 | 59.34 | **50.58 / 44.29** | 27.60 |

Rows 2 and 3 have **identical** per-category F1 on all four categories but Overall differing by 0.93, and
identical tokens are claimed in rows 1–2 but not 3. An aggregate cannot differ when all its components are
equal — so at least one is a transcription error. Either way: **the paper's own default config spans
~0.9 Overall F1 and ~2.8 Temporal BLEU across tables.** Several of their reported gains are smaller than
that spread (e.g. Open-domain 7B: GAM 23.82 → R²-Mem 25.06 = +1.24).

**No run-to-run variance is reported anywhere.** Seed 42 is used for *dataset sampling* [paper, App. B.1],
not for repeated runs. Combined with the above, the small-margin cells are unsupported.

**The retrieval-size ablation is mislabelled and may not exist.** §5.5 and Fig. 5 describe "the retrieval
size k for experience selection." App. C.3 / Table 9 describe the *same* axis (labelled "Exp", values 1–5)
as an "exponential weighting factor" controlling score separation among trajectories. These are different
quantities. So **Top-K=3 is effectively unablated** — which matters, because retrieval is the weakest link
(§3.3).

**RULER is dropped.** GAM evaluates on RULER-128K and posts its most dramatic wins there (multi-hop tracing
90–93% acc vs. 0.00 for RAG and A-Mem). R²-Mem silently omits it. Unexplained, and RULER's MT/AGG tasks are
exactly where planning quality should matter most.

**HotpotQA 7B/448K is a null result presented as a win.** 44.61 → 44.69 = +0.08 F1. [paper, Table 2] The
method does essentially nothing at the longest context, and the paper does not remark on it.

### 3.2 Method: the supervision signal

- **The middle of the distribution is thrown away.** [code, `exp/self_reflection.py`] Steps scoring 6–9 of
  12 hit `continue`. With 4 dims × 0–3 the mass is concentrated there, so most steps teach nothing.
- **The score is an unweighted sum, so failures get blended away.** A step scoring (3,3,0,3)=9 — a total
  breakdown in one dimension — is discarded as "average," identically to (2,2,2,3)=9. **Rubric dimensions
  are collected and then immediately destroyed by summing.** The per-dimension diagnosis, which is the whole
  point of using rubrics, never reaches the Learner. [ours]
- **The rubric is frozen and hand-written.** ARCO (arXiv:2606.21262, Jun 2026) shows step-specific rubrics
  that *co-evolve* with the agent beat static closed-judge rubrics — on HotpotQA/2Wiki/MuSiQue, though not
  for memory. Being static is now a known-inferior design point.
- **The "low-cost" claim mixes currencies.** [ours] App. C.1 adds 0.29M GPT-4o Evaluator tokens to 0.25M
  local-Qwen Learner tokens and compares the sum against Qwen inference tokens as if fungible. At list
  prices GPT-4o is orders of magnitude more expensive per token than self-hosted Qwen-7B. The Self-Evo
  variant (Table 10) is the honest costing, and it is 1.6 F1 worse at 7B.
- **Efficiency nearly vanishes at scale.** Token reduction is −12.9% (3B), −14.9% (7B), **−3.9% (14B)**
  [paper, Table 7], against a one-time 0.53M accumulation cost on a 3.08M/conversation baseline — so at 14B
  the method needs ~4+ conversations just to break even.

### 3.3 Method: the bank and its retrieval

- **Write-once, never maintained.** [code] No dedup, no merge, no decay, no eviction, no re-scoring. The
  bank only grows. Tested only at 10–20% of one dataset. The 2026 consolidation literature (redundancy
  elimination, temporal decay, importance-driven forgetting) is entirely unapplied here.
- **No feedback loop.** [paper, App. F.3 — they admit distribution shift is unhandled] An experience is
  never checked against whether it actually helped. The pipeline is a triple indirection —
  *judge's opinion of a step* → *text rule* → *cosine similarity* → *hoped-for behaviour change* — and
  **not one of the three links is verified.** This is, in our view, the central flaw. [ours]
- **Retrieval is similarity-only.** Flat `IndexFlatIP`, fixed Top-3, no threshold, no applicability check.
  Experiences are IF-THEN rules, so two retrieved rules can carry **contradictory THEN-branches** for the
  same situation and both get injected. Nothing detects this. [code, ours]
- **No transfer demonstrated across datasets or backbones.** Banks are keyed `Planning_{model_size}_*`
  [code]. App. B.1's robustness check is *within* LoCoMo (different source conversations), which is a much
  weaker claim than cross-dataset or cross-model reuse.
- **Injected experience lengthens every prompt.** Savings come only from fewer iterations, which is why the
  gain collapses at 14B where iterations barely drop (1.55 → 1.47).

### 3.4 Protocol

- **LoCoMo evaluation is a replay over GAM's outputs.** [code, `exp/eval/locomo.py`] It reads questions from
  the GAM run's `qa_results.json` and re-runs only the search loop; memory construction is shared. Clean for
  isolating the contribution, but R²-Mem never shows an end-to-end gain including memory building. We should
  match this protocol and *say so*.
- **LoCoMo itself is shaky.** An independent audit found **6.4% of the answer key is wrong** (99/1540:
  hallucinated facts, bad temporal reasoning, 24 speaker-attribution errors), and the standard LLM judge
  accepts **62.8%** of deliberately-wrong-but-topically-adjacent answers. Conversations average only
  16–26K tokens — inside every modern context window. GAM/R²-Mem use word-overlap F1/BLEU-1 rather than the
  LLM judge, so the judge critique doesn't bite them directly, but the wrong answer keys do.
- **Beware cross-protocol number confusion.** Industry systems (Mem0 92.5, ByteRover 92.2) report *LLM-judge
  accuracy* on LoCoMo. GAM/R²-Mem report *token-overlap F1* (~30–60). These are not comparable and must
  never be put in the same table.

---

## 4. Competitive landscape (what already exists — do not re-invent)

| Work | What it does | Overlap risk for us |
|---|---|---|
| **ReasoningBank** (ICLR 2026, Google, 2509.25140) | Distils strategies from success *and* failure; **writes back online at test time**; + MaTTS memory-aware test-time scaling. Web agents (WebArena, Mind2Web) + SWE-Bench. | **High.** "Add online write-back" alone is taken. Ours must be step-level and utility-verified, in the memory-search setting they don't touch. |
| **ARCO** (2606.21262, Jun 2026) | Step-specific rubrics that co-evolve with the agent during RL; same-scale scorer, not a frozen closed judge. HotpotQA/2Wiki/MuSiQue. | **High for the rubric axis.** "Make the rubric adaptive" is no longer novel on its own. |
| **Evo-Memory** (2511.20857) | Streaming benchmark for self-evolving memory + ReMem (act–think–memory-refine). | Medium. Possible evaluation venue for us. |
| **Latent Experiential Memories** (2606.17803, Jun 2026) | Distils inference-time compute into soft-prompt latent memories (~0.001% params) instead of text. | Medium — an orthogonal answer to the prompt-length problem. |
| **Memp** (2508.06433) | Agent procedural memory. | Low. |
| **Locomo-Plus** (2602.10715) | Corrected/extended LoCoMo. | None — useful to us. |
| Mem0 / A-Mem / MemoryOS / LightMem / Memory-R1 / MemAgent | R²-Mem's baseline table. | None. |

Two axes are now crowded: *adaptive rubrics* (ARCO) and *online experience write-back* (ReasoningBank).
Whatever we propose must not be reducible to either.

---

## 5. Open gaps

1. Nobody has made experience selection depend on **realized utility** rather than judged quality or
   embedding similarity.
2. Nobody maintains the bank (dedup / decay / eviction) in the memory-search setting.
3. Nobody uses the **per-dimension** rubric signal — everyone sums it to a scalar first.
4. Nobody handles **conflicting** retrieved rules.
5. The **iso-compute frontier** for deep memory search is uncharted: GAM Fig. 2 says depth/breadth buy F1
   monotonically, and R²-Mem reduces depth. The two have never been compared at matched token budget.
6. Cross-dataset / cross-backbone **experience transfer** is untested.
7. RULER + variance reporting are missing from R²-Mem.
