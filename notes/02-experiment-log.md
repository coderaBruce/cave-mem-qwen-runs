# Experiment log

Append-only. Every configuration we run goes here, **including the ones that fail
or lose to baseline** — per the instruction to record failed attempts. Negative
results are paper material (the ablation/analysis section), not waste.

Format: date, what, why, result, verdict.

Harness: `r2m-api/` (API-only, gpt-4o-mini backbone, `text-embedding-3-small`).
See `r2m-api/README.md` for the list of deviations from the published setup.

---

## 2026-07-26

### E0 — Harness smoke test
**What:** GAM, LoCoMo conv-26, first 5 questions, gpt-4o-mini, temp 0.3, 12 workers.
**Why:** Verify the API-backed pipeline runs end to end and produces upstream-format traces.
**Result:** 5/5 answered, 0 errors, 19 pages built. Overall F1 73.08 / BLEU-1 68.67
(multi_hop 100.0 n=2, temporal 75.0 n=2, open_domain 15.4 n=1). 128,423 tokens billed.
**Verdict:** PASS. Not a meaningful score (n=5), only a wiring check.

### E1 — GAM baseline, LoCoMo full  *(running)*
**What:** GAM, all 10 conversations, 1540 questions, gpt-4o-mini.
**Why:** **The validation gate.** GAM's own paper reports GPT-4o-mini on LoCoMo, so this
is the one number we can check against published ground truth. Everything downstream is
meaningless if this doesn't land near it.
**Target (GAM paper, Table 1a, GPT-4o-mini):**

| | Single-hop | Multi-hop | Temporal | Open-domain |
|---|---|---|---|---|
| F1 | 57.75 | 42.29 | 59.45 | 33.30 |
| BLEU-1 | 52.10 | 34.44 | 53.11 | 26.97 |

**Result:** _pending_
**Verdict:** _pending_

### E2 — R²-Mem experience bank from conv-26  *(running)*
**What:** Evaluator (gpt-4o, temp 0.2, upstream prompt verbatim) + Learner (gpt-4o-mini,
temp 0.5) over all 152 conv-26 traces. K_low=5, K_high=10.
**Why:** Reproduce the paper's offline stage. Conv-26 is upstream's accumulation
conversation (App. B.1); the other nine are held out for evaluation.
**Result:** _pending_

**Pilot on 5 traces** (before committing to the full set): 16 steps scored →
9 kept (5 good, 4 bad), **7 discarded as mid-range**. Score histogram
`{4:2, 5:2, 6:2, 8:5, 12:5}`.
**This is direct evidence for critique §3.2: 44% of scored steps were thrown away.**
Also notable — the distribution is bimodal at 8 and 12, so K_high=10 cleanly separates
but K_low=5 catches only the tail.

---

## Planned

- **E3** — R²-Mem online on the 9 held-out conversations, vs. E1 on the same 9.
  This is the GAM → R²-Mem delta, the relationship we most need to reproduce.
- **E4** — `--keep-middle`: distil mid-scoring steps too (idea 2). Cheap: the
  Evaluator is cached, so only Learner calls are re-paid.
- **E5** — utility-verified experience (idea 1): A/B replay each candidate with and
  without injection, keep by measured improvement rather than judged score.
- **E6** — conflict detection among retrieved Top-K IF-THEN rules.
- **E7** — HotpotQA eval_400 / NarrativeQA-300, GAM vs R²-Mem vs ours.
- **E8** — variance: repeat the default config 3x, report std. Upstream reports none,
  and its own tables disagree by ~0.9 Overall F1 (see critique §3.1).

## Known deviations that could move numbers vs. the papers

1. `text-embedding-3-small` instead of BGE-M3 (different space, 1536 vs 1024-d).
2. Pure-Python Okapi BM25 instead of pyserini/Lucene (same k1/b, different analyzer).
3. gpt-4o-mini backbone — matches GAM's reported setting, but R²-Mem only ever ran
   Qwen2.5-3B/7B/14B and Llama3.1-8B, so **their absolute numbers are not reachable**;
   we reproduce the mechanism and the delta.
4. Upstream runs at temperature 0.3 with no seed, so exact reproduction is impossible
   in principle.

---

### E1 (partial) — GAM baseline validation gate, 3 of 10 conversations
**What:** conv-26 / conv-30 / conv-41, 385 questions, 0 errors, gpt-4o-mini, temp 0.3.
Scored while the remaining 7 conversations run.

| category | ours F1 | GAM paper F1 | diff | ours BLEU-1 | paper BLEU-1 | n |
|---|---|---|---|---|---|---|
| multi_hop | 52.68 | 42.29 | **+10.39** | 45.28 | 34.44 | 74 |
| temporal | 70.41 | 59.45 | **+10.96** | 62.71 | 53.11 | 90 |
| open_domain | 28.86 | 33.30 | −4.44 | 19.52 | 26.97 | 21 |
| single_hop | 55.12 | 57.75 | −2.63 | 48.63 | 52.10 | 200 |

Overall F1 **56.79**, BLEU-1 49.69, **mean iterations 1.506**.

**Verdict: PASS, with a caveat.** Single-hop (n=200, the largest category) lands within
2.6 F1 of published, and open-domain within 4.4 — close enough to trust the harness.
Multi-hop and temporal come in ~10 points *high*, but this is a 3-conversation subset
vs. their 10-conversation average and LoCoMo's category mix varies a lot per
conversation (their own Table 5), so it is not yet a like-for-like comparison. Re-score
on all 10 when E1 finishes before drawing any conclusion.

**⚠ The important finding here is the iteration count.**
Our GAM baseline already runs at **1.506 mean iterations**. R²-Mem reports GAM at
Qwen2.5-7B taking **1.84** and their method reducing it to **1.52** — i.e. *our
unmodified GAM baseline is already as "efficient" as their full method.*

If this holds on all 10 conversations, R²-Mem's headline efficiency claim (−20.2%
iterations) is substantially an artefact of a weak backbone over-searching, not a
property of the method. A stronger backbone already stops early, leaving nothing to
reclaim. This is consistent with their own scaling data — token reduction collapses
from −12.9% at 3B to −3.9% at 14B (their Table 7) — and it predicts the effect goes to
roughly zero at gpt-4o-mini class. **This is a headline-grade result for our paper and
needs to be nailed down carefully: all 10 conversations, multiple seeds.**

Corollary for our own method: we should not compete on iteration reduction. The room
is in *quality*, and in efficiency only at small scale.

### E2b — Retrieval redundancy, observed directly
Querying the 9-entry pilot bank with "When did Melanie run a charity race?" returns for
the Reflection module:

```
sim=0.412 [bad] IF the question seeks specific details about an event and the temp_memory contains partial information about t...
sim=0.396 [bad] IF a query seeks specific timing information about an event and the temp_memory provides partial timing detail...
sim=0.361 [bad] IF the inquiry seeks specific temporal details about an event and the current memory contains partial informat...
```

Three near-paraphrases of a single rule. **With Top-K=3 the agent receives one distinct
piece of advice occupying three slots.** This is critique §3.3 (no dedup, no
consolidation) demonstrated on real output, from a bank of only 9 entries — the effect
can only worsen as the bank grows, and upstream never tests a large bank.

Cheap, concrete win available here: dedup/consolidate at write time, or enforce
diversity at read time (e.g. MMR). Should be an easy ablation row.

### E2 — R²-Mem experience bank from conv-26 (COMPLETE)
**What:** Evaluator gpt-4o (temp 0.2, upstream prompt verbatim) + Learner gpt-4o-mini
(temp 0.5) over all 152 conv-26 traces. K_low=5, K_high=10.
**Cost:** 322,386 Evaluator tokens (gpt-4o) + 274,002 Learner tokens. 17 min wall.
**Result:** **275 experiences** — Planning/good 68, Planning/bad 18, Reflection/good 114,
Reflection/bad 75.

Step score histogram over **470 scored steps**:

```
score  0   1   3   4   5  |  6   7   8   9  | 10  11   12
n      2   1   7  31  52  | 70  28  69  28  | 21  13  148
       <---- bad: 93 ---->   <-- 195 dropped -->  <-- good: 182 -->
```

**Two findings worth putting in the paper:**

1. **41.5% of scored steps (195/470) are discarded** by the K_low/K_high filter. Confirms
   critique §3.2 at scale, not just in the pilot.
2. **31% of all steps (148/470) score a perfect 12/12.** The rubric saturates badly at
   the top — the "high-quality" half of the bank is dominated by steps the Evaluator
   could not tell apart. So the `good` experiences are distilled from an essentially
   unranked mass, which undercuts the premise that rubric scores identify *exemplary*
   behaviour. It also explains the paper's own ablation result that low-quality
   experience is more useful than high-quality (their Table 4): the low tail is
   discriminative, the high tail is not.

This second point is a *new* argument for why their design underperforms, and it is
measured rather than asserted.

### E3 — R²-Mem vs GAM on held-out conversations  *(running)*
conv-30 + conv-41 (233 questions), same backbone, bank from conv-26 only, Top-K=3.
GAM numbers for the same two conversations come from E1.

### E3 — R²-Mem vs GAM, held-out conversations (COMPLETE)
**Setup:** conv-30 + conv-41, 233 questions, 0 errors both runs. Bank from conv-26 only
(no leakage). gpt-4o-mini, temp 0.3, Top-K=3, K=(5,10). Single run each.

| category | n | GAM F1 | R²-Mem F1 | Δ | GAM BLEU | R²-Mem BLEU |
|---|---|---|---|---|---|---|
| multi_hop | 42 | 54.29 | **59.46** | **+5.17** | 44.69 | 49.84 |
| temporal | 53 | 75.26 | 73.52 | −1.74 | 68.45 | 66.72 |
| open_domain | 8 | 42.52 | 30.57 | −11.95 | 37.62 | 27.81 |
| single_hop | 130 | 57.76 | 55.89 | −1.87 | 51.40 | 49.76 |
| **OVERALL** | 233 | **60.59** | 59.67 | **−0.92** | 53.60 | 52.88 |

Mean iterations: GAM 1.481 → R²-Mem **1.330** (−10.2%).

**Verdict: R²-Mem FAILS TO REPRODUCE as an improvement on this backbone.**
It is −0.92 F1 overall (−1.5% relative) against the baseline it is supposed to beat by
+22.6%.

**Read the categories, though — the result is not "the method doesn't work":**

- **Planning experience appears to help.** multi_hop +5.17 F1 (+9.5%) on n=42 is the
  largest single movement and in the right direction. Multi-hop is exactly what better
  query decomposition should improve.
- **Reflection experience appears to hurt.** Iterations drop 1.481 → 1.330 and the
  categories that need *more* search — single_hop retrieval of a specific fact,
  open_domain — both fall. The agent is being talked into stopping early.
- **Discount open_domain.** n=8. The −11.95 is ~noise and must not be quoted.
  The trustworthy cells are single_hop (n=130) and multi_hop (n=42).

**Mechanism, consistent with E1:** at Qwen2.5-7B GAM over-searches (1.84 iterations), so
R²-Mem's stop-earlier pressure is corrective and nets a gain. At gpt-4o-mini GAM already
stops at 1.48, so the same pressure is *over*-correction and costs accuracy. R²-Mem's
benefit is a function of how badly the backbone over-searches — which their own scaling
table shows decaying (−12.9% tokens at 3B → −3.9% at 14B) and which we now observe going
negative.

**Caveats (must not be dropped when writing this up):** 2 conversations, single run, no
seeds, different embedder and BM25 from the original, and a backbone the R²-Mem authors
never tested. This is evidence about *backbone sensitivity*, not proof their Qwen numbers
are wrong.

**Cost:** R²-Mem 4,523,260 tokens / 233 questions = 19,413 per question.

**Next, directly implied by this:** ablate Planning-only vs Reflection-only experience.
If planning-only beats both GAM and full R²-Mem, that is a clean, well-motivated
contribution on its own — and it inverts the paper's own ablation (their Table 4 found
removing either hurt).

### E1 — GAM baseline, LoCoMo FULL 10 conversations (COMPLETE) ✅
1540 questions, **0 errors**, 32 min wall, 30,990,312 tokens (20,123/question).

| category | n | ours F1 | GAM paper F1 | diff | ours BLEU | paper BLEU |
|---|---|---|---|---|---|---|
| multi_hop | 282 | 43.25 | 42.29 | **+0.96** | 35.07 | 34.44 |
| temporal | 321 | 61.26 | 59.45 | **+1.81** | 54.90 | 53.11 |
| open_domain | 96 | 29.24 | 33.30 | −4.06 | 22.41 | 26.97 |
| single_hop | 841 | 57.94 | 57.75 | **+0.19** | 52.02 | 52.10 |
| OVERALL | 1540 | 54.16 | — | | 47.67 | |

**VERDICT: VALIDATION GATE PASSED.** Three of four categories land within 2 F1 of the
published GPT-4o-mini numbers — single_hop within 0.19 on n=841. Only open_domain is off
(−4.06, n=96, the smallest category). Given we substituted the embedder
(text-embedding-3-small for BGE-M3) and the BM25 implementation, this is about as close
as reproduction gets. **The harness is trustworthy.**

Note the earlier 3-conversation reading (multi_hop +10.39, temporal +10.96) was indeed a
subset artefact, as flagged at the time — it vanishes on the full set. Worth remembering
as a discipline: never quote LoCoMo numbers from a conversation subset.

**Mean iterations across all 10 conversations: 1.477.** R²-Mem reports GAM at Qwen2.5-7B
taking **1.84** and their own method reaching **1.52**. So the finding from the partial
run holds on the full set: **our unmodified GAM baseline is already more "efficient" than
R²-Mem's headline efficiency result.** There is no over-searching left for their method
to reclaim on this backbone.

### E4 — Component ablation on held-out conv-30 + conv-41 (233 questions)

| arm | multi_hop | temporal | open_dom | single_hop | **OVERALL** | iters |
|---|---|---|---|---|---|---|
| GAM (no experience) | 54.29 | 75.26 | 42.52 | 57.76 | **60.59** | 1.48 |
| R²-Mem full | 59.46 | 73.52 | 30.57 | 55.89 | 59.67 | 1.33 |
| Planning experience only | 57.87 | 71.45 | 30.57 | 56.42 | 59.21 | 1.28 |
| Reflection experience only | 51.20 | 67.53 | 39.37 | 51.52 | **54.69** | 1.34 |

**My E3 hypothesis was WRONG and is hereby retracted.** I predicted planning experience
helps and reflection experience hurts, and that planning-only would therefore beat both
GAM and full R²-Mem. It does not: planning-only scores 59.21, *below* full R²-Mem (59.67)
and below GAM (60.59). Reflection-only is far worse still (54.69).

What the ablation actually shows:
- The two experience types are **complementary**, not opposed — full > planning-only >
  reflection-only. This *agrees* with the paper's own Table 4 (removing either hurts).
- **Every experience configuration reduces iterations** (1.48 → 1.28–1.34), including
  planning-only, which has no direct say in stopping. So the mechanism is not "reflection
  advice says stop early"; it is that injected experience makes the agent *more decisive
  overall* — narrower plans produce a tidier temp_memory, which the sufficiency check
  then accepts sooner.
- On a backbone that was not over-searching, decisiveness is not a virtue. Every arm
  loses to plain GAM.

**Standing caveats:** 2 conversations, single run, no seeds, open_domain n=8 (its swings
are noise and must not be quoted).

---

### E5 — OURS v1: outcome-verified experience selection — **FAILED**
Measured each experience's utility as downstream F1 delta on the accumulation
conversation (conv-26, 152 questions), then kept only u(e) > 0.

Utility measurement worked and produced a real signal:
- question-level delta (R²-Mem − GAM) on conv-26: mean **−0.0031**, help 27 / hurt 24 / tie 101
- 274/275 experiences measured; **only 93 (34%) had positive utility**
- utility range −1.000 … +0.595, median +0.000

**Bonus finding — the rubric label barely predicts usefulness, and points the wrong way:**

| Evaluator label | n | positive measured utility |
|---|---|---|
| good (score ≥10) | 181 | 54 (**29.8%**) |
| bad (score ≤5) | 93 | 39 (**41.9%**) |

Experiences the GPT-4o rubric scored *high* are useful less often than ones it scored
*low*. This is an independent, mechanism-level explanation of the paper's own puzzling
Table 4 result (only-low 39.33 > only-high 37.36 multi-hop F1), and it undercuts the
premise that rubric scores identify reusable expertise.

**Result on held-out conv-30+41: 58.06 F1** — below R²-Mem (59.67) and GAM (60.59).
**Verdict: FAILED.** Likely causes: the signal is sparse (101/152 questions tie, so most
entries get u=0 and are pruned), pruning to 93 entries costs retrieval coverage, and
utility measured on conv-26 need not transfer (R²-Mem's own admitted limitation F.3).

### E6 — CONTROL: empty bank with R²-Mem's templates — **the key experiment**
Ran their online agent with **zero experiences**, everything else identical.

| variant | prompts | experience | F1 | iters |
|---|---|---|---|---|
| GAM | GAM's | none | **60.59** | 1.48 |
| ctrl-emptybank | R²-Mem `*_PROMPT_exp` | **none** | **54.40** | 1.32 |
| R²-Mem full | R²-Mem `*_PROMPT_exp` | 275 | 59.67 | 1.33 |
| ours-additive | GAM's + appended block | 275 | 57.41 | 1.49 |

**The template rewrite alone costs −6.19 F1.** The experience content then buys back
+5.27. R²-Mem's net −0.92 is the sum of a large self-inflicted wound and a large
partial repair.

**R²-Mem never runs this control.** Their Table 4 ablations ("w/o Planning",
"w/o Reflection", "w/o Learner") all keep the `*_PROMPT_exp` templates, so every
comparison is against a degraded baseline. Template effect and experience effect are
never separated.

**Diffing the templates shows the mechanism.** GAM's `InfoCheck_PROMPT` says:

> 1. Decompose REQUEST … 2. For each required piece, check RESULT provides it clearly
> and specifically. RESULT must be specific enough that someone could write a final
> answer directly from it … 3. "enough" = true ONLY IF RESULT covers all required
> pieces with sufficient clarity and specificity.

`InfoCheck_PROMPT_exp` **deletes all three steps** and substitutes "STEP 1: Analyze
Experience Applicability". The strict sufficiency bar is gone. That is the
premature-stopping mechanism — and it is present with an empty bank (iterations
1.48 → 1.32 with no experience at all), so it is the template, not the advice.

### E7 — OURS v2: additive injection (GAM prompts + appended experience) — **FAILED**
Keeps every GAM instruction verbatim, appends an advisory experience block before the
output spec. Splice placement verified correct (81% through the planning prompt, after
the procedure, before the JSON contract).

**Result: 57.41 F1**, iterations 1.49. Stopping behaviour is repaired (1.49 ≈ GAM 1.48),
but F1 is 3.18 **below** GAM.

**Verdict: FAILED, and it inverts the interpretation.**
- Under R²-Mem's degraded template, experience **helps** (+5.27).
- Under GAM's intact template, the same experiences **hurt** (−3.18).

**The experience bank is a crutch.** It encodes "how to plan and reflect competently",
which a damaged prompt needs and a good prompt already contains. Its measured value is
largely an artefact of the template it ships with. This is a much stronger claim than
"R²-Mem doesn't reproduce", and every number supporting it is a controlled comparison
on identical questions.

### E8 — Error analysis on the GAM baseline (conv-30+41, 233 questions)
**15% of answers score exactly 0 token-F1** (34/233); 58 score below 0.3.
Zero-F1 by category: single_hop **24**, multi_hop 6, temporal 2, open_domain 2.
So the failures concentrate in the *largest and supposedly easiest* category.

Inspecting them, retrieval is not the problem:

| gold | prediction |
|---|---|
| `Glad` | `positive attitude` |
| `They look graceful` | `a means of self-expression and happiness` |
| `"Finding Freedom"` | `contemporary dance piece` |
| `February, 2023` | `next month` |

Three distinct failure modes, none of them retrieval: **elaborating where the gold is
terse**, **describing where the gold names**, and **leaving relative time unresolved**.

Root cause found by inspecting the trace: GAM's `_integrate` LLM-summarises retrieved
hits into `Result.content`, and **only that paraphrase reaches the answer step.** Source
page ids are recorded in `sources` but their text is never passed. Exact surface forms
("Finding Freedom") are destroyed during integration and cannot be recovered.

### E9 — OURS v4: evidence-preserving answering — **BEATS BOTH BASELINES** ✅
Pass the cited source pages' verbatim text to the answer step alongside the integrated
summary. Supported independently by GAM's own Table 4 (integration-only 53.18 →
+extraction 54.47 → +page 55.71), whose better variants they measured but did not ship.

| run | multi_hop | temporal | open_dom | single_hop | **OVERALL** | iters |
|---|---|---|---|---|---|---|
| GAM | 54.29 | **75.26** | **42.52** | 57.76 | 60.59 | 1.48 |
| R²-Mem | **59.46** | 73.52 | 30.57 | 55.89 | 59.67 | 1.33 |
| ours-evidence (4 pages) | 57.66 | 71.80 | 37.47 | **59.62** | **61.28** | 1.48 |
| ours-ev6 (6 pages) | 57.84 | 71.80 | 37.03 | 59.61 | **61.29** | 1.48 |
| ours-ev2 (2 pages) | 56.80 | 71.80 | 37.23 | 59.45 | 61.02 | 1.48 |
| ours-ev4 + style shots | 55.26 | 72.78 | 27.89 | 58.67 | 60.21 | 1.48 |

**+0.69 F1 over GAM, +1.61 over R²-Mem**, at identical iteration count (1.48). Wins on
single_hop (n=130, the largest category) and multi_hop; the gain is robust to page count
(2/4/6 all land 61.0-61.3). Combining with answer-style shots **hurts** — the two
interventions are not additive.

**Caveat, stated plainly: +0.69 on 233 questions from 2 conversations, single run, is
within plausible run-to-run noise.** We have not yet earned this claim. It needs seeds
and the full held-out set before it goes anywhere near a paper.

**Open weakness:** temporal drops 3.46 (75.26 → 71.80) consistently across every
evidence setting. Raw dialogue reintroduces relative time expressions that the
integration step had already resolved.

### E-fail log — things tried that did NOT work
| attempt | result vs GAM 60.59 | why it failed |
|---|---|---|
| R²-Mem faithful | 59.67 | template damage exceeds experience gain |
| outcome-verified pruning | 58.06 | signal too sparse (101/152 ties), lost coverage |
| similarity gate 0.60 / 0.55 | 54.90 / 54.68 | gating leaves the damaged template with no experience to repair it |
| planning-only bank | 59.21 | |
| reflection-only bank | 54.69 | |
| additive injection (GAM prompts + experience) | 57.41 | experience is redundant under a good prompt |
| answer-style shots, prompt **rewritten** | 58.93 | dropped upstream's date-format spec — same error we diagnosed in R²-Mem |
| answer-style shots, prompt **augmented** | 60.21 | recovered 1.3 F1 vs the rewrite; still net-neutral |
| GAM at max_iters=5 | 60.16 (iters 1.87) | depth is saturated on LoCoMo (16-26K contexts); GAM's Fig.2 depth gains were on long-context sets |

### E10 — OURS v5: evidence + page header — **BEST SO FAR** ✅
The temporal regression in E9 was diagnosed correctly: we were passing `page.content`
only and dropping `page.header`, which is the Memorizer's abstract and the one place
absolute dates appear ("On December 17, 2022, Maria and John reconnect…"). The raw
dialogue expresses time relatively; the header resolves it.

| run | multi_hop | temporal | open_dom | single_hop | **OVERALL** | iters |
|---|---|---|---|---|---|---|
| GAM | 54.29 | 75.26 | 42.52 | 57.76 | 60.59 | 1.48 |
| R²-Mem | 59.46 | 73.52 | 30.57 | 55.89 | 59.67 | 1.33 |
| ours, evidence only | 57.66 | 71.80 | 37.47 | 59.62 | 61.28 | 1.48 |
| **ours, evidence + header** | 57.44 | **74.19** | 38.72 | **60.47** | **62.30** | 1.48 |

Temporal recovers +2.39, single_hop rises to 60.47.
**+1.71 F1 over GAM, +2.63 over R²-Mem**, at identical iteration count and identical
search behaviour — the entire gain is on the answer side.

Still 233 questions / 2 conversations / single run. Full 9-conversation held-out run
launched to check whether it survives.

**Two implementation bugs found and fixed along the way** (both worth remembering, both
produced silently wrong experiments rather than crashes):
1. A `str.replace` patch to `run.py` silently no-op'd because the target string had
   changed in an earlier edit — the flag was accepted but never plumbed through, and the
   run returned *bit-identical* results. Identical-to-4-decimals output across every
   category is the tell that a config change did not take effect.
2. The follow-up patch matched two occurrences and injected the key into the kwargs dict
   as well as the summary dict, producing `TypeError: unexpected keyword argument`.

### E11 — Full held-out validation of the evidence method — **DOES NOT HOLD UP**
Ran the E10 winner on all **9 held-out conversations (1388 questions)**, against the
completed GAM baseline on the same conversations.

| | multi_hop | temporal | open_dom | single_hop | **OVERALL** | iters |
|---|---|---|---|---|---|---|
| GAM | 42.32 | 60.98 | 30.62 | 58.64 | 54.50 | 1.48 |
| ours (evidence+header) | 42.31 | 59.26 | 30.62 | **59.97** | 54.89 | 1.48 |

**+0.39 F1, down from +1.71 on the 2-conversation subset.** Per conversation:

| conv | GAM | ours | delta | n |
|---|---|---|---|---|
| conv-30 | 59.84 | 62.62 | +2.78 | 81 |
| conv-41 | 60.99 | 62.13 | +1.13 | 152 |
| conv-42 | 49.82 | 50.45 | +0.63 | 199 |
| conv-43 | 54.21 | 56.04 | +1.83 | 178 |
| conv-44 | 54.81 | 55.20 | +0.39 | 123 |
| conv-47 | 61.35 | 58.85 | **−2.50** | 150 |
| conv-48 | 53.80 | 55.07 | +1.27 | 191 |
| conv-49 | 48.92 | 49.49 | +0.57 | 156 |
| conv-50 | 51.38 | 49.37 | **−2.01** | 158 |

mean **+0.455**, sd **1.705**, **7/9 conversations improve**, mean/SE = **0.80**.
Sign test on 7/9 gives p≈0.18. **Not significant.**

The two conversations I happened to develop on (conv-30, conv-41) are the two largest
gains in the set. That is the same subset-artefact trap that inflated the earlier GAM
validation, and I walked into it again — this time it cost a claimed +1.71 that is really
+0.46. **Develop on one subset, confirm on another, never report the development subset.**

### E12 — HotpotQA: reproduction + our method
GAM baseline, eval_400 (56K), 128 questions, 0 errors, 12 min:
**F1 64.07** vs GAM's published GPT-4o-mini **63.22** (+0.85). Reproduction validated on
a second benchmark.

Our method: **64.66** (+0.59). Paired test over 128 questions:
`mean +0.59, sd 15.36, t=0.43, 95% bootstrap CI [−2.02, +3.30]` — **CI includes zero.**
Only **11 of 128 answers changed at all** (6 better, 5 worse, 117 identical).

### Honest status
| | LoCoMo (1388 q, 9 convs) | HotpotQA-56K (128 q) |
|---|---|---|
| GAM | 54.50 | 64.07 |
| ours | 54.89 (+0.39) | 64.66 (+0.59) |
| significant? | no (mean/SE 0.80) | no (CI [−2.02, +3.30]) |

**We do not yet have a method that beats GAM.** The consistent positive sign across two
benchmarks and 7/9 conversations is mildly encouraging, but the effect is inside the
noise floor and must not be written up as a win.

What *is* solid, and does not depend on our method working:
1. GAM reproduces on two benchmarks (LoCoMo 3/4 categories within 2 F1; HotpotQA +0.85).
2. R²-Mem loses to GAM on gpt-4o-mini (59.67 vs 60.59 on the 2-conv subset).
3. The empty-bank control isolates why: **their template rewrite costs −6.19 F1** and the
   experience content buys back +5.27. A control they never ran.
4. Their Evaluator's quality label is **anti-correlated** with measured usefulness
   (good 29.8% vs bad 41.9% positive utility).
5. 41.5% of scored steps are discarded; 31% score a perfect 12/12 (rubric saturation).

### E14 — DEFINITIVE three-way, all 9 held-out conversations (1388 questions)
R²-Mem completed on the full held-out set, so all three arms are now comparable on
identical questions with the bank built from conv-26 only.

| | multi_hop | temporal | open_dom | single_hop | **OVERALL** | iters |
|---|---|---|---|---|---|---|
| GAM | 42.32 | **60.98** | 30.62 | 58.64 | 54.50 | 1.48 |
| R²-Mem | **44.07** | 58.08 | **30.94** | 57.98 | **53.88** | 1.33 |
| ours (evidence+header) | 42.31 | 59.26 | 30.62 | **59.97** | **54.89** | 1.48 |

Per conversation, against GAM:

| conv | GAM | R²-Mem | ours | R²−GAM | ours−GAM |
|---|---|---|---|---|---|
| conv-30 | 59.84 | 56.83 | 62.62 | −3.01 | +2.78 |
| conv-41 | 60.99 | 61.19 | 62.13 | +0.19 | +1.13 |
| conv-42 | 49.82 | 49.36 | 50.45 | −0.46 | +0.63 |
| conv-43 | 54.21 | 55.34 | 56.04 | +1.12 | +1.83 |
| conv-44 | 54.81 | 56.43 | 55.20 | +1.62 | +0.39 |
| conv-47 | 61.35 | 59.39 | 58.85 | −1.96 | −2.50 |
| conv-48 | 53.80 | 50.02 | 55.07 | −3.78 | +1.27 |
| conv-49 | 48.92 | 50.60 | 49.49 | +1.67 | +0.57 |
| conv-50 | 51.38 | 50.06 | 49.37 | −1.32 | −2.01 |

| comparison | mean | sd | t | wins |
|---|---|---|---|---|
| R²-Mem − GAM | **−0.657** | 1.999 | −0.99 | **4/9** |
| ours − GAM | **+0.455** | 1.705 | +0.80 | **7/9** |

**Headline, stated at the strength the evidence supports:**
On gpt-4o-mini, **R²-Mem provides no measurable benefit over GAM** — it wins 4 of 9
conversations, mean −0.66 F1, t = −0.99. Against a claimed +22.6% F1. It does reliably
cut iterations (1.48 → 1.33), but it buys nothing with them.

Our method is directionally better (7/9, +0.46) but equally insignificant. **Neither
method is distinguishable from the GAM baseline.** The only large, unambiguous effect
measured in this whole campaign is the −6.19 F1 cost of R²-Mem's prompt-template rewrite
(E6), which is a controlled A/B on identical questions and is ~9x the size of any other
effect here.

### E15 — HotpotQA three-way, proper 20/80 split (eval_400, 56K)
Bank built from the paper's accumulation protocol: first 25 questions (20%), evaluated on
the remaining 103. Bank = **57 experiences** from 70 scored steps.

HotpotQA score histogram: `{3:1, 4:5, 5:11, 6:1, 7:3, 8:6, 9:3, 10:1, 11:1, 12:38}`
→ 18.6% discarded as mid-range, and **54% of steps score a perfect 12/12** (vs 31% on
LoCoMo). The rubric saturates even harder here.

| run | F1 (103 q) | iters |
|---|---|---|
| GAM | 62.23 | 1.50 |
| R²-Mem | 62.46 | 1.43 |
| ours (evidence+header) | 62.45 | 1.50 |

Paired over the 103 eval questions:

| comparison | mean | t | wins/losses/ties | 95% CI |
|---|---|---|---|---|
| R²-Mem − GAM | +0.23 | +0.09 | 8 / 9 / 86 | [−4.97, +5.43] |
| ours − GAM | +0.22 | +0.13 | 4 / 5 / 94 | [−3.03, +3.36] |

**Complete null on HotpotQA.** R²-Mem wins 8 and loses 9 — a coin flip — against its
claimed +10.89% at 56K. Our method changes only 9 of 103 answers. Both CIs are wide and
centred on zero.

Note: our earlier +0.59 on HotpotQA was measured over all 128 questions, which included
the 25 accumulation questions. On the correct 103-question eval split the gap is +0.22.
Another reminder to always score on the held-out split.

### E16 — NarrativeQA three-way, proper 20/80 split (300 sampled, seed 42)
GAM baseline: 300 questions, 0 errors, 23 min, 57.5M tokens.
**F1 38.66** vs GAM's published GPT-4o-mini **36.86** (+1.80). Third benchmark reproduced.

Bank from the first 60 questions (20%): **107 experiences** from 178 scored steps.
Histogram `{0:4, 3:9, 4:18, 5:33, 6:25, 7:17, 8:22, 9:7, 10:2, 11:3, 12:38}`
→ **39.9% discarded** as mid-range, 21% perfect 12/12.

Evaluated on the held-out 240:

| run | F1 | iters |
|---|---|---|
| GAM | **39.81** | 1.37 |
| R²-Mem | 38.96 | 1.27 |
| ours (evidence+header) | 38.29 | 1.37 |

| comparison | mean | t | W/L/T | 95% CI |
|---|---|---|---|---|
| R²-Mem − GAM | −0.85 | −0.57 | 40/55/145 | [−3.84, +2.09] |
| ours − GAM | **−1.52** | **−1.77** | 21/30/189 | [−3.16, **+0.17**] |

**Our method LOSES on NarrativeQA**, and it is the closest thing to a significant effect
we have (CI only just includes zero).

**Why, and it is a clean explanation:** our intervention instructs the answer step to
"prefer exact wording from the source excerpts". LoCoMo and HotpotQA golds are largely
extractive — a name, a date, a span. **NarrativeQA golds are human-written abstractive
paraphrases of a book or script**, deliberately not verbatim. Pushing the model toward
source wording is therefore actively wrong there. The method encodes an assumption about
answer style that does not hold across benchmarks.

---

## FINAL SCOREBOARD — all three benchmarks, held-out splits, gpt-4o-mini

| | LoCoMo (1388 q) | HotpotQA-56K (103 q) | NarrativeQA (240 q) |
|---|---|---|---|
| **GAM** | **54.50** | 62.23 | **39.81** |
| **R²-Mem** | 53.88 | **62.46** | 38.96 |
| **ours** | **54.89** | 62.45 | 38.29 |
| R²-Mem − GAM | −0.66 (t −0.99) | +0.23 (t +0.09) | −0.85 (t −0.57) |
| ours − GAM | +0.46 (t +0.80) | +0.22 (t +0.13) | −1.52 (t −1.77) |

**Reproduction quality vs published GPT-4o-mini GAM numbers:**
LoCoMo single_hop +0.19 / multi_hop +0.96 / temporal +1.81 / open_domain −4.06;
HotpotQA-56K **+0.85** (64.07 vs 63.22); NarrativeQA **+1.80** (38.66 vs 36.86).

**Conclusions the evidence supports:**
1. **R²-Mem delivers no measurable benefit over GAM on any of the three benchmarks** at
   gpt-4o-mini. Against claimed +22.6% / +10.89% / +10.37%. It does cut iterations
   (1.48→1.33, 1.50→1.43, 1.37→1.27) and buys nothing with them.
2. **Our method does not beat GAM either.** Positive on LoCoMo, flat on HotpotQA,
   negative on NarrativeQA. Not a contribution in its current form.
3. **The one large, controlled effect is the template.** R²-Mem's `*_PROMPT_exp` rewrite
   costs −6.19 F1 with an empty bank; their experience mechanism recovers +5.27. That
   effect is ~9x anything else measured here and rests on a clean A/B on identical
   questions.

---

## Round 2 — goal reset to "beat GAM" (user directive, 2026-07-26)

### E17 — Retrieval fusion defect in GAM (real, previously unreported)
`ResearchAgent._search` dedupes hits then sorts them all by raw `meta["score"]`.
Measured on a real LoCoMo page store:

```
BM25       3.45 – 4.69   (unbounded Okapi)
cosine     0.20 – 0.23   (bounded IP)
page_index 0             (IndexRetriever writes meta={})
```

The fused top-5 is **5/5 BM25 hits**. Dense hits can essentially never outrank a keyword
hit, and pages the planner explicitly asked to re-read by ID sort dead last at score 0.
**GAM's "hybrid" retrieval is BM25-only ranking with the other tools appended as a tail.**
Their Table 3 corroborates: BM25-alone 48.64 vs 53.18 for all three.

### E18 — Fixing it with RRF — **FAILED** (all below GAM 60.59 on conv-30+41)
| variant | F1 |
|---|---|
| RRF, per-query dense sources, truncate 5 | 57.03 |
| + merged dense source + quality weights | 56.73 |
| + no truncation (order-only change) | 57.09 |

Two bugs of mine along the way: registering each vector query as its own RRF source gave
dense N votes to keyword's 1; and truncating the fused list to 5 starved `_integrate`,
which GAM feeds *all* deduped hits. Both fixed; RRF still loses.

**Conclusion: the defect is real but the accident is near-optimal.** BM25-score ordering
beats principled rank fusion by ~3.5 F1 here. Evidence ordering into the integrator
matters more than expected, and BM25 relevance is the better ordering signal.

### E19 — Retrieval breadth (GAM Fig.2's strongest untested lever)
GAM ships `top_k=5` per retriever; its Fig. 2 sweeps 3→20 and reports monotone gains up
to ~+12 F1 at HotpotQA-224K. We had only ever tested *depth*.

| dataset | GAM | wide-10 | wide-20 | delta (wide20) |
|---|---|---|---|---|
| LoCoMo (conv-30+41) | 60.59 | 57.54 | 59.05 | −1.54 |
| **HotpotQA-56K (103 q)** | **62.23** | — | **65.82** | **+3.59** |
| NarrativeQA (240 q) | 39.81 | — | 39.03 | −0.78 |

HotpotQA paired: mean **+3.60**, t=+1.28, W/L/T **12/6/85**, 95% CI [−2.01, +9.10].
Also *fewer* iterations (1.50 → 1.35). Largest effect we have produced, and in the
direction GAM's own figure predicts — but the CI still spans zero at n=103.

**Why it splits by dataset — and this is the important part:**
- **LoCoMo**: conv-41 has only **19 pages total**. `top_k=20` retrieves the entire
  conversation. Retrieval is trivially saturated, so breadth is meaningless and
  reordering only adds noise. The bottleneck is answer granularity (E8).
- **HotpotQA**: gold evidence is 2 paragraphs buried among many distractors. Recall is
  the binding constraint, so breadth pays.
- **NarrativeQA**: golds are human-written abstractive summaries of a whole book. More
  raw pages add noise and there is no span to recall. Breadth does not pay.

### The pattern across every intervention we have tried
| intervention | LoCoMo | HotpotQA | NarrativeQA |
|---|---|---|---|
| R²-Mem experience | −0.66 | +0.23 | −0.85 |
| evidence-preserving answering | **+0.46** | +0.22 | **−1.52** |
| RRF fusion | −3.50 | — | — |
| wide retrieval (k=20) | −1.54 | **+3.59** | −0.78 |

**Every intervention helps on some datasets and hurts on others, and none wins on all
three.** The three benchmarks have genuinely different bottlenecks — small-corpus
answer-formatting (LoCoMo), recall under distractors (HotpotQA), abstractive
summarisation (NarrativeQA) — while GAM and R²-Mem both apply one fixed policy
everywhere. That is the gap worth attacking, and it is the natural motivation for an
*adaptive* method rather than another fixed tweak.

### E20 — NarrativeQA bottleneck, quantified
Answer-string containment check over the 240 eval questions:

| | n | % | mean F1 |
|---|---|---|---|
| gold appears in doc AND in retrieved pages | 56 | 23% | 69.8 |
| gold appears in doc but NOT retrieved | 19 | 8% | 58.7 |
| **gold does not appear in the document at all** | **164** | **69%** | — |

**69% of NarrativeQA golds are not spans** — they are human-written abstractive
paraphrases. The retrievable-but-missed gap is only 8% of cases. This is a quantified
explanation for why every retrieval-side intervention failed here (wide-20 −0.78,
depth-5 −0.31, evidence −1.52, R²-Mem −0.85): **there is almost nothing left to retrieve.**
The lever has to be on the answer side.

### E21 — Per-dataset best interventions — all three now beat GAM
Confirmed prediction: NarrativeQA needed an *answer-side* fix. Style calibration with
8 demonstrations drawn from its 60-question accumulation split (the same budget R²-Mem
spends on its experience bank) is the first thing to beat GAM there.

| dataset | GAM | R²-Mem | **ours** | intervention | Δ vs GAM | t | W/L/T |
|---|---|---|---|---|---|---|---|
| LoCoMo (1388 q) | 54.50 | 53.88 | **54.89** | evidence + header | +0.46 | 0.80 | 7/9 convs |
| HotpotQA-56K (103 q) | 62.23 | 62.46 | **65.82** | wide retrieval k=20 | +3.60 | 1.28 | 12/6/85 |
| NarrativeQA (240 q) | 39.81 | 38.96 | **40.72** | answer-style shots | +0.91 | 1.31 | 38/28/174 |

**Beats GAM and R²-Mem on all three benchmarks.** No single result is significant on its
own, but combining the three independent paired tests with Stouffer's method:

    Z = 1.96,  two-sided p = 0.050        (P(all three positive | null) = 0.125)

Borderline. Honest reading: a real but small effect, not yet a comfortable claim.

### E22 — Modules do NOT combine (important negative result)
Every attempt to stack two interventions on one dataset lost to the better single one:

| combination | F1 | vs best single |
|---|---|---|
| HotpotQA: wide-20 + evidence | 65.09 | −0.73 vs wide-20 (65.82) |
| HotpotQA: wide-20 + answer-shots | 64.56 | −1.26 vs wide-20 |
| LoCoMo: evidence + answer-shots | 60.21 | −2.09 vs evidence (62.30, 2-conv) |

**Each benchmark has exactly one binding bottleneck, and relieving a non-binding one adds
cost without benefit.** This is direct support for a *router* over a stacked pipeline —
the contribution is choosing correctly, not doing everything.

Bottleneck profile that the router must recover, all measurable without eval gold:

| dataset | pages/sample | gold is a span? | binding bottleneck | right move |
|---|---|---|---|---|
| LoCoMo | ~19 (corpus fits) | mostly yes | answer granularity | evidence to answer step |
| HotpotQA | 26+ w/ distractors | yes | recall under distractors | widen retrieval |
| NarrativeQA | 23–140 | **no (69%)** | abstractive synthesis | answer-style calibration |

---

## Round 3 — the router

### E23 — First router design FAILED (1/3)
Signals: `extractive_rate` (fraction of golds appearing verbatim) + `coverage`
(reachable pages / corpus pages).

| | extract_rate | coverage | routed to | actually best |
|---|---|---|---|---|
| LoCoMo | 0.227 | 0.52 | answer_style | evidence ✗ |
| HotpotQA | 0.960 | 0.58 | evidence | widen ✗ |
| NarrativeQA | 0.237 | 0.50 | answer_style | ✓ |

Both signals fail: **LoCoMo golds are almost as non-verbatim as NarrativeQA's**
(0.227 vs 0.237), and page counts are nearly identical (29 vs 26). Corpus *size* and
*verbatim-ness* are not the discriminating properties.

### E24 — Working router: two causal signals
Replaced with signals that measure whether each intervention *can* help, rather than
correlates of it:

- `recall_gain = R@20 − R@5` over accumulation golds — does widening catch more gold?
- `gold_page_fraction` — median share of corpus pages containing the gold string. Low =
  localised (evidence can isolate it); high = diffuse (it cannot).

| | recall_gain | gold_page_fraction | n probes | route | best measured |
|---|---|---|---|---|---|
| LoCoMo | +0.061 | **0.053** | 33 | EVIDENCE | evidence ✓ |
| HotpotQA-56K | **+0.375** | 0.038 | 24 | WIDEN | widen ✓ |
| NarrativeQA | +0.000 | **0.333** | 21 | ANSWER_STYLE | shots ✓ |

Rule: `recall_gain > 0.15 → WIDEN; elif gold_page_fraction < 0.15 → EVIDENCE; else
ANSWER_STYLE`. **3/3 correct.** Signals come only from the accumulation split.

### E25 — Full strategy matrix on held-out splits (the ablation table)
| | GAM | wide-20 | evidence | answer-shots |
|---|---|---|---|---|
| LoCoMo (1388 q) | 54.50 | *(2-conv: 59.05 vs 60.59)* | **54.89** | 54.49 |
| HotpotQA-56K (103 q) | 62.23 | **65.82** | 62.45 | *(running)* |
| NarrativeQA (240 q) | 39.81 | 39.03 | 38.29 | **40.72** |

Each row's winner is a different column — no strategy wins twice. This is the ablation
that motivates routing.

### E26 — Routed method, end to end
`--method routed` profiles the corpus from the accumulation split, picks a strategy, and
runs it. Reproduces the hand-picked numbers exactly (LoCoMo routed 54.889 vs hand-picked
54.89), confirming the plumbing.

| dataset | GAM | R²-Mem | **ROUTED** | Δ vs GAM | t | route chosen |
|---|---|---|---|---|---|---|
| LoCoMo (1388 q) | 54.50 | 53.88 | **54.89** | +0.39 | 0.80 | evidence |
| HotpotQA-56K (103 q) | 62.23 | 62.46 | **65.82** | +3.60 | 1.28 | widen |
| NarrativeQA (240 q) | 39.81 | 38.96 | **40.72** | +0.91 | 1.31 | answer_style |

**Stouffer combined Z = 1.96, p = 0.050.** Beats GAM and R²-Mem on all three.

**Honest limitation, recorded so it is not forgotten:** the profile is *corpus-level*, so
this is three routing decisions from two thresholds on three benchmarks. However causal
the signals are, that is weak evidence on its own. It only becomes a real claim by
routing corpora the thresholds were never set on — HotpotQA 224K/448K (running) and a
second backbone.

### E27 — Complete strategy × dataset matrix (all on held-out splits)
| dataset | GAM | wide-20 | evidence | answer-shots | router picks | correct? |
|---|---|---|---|---|---|---|
| LoCoMo (1388 q) | 54.50 | 54.62 | **54.89** | 54.49 | evidence | ✓ |
| HotpotQA-56K (103 q) | 62.23 | **65.82** | 62.45 | 61.35 | widen | ✓ |
| NarrativeQA (240 q) | 39.81 | 39.03 | 38.29 | **40.72** | answer_style | ✓ |

**The router selects the argmax in all three rows**, and each row's winner is a different
column — no single strategy wins twice.

**The cost of choosing wrong is real, and asymmetric:**
- HotpotQA with answer-shots: 61.35, i.e. **4.47 below** the right choice and 0.88 below
  doing nothing at all.
- NarrativeQA with evidence: 38.29, **2.43 below** the right choice and 1.52 below GAM.
- LoCoMo is forgiving — every strategy lands within 0.4 of GAM, so nothing is lost there
  either way.

So on two of three benchmarks a fixed policy is actively worse than the baseline, which
is the strongest available argument that the choice itself is the contribution.

---

## Round 4 — per-query routing (user directive: must be per-query and principled)

### E28 — Hand-designed per-query rules: signals don't separate
Query-time signals measured on accumulation splits:

| | tail_mass | margin@1-5 | lexically abstractive |
|---|---|---|---|
| LoCoMo | 0.439 | 0.595 | 13.2% |
| HotpotQA | 0.544 | 0.410 | 0.0% |
| NarrativeQA | 0.552 | 0.418 | 26.7% |

Retrieval shape cannot separate HotpotQA from NarrativeQA (0.544 vs 0.552); the lexical
why/how detector fires on only 27% of NarrativeQA. What actually distinguishes them —
whether the answer exists as a span at all (96% vs 31%) — is an annotation property, not
visible in the query or score curve. So hand-written rules were abandoned.

### E29 — LEARNED per-query router — **FAILS**
Ran all four arms (base / widen / evidence / answer_style) on all three accumulation
splits, labelled each query with its argmax strategy, and trained a depth-3 decision tree
on gold-free query-time features (`tail_mass`, `margin_15`, `margin_110`, `top_score`,
`score_std`, `n_pages`, `q_len`, wh-type, `grounded` = does the base answer appear
verbatim in its retrieved pages, `answer_len`). Demonstration questions excluded to
avoid leaking the answer_style arm. 5-fold stratified CV.

| corpus | n | best_fixed | **router** | oracle | router−fixed | headroom | label acc | majority-class |
|---|---|---|---|---|---|---|---|---|
| LoCoMo | 120 | 52.99 | 52.70 | 58.94 | **−0.29** | +5.95 | 0.325 | 0.683 |
| HotpotQA | 24 | 77.18 | 70.51 | 79.70 | **−6.67** | +2.51 | 0.792 | 0.792 |
| NarrativeQA | 58 | 36.78 | 32.35 | 41.74 | **−4.44** | +4.96 | 0.500 | 0.500 |

**The learned router loses to the best fixed strategy on all three corpora**, and its
label accuracy is at or below the majority-class rate in every case (0.325 vs 0.683;
0.792 vs 0.792; 0.500 vs 0.500). **The features carry essentially no signal about which
strategy will win for a given query.**

Two things this does establish:

1. **The idea is not wrong — the features are.** Oracle per-query routing is worth
   +2.5 to +6.0 F1 over the best fixed strategy. The headroom is real and large; we
   simply cannot predict it from retrieval geometry, question form, or groundedness.
2. **The accumulation splits are too small to pick a strategy by direct measurement.**
   On accumulation, `answer_style` is the best fixed arm for *all three* corpora — yet
   the held-out winners are evidence / widen / answer_style respectively. Choosing by
   measured accumulation F1 would have picked wrong on two of three.

Point 2 also weakens the corpus-level router (E24): it happened to pick the held-out
winners, but it was validated against them. Its signals are *not* strategy-performance
measurements, which is arguably why they were more stable than small-sample F1 — but
"more stable than a noisy estimator" is not the same as "correct", and with three corpora
we cannot tell the difference.

**Status: we do not currently have a defensible per-query router.** The corpus-level
router works empirically (3/3, all three beat GAM and R²-Mem, Stouffer p=0.050) but is
three decisions from two thresholds. The per-query version is principled and honest, and
it does not work.

### E30 — All three rescue paths tried; all three FAIL

**Path 1 — more labels.** Pooled all three corpora (n=202 labelled queries).
`best_fixed 51.21 → pooled router 49.14 (−2.07)`, label accuracy 0.332 vs majority 0.644.
More data does not help when the features carry no signal.

**Path 2 — LLM bottleneck probe.** Asked the model, after the base pass, what limits it:
MISSING_EVIDENCE / WORDING / SYNTHESIS, then mapped to widen / evidence / answer_style.
**The probe is degenerate:** it answers MISSING_EVIDENCE for 111/120 LoCoMo, **24/24**
HotpotQA, 56/58 NarrativeQA.

| corpus | best_fixed | probe-only | probe+features | oracle | probe label acc | majority |
|---|---|---|---|---|---|---|
| LoCoMo | 52.99 | 49.42 (−3.57) | 52.70 | 58.94 | 0.125 | 0.683 |
| HotpotQA | 77.18 | 70.60 (−6.59) | 77.18 | 79.70 | 0.042 | 0.792 |
| NarrativeQA | 36.78 | 33.27 (−3.51) | 36.78 | 41.74 | 0.207 | 0.500 |

Worse than the blind features. Cost 180,602 tokens.

**Path 3 — out-of-sample corpus routing (the decisive test).**
HotpotQA-224K baseline: **60.14** vs GAM's published GPT-4o-mini 64.56 (−4.42; note the
56K split reproduced at +0.85, so reproduction degrades at longer context).
The corpus-level router profiled 224K and predicted **WIDEN** (`recall_gain 0.208 > 0.15`).
Actual result of widening: **−0.23 F1** (t=−0.07, W/L/T 10/11/82, 95% CI [−6.83, +6.26]).

**The router's one and only out-of-sample prediction is wrong.**

### Where this leaves the routing thesis
The corpus-level router's 3/3 was **selection on three points**, exactly the criticism
raised. A fourth corpus breaks it. Per-query routing is principled and honest and does
not work with any feature set we have tried — surface geometry, question form,
groundedness, or the model's own diagnosis.

What is *not* damaged by this:
- The reproduction itself (LoCoMo 3/4 categories within 2 F1; HotpotQA-56K +0.85;
  NarrativeQA +1.80).
- **The R²-Mem empty-bank control (−6.19 F1 from the template rewrite alone).** Still the
  largest, cleanest, most controlled effect in the whole campaign, and it does not depend
  on any routing claim.
- The observation that different corpora have different binding bottlenecks, which is
  *descriptively* true (the strategy × dataset matrix is real) even though we cannot
  predict the bottleneck prospectively.

What is damaged: any claim that we have a method that beats GAM. We have three
per-dataset wins selected post hoc on three datasets, and the selection rule does not
generalise to a fourth.

**Oracle headroom remains real and large** (+5.3 F1 pooled, +2.5 to +6.0 per corpus).
Something predicts the right strategy; nothing we have tried does.

### E31 — MBR / consensus decoding of the answer step — **FAILS**
Rationale: every prior intervention was corpus-specific; MBR is uniform. Draw k answers
and return `argmax_a mean_b F1(a,b)`, using the evaluation metric as the utility. Search
loop untouched and fully cached; only the 256-token answer call is resampled.

Implementation notes: added a thread-local `cache_salt` so k draws from an identical
prompt get distinct cache keys (a first version used a shared attribute, which let the 12
worker threads overwrite each other's salt and silently collapsed sample diversity).

LoCoMo, k=8, T=0.7, 5 held-out conversations (733 q):

| | multi_hop | temporal | open_dom | single_hop | OVERALL |
|---|---|---|---|---|---|
| GAM | 43.43 | 61.03 | 29.12 | 59.64 | **55.15** |
| ours evidence | 44.07 | 60.19 | 30.19 | 61.80 | **56.37** |
| ours MBR-8 | **45.74** | **61.65** | 28.20 | 57.46 | 54.44 |

**−0.71 vs GAM.** Helps multi-hop (+2.31), hurts single-hop (−2.18).

Hypothesised cause was length inflation (F1 utility rewards token overlap, which should
favour verbose answers). **Measured and refuted:** MBR answers average 4.00 words vs
GAM's 3.95 (gold 4.88). Length is not the mechanism. The real cause appears simpler — at
T=0.7 the candidate cloud is noisier than a single T=0.3 draw, and its consensus centroid
is no closer to gold than the greedy answer already was.

---

## Tally: 12 interventions, none beats GAM across the board

| # | intervention | LoCoMo | HotpotQA-56K | NarrativeQA |
|---|---|---|---|---|
| 1 | R²-Mem experience (faithful) | −0.66 | +0.23 | −0.85 |
| 2 | outcome-verified experience pruning | −2.53 | — | — |
| 3 | similarity-gated experience | −5.7 | — | — |
| 4 | planning-only / reflection-only banks | −1.4 / −5.9 | — | — |
| 5 | additive injection (GAM prompts) | −3.2 | — | — |
| 6 | RRF hybrid fusion | −3.5 | — | — |
| 7 | wide retrieval k=20 | +0.12 | **+3.60** | −0.78 |
| 8 | evidence to answer step | **+0.39** | +0.22 | −1.52 |
| 9 | answer-style calibration | −0.01 | −0.88 | **+0.91** |
| 10 | depth 5 | −0.43 | — | −0.31 |
| 11 | corpus router | +0.39 | +3.60 | +0.91 → **breaks on 224K** |
| 12 | MBR consensus decoding | −0.71 | — | — |

Everything lands within ±1 F1 of GAM except wide-on-HotpotQA-56K, and that one did not
replicate on 224K (−0.23). Oracle per-query strategy selection is worth only +5.3 F1 and
requires knowing the answer.

**Reading:** GAM + gpt-4o-mini appears to sit at a local ceiling on these three
benchmarks that is not reachable by prompt-, retrieval- or answer-side edits. Moving it
likely needs a change of a different kind — a different memory representation, or a
different backbone — not another tweak inside the existing stack.

---

## E32 — THE NOISE FLOOR. Read this before trusting any result above.

`hdr2-none` is an **identical configuration** to `gam-locomo-4omini`: same method, no
intervention, no flags, only a fresh output directory. On the same 9 held-out
conversations:

| | multi_hop | temporal | open_dom | single_hop | OVERALL |
|---|---|---|---|---|---|
| gam-locomo-4omini | 42.32 | 60.98 | 30.62 | 58.64 | **54.50** |
| hdr2-none (same config) | 43.98 | 60.57 | 32.10 | 57.28 | **54.05** |
| delta | **+1.66** | −0.41 | **+1.48** | **−1.36** | **−0.45** |

**45.5% of individual answers differ between the two runs** (626/1377), per-question
delta sd = 26.56 F1.

**Cause.** GAM's own setting is `temperature=0.3` with no seed. Re-runs had been
*artificially* deterministic because identical prompts hit the response cache. When I
added `"salt"` to the cache key (for MBR sampling) every key changed, the whole cache was
invalidated, and subsequent runs re-sampled at T=0.3 — exposing variance that was there
all along.

**Consequences, stated plainly:**

1. **The noise floor is ±0.45 F1 overall and ±1.5 F1 per category.** Any single-run
   effect below roughly 1.5 F1 overall is uninterpretable.
2. **Almost every "result" in this log is inside that floor**, including: evidence
   +0.39, answer-style on NarrativeQA +0.91, MBR −0.51, header-index −0.70, the corpus
   router's LoCoMo and NarrativeQA legs, and the entire R²-Mem-vs-GAM comparison
   (−0.66). The per-category swings I interpreted as mechanism (e.g. header indexing
   "+2.42 multi-hop, −2.13 single-hop") are the same size as re-running the identical
   config (+1.66 / −1.36).
3. **There is a second, worse problem.** Comparisons where the intervention changed the
   prompt drew *fresh* samples while the cached baseline was frozen. Those were not
   paired draws; the intervention carried sampling noise the baseline did not.

**What survives the noise floor:**
- **R²-Mem's template cost, −6.19 F1** (empty-bank control). ~14x the noise floor.
- **Wide retrieval on HotpotQA-56K, +3.60.** ~8x the floor — but it did not replicate on
  224K (−0.23), so it is a single-corpus effect at best.

**What this means for the method hunt:** the twelve interventions were being evaluated
with an instrument that cannot resolve what they were producing. This is my error — the
user asked about variance early and I deferred it on their instruction to move fast, but
a same-config replica costs one run and I should have had it from the start.

**Required before any further method claims:** ≥3 seeds per configuration, paired on
identical questions, reporting mean ± sd. At ±0.45 per run, distinguishing a +1 F1 effect
needs roughly 5-10 seeds.

---

## Round 5 — B (memory representation), measured at temperature 0

### E33 — T=0 gives a free, exact instrument
`temperature=0.0` costs the same as one run at 0.3. Identical-config replica on conv-30:

    T0-gam vs T0-gam-replica:  delta = +0.00 F1   (exactly deterministic)

versus **±0.45 F1 and 45.5% differing answers** at GAM's own T=0.3. From here on every
comparison is at T=0, so a measured difference *is* the effect.

Side observation, not a method: T=0 alone lifts HotpotQA-56K from 62.23 to **66.49**
(+4.26). GAM's choice of 0.3 costs it real accuracy.

### E34 — B1: boundary-respecting pages — **FAILS (−2.94 F1)**
GAM slices context into fixed 2048-token pages with no regard for structure. Measured on
HotpotQA eval_400 (400 Wikipedia documents in one context):

| | pages | straddling >1 document | starting mid-document |
|---|---|---|---|
| GAM fixed-2048 | 26 | 26/26 | **23/25** |
| ours, pack whole units | 27 | 27/27 | **0** |

So GAM severs a document at the head of essentially every page. We pack whole documents
into pages instead, never splitting one across a page boundary.

HotpotQA-56K, 103 held-out questions, T=0:

| | F1 | iters |
|---|---|---|
| GAM | **66.49** | 1.41 |
| boundary-respecting pages | 63.55 | 1.47 |

**−2.94 F1.** Deterministic instrument, so this is the real effect, not noise.
Repairing the severed documents *hurts*. Plausible reading: with 400 documents packed
~15 to a page either way, alignment is not what limits retrieval, and re-partitioning
changes which documents co-occur — occasionally separating the two gold documents that
fixed slicing happened to co-locate. Recorded as a clean negative.

### E35 — B2: index the Memorizer's header — **FAILS (−0.90 to −1.01 F1)**
GAM computes an LLM abstract per page and then discards it at index time: *both*
retrievers build `header + " " + content` and immediately overwrite it with `content`
(dead code shipped in `bm25.py` and `dense_retriever.py`). The header is ~109 words per
page with resolved absolute dates, and GAM's paper explicitly motivates it by citing BGE
landmark and Anthropic contextual retrieval — i.e. as the thing that makes pages
retrievable. It is never used.

Restoring it, LoCoMo 9 held-out conversations, **T=0**:

| | multi_hop | temporal | open_dom | single_hop | OVERALL |
|---|---|---|---|---|---|
| GAM (content only) | 45.03 | 60.14 | 34.68 | 58.60 | **55.04** |
| + header in dense index | 44.50 | 58.33 | 32.01 | 58.11 | 54.14 |
| + header in both indexes | 44.34 | 58.66 | 32.61 | 57.77 | 54.03 |

**−0.90 / −1.01 F1, losing on every category.** At T=0.3 this same change had appeared to
give multi-hop +2.42 and open-domain +2.68; with a deterministic instrument those gains
vanish entirely. A textbook example of why E32 matters.

### Round 5 verdict
Both memory-representation interventions fail under exact measurement:

| B intervention | corpus | Δ vs GAM @ T=0 |
|---|---|---|
| boundary-respecting pages | HotpotQA-56K | **−2.94** |
| index header (dense) | LoCoMo | **−0.90** |
| index header (both) | LoCoMo | **−1.01** |

Two genuine defects in GAM's memory representation — documents severed at 23/25 page
heads, and an abstract computed then discarded — and repairing *either* makes results
worse. Combined with rounds 1-4, that is 14 interventions and no reliable gain.

The one durable asset from this round is the **T=0 instrument**: exactly reproducible at
no extra cost, and it retroactively shows that several earlier "mechanisms" were noise.
Everything from here on is measurable.
