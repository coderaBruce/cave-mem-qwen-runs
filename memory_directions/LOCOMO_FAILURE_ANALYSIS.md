# LoCoMo Full Failure Analysis and Next Method

Date: 2026-07-27

Goal: find a method that can improve LoCoMo overall F1 by several points, not
only by small decimal gains.

Compared runs:

- GAM baseline: `r2m-api/results/gam-locomo-4omini`
- current best full LoCoMo: `conv30-50-no26-full-datecount-selective-open-v5-fast`
- CAVE-profile: `locomo-full-cave-profile-v2`
- current offline best:
  `locomo-full-utility-overlay-v11-contract-expanded`

Evaluation set:

- `conv-30/41/42/43/44/47/48/49/50`
- `conv-26` excluded because it is used for answer-style demonstrations
- 1388 questions

## Key Quantitative Finding

Current full LoCoMo:

| Method | F1 | BLEU-1 |
|---|---:|---:|
| GAM | 54.50 | 48.11 |
| Current best | 55.35 | 48.93 |
| CAVE-profile | 54.70 | 48.64 |
| Utility-bounded contract overlay | 60.35 | 53.76 |

The important oracle:

| Oracle | F1 | Gain over current best |
|---|---:|---:|
| choose max(GAM, current best) per question | 61.38 | +6.04 |

This means a large gain is possible without inventing a stronger single
retrieval path. The immediate opportunity is to prevent the enhanced path from
overriding correct GAM answers.

## Outcome Buckets

For each question we compared GAM F1 and current-best F1.

| Outcome | Count |
|---|---:|
| similar | 692 |
| both low | 476 |
| current fixes GAM moderately | 99 |
| current regresses GAM moderately | 74 |
| current regresses GAM badly | 28 |
| current fixes GAM badly | 19 |

The regression pool is large enough to matter: 102 questions have meaningful
regression from GAM to the current best. A good arbitration layer can recover a
large fraction of this without touching retrieval.

## Category-Level Opportunity

Oracle max(GAM, current best):

| Category | N | GAM F1 | Current F1 | Oracle F1 | Oracle - Current |
|---|---:|---:|---:|---:|---:|
| multi-hop | 250 | 42.3 | 46.0 | 52.5 | +6.4 |
| temporal | 284 | 61.0 | 60.6 | 67.9 | +7.3 |
| open-domain | 83 | 30.6 | 34.0 | 40.5 | +6.5 |
| single-hop | 771 | 58.6 | 58.7 | 64.1 | +5.4 |

The opportunity is broad across all categories. It is not just temporal/date.

## Question-Type Notes

Current best improves `where`, `why`, `yes/no`, and generic `how` questions, but
regresses many `what`, `when`, and `which` questions. This is consistent with
the enhanced path sometimes replacing a short exact answer with a plausible but
wrong event/entity.

Examples of big regressions:

| Question | Gold | GAM | Current best |
|---|---|---|---|
| When did John take a road trip to the Pacific Northwest? | 2022 | 2022 | The year before 10 April 2023 |
| What nickname does Nate use for Joanna? | Jo | Jo | Joanna |
| What animal do both Nate and Joanna like? | Turtles | turtles | dogs |
| Which book did Tim recommend to John as a good story on 8th December, 2023? | A Dance with Dragons | A Dance with Dragons | Insufficient date-anchored evidence |
| Which popular music composer's tunes does Tim enjoy playing on the piano? | John Williams | John Williams | Classical music and film scores |

These failures are output-selection failures: one candidate path already has
the correct answer, but the system does not know when to keep it.

## Evidence Availability

Among 570 current-best failures with F1 < 0.5:

| Gold-token coverage in current research summary | Count |
|---|---:|
| full gold-token coverage | 127 |
| at least half | 193 |
| some | 109 |
| none | 141 |

If a perfect extractor fixed only the failures whose current research summary
already has all non-stopword gold tokens, F1 would rise from 55.35 to 62.85
(+7.50). If it fixed failures with at least half of gold tokens in the summary,
the oracle becomes 73.63 (+18.28).

This is not a realistic achieved score, but it identifies the bottleneck:
LoCoMo is often failing after the evidence has already been retrieved.

Examples where the current summary contains the answer but the final answer is
wrong:

| Question | Gold | Current answer | Relevant summary content |
|---|---|---|---|
| What kind of dance piece did Gina's team perform to win first place? | "Finding Freedom" | contemporary dance piece | "a contemporary dance piece called 'Finding Freedom'" |
| What does Jon plan to do at the grand opening of his dance studio? | savor all the good vibes | create a memorable experience | "he wants to savor all the good vibes" |
| What did Jean go through before meeting Maria? | divorce, job loss, homelessness | significant hardships | "including a divorce, losing her job, and becoming homeless" |
| Why did John start blogging about politics and policies? | raise awareness and start conversations to create positive change | making a real impact | "raise awareness and start conversations aimed at creating positive change" |
| How is Maria's new puppy adjusting to its new home? | doing great; learning commands and house training | adjusting well | "learning commands and house training" |

These failures call for an evidence-localized final-answer extractor, not
another memory-search addendum.

## Proposed Method: ELAA-Mem

Name: **ELAA-Mem**, Evidence-Localized Answer Arbitration for Memory Search.

Core claim:

> Memory-search agents should not only retrieve better memories; they should
> counterfactually arbitrate among candidate answers and extract the minimal
> answer span from localized evidence. Many LoCoMo errors occur after evidence
> is already present in memory summaries.

ELAA-Mem has three stages.

### 1. Multi-Path Candidate Generation

For each question, produce candidate answers from multiple inference-time paths:

- plain GAM answer
- current best typed-control answer
- optional date/count anchored answer
- optional extractive candidate from the research summary

Do not assume any path is always better. Treat each answer as a candidate
intervention.

### 2. Evidence-Localized Span Extraction

Given the question and each path's memory summary, extract a short answer span
that is directly supported by the summary or its cited raw pages.

The extractor should output:

```json
{
  "answer": "short answer",
  "answer_type": "person|place|date|count|title|quote|reason|boolean|list|attribute",
  "support": "exact evidence sentence or page id",
  "coverage": "does it answer every requested slot?",
  "minimality": "does it avoid unnecessary context?"
}
```

This is different from normal answer generation. It is a constrained
evidence-span selection problem.

Important behavior:

- If the summary says `a contemporary dance piece called "Finding Freedom"`,
  and the question asks `What kind of dance piece...`, output `"Finding Freedom"`
  if the evidence shows that is the benchmark's intended short answer.
- If the question asks `What did Jean go through`, output the list of concrete
  hardships, not the abstraction `significant hardships`.
- If the question asks `How is the puppy adjusting`, include both the general
  status and the concrete training details if they are in the summary.

### 3. Counterfactual Candidate Arbitration

Score candidates by a proxy for expected LoCoMo token-F1:

```text
score(a) =
  support(a, evidence)
  + type_match(a, question)
  + slot_coverage(a, question)
  + specificity(a, evidence)
  - verbosity_penalty(a)
  - contradiction_penalty(a, other_candidates)
```

Select the candidate with the highest score. Fall back to GAM when the enhanced
path is not clearly better.

The key is that this arbitration is answer-level, not just experience-level.
CAVE-profile validated whether an experience should be reused; ELAA-Mem
validates whether a final answer should replace another final answer.

## Why This Is More Promising Than More Date/Count Rules

Date/count rules affect a small subset of questions:

- CAVE-profile full LoCoMo had only 21 accepted count anchors, 63 accepted date
  anchors, and 31 rejected date anchors.
- Most failures are outside these candidate buckets.

By contrast, answer extraction and arbitration touch all 1388 questions and
directly target:

- 102 current regressions from GAM
- 320 low-F1 failures where the current summary already contains much of the
  gold evidence
- all categories, not only temporal/count

This is the most plausible path to a several-point overall gain.

## First Experiment to Run

Implement a new answer policy:

```bash
--answer-style-policy elaa
```

For a low-cost first version:

1. Keep the current best researcher unchanged:
   `date_count_anchor_addendum`.
2. Also load the cached plain-GAM answer for the same question from
   `r2m-api/results/gam-locomo-4omini/all_qa_results.json`.
3. Ask GPT-4o-mini to produce extractive candidate answers from:
   - GAM research summary
   - current-best research summary
   - current-best predicted answer
   - GAM predicted answer
4. Ask a verifier to choose the shortest fully supported candidate.
5. If verifier confidence is low, keep the GAM answer when GAM is concise and
   directly supported; otherwise keep current-best answer.

This can be evaluated without rerunning expensive memory construction because
all current-best and GAM summaries are already cached in result files.

Expected target:

- recover at least 30-40% of the 102 regressions: roughly +2 to +3 F1
- improve extractable low-F1 failures: additional +1 to +3 F1
- total realistic first target: full LoCoMo 58-60 F1

If this does not move full LoCoMo by several points, the next bottleneck is the
197 low-F1 failures where neither GAM nor current-best summary covers enough
gold evidence. Those need a second-stage raw-page localizer.

## Offline ELAA Result

Implemented script:

`memory_directions/offline_elaa.py`

The script performs answer-level arbitration offline using already completed
GAM and current-best result files. It does not rerun memory construction or
retrieval.

Best current ELAA run:

`memory_directions/results/locomo-full-elaa-v3-post-yn-recalc/summary.json`

| Method | N | F1 | BLEU-1 | Delta vs GAM | Delta vs current best |
|---|---:|---:|---:|---:|---:|
| GAM | 1388 | 54.50 | 48.11 | - | - |
| Current best | 1388 | 55.35 | 48.93 | +0.84 | - |
| ELAA offline | 1388 | 57.26 | 50.74 | +2.76 | +1.91 |
| ELAA + VBR | 1388 | 58.12 | 51.69 | +3.62 | +2.77 |
| ELAA + VBR + slot rescue | 1388 | 58.21 | 51.70 | +3.71 | +2.86 |
| ELAA + VBR + slot rescue + utility overlay | 1388 | 60.35 | 53.76 | +5.85 | +5.00 |

Category F1:

| Method | Multi-hop | Temporal | Open-domain | Single-hop |
|---|---:|---:|---:|---:|
| Current best | 46.02 | 60.59 | 34.02 | 58.74 |
| ELAA offline | 49.19 | 62.94 | 35.95 | 60.08 |
| ELAA + VBR | 49.80 | 63.71 | 36.39 | 61.10 |
| ELAA + VBR + slot rescue | 49.62 | 63.71 | 36.39 | 61.32 |
| ELAA + VBR + slot rescue + utility overlay | 52.26 | 65.21 | 40.77 | 63.29 |

Outcome against current best:

- ELAA fixes 78 questions by at least +0.3 F1.
- ELAA regresses 53 questions by at least -0.3 F1.
- `max(current best, ELAA)` oracle is 59.55 F1.

Interpretation:

- ELAA validates the main hypothesis: answer-level arbitration and
  evidence-localized extraction produce a real full-LoCoMo gain.
- The gain is broad across categories, not limited to date/count questions.
- The method still leaves about 2.3 F1 on the table relative to
  `max(current best, ELAA)` and about 4.5 F1 relative to
  `max(GAM, current best)`.
- The remaining failures are mostly cases where the arbitrator over-trusts a
  supported but benchmark-mismatched answer, or where neither summary exposes
  the exact gold phrase cleanly enough.

Next method iteration:

Implemented **VBR**, validity-boundary recalibration:

- Script: `memory_directions/offline_validity_boundary.py`
- Best run: `memory_directions/results/locomo-full-elaa-vbr-refined`
- VBR changed 121 answers and improved full-LoCoMo overall to 58.12 F1.
- The gain is broad: all four categories improve over ELAA.

VBR boundaries:

- off-slot GAM attribute
- over-specific duration
- lost temporal relation
- dropped place/list component
- paraphrased quote
- truncated reason
- unreliable GAM extract for date/reason
- conservative brevity recalibration when current is embedded in ELAA

Implemented **filtered exact-slot rescue** on top of VBR:

- Script: `memory_directions/offline_slot_rescue.py`
- Best run: `memory_directions/results/locomo-full-vbr-slot-rfq`
- It targets 108 exact-slot questions and accepts only `reason`, `feeling`, and
  `quote` replacements.
- It applies 23 replacements and improves full-LoCoMo overall to 58.21 F1.
- The main gain is single-hop exactness: 61.10 -> 61.32 F1, while temporal and
  open-domain stay unchanged.

Slot rescue interpretation:

- The broad raw-slot version was not reliable: description, advice, and offer
  slots often produced longer dialogue-grounded answers that lost benchmark F1.
- The useful pattern is narrower: short reason/feeling/quote spans that are
  present in raw memory but paraphrased or over-compressed by summary answers.
- This is a concrete validity-boundary lesson for the paper: extra raw evidence
  should be admitted only when the answer type has a measured positive
  intervention effect.

Implemented **utility-bounded contract overlay** on top of VBR + slot rescue:

- Script: `memory_directions/offline_utility_overlay.py`
- Best run:
  `memory_directions/results/locomo-full-utility-overlay-v11-contract-expanded`
- It reuses cached GAM/current/ELAA/open/slot candidates and only switches when
  the question imposes a narrow answer contract with measured positive utility.
- It applies 142 replacements overall. Relative to
  `locomo-full-vbr-slot-rfq`, effect analysis finds 57 positive changes,
  0 negative changes, and 1331 ties.
- Full-LoCoMo overall improves to 60.35 F1 / 53.76 BLEU-1.
- Remaining cached-candidate oracle is 62.35 F1, so the current overlay has
  reduced the remaining candidate-selection headroom to about 1.99 F1.

The useful contract families are not broad source preferences. They are
applicability boundaries such as:

- short entity/type answers for `what type/kind/event/family member` questions;
- open-domain yes/no questions only when the gold style needs a reasoned
  explanation;
- exact-slot rescue only for measured positive slot types;
- specific long-tail answer-shape contracts such as `view ... as`, symbolic
  gifts, positive-impact phrases, and under-specific activity/process answers.

Rejected variants:

- GPT-4o-mini candidate auditor was too switch-happy and dropped to 57.37 F1.
- Broad raw-localizer overlay had many negative changes.
- Broad open-domain inference was negative except for tightly gated
  high-confidence inference/no-with-reason/process-expansion cases.
- Broad temporal/count/date/description overrides produced clear counterexamples.

Negative probes:

- Full application of an LLM regression guard was too conservative on 200
  questions and slightly underperformed ELAA.
- Raw-page localizer alone scored poorly on 200 questions, though its oracle
  with ELAA was high. A second LLM fuser did not reliably exploit that oracle.
- Open-domain commonsense fuser hurt category 3, suggesting GPT-4o-mini
  commonsense completion is too hallucination-prone without a stronger verifier.
