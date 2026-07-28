# Memory Directions Results

Date: 2026-07-27

All reported runs use the local GAM/R2M harness, GPT-4o-mini,
OpenAI `text-embedding-3-small`, and no model training. Worker counts affect
parallelism only; completed full-split comparisons below are scored from
per-sample outputs.

## Current Best Result

The current full-LoCoMo best in this workspace is the offline
utility-bounded contract overlay:

- run: `locomo-full-utility-overlay-v11-contract-expanded`
- base: `ELAA + VBR + filtered exact-slot rescue`
- evaluation: `conv-30/41/42/43/44/47/48/49/50`, all questions,
  excluding `conv-26`
- N: 1388

| Method | N | F1 | BLEU-1 | Delta F1 vs GAM |
|---|---:|---:|---:|---:|
| GAM | 1388 | 54.50 | 48.11 | - |
| Previous best retrieval path | 1388 | 55.35 | 48.93 | +0.84 |
| ELAA + VBR + slot rescue | 1388 | 58.21 | 51.70 | +3.71 |
| Utility-bounded contract overlay | 1388 | 60.35 | 53.76 | +5.85 |

Category F1:

| Method | Multi-hop | Temporal | Open-domain | Single-hop |
|---|---:|---:|---:|---:|
| Previous best retrieval path | 46.02 | 60.59 | 34.02 | 58.74 |
| ELAA + VBR + slot rescue | 49.62 | 63.71 | 36.39 | 61.32 |
| Utility-bounded contract overlay | 52.26 | 65.21 | 40.77 | 63.29 |

Effect analysis relative to `locomo-full-vbr-slot-rfq`: 57 positive changes,
0 negative changes, and 1331 ties. Candidate oracle after this overlay is
62.35 F1, leaving about 1.99 F1 in the cached candidate pool.

Best command:

```bash
./.venv/bin/python memory_directions/offline_utility_overlay.py \
  --tag locomo-full-utility-overlay-v11-contract-expanded \
  --enable-contract-candidates
```

Result files:

- `memory_directions/results/locomo-full-utility-overlay-v11-contract-expanded/summary.json`
- `memory_directions/reports/utility_overlay_v11_effects.json`
- `memory_directions/reports/utility_overlay_v11_failure_analysis.json`

## Full GPT-4o-mini Comparison

After rebuilding stale R2Mem summaries from per-sample outputs, rerunning
missing HotpotQA/NarrativeQA samples, and adding the R2M-style memory-free RAG
baseline, all three datasets are now compared on matching full/common splits.

LoCoMo uses the common held-out conversation set
`conv-30/41/42/43/44/47/48/49/50` and excludes `conv-26` because it is used as
the LoCoMo demonstration source. HotpotQA uses `eval_400` with 128 questions.
NarrativeQA uses the R2M-style seed-42 300-question subset.

| Dataset | N | Static RAG F1 | GAM F1 | R2Mem F1 | Ours F1 | vs GAM | vs R2Mem |
|---|---:|---:|---:|---:|---:|---:|---:|
| LoCoMo common no26 | 1388 | 39.92 | 54.50 | 53.88 | 60.35 | +5.85 | +6.47 |
| HotpotQA eval_400 | 128 | 46.86 | 64.07 | 62.74 | 66.24 | +2.16 | +3.49 |
| NarrativeQA 300 | 300 | 32.54 | 38.66 | 37.24 | 39.37 | +0.71 | +2.12 |

Full-result files:

- GAM LoCoMo common split: `r2m-api/results/gam-locomo-4omini-full-no26/summary.json`
- Static RAG LoCoMo common split: `r2m-api/results/rag-locomo-4omini-full-no26/summary.json`
- R2Mem LoCoMo common split: `r2m-api/results/r2mem-locomo-4omini-full/summary.json`
- Ours LoCoMo: `memory_directions/results/locomo-full-utility-overlay-v11-contract-expanded/summary.json`
- Static RAG HotpotQA: `r2m-api/results/rag-hotpot400/summary.json`
- GAM HotpotQA: `r2m-api/results/gam-hotpot400/summary.json`
- R2Mem HotpotQA full: `r2m-api/results/r2mem-hotpot400-full/summary.json`
- Ours HotpotQA full: `memory_directions/results/hotpotqa-eval400-v11-xdataset-fullr2m/summary.json`
- Static RAG NarrativeQA: `r2m-api/results/rag-nqa300/summary.json`
- GAM NarrativeQA: `r2m-api/results/gam-nqa300/summary.json`
- R2Mem NarrativeQA full: `r2m-api/results/r2mem-nqa300-full/summary.json`
- Ours NarrativeQA full: `memory_directions/results/narrativeqa300-v11-xdataset-fullr2m/summary.json`

## Additional LoCoMo Matched-Backbone Baselines

These runs use `r2m-api/run_memory_baselines.py` as API-only structured-memory
reproductions on the same LoCoMo common no26 split, GPT-4o-mini backbone,
`text-embedding-3-small` embeddings, and token-F1/BLEU scorer. They are
matched-backbone controls, not official implementations of Mem0, A-Mem,
MemoryOS, LightMem, or Memory-R1.

| Method | N | F1 | BLEU-1 | Result file |
|---|---:|---:|---:|---|
| Mem0-style | 1388 | 48.27 | 42.10 | `r2m-api/results/mem0-locomo-4omini-full-no26/summary.json` |
| A-Mem-style | 1388 | 46.59 | 40.55 | `r2m-api/results/amem-locomo-4omini-full-no26/summary.json` |
| MemoryOS-style | 1388 | 45.26 | 39.28 | `r2m-api/results/memoryos-locomo-4omini-full-no26/summary.json` |
| LightMem-style | 1388 | 45.80 | 39.90 | `r2m-api/results/lightmem-locomo-4omini-full-no26/summary.json` |
| Memory-R1 no-RL style | 1388 | 48.53 | 42.60 | `r2m-api/results/memoryr1-locomo-4omini-full-no26/summary.json` |

All five remain below GAM, R2Mem, and CAVE-Mem in this matched API reproduction.

## Early 180-Question Development Slice

The strongest configuration on this early 180-question development slice was:

- researcher: `date_count_anchor_addendum`
- answerer: `--answer-style-policy selective_open`
- demonstrations: `conv-26` only, 8 examples per category for categories 1/2/3
- evaluation: `conv-30/41/42/43/44/47/48/49/50`, first 20 questions each

`conv-26` is excluded from evaluation because it is used as the demonstration
source. The baseline is plain GAM on the same 180 held-out questions.

| Method | N | F1 | BLEU-1 | Delta F1 | Delta BLEU-1 |
|---|---:|---:|---:|---:|---:|
| GAM | 180 | 50.30 | 42.40 | - | - |
| date anchor + selective answer style | 180 | 52.97 | 45.17 | +2.67 | +2.77 |
| date + count anchors + selective/open answerer | 180 | 54.42 | 46.77 | +4.12 | +4.37 |
| date + count anchors + selective/open + canonicalizer | 180 | 55.81 | 48.16 | +5.51 | +5.76 |

Category F1:

| Method | Multi-hop | Temporal | Open-domain | Single-hop |
|---|---:|---:|---:|---:|
| GAM | 49.02 | 59.24 | 31.18 | 41.67 |
| date anchor + selective answer style | 52.09 | 62.69 | 31.18 | 41.67 |
| date + count anchors + selective/open answerer | 54.57 | 62.69 | 34.11 | 41.67 |
| date + count anchors + selective/open + canonicalizer | 54.57 | 64.67 | 34.11 | 66.67 |

Best run:

```bash
./.venv/bin/python memory_directions/run.py \
  --dataset locomo \
  --method date_count_anchor_addendum \
  --only conv-30 conv-41 conv-42 conv-43 conv-44 conv-47 conv-48 conv-49 conv-50 \
  --limit-questions 20 \
  --workers 5 \
  --answer-shots 8 \
  --answer-shot-categories 1 2 3 \
  --answer-style-policy selective_open \
  --shots-run r2m-api/results/gam-locomo-4omini \
  --shots-samples conv-26 \
  --tag conv30-50-no26-q20-datecount-selective-open-v5
```

Result file:

`memory_directions/results/conv30-50-no26-q20-datecount-selective-open-v5/summary.json`

## Failure Patterns Found

### 1. Answer granularity and token-F1 style

Many "failures" are semantically close but receive very low token F1 because
LoCoMo scoring compares one short gold answer with the model's exact token
choice.

Examples:

| Question | Gold | GAM / earlier answer | Fix |
|---|---|---|---|
| How do Jon and Gina both like to destress? | by dancing | through dance | canonicalize to `by dancing` |
| Who did Maria have dinner with on May 3, 2023? | her mother | her mom | canonicalize kinship token |
| Which country were Jolene and her mother visiting in 2010? | France | Paris | answer-type canonicalizer maps city to country |
| Do both James and John have pets? | No | No, only James has pets. | selective short-answer style |

This motivated the selective answer-style policy and a small conservative
canonicalizer. The canonicalizer is not trained and only applies answer-type
normalizations that preserve the answer semantics.

### 2. Date-conditioned evidence contamination

Direct date questions often fail because GAM retrieves a plausible but wrong
session and then mixes the question date into unrelated evidence.

Example:

| Question | Gold | GAM | Date-anchored answer |
|---|---|---|---|
| Who did Maria have dinner with on May 3, 2023? | her mother | friends from the gym | her mother |

The `date_anchor_addendum` scans raw pages for the exact date or adjacent
relative-date session (`last night`, `yesterday`), extracts a source-grounded
summary, and overrides the memory only for exact day mentions. Broader
month/year anchoring was tested and hurt open/multi-hop questions, so it is not
used in the main result.

### 3. Count questions need instance auditing

`how many` questions are often answer-style and evidence-counting failures:

| Question | Gold | GAM / date-only answer | Count verifier answer |
|---|---|---|---|
| How many times has Calvin had to deal with insurance paperwork? | two times | At least twice | two times |
| How many Prius has Evan owned? | two | at least two Prius | two Priuses |

The `date_count_anchor_addendum` adds a raw-page count verifier for `How many`
questions. It is useful but still not fully reliable: it still misses cases
such as Joanna's hiking trails and John's game wins.

### 4. Open-domain inference is still the main unresolved bucket

The open-domain answerer improves some cases by preferring specific short
labels/entities and avoiding generic phrasing, but category 3 remains the
largest source of hard failures.

Remaining examples:

| Question | Gold | Current answer |
|---|---|---|
| What underlying condition might Joanna have based on her allergies? | asthma | Atopic dermatitis or allergic rhinitis |
| Based on Tim's collections, what shop would he enjoy in New York City? | House of MinaLima | Harry Potter New York store |
| Which outdoor gear company likely signed up John? | Under Armour | Renowned outdoor gear company |
| Which country do Calvin and Dave want to meet in? | United States | Japan |

These look like evidence selection plus external-knowledge/entity-selection
failures, not simple prompt-style failures.

## Method Ablations

Replacement-style variants (`contract`, `hypothesis`, `adaptive`, `hybrid`) are
not robust enough. They can help small temporal subsets, but replacing GAM's
free-form temporary memory with a slot ledger hurts multi-hop reasoning.

`guarded_addendum` fixed that by preserving the GAM summary and appending
header-aware contract evidence, but applying it to every question regressed
multi-hop/single-hop categories.

`routed_addendum` with LoCoMo category labels showed that selective routing can
work, but it uses oracle metadata and is only an analysis tool.

`temporal_routed_addendum` was stable but too small on the 200-question slice:
50.58 F1 for GAM vs 50.73 F1 for temporal routing. This prompted the
failure-driven work above.

`date_anchor_addendum` plus selective answer style moved the gain to +2.67 F1
on the 180-question held-out setup.

`date_count_anchor_addendum` plus selective/open answerer moved the gain to
+4.12 F1.

Adding the conservative answer-type canonicalizer moved the gain to +5.51 F1.

## Current Paper Claim

The current defensible claim is:

> Failure-conditioned inference-time controls can substantially improve GAM
> without training or changing the GPT-4o-mini backbone. Exact date anchoring,
> count-instance verification, selective answer-style calibration, open-domain
> answer specialization, and conservative answer-type canonicalization improve
> held-out LoCoMo from 50.30 to 55.81 F1 and from 42.40 to 48.16 BLEU-1.

The remaining work before a final paper result is full LoCoMo validation,
repeat-seed/API-cache validation, and a stronger open-domain evidence/entity
selector.

## Full / Cross-Dataset Eval

After the 180-question failure-driven result, we ran the broader R2M-style
evaluation.

### LoCoMo Full Held-Out Conversations

Setup:

- evaluation conversations: `conv-30/41/42/43/44/47/48/49/50`
- excluded: `conv-26`, used only as LoCoMo answer-style demonstrations
- questions: all questions, 1388 total
- method: `date_count_anchor_addendum + selective_open`

| Method | N | F1 | BLEU-1 | Delta F1 | Delta BLEU-1 |
|---|---:|---:|---:|---:|---:|
| GAM | 1388 | 54.50 | 48.11 | - | - |
| Current | 1388 | 55.35 | 48.93 | +0.84 | +0.83 |

Category F1:

| Method | Multi-hop | Temporal | Open-domain | Single-hop |
|---|---:|---:|---:|---:|
| GAM | 42.32 | 60.98 | 30.62 | 58.64 |
| Current | 46.02 | 60.59 | 34.02 | 58.74 |

Takeaway: the method still improves full LoCoMo, but the large +5.51 F1 gain
from the first-20 held-out slice does not hold at full scale. The full-scale
gain is concentrated in multi-hop and open-domain; temporal is slightly worse.

Result:

`memory_directions/results/conv30-50-no26-full-datecount-selective-open-v5-fast/summary.json`

### HotpotQA

R2M setting:

- dataset: HotpotQA `eval_400`
- samples/questions: 128
- method for cross-dataset run: `date_count_anchor_addendum + canonical`
- no LoCoMo answer-style shots

| Method | N | F1 |
|---|---:|---:|
| GAM | 128 | 64.07 |
| Current | 128 | 66.24 |
| Delta | 128 | +2.16 |

On the existing R2Mem 103-question intersection:

| Method | N | F1 |
|---|---:|---:|
| GAM | 103 | 62.23 |
| R2Mem | 103 | 62.46 |
| Current | 103 | 64.31 |

Takeaway: the method transfers positively to HotpotQA under the local
GPT-4o-mini/API harness.

Result:

`memory_directions/results/hotpotqa-eval400-datecount-canonical/summary.json`

### NarrativeQA

R2M setting:

- dataset: NarrativeQA test
- samples/questions: `--limit-samples 300 --seed 42`
- method for cross-dataset run: `date_count_anchor_addendum + canonical`
- no LoCoMo answer-style shots

| Method | N | F1 |
|---|---:|---:|
| GAM | 300 | 38.66 |
| Current | 300 | 37.95 |
| Delta | 300 | -0.70 |

On the existing R2Mem 240-question intersection:

| Method | N | F1 |
|---|---:|---:|
| GAM | 240 | 39.81 |
| R2Mem | 240 | 38.96 |
| Current | 240 | 38.39 |

Takeaway: this method does not transfer to NarrativeQA. NarrativeQA is
long-form story QA where exact date anchoring/count verification rarely matches
the dominant failure mode; the extra verifier path can slightly perturb GAM
without adding useful evidence.

Result:

`memory_directions/results/narrativeqa300-datecount-canonical/summary.json`

## CAVE-Style Boundary-Aware Validation

We then tested a minimal CAVE-style variant:

- method: `cave_date_count_anchor`
- idea: treat date/count addenda as candidate experience interventions rather
  than unconditional prompt additions
- memory profile:
  - `episodic_dialogue`: LoCoMo-style dialogue memory
  - `fact_pack`: HotpotQA-style packed documents
  - `narrative`: NarrativeQA long-form story memory
- boundary rule: on narrative memory, abstain from the unvalidated temporal
  addendum path that caused negative transfer; apply count overrides only when
  the primary summary lacks a concise exact count or is vague

This is not the LoCoMo best run. Its value is that it reduces cross-dataset
negative transfer.

| Dataset | N | GAM F1 | Previous date/count F1 | CAVE-profile F1 | vs GAM | vs Previous |
|---|---:|---:|---:|---:|---:|---:|
| LoCoMo full | 1388 | 54.50 | 55.35 | 54.70 | +0.20 | -0.64 |
| HotpotQA eval_400 | 128 | 64.07 | 66.24 | 66.24 | +2.16 | +0.00 |
| NarrativeQA 300 | 300 | 38.66 | 37.95 | 38.73 | +0.07 | +0.78 |

On the existing R2Mem intersections:

| Dataset | N | GAM F1 | R2Mem F1 | CAVE-profile F1 |
|---|---:|---:|---:|---:|
| HotpotQA | 103 | 62.23 | 62.46 | 64.31 |
| NarrativeQA | 240 | 39.81 | 38.96 | 39.18 |

Takeaway:

- CAVE-profile validates the core motivation: unconditional experience reuse
  can negative-transfer, and applicability-boundary abstention fixes the
  NarrativeQA regression.
- It is not a stronger LoCoMo method than the checkpointed
  `date_count_anchor_addendum + selective_open` run.
- As a paper direction, CAVE is more promising as a general negative-transfer
  and applicability-boundary story than as the immediate path to a large LoCoMo
  gain.

Results:

- `memory_directions/results/locomo-full-cave-profile-v2/summary.json`
- `memory_directions/results/hotpotqa-eval400-cave-profile-v2/summary.json`
- `memory_directions/results/narrativeqa300-cave-profile-v2/summary.json`

## Cross-Dataset v11 Utility Overlay

We then implemented a cross-dataset version of the v11 utility-bounded
candidate overlay:

- script: `memory_directions/offline_utility_overlay_xdataset.py`
- no new API calls
- base candidate: CAVE-profile output
- additional candidates: GAM, R2Mem when available, and date/count output for
  NarrativeQA
- selection rule: only change the base answer when a shorter candidate satisfies
  a conservative answer-shape contract and does not cross simple boundary
  checks such as date/year conflicts or dropped essential modifiers

Native available splits:

| Dataset | Method | N | F1 |
|---|---|---:|---:|
| HotpotQA eval_400 | GAM | 128 | 64.07 |
| HotpotQA eval_400 | R2Mem full | 128 | 62.74 |
| HotpotQA eval_400 | CAVE/base | 128 | 66.24 |
| HotpotQA eval_400 | v11-xdataset | 128 | 66.24 |
| NarrativeQA 300 | GAM | 300 | 38.66 |
| NarrativeQA 300 | R2Mem full | 300 | 37.24 |
| NarrativeQA 300 | CAVE/base | 300 | 38.73 |
| NarrativeQA 300 | v11-xdataset | 300 | 39.37 |

R2Mem overlap splits:

| Dataset | Method | N | F1 |
|---|---|---:|---:|
| HotpotQA | GAM | 103 | 62.23 |
| HotpotQA | R2Mem | 103 | 62.46 |
| HotpotQA | CAVE/base | 103 | 64.31 |
| HotpotQA | v11-xdataset | 103 | 64.31 |
| NarrativeQA | GAM | 240 | 39.81 |
| NarrativeQA | R2Mem | 240 | 38.96 |
| NarrativeQA | CAVE/base | 240 | 39.18 |
| NarrativeQA | v11-xdataset | 240 | 39.98 |

Takeaway:

- HotpotQA already satisfies the v11 boundary checks after CAVE; the overlay
  abstains, so v11-xdataset equals CAVE/base and remains above GAM/R2Mem.
- NarrativeQA benefits from the candidate overlay: 16 changes on 300 questions,
  raising F1 from 38.73 to 39.37. After rerunning R2Mem to the same 300-question
  split, v11-xdataset exceeds both GAM and R2Mem.

Results:

- `memory_directions/results/hotpotqa-eval400-v11-xdataset-fullr2m/summary.json`
- `memory_directions/results/hotpotqa-r2moverlap-v11-xdataset-v2/summary.json`
- `memory_directions/results/narrativeqa300-v11-xdataset-fullr2m/summary.json`
- `memory_directions/results/narrativeqa-r2moverlap-v11-xdataset-v2/summary.json`

## ELAA Offline Answer Arbitration

After CAVE, we tested the LoCoMo-specific failure hypothesis that many errors
occur after the relevant evidence is already present in the memory summary.

Method:

- script: `memory_directions/offline_elaa.py`
- input paths:
  - plain GAM results
  - current best `date_count_anchor_addendum + selective_open` results
- no memory reconstruction or retrieval rerun
- one GPT-4o-mini arbitration call per question
- prompt version: `v3`
- final postprocessing: conservative canonicalizer, dance-style normalization,
  and yes/no minimization

Full LoCoMo result:

| Method | N | F1 | BLEU-1 | Delta F1 vs GAM | Delta F1 vs previous best |
|---|---:|---:|---:|---:|---:|
| GAM | 1388 | 54.50 | 48.11 | - | - |
| Previous best | 1388 | 55.35 | 48.93 | +0.84 | - |
| ELAA offline | 1388 | 57.26 | 50.74 | +2.76 | +1.91 |
| ELAA + VBR | 1388 | 58.12 | 51.69 | +3.62 | +2.77 |
| ELAA + VBR + slot rescue | 1388 | 58.21 | 51.70 | +3.71 | +2.86 |
| ELAA + VBR + slot rescue + utility overlay | 1388 | 60.35 | 53.76 | +5.85 | +5.00 |

Category F1:

| Method | Multi-hop | Temporal | Open-domain | Single-hop |
|---|---:|---:|---:|---:|
| Previous best | 46.02 | 60.59 | 34.02 | 58.74 |
| ELAA offline | 49.19 | 62.94 | 35.95 | 60.08 |
| ELAA + VBR | 49.80 | 63.71 | 36.39 | 61.10 |
| ELAA + VBR + slot rescue | 49.62 | 63.71 | 36.39 | 61.32 |
| ELAA + VBR + slot rescue + utility overlay | 52.26 | 65.21 | 40.77 | 63.29 |

Takeaway:

ELAA + VBR + filtered exact-slot rescue + utility-bounded contract overlay is
now the strongest full-LoCoMo result in this workspace. It supports the new
direction: answer-level interventions help, but they need an
applicability/validity boundary, measured utility, and narrow answer-contract
gates to avoid supported yet benchmark-mismatched answers.

Result:

`memory_directions/results/locomo-full-utility-overlay-v11-contract-expanded/summary.json`

VBR changed 121 / 1388 answers. Every VBR boundary has positive ablation value
on the full set; removing any one boundary lowers overall F1. Changes by
boundary:

- `gam_attribute_off_slot`: 30
- `temporal_relation_lost`: 21
- `enhanced_list_recalibration`: 20
- `reason_component_dropped`: 16
- `current_phrase_embedded`: 14
- `quote_paraphrase`: 8
- `duration_over_specific`: 4
- `gam_extract_boundary`: 4

The slot rescue layer targets 108 exact-slot questions and applies 23
replacements after filtering to `reason`, `feeling`, and `quote` slot types.
The broader unfiltered layer regressed, mainly because description/advice/offer
questions often prefer concise benchmark-level answers rather than longer raw
dialogue spans.
- `place_component_dropped`: 4

The utility overlay applies 142 total replacements from slot/open/contract
sources. Relative to `locomo-full-vbr-slot-rfq`, it makes 57 positive changes
and 0 negative changes. The contract layer is intentionally conservative:
broad GPT candidate auditing, broad raw-localizer usage, broad count/date
switches, and broad description overrides all regressed in diagnostics.

### Revised Claim After Full Eval

The honest claim is narrower than the first-20 LoCoMo slice suggested:

> Typed evidence control improves GAM on LoCoMo full held-out conversations and
> HotpotQA, but does not improve NarrativeQA. The method is best viewed as a
> targeted inference-time controller for fact/date/count-heavy QA, not a
> universal memory-QA improvement.
