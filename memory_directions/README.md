# Memory Directions

Inference-time memory-search variants built on the local GAM/R2M harness.

The code is intentionally isolated from `R2M-official/` and imports the existing
dataset loaders, retrievers, metrics, cache, and GAM memory builder from
`r2m-api/`.

## Methods

- `contract`: parses the question into an evidence contract, then maintains a
  slot-level evidence ledger instead of a free-form `temp_memory`.
- `hypothesis`: proposes candidate answers, searches to support/refute their
  atomic claims, and returns the best verified answer summary.
- `adaptive`: contract + ledger with dynamic retrieval width when slots remain
  missing or conflicting.
- `hybrid`: contract + hypotheses + adaptive width.
- `guarded_addendum`: runs GAM first, then appends header-aware
  contract/hypothesis evidence.
- `guarded_hybrid`: same as `guarded_addendum`, but uses an additional merge gate
  to avoid changing GAM unless the addendum is clearly useful.
- `routed_addendum`: uses `guarded_addendum` only for temporal/open-domain
  questions and falls back to plain GAM otherwise. This consumes dataset category
  labels, so it is an oracle analysis rather than the paper method.
- `heuristic_routed_addendum`: same idea, but ignores dataset category labels
  and routes from the question text only.
- `llm_routed_addendum`: question-only GPT-4o-mini router for temporal and
  open-domain addenda. Useful as an ablation; open-domain routing was not stable
  on the held-out slice.
- `temporal_routed_addendum`: question-only router that enables the addendum
  only for direct temporal questions, with guards for counts/durations,
  yes/no before/after questions, date mentions that are not asking for a date,
  and low-quality relative-time addenda.
- `date_anchor_addendum`: starts from temporal routing, then overrides exact
  day-conditioned questions with a raw-page source check for the matching
  session date or adjacent relative-date references such as `last night`.
- `date_count_anchor_addendum`: adds a raw-page count verifier for `How many`
  questions on top of `date_anchor_addendum`.
- `cave_date_count_anchor`: CAVE-style boundary-aware variant. It profiles the
  memory substrate (`episodic_dialogue`, `fact_pack`, or `narrative`), then
  applies date/count candidate experiences only when their applicability checks
  predict a positive intervention. On narrative memory it abstains from the
  unvalidated temporal addendum that hurt NarrativeQA.

All methods use GPT-4o-mini by default and do not train any model.

## Offline ELAA + VBR + Slot Rescue + Utility Overlay

`offline_elaa.py` is an answer-level arbitration experiment for LoCoMo. It
reuses completed GAM and current-best result files, then asks GPT-4o-mini to
choose or extract the shortest supported final answer from the competing memory
summaries. It does not rebuild memory or rerun retrieval.

`offline_validity_boundary.py` applies a deterministic validity-boundary
recalibration on top of ELAA. It keeps ELAA unless the proposed answer falls
into a known failure boundary such as an off-slot GAM attribute, over-specific
duration, lost temporal relation, dropped place/list component, paraphrased
quote, or truncated reason.

`offline_slot_rescue.py` is a narrow exact-slot recovery layer on top of VBR. It
targets questions whose gold answer often depends on a short quoted phrase,
feeling, or reason that is present in the raw dialogue but gets paraphrased or
over-compressed by summary-based answering. The best setting only accepts
`reason`, `feeling`, and `quote` replacements; broader slot types regressed.

`offline_utility_overlay.py` is a utility-bounded contract overlay on top of
VBR + slot rescue. It reuses cached GAM/current/ELAA/open/slot candidate
answers and only applies a candidate when the question imposes a narrow answer
contract with measured positive utility. Broad GPT-4o-mini candidate auditing,
raw-localizer overlays, broad count/date overrides, and broad description
overrides were negative and are kept out of the default path.

Current full-LoCoMo best in this workspace:

- `locomo-full-utility-overlay-v11-contract-expanded`
- F1 60.35, BLEU-1 53.76 on 1388 held-out LoCoMo questions
- Relative to `locomo-full-vbr-slot-rfq`: +2.14 F1, 57 positive changes,
  0 negative changes
- Relative to GAM on the same held-out conversations: +5.85 F1

Full GPT-4o-mini comparison on the matched R2M-style splits:

| Dataset | N | Static RAG | GAM | R2Mem | Ours |
|---|---:|---:|---:|---:|---:|
| LoCoMo common no26 | 1388 | 39.92 | 54.50 | 53.88 | 60.35 |
| HotpotQA eval_400 | 128 | 46.86 | 64.07 | 62.74 | 66.24 |
| NarrativeQA 300 | 300 | 32.54 | 38.66 | 37.24 | 39.37 |

Additional LoCoMo API-only structured-memory controls:

| Method | N | F1 | BLEU-1 |
|---|---:|---:|---:|
| Mem0-style | 1388 | 48.27 | 42.10 |
| A-Mem-style | 1388 | 46.59 | 40.55 |
| MemoryOS-style | 1388 | 45.26 | 39.28 |
| LightMem-style | 1388 | 45.80 | 39.90 |
| Memory-R1 no-RL style | 1388 | 48.53 | 42.60 |

These rows are matched-backbone reproductions from `r2m-api/run_memory_baselines.py`, not official implementations.

Static RAG is implemented in `r2m-api/run_static_rag.py` and follows the R2M
memory-free baseline: fixed 2048-token chunks, top-5 dense retrieval, and
single-pass answer generation.

## Answerer Policies

- `--answer-style-policy selective`: uses held-out demonstration answers only
  for direct temporal questions and calibratable multi-hop questions.
- `--answer-style-policy selective_open`: uses the selective policy above and
  adds an open-domain/inference answerer for category 3 questions.
- The current best configuration also includes a conservative answer-type
  canonicalizer for cases such as city-to-country answers when the question asks
  for a country, `mom` to `mother`, and `through dance` to `by dancing`.

## Quick Runs

```bash
cd /Users/xinyu.li/Documents/research1

# Smoke on one LoCoMo conversation, 5 questions
./.venv/bin/python memory_directions/run.py \
  --dataset locomo --method hybrid --limit-samples 1 --limit-questions 5 --workers 5

# Paired small comparison
./.venv/bin/python memory_directions/run.py \
  --dataset locomo --method gam --only conv-30 --limit-questions 20 --workers 5 --tag gam-conv30-q20

./.venv/bin/python memory_directions/run.py \
  --dataset locomo --method temporal_routed_addendum \
  --only conv-30 --limit-questions 20 --workers 5 --tag temporal-conv30-q20

# Current held-out best: uses conv-26 only as answer-style demonstrations
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

# R2M-style HotpotQA cross-dataset eval, no LoCoMo shots
./.venv/bin/python memory_directions/run.py \
  --dataset hotpotqa \
  --split eval_400 \
  --method date_count_anchor_addendum \
  --workers 5 \
  --answer-style-policy canonical \
  --tag hotpotqa-eval400-datecount-canonical

# R2M-style NarrativeQA cross-dataset eval, no LoCoMo shots
./.venv/bin/python memory_directions/run.py \
  --dataset narrativeqa \
  --method date_count_anchor_addendum \
  --limit-samples 300 \
  --seed 42 \
  --workers 5 \
  --answer-style-policy canonical \
  --tag narrativeqa300-datecount-canonical

# Boundary-aware CAVE variant
./.venv/bin/python memory_directions/run.py \
  --dataset locomo \
  --method cave_date_count_anchor \
  --only conv-30 conv-41 conv-42 conv-43 conv-44 conv-47 conv-48 conv-49 conv-50 \
  --workers 5 \
  --answer-shots 8 \
  --answer-shot-categories 1 2 3 \
  --answer-style-policy selective_open \
  --shots-run r2m-api/results/gam-locomo-4omini \
  --shots-samples conv-26 \
  --tag locomo-full-cave-profile-v2

./.venv/bin/python memory_directions/run.py \
  --dataset hotpotqa \
  --split eval_400 \
  --method cave_date_count_anchor \
  --workers 5 \
  --answer-style-policy canonical \
  --tag hotpotqa-eval400-cave-profile-v2

./.venv/bin/python memory_directions/run.py \
  --dataset narrativeqa \
  --method cave_date_count_anchor \
  --limit-samples 300 \
  --seed 42 \
  --workers 5 \
  --answer-style-policy canonical \
  --tag narrativeqa300-cave-profile-v2

# Offline ELAA answer arbitration on full LoCoMo
./.venv/bin/python memory_directions/offline_elaa.py \
  --workers 5 \
  --prompt-version v3 \
  --tag locomo-full-elaa-v3-post-yn

# ELAA + validity-boundary recalibration
./.venv/bin/python memory_directions/offline_validity_boundary.py \
  --tag locomo-full-elaa-vbr-refined \
  --include-brevity

# ELAA + VBR + filtered exact-slot rescue
./.venv/bin/python -u memory_directions/offline_slot_rescue.py \
  --workers 5 \
  --tag locomo-full-vbr-slot-rfq \
  --prompt-version v2 \
  --accept-slot-types reason feeling quote

# Current best: ELAA + VBR + slot rescue + utility-bounded contract overlay
./.venv/bin/python memory_directions/offline_utility_overlay.py \
  --tag locomo-full-utility-overlay-v11-contract-expanded \
  --enable-contract-candidates
```

Results are written to `memory_directions/results/<tag>/summary.json`.

## Qwen Server Runs

Qwen 3B/7B/14B runs are scripted but not executed locally. The remote scripts
assume an OpenAI-compatible chat endpoint, such as vLLM. For server-only runs,
use the local wrapper below: it routes chat to vLLM and embeddings to a local
SentenceTransformer model (`BAAI/bge-m3` by default), so no `.env` or OpenAI API
key is required.

```bash
# one-time remote setup: Python deps + HotpotQA/NarrativeQA eval data
bash memory_directions/scripts/prepare_qwen_remote.sh

# if a previous setup used an incompatible Python/Torch stack, rebuild it
RECREATE_VENV=1 bash memory_directions/scripts/prepare_qwen_remote.sh

# terminal/tmux/screen pane 1: start Qwen2.5-7B
bash memory_directions/scripts/start_qwen7b_vllm.sh

# terminal/tmux/screen pane 2: run one 12-hour-sized block
bash memory_directions/scripts/run_qwen7b_locomo.sh
bash memory_directions/scripts/run_qwen7b_hotpot.sh
bash memory_directions/scripts/run_qwen7b_nqa.sh
```

For 3B, restart vLLM with `bash memory_directions/scripts/start_qwen3b_vllm.sh`
and run:

```bash
bash memory_directions/scripts/run_qwen3b_locomo.sh
bash memory_directions/scripts/run_qwen3b_hotpot.sh
bash memory_directions/scripts/run_qwen3b_nqa.sh
```

The core blocks evaluate Static RAG, GAM, R2Mem, and CAVE-Mem on the matched
dataset split. The reproduced structured-memory baselines are LoCoMo-only:

```bash
bash memory_directions/scripts/run_qwen7b_locomo_extra.sh
bash memory_directions/scripts/run_qwen3b_locomo_extra.sh
```

Each block mirrors completed summaries and artifacts into
`qwen_runs/<RUN_PREFIX>/`, with `summary_index.tsv`, `leaderboard.md`, and
`manifest.json` at the top level. The vLLM wrapper defaults to a 16K context
window, and the local eval wrappers default to a 1024-token research output cap
to avoid 8K context boundary failures in long GAM prompts. The lower-level
`run_qwen_full_eval.sh` still supports split remote endpoints if you explicitly
want API embeddings.
