#!/usr/bin/env bash
set -euo pipefail

# Run the full CAVE-Mem/GAM/R2Mem evaluation pipeline for one OpenAI-compatible
# Qwen endpoint. This script does not start a model server.
#
# Required:
#   MODEL_NAME                 served chat model name, e.g. Qwen/Qwen2.5-7B-Instruct
#
# Endpoint options:
#   OPENAI_LLM_BASE_URL        chat endpoint, e.g. http://127.0.0.1:8001/v1
#   OPENAI_LLM_API_KEY         chat API key, often EMPTY for local vLLM
#   OPENAI_EMBED_API_KEY       embedding key; defaults to OPENAI_API_KEY
#   OPENAI_EMBED_BASE_URL      optional embedding endpoint
#
# Other options:
#   RUN_PREFIX                 result tag suffix; defaults to a slug from MODEL_NAME
#   WORKERS                    default 5
#   EMBED_MODEL                default text-embedding-3-small
#   RUN_BASELINES              1/0, default 1
#   RUN_OURS                   1/0, default 1
#   RUN_LOCOMO                 1/0, default 1
#   RUN_HOTPOT                 1/0, default 1
#   RUN_NARRATIVE              1/0, default 1
#
# Examples:
#   OPENAI_API_KEY=sk-... \
#   OPENAI_LLM_BASE_URL=http://127.0.0.1:8001/v1 \
#   OPENAI_LLM_API_KEY=EMPTY \
#   MODEL_NAME=Qwen/Qwen2.5-7B-Instruct \
#   RUN_PREFIX=qwen25-7b \
#   bash memory_directions/scripts/run_qwen_full_eval.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

: "${MODEL_NAME:?Set MODEL_NAME to the served Qwen model name}"

PY="${PY:-./.venv/bin/python}"
PY_ABS="$("$PY" -c 'import sys; print(sys.executable)')"
WORKERS="${WORKERS:-5}"
EMBED_MODEL="${EMBED_MODEL:-text-embedding-3-small}"
RUN_BASELINES="${RUN_BASELINES:-1}"
RUN_OURS="${RUN_OURS:-1}"
RUN_LOCOMO="${RUN_LOCOMO:-1}"
RUN_HOTPOT="${RUN_HOTPOT:-1}"
RUN_NARRATIVE="${RUN_NARRATIVE:-1}"

if [[ -z "${RUN_PREFIX:-}" ]]; then
  RUN_PREFIX="$(printf "%s" "$MODEL_NAME" \
    | tr '[:upper:]' '[:lower:]' \
    | sed -E 's/[^a-z0-9]+/-/g; s/^-//; s/-$//')"
fi

LOCOMO_EVAL=(conv-30 conv-41 conv-42 conv-43 conv-44 conv-47 conv-48 conv-49 conv-50)

run_r2m() {
  (cd r2m-api && "$PY_ABS" run.py "$@")
}

echo "MODEL_NAME=$MODEL_NAME"
echo "RUN_PREFIX=$RUN_PREFIX"
echo "WORKERS=$WORKERS"
echo "EMBED_MODEL=$EMBED_MODEL"

if [[ "$RUN_BASELINES" == "1" && "$RUN_LOCOMO" == "1" ]]; then
  run_r2m \
    --dataset locomo \
    --method gam \
    --model "$MODEL_NAME" \
    --embed-model "$EMBED_MODEL" \
    --workers "$WORKERS" \
    --tag "gam-locomo-${RUN_PREFIX}"

  run_r2m \
    --dataset locomo \
    --method r2mem \
    --bank banks/locomo-faithful.json \
    --model "$MODEL_NAME" \
    --embed-model "$EMBED_MODEL" \
    --workers "$WORKERS" \
    --only "${LOCOMO_EVAL[@]}" \
    --tag "r2mem-locomo-${RUN_PREFIX}"

  "$PY_ABS" memory_directions/aggregate_per_sample_results.py \
    --run-dir "r2m-api/results/gam-locomo-${RUN_PREFIX}" \
    --dataset locomo \
    --out-dir "r2m-api/results/gam-locomo-${RUN_PREFIX}-full-no26" \
    --exclude-sample conv-26

  "$PY_ABS" memory_directions/aggregate_per_sample_results.py \
    --run-dir "r2m-api/results/r2mem-locomo-${RUN_PREFIX}" \
    --dataset locomo \
    --out-dir "r2m-api/results/r2mem-locomo-${RUN_PREFIX}-full"
fi

if [[ "$RUN_BASELINES" == "1" && "$RUN_HOTPOT" == "1" ]]; then
  run_r2m \
    --dataset hotpotqa \
    --split eval_400 \
    --method gam \
    --model "$MODEL_NAME" \
    --embed-model "$EMBED_MODEL" \
    --workers "$WORKERS" \
    --tag "gam-hotpot400-${RUN_PREFIX}"

  run_r2m \
    --dataset hotpotqa \
    --split eval_400 \
    --method r2mem \
    --bank banks/hotpot-faithful.json \
    --model "$MODEL_NAME" \
    --embed-model "$EMBED_MODEL" \
    --workers "$WORKERS" \
    --tag "r2mem-hotpot400-${RUN_PREFIX}"

  "$PY_ABS" memory_directions/aggregate_per_sample_results.py \
    --run-dir "r2m-api/results/r2mem-hotpot400-${RUN_PREFIX}" \
    --dataset hotpotqa-eval_400 \
    --out-dir "r2m-api/results/r2mem-hotpot400-${RUN_PREFIX}-full"
fi

if [[ "$RUN_BASELINES" == "1" && "$RUN_NARRATIVE" == "1" ]]; then
  run_r2m \
    --dataset narrativeqa \
    --method gam \
    --model "$MODEL_NAME" \
    --embed-model "$EMBED_MODEL" \
    --workers "$WORKERS" \
    --limit-samples 300 \
    --seed 42 \
    --tag "gam-nqa300-${RUN_PREFIX}"

  run_r2m \
    --dataset narrativeqa \
    --method r2mem \
    --bank banks/nqa-faithful.json \
    --model "$MODEL_NAME" \
    --embed-model "$EMBED_MODEL" \
    --workers "$WORKERS" \
    --limit-samples 300 \
    --seed 42 \
    --tag "r2mem-nqa300-${RUN_PREFIX}"

  "$PY_ABS" memory_directions/aggregate_per_sample_results.py \
    --run-dir "r2m-api/results/r2mem-nqa300-${RUN_PREFIX}" \
    --dataset narrativeqa \
    --out-dir "r2m-api/results/r2mem-nqa300-${RUN_PREFIX}-full"
fi

if [[ "$RUN_OURS" == "1" && "$RUN_LOCOMO" == "1" ]]; then
  "$PY_ABS" memory_directions/run.py \
    --dataset locomo \
    --method date_count_anchor_addendum \
    --model "$MODEL_NAME" \
    --embed-model "$EMBED_MODEL" \
    --only "${LOCOMO_EVAL[@]}" \
    --workers "$WORKERS" \
    --answer-shots 8 \
    --answer-shot-categories 1 2 3 \
    --answer-style-policy selective_open \
    --shots-run "r2m-api/results/gam-locomo-${RUN_PREFIX}" \
    --shots-samples conv-26 \
    --tag "locomo-current-${RUN_PREFIX}"

  "$PY_ABS" memory_directions/offline_elaa.py \
    --gam-run "r2m-api/results/gam-locomo-${RUN_PREFIX}" \
    --enhanced-run "memory_directions/results/locomo-current-${RUN_PREFIX}" \
    --model "$MODEL_NAME" \
    --workers "$WORKERS" \
    --prompt-version v3 \
    --tag "locomo-elaa-${RUN_PREFIX}"

  "$PY_ABS" memory_directions/offline_validity_boundary.py \
    --current-run "memory_directions/results/locomo-current-${RUN_PREFIX}" \
    --elaa-run "memory_directions/results/locomo-elaa-${RUN_PREFIX}" \
    --include-brevity \
    --tag "locomo-vbr-${RUN_PREFIX}"

  "$PY_ABS" memory_directions/offline_slot_rescue.py \
    --gam-run "r2m-api/results/gam-locomo-${RUN_PREFIX}" \
    --current-run "memory_directions/results/locomo-current-${RUN_PREFIX}" \
    --elaa-run "memory_directions/results/locomo-elaa-${RUN_PREFIX}" \
    --base-run "memory_directions/results/locomo-vbr-${RUN_PREFIX}" \
    --model "$MODEL_NAME" \
    --workers "$WORKERS" \
    --prompt-version v2 \
    --accept-slot-types reason feeling quote \
    --tag "locomo-slot-rfq-${RUN_PREFIX}"

  "$PY_ABS" memory_directions/offline_slot_rescue.py \
    --gam-run "r2m-api/results/gam-locomo-${RUN_PREFIX}" \
    --current-run "memory_directions/results/locomo-current-${RUN_PREFIX}" \
    --elaa-run "memory_directions/results/locomo-elaa-${RUN_PREFIX}" \
    --base-run "memory_directions/results/locomo-vbr-${RUN_PREFIX}" \
    --model "$MODEL_NAME" \
    --workers "$WORKERS" \
    --prompt-version v2 \
    --tag "locomo-slot-broad-${RUN_PREFIX}"

  "$PY_ABS" memory_directions/offline_open_inference.py \
    --gam-run "r2m-api/results/gam-locomo-${RUN_PREFIX}" \
    --current-run "memory_directions/results/locomo-current-${RUN_PREFIX}" \
    --elaa-run "memory_directions/results/locomo-elaa-${RUN_PREFIX}" \
    --model "$MODEL_NAME" \
    --workers "$WORKERS" \
    --prompt-version v2 \
    --tag "locomo-open-${RUN_PREFIX}"

  "$PY_ABS" memory_directions/offline_utility_overlay.py \
    --base-run "memory_directions/results/locomo-slot-rfq-${RUN_PREFIX}" \
    --gam-run "r2m-api/results/gam-locomo-${RUN_PREFIX}" \
    --current-run "memory_directions/results/locomo-current-${RUN_PREFIX}" \
    --elaa-run "memory_directions/results/locomo-elaa-${RUN_PREFIX}" \
    --slot-run "memory_directions/results/locomo-slot-broad-${RUN_PREFIX}" \
    --open-run "memory_directions/results/locomo-open-${RUN_PREFIX}" \
    --enable-contract-candidates \
    --tag "locomo-v11-${RUN_PREFIX}"
fi

if [[ "$RUN_OURS" == "1" && "$RUN_HOTPOT" == "1" ]]; then
  "$PY_ABS" memory_directions/run.py \
    --dataset hotpotqa \
    --split eval_400 \
    --method cave_date_count_anchor \
    --model "$MODEL_NAME" \
    --embed-model "$EMBED_MODEL" \
    --workers "$WORKERS" \
    --answer-style-policy canonical \
    --tag "hotpotqa-eval400-cave-${RUN_PREFIX}"

  "$PY_ABS" memory_directions/offline_utility_overlay_xdataset.py \
    --dataset hotpotqa \
    --base-run "memory_directions/results/hotpotqa-eval400-cave-${RUN_PREFIX}" \
    --candidate-run "gam=r2m-api/results/gam-hotpot400-${RUN_PREFIX}" \
    --candidate-run "r2m=r2m-api/results/r2mem-hotpot400-${RUN_PREFIX}-full" \
    --tag "hotpotqa-eval400-v11-${RUN_PREFIX}"
fi

if [[ "$RUN_OURS" == "1" && "$RUN_NARRATIVE" == "1" ]]; then
  "$PY_ABS" memory_directions/run.py \
    --dataset narrativeqa \
    --method date_count_anchor_addendum \
    --model "$MODEL_NAME" \
    --embed-model "$EMBED_MODEL" \
    --workers "$WORKERS" \
    --answer-style-policy canonical \
    --limit-samples 300 \
    --seed 42 \
    --tag "narrativeqa300-datecount-${RUN_PREFIX}"

  "$PY_ABS" memory_directions/run.py \
    --dataset narrativeqa \
    --method cave_date_count_anchor \
    --model "$MODEL_NAME" \
    --embed-model "$EMBED_MODEL" \
    --workers "$WORKERS" \
    --answer-style-policy canonical \
    --limit-samples 300 \
    --seed 42 \
    --tag "narrativeqa300-cave-${RUN_PREFIX}"

  "$PY_ABS" memory_directions/offline_utility_overlay_xdataset.py \
    --dataset narrativeqa \
    --base-run "memory_directions/results/narrativeqa300-cave-${RUN_PREFIX}" \
    --candidate-run "gam=r2m-api/results/gam-nqa300-${RUN_PREFIX}" \
    --candidate-run "r2m=r2m-api/results/r2mem-nqa300-${RUN_PREFIX}-full" \
    --candidate-run "datecount=memory_directions/results/narrativeqa300-datecount-${RUN_PREFIX}" \
    --tag "narrativeqa300-v11-${RUN_PREFIX}"
fi

"$PY_ABS" memory_directions/scripts/summarize_full_eval.py --prefix "$RUN_PREFIX" || true
