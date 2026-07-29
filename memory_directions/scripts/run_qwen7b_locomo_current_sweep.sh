#!/usr/bin/env bash
set -euo pipefail

# LoCoMo-only Qwen tuning sweep for the online CAVE current/research stage.
# It does not rerun GAM/R2Mem/extra baselines and does not overwrite the main
# qwen25-7b results. The key diagnostic is whether using the exact GAM memory
# store restores non-temporal categories while preserving temporal gains.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PY="${PY:-./.venv/bin/python}"
PREFIX="${PREFIX:-qwen25-7b}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
WORKERS="${WORKERS:-5}"
PORT="${PORT:-8001}"

if [[ -z "${HF_HOME:-}" ]]; then
  if [[ -d "/data/${USER:-}" ]]; then
    export HF_HOME="/data/${USER}/hf_cache"
  else
    export HF_HOME="$ROOT/.hf_cache"
  fi
fi
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
if [[ -z "${TMPDIR:-}" ]]; then
  if [[ -d "/data/${USER:-}" ]]; then
    export TMPDIR="/data/${USER}/tmp"
  else
    export TMPDIR="$ROOT/.tmp"
  fi
fi
mkdir -p "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$TMPDIR"

export OPENAI_LLM_BASE_URL="${OPENAI_LLM_BASE_URL:-http://127.0.0.1:${PORT}/v1}"
export OPENAI_LLM_API_KEY="${OPENAI_LLM_API_KEY:-EMPTY}"
export EMBED_BACKEND="${EMBED_BACKEND:-local}"
export EMBED_MODEL="${EMBED_MODEL:-local:BAAI/bge-m3}"
export LOCAL_EMBED_DEVICE="${LOCAL_EMBED_DEVICE:-cuda}"
export EMBED_BATCH_SIZE="${EMBED_BATCH_SIZE:-16}"
export MEMORY_MAX_TOKENS="${MEMORY_MAX_TOKENS:-256}"
export RESEARCH_MAX_TOKENS="${RESEARCH_MAX_TOKENS:-768}"
export WORKING_MAX_TOKENS="${WORKING_MAX_TOKENS:-256}"

LOCOMO_EVAL=(conv-30 conv-41 conv-42 conv-43 conv-44 conv-47 conv-48 conv-49 conv-50)
GAM_RAW="r2m-api/results/gam-locomo-${PREFIX}"
GAM_FULL="r2m-api/results/gam-locomo-${PREFIX}-full-no26"
R2M_FULL="r2m-api/results/r2mem-locomo-${PREFIX}-full"
MAIN_CURRENT="memory_directions/results/locomo-current-${PREFIX}"
MAIN_CAVE="memory_directions/results/locomo-v11-${PREFIX}"

if [[ ! -f "$GAM_RAW/conv-26/qa_results.json" ]]; then
  echo "missing GAM raw run for memory/shots: $GAM_RAW/conv-26/qa_results.json" >&2
  exit 2
fi
if [[ ! -f "$GAM_FULL/all_qa_results.json" ]]; then
  echo "missing GAM full-no26 aggregate: $GAM_FULL/all_qa_results.json" >&2
  exit 2
fi
for required in "$R2M_FULL" "$MAIN_CURRENT" "$MAIN_CAVE"; do
  if [[ ! -f "$required/all_qa_results.json" ]]; then
    echo "missing comparison artifact: $required/all_qa_results.json" >&2
    exit 2
  fi
done

echo "Qwen LoCoMo current-stage sweep:"
echo "  PREFIX=$PREFIX"
echo "  MODEL_NAME=$MODEL_NAME"
echo "  OPENAI_LLM_BASE_URL=$OPENAI_LLM_BASE_URL"
echo "  EMBED_MODEL=$EMBED_MODEL"
echo "  WORKERS=$WORKERS"
echo "  GAM_RAW=$GAM_RAW"
echo "  shared memory source: enabled"

run_current() {
  local tag="$1"
  local method="$2"
  shift 2
  local outdir="memory_directions/results/${tag}"
  if [[ -f "$outdir/all_qa_results.json" ]]; then
    echo "skip existing complete run: $tag"
    return
  fi
  "$PY" memory_directions/run.py \
    --dataset locomo \
    --method "$method" \
    --model "$MODEL_NAME" \
    --embed-model "$EMBED_MODEL" \
    --only "${LOCOMO_EVAL[@]}" \
    --workers "$WORKERS" \
    --memory-source-run "$GAM_RAW" \
    --resume-samples \
    --tag "$tag" \
    "$@"
}

# 1) Isolate the effect of sharing the exact GAM memory store while keeping the
# original Qwen LoCoMo current-stage answer policy.
run_current "locomo-tune-shared-current-${PREFIX}" \
  date_count_anchor_addendum \
  --answer-shots 8 \
  --answer-shot-categories 1 2 3 \
  --answer-style-policy selective_open \
  --shots-run "$GAM_RAW" \
  --shots-samples conv-26

# 2) Same research method, but avoid open-domain answer-style prompting.
run_current "locomo-tune-shared-datecount-c2-${PREFIX}" \
  date_count_anchor_addendum \
  --answer-shots 8 \
  --answer-shot-categories 2 \
  --answer-style-policy selective \
  --shots-run "$GAM_RAW" \
  --shots-samples conv-26

# 3) Remove count/open interventions from the online stage; only direct temporal
# questions receive an addendum.
run_current "locomo-tune-shared-temporal-c2-${PREFIX}" \
  temporal_routed_addendum \
  --answer-shots 8 \
  --answer-shot-categories 2 \
  --answer-style-policy selective \
  --shots-run "$GAM_RAW" \
  --shots-samples conv-26

# 4) Use the existing CAVE-style validator before allowing date/count anchors to
# override the primary memory.
run_current "locomo-tune-shared-caveanchor-c2-${PREFIX}" \
  cave_date_count_anchor \
  --answer-shots 8 \
  --answer-shot-categories 2 \
  --answer-style-policy selective \
  --shots-run "$GAM_RAW" \
  --shots-samples conv-26

# 5) Conservative retrieval budget: useful if Qwen over-expands evidence and
# dilutes direct facts.
run_current "locomo-tune-shared-datecount-c2-k3i2-${PREFIX}" \
  date_count_anchor_addendum \
  --max-iters 2 \
  --base-k 3 \
  --wide-k 8 \
  --answer-shots 8 \
  --answer-shot-categories 2 \
  --answer-style-policy selective \
  --shots-run "$GAM_RAW" \
  --shots-samples conv-26

mkdir -p "qwen_runs/${PREFIX}"
"$PY" memory_directions/scripts/score_locomo_variants.py \
  --run "GAM=${GAM_FULL}" \
  --run "R2Mem=${R2M_FULL}" \
  --run "Main Current=${MAIN_CURRENT}" \
  --run "Main CAVE=${MAIN_CAVE}" \
  --run "Shared current=memory_directions/results/locomo-tune-shared-current-${PREFIX}" \
  --run "Shared datecount c2=memory_directions/results/locomo-tune-shared-datecount-c2-${PREFIX}" \
  --run "Shared temporal c2=memory_directions/results/locomo-tune-shared-temporal-c2-${PREFIX}" \
  --run "Shared validated c2=memory_directions/results/locomo-tune-shared-caveanchor-c2-${PREFIX}" \
  --run "Shared datecount c2 k3i2=memory_directions/results/locomo-tune-shared-datecount-c2-k3i2-${PREFIX}" \
  --out "qwen_runs/${PREFIX}/locomo_current_sweep.md"

echo "wrote qwen_runs/${PREFIX}/locomo_current_sweep.md"
