#!/usr/bin/env bash
set -euo pipefail

# Run one bounded Qwen evaluation block against an already-running local vLLM
# endpoint. Defaults are 7B + LoCoMo core, so the simplest command is:
#   bash memory_directions/scripts/run_qwen_block.sh
#
# Direct form:
#   bash memory_directions/scripts/run_qwen_block.sh 7b locomo-core
#   bash memory_directions/scripts/run_qwen_block.sh 3b hotpot-core
#
# Blocks:
#   locomo-core       Static RAG + GAM + R2Mem + CAVE-Mem on LoCoMo
#   hotpot-core       Static RAG + GAM + R2Mem + CAVE-Mem on HotpotQA eval_400
#   nqa-core          Static RAG + GAM + R2Mem + CAVE-Mem on NarrativeQA 300
#   locomo-extra      LoCoMo-only reproduced memory baselines
#   all-core          locomo-core + hotpot-core + nqa-core

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

MODEL_SIZE="${MODEL_SIZE:-7b}"
BLOCK="${BLOCK:-locomo-core}"
if [[ $# -ge 1 ]]; then
  MODEL_SIZE="$1"
fi
if [[ $# -ge 2 ]]; then
  BLOCK="$2"
fi

case "$MODEL_SIZE" in
  3b)
    MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-3B-Instruct}"
    RUN_PREFIX="${RUN_PREFIX:-qwen25-3b}"
    ;;
  7b)
    MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
    RUN_PREFIX="${RUN_PREFIX:-qwen25-7b}"
    ;;
  14b)
    MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-14B-Instruct}"
    RUN_PREFIX="${RUN_PREFIX:-qwen25-14b}"
    ;;
  *)
    : "${MODEL_NAME:?Set MODEL_NAME for custom MODEL_SIZE}"
    RUN_PREFIX="${RUN_PREFIX:-$(printf "%s" "$MODEL_NAME" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/-/g; s/^-//; s/-$//')}"
    ;;
esac

PY="${PY:-./.venv/bin/python}"
PY_ABS="$("$PY" -c 'import sys; print(sys.executable)')"
PORT="${PORT:-8001}"
WORKERS="${WORKERS:-5}"
COPY_ARTIFACTS="${COPY_ARTIFACTS:-1}"

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

export MODEL_NAME RUN_PREFIX WORKERS
export OPENAI_LLM_BASE_URL="${OPENAI_LLM_BASE_URL:-http://127.0.0.1:${PORT}/v1}"
export OPENAI_LLM_API_KEY="${OPENAI_LLM_API_KEY:-EMPTY}"
export EMBED_BACKEND="${EMBED_BACKEND:-local}"
export EMBED_MODEL="${EMBED_MODEL:-local:BAAI/bge-m3}"
export LOCAL_EMBED_DEVICE="${LOCAL_EMBED_DEVICE:-cuda}"
export EMBED_BATCH_SIZE="${EMBED_BATCH_SIZE:-16}"
export MEMORY_MAX_TOKENS="${MEMORY_MAX_TOKENS:-256}"
export RESEARCH_MAX_TOKENS="${RESEARCH_MAX_TOKENS:-1024}"
export WORKING_MAX_TOKENS="${WORKING_MAX_TOKENS:-256}"

LOCOMO_EVAL=(conv-30 conv-41 conv-42 conv-43 conv-44 conv-47 conv-48 conv-49 conv-50)

collect_results() {
  local args=(--prefix "$RUN_PREFIX")
  if [[ "$COPY_ARTIFACTS" == "1" ]]; then
    args+=(--copy-artifacts)
  fi
  "$PY_ABS" memory_directions/scripts/collect_qwen_results.py "${args[@]}" || true
}

trap collect_results EXIT

check_endpoint() {
  "$PY_ABS" -c 'import os
from openai import OpenAI
client = OpenAI(base_url=os.environ["OPENAI_LLM_BASE_URL"], api_key=os.environ["OPENAI_LLM_API_KEY"])
resp = client.chat.completions.create(model=os.environ["MODEL_NAME"], messages=[{"role": "user", "content": "Say OK only."}], max_tokens=4)
print("endpoint ok:", resp.choices[0].message.content.strip())'
}

run_static_rag() {
  local dataset="$1"
  shift
  (cd r2m-api && "$PY_ABS" run_static_rag.py \
    --dataset "$dataset" \
    --model "$MODEL_NAME" \
    --embed-model "$EMBED_MODEL" \
    --workers "$WORKERS" \
    "$@")
}

run_locomo_core() {
  run_static_rag locomo \
    --only "${LOCOMO_EVAL[@]}" \
    --tag "rag-locomo-${RUN_PREFIX}-full-no26"

  RUN_BASELINES=1 RUN_OURS=1 RUN_LOCOMO=1 RUN_HOTPOT=0 RUN_NARRATIVE=0 \
    bash memory_directions/scripts/run_qwen_full_eval.sh
}

run_hotpot_core() {
  run_static_rag hotpotqa \
    --split eval_400 \
    --tag "rag-hotpot400-${RUN_PREFIX}"

  RUN_BASELINES=1 RUN_OURS=1 RUN_LOCOMO=0 RUN_HOTPOT=1 RUN_NARRATIVE=0 \
    bash memory_directions/scripts/run_qwen_full_eval.sh
}

run_nqa_core() {
  run_static_rag narrativeqa \
    --limit-samples 300 \
    --seed 42 \
    --tag "rag-nqa300-${RUN_PREFIX}"

  RUN_BASELINES=1 RUN_OURS=1 RUN_LOCOMO=0 RUN_HOTPOT=0 RUN_NARRATIVE=1 \
    bash memory_directions/scripts/run_qwen_full_eval.sh
}

run_locomo_extra() {
  for method in mem0 amem memoryos lightmem memoryr1; do
    (cd r2m-api && "$PY_ABS" run_memory_baselines.py \
      --method "$method" \
      --model "$MODEL_NAME" \
      --embed-model "$EMBED_MODEL" \
      --workers "$WORKERS" \
      --only "${LOCOMO_EVAL[@]}" \
      --tag "${method}-locomo-${RUN_PREFIX}-full-no26")
  done
}

echo "Qwen block eval:"
echo "  MODEL_SIZE=$MODEL_SIZE"
echo "  MODEL_NAME=$MODEL_NAME"
echo "  RUN_PREFIX=$RUN_PREFIX"
echo "  BLOCK=$BLOCK"
echo "  OPENAI_LLM_BASE_URL=$OPENAI_LLM_BASE_URL"
echo "  EMBED_MODEL=$EMBED_MODEL"
echo "  WORKERS=$WORKERS"
echo "  RESULTS=qwen_runs/$RUN_PREFIX"

check_endpoint

case "$BLOCK" in
  locomo|locomo-core)
    run_locomo_core
    ;;
  hotpot|hotpot-core)
    run_hotpot_core
    ;;
  nqa|nqa-core|narrative|narrative-core)
    run_nqa_core
    ;;
  locomo-extra|extra)
    run_locomo_extra
    ;;
  all-core)
    run_locomo_core
    run_hotpot_core
    run_nqa_core
    ;;
  *)
    echo "Unknown BLOCK=$BLOCK" >&2
    exit 2
    ;;
esac

collect_results
trap - EXIT
echo "block complete: $BLOCK for $RUN_PREFIX"
