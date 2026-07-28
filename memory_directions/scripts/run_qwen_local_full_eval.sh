#!/usr/bin/env bash
set -euo pipefail

# Run the full GAM/R2Mem/CAVE-Mem pipeline against a local vLLM Qwen endpoint
# and local embeddings. No .env and no OpenAI API key are required.
#
# Example:
#   MODEL_NAME=Qwen/Qwen2.5-7B-Instruct RUN_PREFIX=qwen25-7b \
#     bash memory_directions/scripts/run_qwen_local_full_eval.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

: "${MODEL_NAME:=Qwen/Qwen2.5-7B-Instruct}"

PORT="${PORT:-8001}"
export OPENAI_LLM_BASE_URL="${OPENAI_LLM_BASE_URL:-http://127.0.0.1:${PORT}/v1}"
export OPENAI_LLM_API_KEY="${OPENAI_LLM_API_KEY:-EMPTY}"

export EMBED_BACKEND="${EMBED_BACKEND:-local}"
export EMBED_MODEL="${EMBED_MODEL:-local:BAAI/bge-m3}"
export LOCAL_EMBED_DEVICE="${LOCAL_EMBED_DEVICE:-cuda}"
export EMBED_BATCH_SIZE="${EMBED_BATCH_SIZE:-16}"

export MODEL_NAME
export RUN_PREFIX="${RUN_PREFIX:-$(printf "%s" "$MODEL_NAME" \
  | tr '[:upper:]' '[:lower:]' \
  | sed -E 's/[^a-z0-9]+/-/g; s/^-//; s/-$//')}"
export WORKERS="${WORKERS:-5}"

echo "Local Qwen eval:"
echo "  MODEL_NAME=$MODEL_NAME"
echo "  RUN_PREFIX=$RUN_PREFIX"
echo "  OPENAI_LLM_BASE_URL=$OPENAI_LLM_BASE_URL"
echo "  EMBED_MODEL=$EMBED_MODEL"
echo "  WORKERS=$WORKERS"

exec bash memory_directions/scripts/run_qwen_full_eval.sh
