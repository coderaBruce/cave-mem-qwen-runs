#!/usr/bin/env bash
set -euo pipefail

# Run the full GAM/R2Mem/CAVE-Mem pipeline against a local vLLM Qwen endpoint
# and local embeddings. No .env and no OpenAI API key are required.
#
# Default is Qwen2.5-7B-Instruct with RUN_PREFIX=qwen25-7b, so the normal
# command is just:
#   bash memory_directions/scripts/run_qwen_local_full_eval.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
if [[ -z "${RUN_PREFIX:-}" ]]; then
  case "$MODEL_NAME" in
    Qwen/Qwen2.5-3B-Instruct) RUN_PREFIX="qwen25-3b" ;;
    Qwen/Qwen2.5-7B-Instruct) RUN_PREFIX="qwen25-7b" ;;
    Qwen/Qwen2.5-14B-Instruct) RUN_PREFIX="qwen25-14b" ;;
    *)
      RUN_PREFIX="$(printf "%s" "$MODEL_NAME" \
        | tr '[:upper:]' '[:lower:]' \
        | sed -E 's/[^a-z0-9]+/-/g; s/^-//; s/-$//')"
      ;;
  esac
fi

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

PORT="${PORT:-8001}"
export OPENAI_LLM_BASE_URL="${OPENAI_LLM_BASE_URL:-http://127.0.0.1:${PORT}/v1}"
export OPENAI_LLM_API_KEY="${OPENAI_LLM_API_KEY:-EMPTY}"

export EMBED_BACKEND="${EMBED_BACKEND:-local}"
export EMBED_MODEL="${EMBED_MODEL:-local:BAAI/bge-m3}"
export LOCAL_EMBED_DEVICE="${LOCAL_EMBED_DEVICE:-cuda}"
export EMBED_BATCH_SIZE="${EMBED_BATCH_SIZE:-16}"
export MEMORY_MAX_TOKENS="${MEMORY_MAX_TOKENS:-256}"
export RESEARCH_MAX_TOKENS="${RESEARCH_MAX_TOKENS:-1024}"
export WORKING_MAX_TOKENS="${WORKING_MAX_TOKENS:-256}"

export MODEL_NAME
export RUN_PREFIX
export WORKERS="${WORKERS:-5}"

echo "Local Qwen eval:"
echo "  MODEL_NAME=$MODEL_NAME"
echo "  RUN_PREFIX=$RUN_PREFIX"
echo "  OPENAI_LLM_BASE_URL=$OPENAI_LLM_BASE_URL"
echo "  EMBED_MODEL=$EMBED_MODEL"
echo "  WORKERS=$WORKERS"
echo "  RESEARCH_MAX_TOKENS=$RESEARCH_MAX_TOKENS"
echo "  HF_HOME=$HF_HOME"
echo "  TMPDIR=$TMPDIR"

exec bash memory_directions/scripts/run_qwen_full_eval.sh
