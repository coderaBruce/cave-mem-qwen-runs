#!/usr/bin/env bash
set -euo pipefail

# Run the Qwen2.5-3B core scaling suite against an already-running local vLLM
# endpoint. This intentionally runs only the core methods:
#   Static RAG / GAM / R2Mem / CAVE-Mem
#
# It does not start or stop vLLM. Start the server in a separate screen first:
#   bash memory_directions/scripts/start_qwen3b_vllm.sh
#
# Then run this script in another screen:
#   bash memory_directions/scripts/run_qwen3b_core_suite.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PY="${PY:-./.venv/bin/python}"
PORT="${PORT:-8001}"

export MODEL_SIZE=3b
export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-3B-Instruct}"
export RUN_PREFIX="${RUN_PREFIX:-qwen25-3b}"
export BLOCK=all-core
export WORKERS="${WORKERS:-5}"
export COLLECT_CORE_ONLY=1
export COPY_ARTIFACTS="${COPY_ARTIFACTS:-1}"

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

echo "Qwen2.5-3B core suite:"
echo "  MODEL_NAME=$MODEL_NAME"
echo "  RUN_PREFIX=$RUN_PREFIX"
echo "  OPENAI_LLM_BASE_URL=$OPENAI_LLM_BASE_URL"
echo "  EMBED_MODEL=$EMBED_MODEL"
echo "  WORKERS=$WORKERS"
echo "  RESULTS=qwen_runs/$RUN_PREFIX"
echo "  methods=Static RAG / GAM / R2Mem / CAVE-Mem"
echo "  datasets=LoCoMo full+category, HotpotQA eval_400, NarrativeQA 300"

bash memory_directions/scripts/run_qwen_block.sh

"$PY" memory_directions/scripts/check_qwen_result_health.py \
  --prefix "$RUN_PREFIX" \
  --core-only | tee "qwen_runs/${RUN_PREFIX}/health_core.md"

echo
echo "Leaderboard:"
cat "qwen_runs/${RUN_PREFIX}/leaderboard.md"
echo
echo "LoCoMo category leaderboard:"
cat "qwen_runs/${RUN_PREFIX}/locomo_category_leaderboard.md"
echo
echo "Qwen2.5-3B core suite complete"
