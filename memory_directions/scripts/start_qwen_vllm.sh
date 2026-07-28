#!/usr/bin/env bash
set -euo pipefail

# Start an OpenAI-compatible vLLM server for one Qwen model. This command blocks;
# run it inside tmux/screen, or in a separate terminal, before launching eval.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$MODEL_NAME}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8001}"
DTYPE="${DTYPE:-auto}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
if [[ -z "${VLLM_BIN:-}" ]]; then
  if [[ -x .venv/bin/vllm ]]; then
    VLLM_BIN=".venv/bin/vllm"
  else
    VLLM_BIN="vllm"
  fi
fi

echo "Starting vLLM:"
echo "  MODEL_NAME=$MODEL_NAME"
echo "  SERVED_MODEL_NAME=$SERVED_MODEL_NAME"
echo "  HOST=$HOST"
echo "  PORT=$PORT"
echo "  MAX_MODEL_LEN=$MAX_MODEL_LEN"
echo "  TENSOR_PARALLEL_SIZE=$TENSOR_PARALLEL_SIZE"
echo "  VLLM_BIN=$VLLM_BIN"

exec "$VLLM_BIN" serve "$MODEL_NAME" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --host "$HOST" \
  --port "$PORT" \
  --dtype "$DTYPE" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --enable-prefix-caching
