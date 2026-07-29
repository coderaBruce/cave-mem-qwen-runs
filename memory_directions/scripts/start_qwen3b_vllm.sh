#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_NAME=Qwen/Qwen2.5-3B-Instruct \
SERVED_MODEL_NAME=Qwen/Qwen2.5-3B-Instruct \
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}" \
exec bash "$SCRIPT_DIR/start_qwen_vllm.sh"
