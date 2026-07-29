#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_NAME=Qwen/Qwen2.5-3B-Instruct \
SERVED_MODEL_NAME=Qwen/Qwen2.5-3B-Instruct \
exec bash "$SCRIPT_DIR/start_qwen_vllm.sh"
