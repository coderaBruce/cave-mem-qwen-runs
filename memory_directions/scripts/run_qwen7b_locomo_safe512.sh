#!/usr/bin/env bash
set -euo pipefail

# Clean Qwen2.5-7B LoCoMo core rerun with a safer generation budget for
# 16K-context vLLM endpoints. Uses a new prefix so existing qwen25-7b artifacts
# remain untouched.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_SIZE=7b \
BLOCK=locomo-core \
RUN_PREFIX=qwen25-7b-safe512 \
RESEARCH_MAX_TOKENS=512 \
exec bash "$SCRIPT_DIR/run_qwen_block.sh"
