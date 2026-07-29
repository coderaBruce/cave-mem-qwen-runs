#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_SIZE=7b BLOCK=nqa-extra exec bash "$SCRIPT_DIR/run_qwen_block.sh"
