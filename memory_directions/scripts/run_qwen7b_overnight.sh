#!/usr/bin/env bash
set -euo pipefail

# Run the planned unattended 7B suite against an already-running Qwen2.5-7B
# vLLM endpoint. This does not start or stop vLLM.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Starting Qwen2.5-7B overnight suite"
echo "Order:"
echo "  1. LoCoMo core"
echo "  2. LoCoMo extra baselines"
echo "  3. HotpotQA core"
echo "  4. NarrativeQA core"

bash "$SCRIPT_DIR/run_qwen7b_locomo.sh"
bash "$SCRIPT_DIR/run_qwen7b_locomo_extra.sh"
bash "$SCRIPT_DIR/run_qwen7b_hotpot.sh"
bash "$SCRIPT_DIR/run_qwen7b_nqa.sh"

echo "Qwen2.5-7B overnight suite complete"
