#!/usr/bin/env bash
set -euo pipefail

# Prepare a fresh remote machine for local Qwen evaluation:
#   - create .venv if needed
#   - install Python dependencies, including vLLM and local embeddings
#   - download HotpotQA/NarrativeQA eval data
#
# Set INSTALL_VLLM=0 if vLLM is already installed in the environment.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3.11}"
INSTALL_VLLM="${INSTALL_VLLM:-1}"

if [[ ! -d .venv ]]; then
  "$PYTHON_BIN" -m venv .venv
fi

source .venv/bin/activate
python -m pip install -U pip setuptools wheel
python -m pip install \
  openai anthropic numpy tqdm tiktoken rank_bm25 python-dotenv google-genai \
  pyarrow pydantic sentence-transformers

if [[ "$INSTALL_VLLM" == "1" ]]; then
  python -m pip install vllm
fi

bash memory_directions/scripts/fetch_eval_data.sh

echo "remote environment is ready"
