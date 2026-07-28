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

if [[ -z "${PYTHON_BIN:-}" ]]; then
  for candidate in python3.11 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
      PYTHON_BIN="$candidate"
      break
    fi
  done
fi
if [[ -z "${PYTHON_BIN:-}" ]]; then
  echo "No Python interpreter found. Set PYTHON_BIN=/path/to/python and rerun." >&2
  exit 1
fi
INSTALL_VLLM="${INSTALL_VLLM:-1}"

if [[ ! -d .venv ]]; then
  echo "Using Python: $PYTHON_BIN ($("$PYTHON_BIN" --version 2>&1))"
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
