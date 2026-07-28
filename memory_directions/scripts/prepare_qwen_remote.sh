#!/usr/bin/env bash
set -euo pipefail

# Prepare a fresh remote machine for local Qwen evaluation:
#   - create .venv if needed, preferably as a Python 3.11 conda env
#   - install Python dependencies, including vLLM and local embeddings
#   - download HotpotQA/NarrativeQA eval data
#
# Useful options:
#   RECREATE_VENV=1          move the old .venv aside and rebuild it
#   INSTALL_VLLM=0           skip vLLM/PyTorch installation
#   PYTORCH_CUDA=cu128       PyTorch CUDA wheel family, default cu128 for H100
#   VLLM_VERSION=...         optional vLLM pin

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

INSTALL_VLLM="${INSTALL_VLLM:-1}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
PYTORCH_CUDA="${PYTORCH_CUDA:-cu128}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/${PYTORCH_CUDA}}"

if [[ "${RECREATE_VENV:-0}" == "1" && -e .venv ]]; then
  backup=".venv.bak.$(date +%Y%m%d-%H%M%S)"
  echo "Moving existing .venv to $backup"
  mv .venv "$backup"
fi

if [[ ! -d .venv ]]; then
  if command -v conda >/dev/null 2>&1; then
    echo "Creating conda env at .venv with python=${PYTHON_VERSION}"
    conda create -y -p "$ROOT/.venv" "python=${PYTHON_VERSION}"
  else
    if [[ -z "${PYTHON_BIN:-}" ]]; then
      for candidate in python3.11 python3.12 python3 python; do
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
    echo "Creating venv with Python: $PYTHON_BIN ($("$PYTHON_BIN" --version 2>&1))"
    "$PYTHON_BIN" -m venv .venv
  fi
fi

PY="$ROOT/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "Missing .venv/bin/python after environment creation." >&2
  exit 1
fi

echo "Using environment Python: $("$PY" --version 2>&1)"
"$PY" -m pip install -U pip setuptools wheel
"$PY" -m pip install \
  openai anthropic numpy tqdm tiktoken rank_bm25 python-dotenv google-genai \
  pyarrow pydantic sentence-transformers

if [[ "$INSTALL_VLLM" == "1" ]]; then
  echo "Installing PyTorch from $PYTORCH_INDEX_URL"
  "$PY" -m pip uninstall -y torch torchvision torchaudio vllm || true
  "$PY" -m pip install --index-url "$PYTORCH_INDEX_URL" torch torchvision torchaudio
  if [[ -n "${VLLM_VERSION:-}" ]]; then
    "$PY" -m pip install "vllm==${VLLM_VERSION}" --extra-index-url "$PYTORCH_INDEX_URL"
  else
    "$PY" -m pip install vllm --extra-index-url "$PYTORCH_INDEX_URL"
  fi
  "$PY" - <<'PY'
import torch
print("torch", torch.__version__)
print("torch cuda", torch.version.cuda)
print("cuda available", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
fi

bash memory_directions/scripts/fetch_eval_data.sh

echo "remote environment is ready"
