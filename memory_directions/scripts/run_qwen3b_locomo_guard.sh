#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PY="${PY:-./.venv/bin/python}"

"$PY" memory_directions/offline_category_guard.py \
  --source cave="memory_directions/results/locomo-v11-qwen25-3b" \
  --source gam="r2m-api/results/gam-locomo-qwen25-3b-full-no26" \
  --category-source temporal=cave \
  --default-source gam \
  --tag "locomo-v11-category-guard-qwen25-3b"

"$PY" memory_directions/scripts/collect_qwen_results.py \
  --prefix qwen25-3b \
  --copy-artifacts
