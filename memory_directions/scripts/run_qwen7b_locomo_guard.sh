#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PY="${PY:-./.venv/bin/python}"

"$PY" memory_directions/offline_category_guard.py \
  --source cave="memory_directions/results/locomo-v11-qwen25-7b" \
  --source gam="r2m-api/results/gam-locomo-qwen25-7b-full-no26" \
  --category-source temporal=cave \
  --default-source gam \
  --tag "locomo-v11-category-guard-qwen25-7b"

"$PY" memory_directions/offline_category_guard.py \
  --source static="r2m-api/results/rag-locomo-qwen25-7b-full-no26" \
  --source mem0="r2m-api/results/mem0-locomo-qwen25-7b-full-no26" \
  --source gam="r2m-api/results/gam-locomo-qwen25-7b-full-no26" \
  --source cave="memory_directions/results/locomo-v11-qwen25-7b" \
  --category-source multi_hop=gam \
  --category-source temporal=mem0 \
  --category-source open=gam \
  --category-source single_hop=gam \
  --default-source gam \
  --tag "locomo-category-oracle-qwen25-7b"

"$PY" memory_directions/scripts/collect_qwen_results.py \
  --prefix qwen25-7b \
  --copy-artifacts
