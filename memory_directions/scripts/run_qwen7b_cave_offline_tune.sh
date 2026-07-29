#!/usr/bin/env bash
set -euo pipefail

# Offline CAVE-Mem LoCoMo tuning sweep for an existing Qwen run.
# This does not call vLLM and does not alter baseline results. It only recombines
# already-produced CAVE intermediate artifacts to locate which validation layer
# hurts or helps each LoCoMo category.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PY="${PY:-./.venv/bin/python}"
PREFIX="${PREFIX:-qwen25-7b}"

GAM="r2m-api/results/gam-locomo-${PREFIX}-full-no26"
CURRENT="memory_directions/results/locomo-current-${PREFIX}"
ELAA="memory_directions/results/locomo-elaa-${PREFIX}"
VBR="memory_directions/results/locomo-vbr-${PREFIX}"
RFQ="memory_directions/results/locomo-slot-rfq-${PREFIX}"
BROAD="memory_directions/results/locomo-slot-broad-${PREFIX}"
OPEN="memory_directions/results/locomo-open-${PREFIX}"
FINAL="memory_directions/results/locomo-v11-${PREFIX}"

for path in "$GAM" "$CURRENT" "$ELAA" "$VBR" "$RFQ" "$BROAD" "$OPEN" "$FINAL"; do
  if [[ ! -f "$path/all_qa_results.json" ]]; then
    echo "missing required artifact: $path/all_qa_results.json" >&2
    exit 2
  fi
done

echo "Offline CAVE tuning for PREFIX=$PREFIX"

"$PY" memory_directions/offline_utility_overlay.py \
  --base-run "$RFQ" \
  --gam-run "$GAM" \
  --current-run "$CURRENT" \
  --elaa-run "$ELAA" \
  --slot-run "$BROAD" \
  --open-run "$RFQ" \
  --enable-contract-candidates \
  --tag "locomo-v11-noopen-${PREFIX}"

"$PY" memory_directions/offline_utility_overlay.py \
  --base-run "$RFQ" \
  --gam-run "$GAM" \
  --current-run "$CURRENT" \
  --elaa-run "$ELAA" \
  --slot-run "$BROAD" \
  --open-run "$OPEN" \
  --tag "locomo-v11-no-contract-${PREFIX}"

"$PY" memory_directions/offline_utility_overlay.py \
  --base-run "$RFQ" \
  --gam-run "$GAM" \
  --current-run "$CURRENT" \
  --elaa-run "$ELAA" \
  --slot-run "$BROAD" \
  --open-run "$RFQ" \
  --tag "locomo-v11-noopen-no-contract-${PREFIX}"

"$PY" memory_directions/offline_utility_overlay.py \
  --base-run "$RFQ" \
  --gam-run "$GAM" \
  --current-run "$CURRENT" \
  --elaa-run "$ELAA" \
  --slot-run "$RFQ" \
  --open-run "$OPEN" \
  --enable-contract-candidates \
  --tag "locomo-v11-no-broadslot-${PREFIX}"

mkdir -p "qwen_runs/${PREFIX}"
"$PY" memory_directions/scripts/score_locomo_variants.py \
  --run "GAM=${GAM}" \
  --run "R2Mem=r2m-api/results/r2mem-locomo-${PREFIX}-full" \
  --run "CAVE=${FINAL}" \
  --run "VBR-only=${VBR}" \
  --run "RFQ-only=${RFQ}" \
  --run "CAVE no-open=memory_directions/results/locomo-v11-noopen-${PREFIX}" \
  --run "CAVE no-contract=memory_directions/results/locomo-v11-no-contract-${PREFIX}" \
  --run "CAVE no-open/no-contract=memory_directions/results/locomo-v11-noopen-no-contract-${PREFIX}" \
  --run "CAVE no-broad-slot=memory_directions/results/locomo-v11-no-broadslot-${PREFIX}" \
  --out "qwen_runs/${PREFIX}/cave_offline_tune.md"

echo "wrote qwen_runs/${PREFIX}/cave_offline_tune.md"
