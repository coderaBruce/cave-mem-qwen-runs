#!/usr/bin/env bash
set -euo pipefail

# LoCoMo-only Qwen recovery sweep. This assumes
# run_qwen7b_locomo_current_sweep.sh has already produced the shared-memory
# current-stage candidates, then applies ELAA answer arbitration to recover
# multi-hop/direct answer spans without rerunning memory construction.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PY="${PY:-./.venv/bin/python}"
PREFIX="${PREFIX:-qwen25-7b}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
WORKERS="${WORKERS:-5}"
PORT="${PORT:-8001}"

export OPENAI_LLM_BASE_URL="${OPENAI_LLM_BASE_URL:-http://127.0.0.1:${PORT}/v1}"
export OPENAI_LLM_API_KEY="${OPENAI_LLM_API_KEY:-EMPTY}"
export WORKING_MAX_TOKENS="${WORKING_MAX_TOKENS:-256}"

GAM_FULL="r2m-api/results/gam-locomo-${PREFIX}-full-no26"
R2M_FULL="r2m-api/results/r2mem-locomo-${PREFIX}-full"
MAIN_CAVE="memory_directions/results/locomo-v11-${PREFIX}"
SHARED_CURRENT="memory_directions/results/locomo-tune-shared-current-${PREFIX}"
SHARED_TEMPORAL="memory_directions/results/locomo-tune-shared-temporal-c2-${PREFIX}"
SHARED_VALIDATED="memory_directions/results/locomo-tune-shared-caveanchor-c2-${PREFIX}"

for required in "$GAM_FULL" "$R2M_FULL" "$MAIN_CAVE" "$SHARED_CURRENT" "$SHARED_TEMPORAL" "$SHARED_VALIDATED"; do
  if [[ ! -f "$required/all_qa_results.json" ]]; then
    echo "missing required artifact: $required/all_qa_results.json" >&2
    exit 2
  fi
done

echo "Qwen LoCoMo recovery sweep:"
echo "  PREFIX=$PREFIX"
echo "  MODEL_NAME=$MODEL_NAME"
echo "  OPENAI_LLM_BASE_URL=$OPENAI_LLM_BASE_URL"
echo "  WORKERS=$WORKERS"

run_elaa() {
  local tag="$1"
  local enhanced="$2"
  local prompt_version="$3"
  local outdir="memory_directions/results/${tag}"
  if [[ -f "$outdir/all_qa_results.json" ]]; then
    echo "skip existing ELAA run: $tag"
    return
  fi
  "$PY" memory_directions/offline_elaa.py \
    --gam-run "$GAM_FULL" \
    --enhanced-run "$enhanced" \
    --model "$MODEL_NAME" \
    --workers "$WORKERS" \
    --prompt-version "$prompt_version" \
    --tag "$tag"
}

run_elaa "locomo-tune-elaa-shared-current-v3-${PREFIX}" "$SHARED_CURRENT" v3
run_elaa "locomo-tune-elaa-shared-temporal-v3-${PREFIX}" "$SHARED_TEMPORAL" v3
run_elaa "locomo-tune-elaa-shared-validated-v3-${PREFIX}" "$SHARED_VALIDATED" v3
run_elaa "locomo-tune-elaa-shared-validated-v5-${PREFIX}" "$SHARED_VALIDATED" v5

mkdir -p "qwen_runs/${PREFIX}"
"$PY" memory_directions/scripts/score_locomo_variants.py \
  --run "GAM=${GAM_FULL}" \
  --run "R2Mem=${R2M_FULL}" \
  --run "Main CAVE=${MAIN_CAVE}" \
  --run "Shared current=${SHARED_CURRENT}" \
  --run "Shared temporal=${SHARED_TEMPORAL}" \
  --run "Shared validated=${SHARED_VALIDATED}" \
  --run "ELAA shared current=memory_directions/results/locomo-tune-elaa-shared-current-v3-${PREFIX}" \
  --run "ELAA shared temporal=memory_directions/results/locomo-tune-elaa-shared-temporal-v3-${PREFIX}" \
  --run "ELAA shared validated v3=memory_directions/results/locomo-tune-elaa-shared-validated-v3-${PREFIX}" \
  --run "ELAA shared validated v5=memory_directions/results/locomo-tune-elaa-shared-validated-v5-${PREFIX}" \
  --out "qwen_runs/${PREFIX}/locomo_recovery_sweep.md"

echo "wrote qwen_runs/${PREFIX}/locomo_recovery_sweep.md"
