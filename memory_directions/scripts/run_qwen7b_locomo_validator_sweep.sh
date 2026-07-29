#!/usr/bin/env bash
set -euo pipefail

# Candidate-utility validation sweep for Qwen LoCoMo. Uses existing artifacts
# only. The base is the strongest honest current-stage run; GAM and ELAA runs
# are treated as candidate interventions that must pass an abstaining validator.

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
SHARED_VALIDATED="memory_directions/results/locomo-tune-shared-caveanchor-c2-${PREFIX}"
ELAA_VALIDATED_V3="memory_directions/results/locomo-tune-elaa-shared-validated-v3-${PREFIX}"
ELAA_VALIDATED_V5="memory_directions/results/locomo-tune-elaa-shared-validated-v5-${PREFIX}"
SHARED_TEMPORAL="memory_directions/results/locomo-tune-shared-temporal-c2-${PREFIX}"

for required in "$GAM_FULL" "$R2M_FULL" "$MAIN_CAVE" "$SHARED_VALIDATED" "$ELAA_VALIDATED_V3" "$ELAA_VALIDATED_V5" "$SHARED_TEMPORAL"; do
  if [[ ! -f "$required/all_qa_results.json" ]]; then
    echo "missing required artifact: $required/all_qa_results.json" >&2
    exit 2
  fi
done

echo "Qwen LoCoMo candidate-validator sweep:"
echo "  PREFIX=$PREFIX"
echo "  MODEL_NAME=$MODEL_NAME"
echo "  OPENAI_LLM_BASE_URL=$OPENAI_LLM_BASE_URL"
echo "  WORKERS=$WORKERS"

run_validator() {
  local tag="$1"
  local policy="$2"
  local outdir="memory_directions/results/${tag}"
  if [[ -f "$outdir/all_qa_results.json" ]]; then
    echo "skip existing validator run: $tag"
    return
  fi
  "$PY" memory_directions/offline_candidate_validator.py \
    --base-run "$SHARED_VALIDATED" \
    --base-label "shared_validated" \
    --candidate-run "gam=${GAM_FULL}" \
    --candidate-run "elaa_v3=${ELAA_VALIDATED_V3}" \
    --candidate-run "elaa_v5=${ELAA_VALIDATED_V5}" \
    --candidate-run "shared_temporal=${SHARED_TEMPORAL}" \
    --model "$MODEL_NAME" \
    --workers "$WORKERS" \
    --policy "$policy" \
    --tag "$tag"
}

run_validator "locomo-tune-validator-balanced-${PREFIX}" balanced
run_validator "locomo-tune-validator-directsafe-${PREFIX}" direct_safe

mkdir -p "qwen_runs/${PREFIX}"
"$PY" memory_directions/scripts/score_locomo_variants.py \
  --run "GAM=${GAM_FULL}" \
  --run "R2Mem=${R2M_FULL}" \
  --run "Main CAVE=${MAIN_CAVE}" \
  --run "Shared validated=${SHARED_VALIDATED}" \
  --run "ELAA shared validated v3=${ELAA_VALIDATED_V3}" \
  --run "ELAA shared validated v5=${ELAA_VALIDATED_V5}" \
  --run "Validator balanced=memory_directions/results/locomo-tune-validator-balanced-${PREFIX}" \
  --run "Validator direct-safe=memory_directions/results/locomo-tune-validator-directsafe-${PREFIX}" \
  --out "qwen_runs/${PREFIX}/locomo_validator_sweep.md"

echo "wrote qwen_runs/${PREFIX}/locomo_validator_sweep.md"
