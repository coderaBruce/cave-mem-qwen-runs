#!/usr/bin/env bash
set -euo pipefail

# Cheap Qwen answer-style sweep. Reuses the existing CAVE Current research
# summaries and only re-runs the final answerer with different calibration
# policies. Does not run GAM/R2Mem or rebuild memory.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PY="${PY:-./.venv/bin/python}"
PREFIX="${PREFIX:-qwen25-7b}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
WORKERS="${WORKERS:-5}"
SOURCE_RUN="memory_directions/results/locomo-current-${PREFIX}"
SHOTS_RUN="${SHOTS_RUN:-}"

export OPENAI_LLM_BASE_URL="${OPENAI_LLM_BASE_URL:-http://127.0.0.1:8001/v1}"
export OPENAI_LLM_API_KEY="${OPENAI_LLM_API_KEY:-EMPTY}"
export WORKING_MAX_TOKENS="${WORKING_MAX_TOKENS:-256}"

if [[ ! -f "$SOURCE_RUN/all_qa_results.json" ]]; then
  echo "missing source run: $SOURCE_RUN/all_qa_results.json" >&2
  exit 2
fi
if [[ -z "$SHOTS_RUN" ]]; then
  for candidate in \
    "r2m-api/results/gam-locomo-${PREFIX}" \
    "r2m-api/results/gam-locomo-${PREFIX}-full" \
    "r2m-api/results/gam-locomo-${PREFIX}-full-no26"; do
    if [[ -f "$candidate/conv-26/qa_results.json" ]]; then
      SHOTS_RUN="$candidate"
      break
    fi
  done
fi
if [[ ! -f "$SHOTS_RUN/conv-26/qa_results.json" ]]; then
  echo "missing shot source: $SHOTS_RUN/conv-26/qa_results.json" >&2
  exit 2
fi
echo "Re-answer sweep:"
echo "  SOURCE_RUN=$SOURCE_RUN"
echo "  SHOTS_RUN=$SHOTS_RUN"
echo "  MODEL_NAME=$MODEL_NAME"
echo "  WORKERS=$WORKERS"
echo "  OPENAI_LLM_BASE_URL=$OPENAI_LLM_BASE_URL"
echo "  WORKING_MAX_TOKENS=$WORKING_MAX_TOKENS"

run_reanswer() {
  local tag="$1"
  shift
  "$PY" memory_directions/scripts/reanswer_locomo_run.py \
    --source-run "$SOURCE_RUN" \
    --model "$MODEL_NAME" \
    --workers "$WORKERS" \
    --tag "$tag" \
    "$@"
}

run_reanswer "locomo-current-reanswer-plain-${PREFIX}" \
  --answer-style-policy plain

run_reanswer "locomo-current-reanswer-canonical-${PREFIX}" \
  --answer-style-policy canonical

run_reanswer "locomo-current-reanswer-selective-c2-${PREFIX}" \
  --answer-style-policy selective \
  --answer-shots 8 \
  --answer-shot-categories 2 \
  --shots-run "$SHOTS_RUN" \
  --shots-samples conv-26

run_reanswer "locomo-current-reanswer-selective-c12-${PREFIX}" \
  --answer-style-policy selective \
  --answer-shots 8 \
  --answer-shot-categories 1 2 \
  --shots-run "$SHOTS_RUN" \
  --shots-samples conv-26

run_reanswer "locomo-current-reanswer-selective-c123-${PREFIX}" \
  --answer-style-policy selective \
  --answer-shots 8 \
  --answer-shot-categories 1 2 3 \
  --shots-run "$SHOTS_RUN" \
  --shots-samples conv-26

mkdir -p "qwen_runs/${PREFIX}"
"$PY" memory_directions/scripts/score_locomo_variants.py \
  --run "GAM=r2m-api/results/gam-locomo-${PREFIX}-full-no26" \
  --run "Current=memory_directions/results/locomo-current-${PREFIX}" \
  --run "Reanswer plain=memory_directions/results/locomo-current-reanswer-plain-${PREFIX}" \
  --run "Reanswer canonical=memory_directions/results/locomo-current-reanswer-canonical-${PREFIX}" \
  --run "Reanswer selective-c2=memory_directions/results/locomo-current-reanswer-selective-c2-${PREFIX}" \
  --run "Reanswer selective-c12=memory_directions/results/locomo-current-reanswer-selective-c12-${PREFIX}" \
  --run "Reanswer selective-c123=memory_directions/results/locomo-current-reanswer-selective-c123-${PREFIX}" \
  --run "CAVE=memory_directions/results/locomo-v11-${PREFIX}" \
  --out "qwen_runs/${PREFIX}/reanswer_sweep.md"

echo "wrote qwen_runs/${PREFIX}/reanswer_sweep.md"
