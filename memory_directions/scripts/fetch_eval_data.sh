#!/usr/bin/env bash
set -euo pipefail

# Download the HotpotQA and NarrativeQA evaluation files used by the local
# GAM/R2Mem/CAVE-Mem harness. Existing files are kept.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

mkdir -p data/hotpotqa data/narrativeqa

download() {
  local url="$1"
  local out="$2"
  if [[ -s "$out" ]]; then
    echo "skip existing: $out"
    return
  fi
  echo "download: $out"
  curl -L --fail --retry 5 --retry-delay 3 -o "$out" "$url"
}

for split in eval_400 eval_1600 eval_3200; do
  download \
    "https://huggingface.co/datasets/BytedTsinghua-SIA/hotpotqa/resolve/main/${split}.json" \
    "data/hotpotqa/${split}.json"
done

for idx in 0 1 2 3 4 5 6 7; do
  shard="$(printf 'test-%05d-of-00008.parquet' "$idx")"
  download \
    "https://huggingface.co/datasets/deepmind/narrativeqa/resolve/main/data/${shard}" \
    "data/narrativeqa/${shard}"
done

echo "done"
