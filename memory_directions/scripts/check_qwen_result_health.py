#!/usr/bin/env python
"""Check completed Qwen result artifacts for obvious run-health issues."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


ROOT = Path(__file__).resolve().parents[2]


RESULT_GROUPS = [
    ("locomo", "static_rag", "r2m-api/results/rag-locomo-{prefix}-full-no26"),
    ("locomo", "gam", "r2m-api/results/gam-locomo-{prefix}-full-no26"),
    ("locomo", "r2mem", "r2m-api/results/r2mem-locomo-{prefix}-full"),
    ("locomo", "cave_mem", "memory_directions/results/locomo-v11-{prefix}"),
    ("hotpotqa", "static_rag", "r2m-api/results/rag-hotpot400-{prefix}"),
    ("hotpotqa", "gam", "r2m-api/results/gam-hotpot400-{prefix}"),
    ("hotpotqa", "r2mem", "r2m-api/results/r2mem-hotpot400-{prefix}-full"),
    ("hotpotqa", "cave_mem", "memory_directions/results/hotpotqa-eval400-v11-{prefix}"),
    ("narrativeqa", "static_rag", "r2m-api/results/rag-nqa300-{prefix}"),
    ("narrativeqa", "gam", "r2m-api/results/gam-nqa300-{prefix}"),
    ("narrativeqa", "r2mem", "r2m-api/results/r2mem-nqa300-{prefix}-full"),
    ("narrativeqa", "cave_mem", "memory_directions/results/narrativeqa300-v11-{prefix}"),
    ("locomo", "mem0", "r2m-api/results/mem0-locomo-{prefix}-full-no26"),
    ("locomo", "amem", "r2m-api/results/amem-locomo-{prefix}-full-no26"),
    ("locomo", "memoryos", "r2m-api/results/memoryos-locomo-{prefix}-full-no26"),
    ("locomo", "lightmem", "r2m-api/results/lightmem-locomo-{prefix}-full-no26"),
    ("locomo", "memoryr1", "r2m-api/results/memoryr1-locomo-{prefix}-full-no26"),
]

BAD_PATTERNS = [
    "maximum context length",
    "context length",
    "input_tokens",
    "badrequest",
    "vllmvalidationerror",
    "400 bad request",
]


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def answer_of(row: Dict[str, Any]) -> str:
    return str(row.get("pred") or row.get("summary_answer") or "").strip()


def walk_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from walk_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from walk_strings(item)


def count_bad_strings(paths: Iterable[Path]) -> int:
    count = 0
    for path in paths:
        data = load_json(path)
        if data is None:
            continue
        for text in walk_strings(data):
            lower = text.lower()
            if any(pattern in lower for pattern in BAD_PATTERNS):
                count += 1
    return count


def json_health_paths(run_dir: Path) -> List[Path]:
    paths = []
    for name in ("summary.json", "all_qa_results.json", "run_stats.json"):
        path = run_dir / name
        if path.exists():
            paths.append(path)
    paths.extend(run_dir.glob("*/run_stats.json"))
    paths.extend(run_dir.glob("*/qa_results.json"))
    return paths


def inspect_run(dataset: str, method: str, rel_path: str) -> Dict[str, Any]:
    run_dir = ROOT / rel_path
    out: Dict[str, Any] = {
        "dataset": dataset,
        "method": method,
        "path": rel_path,
        "status": "missing",
        "rows": 0,
        "row_errors": 0,
        "empty_answers": 0,
        "summary_errors": "",
        "bad_strings": 0,
    }
    if not run_dir.exists():
        return out
    out["status"] = "present"
    summary = load_json(run_dir / "summary.json")
    if isinstance(summary, dict):
        if "n_errors" in summary:
            out["summary_errors"] = str(summary.get("n_errors"))
        elif "n_errors" in summary.get("summary", {}):
            out["summary_errors"] = str(summary["summary"].get("n_errors"))
    rows = load_json(run_dir / "all_qa_results.json")
    if isinstance(rows, list):
        out["rows"] = len(rows)
        out["row_errors"] = sum(1 for row in rows if isinstance(row, dict) and "error" in row)
        ok_rows = [row for row in rows if isinstance(row, dict) and "error" not in row]
        out["empty_answers"] = sum(1 for row in ok_rows if not answer_of(row))
    out["bad_strings"] = count_bad_strings(json_health_paths(run_dir))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", default="qwen25-7b")
    parser.add_argument("--dataset", choices=["locomo", "hotpotqa", "narrativeqa"])
    parser.add_argument("--method")
    args = parser.parse_args()

    groups = []
    for dataset, method, tmpl in RESULT_GROUPS:
        if args.dataset and dataset != args.dataset:
            continue
        if args.method and method != args.method:
            continue
        groups.append((dataset, method, tmpl.format(prefix=args.prefix)))

    rows = [inspect_run(*group) for group in groups]
    print("| dataset | method | status | rows | row_errors | empty_answers | summary_errors | bad_strings |")
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row['dataset']} | {row['method']} | {row['status']} | "
            f"{row['rows']} | {row['row_errors']} | {row['empty_answers']} | "
            f"{row['summary_errors']} | {row['bad_strings']} |"
        )
    bad_total = sum(int(row["bad_strings"]) for row in rows)
    empty_total = sum(int(row["empty_answers"]) for row in rows)
    error_total = sum(int(row["row_errors"]) for row in rows)
    print()
    print(
        f"totals: row_errors={error_total} empty_answers={empty_total} "
        f"bad_strings={bad_total}"
    )
    if bad_total or empty_total or error_total:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
