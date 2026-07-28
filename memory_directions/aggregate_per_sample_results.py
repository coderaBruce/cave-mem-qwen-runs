#!/usr/bin/env python
"""Aggregate per-sample QA result files into a run-level summary.

Some baseline runs were completed in multiple missing-only batches, which can
leave the top-level summary stale because the runner overwrites it with the
latest batch. This script treats each ``<sample_id>/qa_results.json`` as the
source of truth and rebuilds the run-level files.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
R2M_API = ROOT / "r2m-api"
UPSTREAM = ROOT / "R2M-official"
for p in (str(R2M_API), str(UPSTREAM), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from apiharness.scoring import score


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_per_sample(
    run_dir: Path,
    *,
    only_samples: set[str] | None = None,
    exclude_samples: set[str] | None = None,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    stats: List[Dict[str, Any]] = []
    for qa_file in sorted(run_dir.glob("*/qa_results.json")):
        sid = qa_file.parent.name
        if only_samples is not None and sid not in only_samples:
            continue
        if exclude_samples is not None and sid in exclude_samples:
            continue
        sample_rows = _read_json(qa_file)
        rows.extend(sample_rows)
        ok = [r for r in sample_rows if "error" not in r]
        iters = [r.get("iterations", 0) for r in ok if r.get("iterations")]
        stats.append(
            {
                "sample_id": sid,
                "n_questions": len(sample_rows),
                "n_errors": len(sample_rows) - len(ok),
                "mean_iterations": (
                    sum(iters) / len(iters) if iters else None
                ),
            }
        )
    return rows, stats


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--only-sample", action="append", default=None)
    parser.add_argument("--exclude-sample", action="append", default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    if not run_dir.exists():
        raise SystemExit(f"missing run dir: {run_dir}")

    rows, per_sample = _load_per_sample(
        run_dir,
        only_samples=set(args.only_sample) if args.only_sample else None,
        exclude_samples=set(args.exclude_sample) if args.exclude_sample else None,
    )
    if not rows:
        raise SystemExit(f"no per-sample qa_results.json files under {run_dir}")

    metrics = score(rows, args.dataset)
    base_summary: Dict[str, Any] = {}
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        try:
            base_summary = _read_json(summary_path)
        except Exception:
            base_summary = {}

    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or out_dir.name
    summary = {
        **base_summary,
        "tag": tag,
        "dataset": args.dataset,
        "source_run": str(run_dir),
        "aggregation": "per_sample_qa_results",
        "only_sample": sorted(args.only_sample or []),
        "exclude_sample": sorted(args.exclude_sample or []),
        "n_samples": len(per_sample),
        "n_questions": len(rows),
        "n_errors": sum(1 for r in rows if "error" in r),
        "metrics": metrics,
        "per_sample": per_sample,
    }

    (out_dir / "all_qa_results.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
