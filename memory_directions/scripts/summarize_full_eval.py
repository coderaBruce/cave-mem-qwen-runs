#!/usr/bin/env python
"""Print a compact full-eval table for a run prefix."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional


ROOT = Path(__file__).resolve().parents[2]


def _summary(path: str) -> Optional[Dict[str, Any]]:
    file = ROOT / path / "summary.json"
    if not file.exists():
        return None
    return json.loads(file.read_text(encoding="utf-8"))


def _f1(path: str) -> tuple[Optional[int], Optional[float]]:
    summary = _summary(path)
    if summary is None:
        return None, None
    overall = summary.get("metrics", {}).get("overall", {})
    return overall.get("n") or summary.get("n_questions"), overall.get("f1")


def _fmt(value: Optional[float]) -> str:
    return "-" if value is None else f"{value:.4f}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", required=True)
    args = parser.parse_args()

    rows = [
        (
            "LoCoMo common no26",
            f"r2m-api/results/gam-locomo-{args.prefix}-full-no26",
            f"r2m-api/results/r2mem-locomo-{args.prefix}-full",
            f"memory_directions/results/locomo-v11-{args.prefix}",
        ),
        (
            "HotpotQA eval_400",
            f"r2m-api/results/gam-hotpot400-{args.prefix}",
            f"r2m-api/results/r2mem-hotpot400-{args.prefix}-full",
            f"memory_directions/results/hotpotqa-eval400-v11-{args.prefix}",
        ),
        (
            "NarrativeQA 300",
            f"r2m-api/results/gam-nqa300-{args.prefix}",
            f"r2m-api/results/r2mem-nqa300-{args.prefix}-full",
            f"memory_directions/results/narrativeqa300-v11-{args.prefix}",
        ),
    ]

    print(f"Full evaluation summary for prefix={args.prefix}")
    print(f"{'dataset':<24} {'n':>6} {'GAM':>10} {'R2Mem':>10} {'Ours':>10} {'vsGAM':>10} {'vsR2Mem':>10}")
    print("-" * 86)
    for dataset, gam_path, r2m_path, ours_path in rows:
        n_gam, gam = _f1(gam_path)
        n_r2m, r2m = _f1(r2m_path)
        n_ours, ours = _f1(ours_path)
        n = n_ours or n_gam or n_r2m
        vs_gam = ours - gam if ours is not None and gam is not None else None
        vs_r2m = ours - r2m if ours is not None and r2m is not None else None
        print(
            f"{dataset:<24} {str(n or '-'):>6} "
            f"{_fmt(gam):>10} {_fmt(r2m):>10} {_fmt(ours):>10} "
            f"{_fmt(vs_gam):>10} {_fmt(vs_r2m):>10}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
