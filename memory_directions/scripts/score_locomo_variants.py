#!/usr/bin/env python
"""Score arbitrary LoCoMo result runs with category breakdowns."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


ROOT = Path(__file__).resolve().parents[2]
R2M_API = ROOT / "r2m-api"
UPSTREAM = ROOT / "R2M-official"
for p in (str(R2M_API), str(UPSTREAM), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from apiharness.scoring import score


CATEGORIES = [
    ("multi_hop", "Multi-hop"),
    ("temporal", "Temporal"),
    ("open_domain", "Open"),
    ("single_hop", "Single-hop"),
]


def resolve_path(value: str) -> Path:
    path = Path(value)
    if path.exists():
        return path
    path = ROOT / value
    if path.exists():
        return path
    path = ROOT / "memory_directions" / "results" / value
    if path.exists():
        return path
    raise FileNotFoundError(value)


def parse_run(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        path = resolve_path(value)
        return path.name, path
    label, raw_path = value.split("=", 1)
    return label.strip(), resolve_path(raw_path.strip())


def load_metrics(run_dir: Path) -> Dict[str, Any]:
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        metrics = summary.get("metrics")
        if isinstance(metrics, dict) and metrics.get("overall"):
            return metrics
    rows_path = run_dir / "all_qa_results.json"
    rows = json.loads(rows_path.read_text(encoding="utf-8"))
    return score([r for r in rows if "error" not in r], "locomo")


def metric_pair(metrics: Dict[str, Any], category: str | None = None) -> Tuple[float | None, float | None]:
    block = metrics.get("overall", {}) if category is None else metrics.get("by_category", {}).get(category, {})
    f1 = block.get("f1")
    bleu = block.get("bleu1")
    return f1, bleu


def fmt(value: float | None) -> str:
    return "--" if value is None else f"{value:.2f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, help="LABEL=path/to/run")
    parser.add_argument("--out", default=None, help="Optional markdown output path.")
    args = parser.parse_args()

    rows: List[str] = []
    header = (
        "| variant | Overall F1 | Overall BLEU | Multi-hop F1 | Multi-hop BLEU | "
        "Temporal F1 | Temporal BLEU | Open F1 | Open BLEU | Single-hop F1 | Single-hop BLEU |"
    )
    sep = "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    rows.extend([header, sep])
    for item in args.run:
        label, run_dir = parse_run(item)
        metrics = load_metrics(run_dir)
        values: List[str] = [label]
        f1, bleu = metric_pair(metrics)
        values.extend([fmt(f1), fmt(bleu)])
        for key, _ in CATEGORIES:
            f1, bleu = metric_pair(metrics, key)
            values.extend([fmt(f1), fmt(bleu)])
        rows.append("| " + " | ".join(values) + " |")

    text = "\n".join(rows) + "\n"
    print(text)
    if args.out:
        out = Path(args.out)
        if not out.is_absolute():
            out = ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
