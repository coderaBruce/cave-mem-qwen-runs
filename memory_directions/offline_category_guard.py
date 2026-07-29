#!/usr/bin/env python
"""Category-aware selection over completed LoCoMo runs.

This is a cheap post-hoc policy for testing whether a backbone needs
category-specific validity boundaries. It does not call an LLM. The default
mode selects a primary run for configured categories and otherwise abstains to a
fallback run. A more general mixer mode can select an arbitrary named source
per category, which is useful for diagnostic upper-bound analysis.
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
from memory_directions.offline_elaa import RESULTS, _answer, _load_rows, _resolve_path


CATEGORY_ALIASES = {
    "1": 1,
    "multi_hop": 1,
    "multi-hop": 1,
    "2": 2,
    "temporal": 2,
    "3": 3,
    "open": 3,
    "open_domain": 3,
    "open-domain": 3,
    "4": 4,
    "single_hop": 4,
    "single-hop": 4,
}


def parse_categories(values: List[str]) -> set[int]:
    out: set[int] = set()
    for value in values:
        key = str(value).strip().lower()
        if key not in CATEGORY_ALIASES:
            raise SystemExit(f"unknown category: {value}")
        out.add(CATEGORY_ALIASES[key])
    return out


def parse_category_source(values: List[str]) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"--category-source expects CATEGORY=SOURCE, got: {value}")
        cat, source = value.split("=", 1)
        cats = parse_categories([cat])
        out[next(iter(cats))] = source.strip()
    return out


def parse_source(values: List[str]) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"--source expects NAME=RUN, got: {value}")
        name, run = value.split("=", 1)
        name = name.strip()
        if not name:
            raise SystemExit(f"empty source name in: {value}")
        out[name] = _resolve_path(run.strip())
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="locomo", choices=["locomo"])
    parser.add_argument("--primary-run", default=None)
    parser.add_argument("--fallback-run", default=None)
    parser.add_argument("--primary-categories", nargs="+", default=["temporal"])
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help="Named source for mixer mode, formatted NAME=RUN.",
    )
    parser.add_argument(
        "--category-source",
        action="append",
        default=[],
        help="Category-specific source for mixer mode, formatted CATEGORY=NAME.",
    )
    parser.add_argument("--default-source", default=None)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--exclude-prefix", nargs="*", default=["conv-26"])
    args = parser.parse_args()

    if args.source:
        source_paths = parse_source(args.source)
        source_rows = {
            name: _load_rows(path / "all_qa_results.json")
            for name, path in source_paths.items()
        }
        category_source = parse_category_source(args.category_source)
        default_source = args.default_source
        if not default_source:
            raise SystemExit("--default-source is required with --source")
        if default_source not in source_rows:
            raise SystemExit(f"--default-source {default_source} is not a named source")
        for source_name in category_source.values():
            if source_name not in source_rows:
                raise SystemExit(f"unknown category source: {source_name}")
        id_sets = [set(rows) for rows in source_rows.values()]
        ids = sorted(set.intersection(*id_sets))
        primary_categories = set(category_source)
    else:
        if not args.primary_run or not args.fallback_run:
            raise SystemExit("set --primary-run/--fallback-run or use --source mixer mode")
        source_paths = {
            "primary": _resolve_path(args.primary_run),
            "fallback": _resolve_path(args.fallback_run),
        }
        source_rows = {
            "primary": _load_rows(source_paths["primary"] / "all_qa_results.json"),
            "fallback": _load_rows(source_paths["fallback"] / "all_qa_results.json"),
        }
        primary_categories = parse_categories(args.primary_categories)
        category_source = {cat: "primary" for cat in primary_categories}
        default_source = "fallback"
        ids = sorted(set(source_rows["primary"]) & set(source_rows["fallback"]))

    for prefix in args.exclude_prefix or []:
        ids = [qid for qid in ids if not qid.startswith(prefix)]

    rows: List[Dict[str, Any]] = []
    changes = {name: 0 for name in source_rows}
    for qid in ids:
        ref = source_rows[default_source][qid]
        category = ref.get("category")
        try:
            category_int = int(category)
        except Exception:
            category_int = -1
        selected_source = category_source.get(category_int, default_source)
        selected = source_rows[selected_source][qid]
        row = dict(selected)
        row["summary_answer"] = _answer(selected)
        row["pred"] = _answer(selected)
        row["category_guard"] = {
            "qid": qid,
            "category": category,
            "selected": selected_source,
            "answers": {name: _answer(source_rows[name][qid]) for name in source_rows},
        }
        changes[selected_source] += 1
        rows.append(row)

    metrics = score(rows, args.dataset)
    outdir = RESULTS / args.tag
    outdir.mkdir(parents=True, exist_ok=True)
    summary = {
        "tag": args.tag,
        "dataset": args.dataset,
        "method": "offline_category_guard",
        "source_runs": {name: str(path) for name, path in source_paths.items()},
        "primary_categories": sorted(primary_categories),
        "category_source": dict(sorted(category_source.items())),
        "default_source": default_source,
        "n_questions": len(rows),
        "selection_counts": changes,
        "metrics": metrics,
    }
    (outdir / "all_qa_results.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (outdir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
