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
from memory_directions.offline_elaa import RESULTS, _answer, _resolve_path


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


def _load_raw_rows(path: Path) -> List[Dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [r for r in rows if isinstance(r, dict) and "error" not in r]


def _signature(row: Dict[str, Any]) -> tuple[str, str]:
    question = str(row.get("question") or row.get("query") or "").strip().lower()
    category = str(row.get("category") or "").strip()
    return (category, " ".join(question.split()))


def _signature_to_ids(rows_by_id: Dict[str, Dict[str, Any]]) -> Dict[tuple[str, str], List[str]]:
    out: Dict[tuple[str, str], List[str]] = {}
    for qid, row in rows_by_id.items():
        out.setdefault(_signature(row), []).append(qid)
    return out


def _index_rows(
    raw_rows: List[Dict[str, Any]],
    *,
    default_ids: List[str] | None = None,
    signature_ids: Dict[tuple[str, str], List[str]] | None = None,
) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    use_position = bool(default_ids) and len(default_ids or []) == len(raw_rows)
    for idx, raw in enumerate(raw_rows):
        qid = raw.get("_id")
        if not qid and use_position and default_ids is not None:
            qid = default_ids[idx]
        if not qid and signature_ids is not None:
            candidates = signature_ids.get(_signature(raw), [])
            if len(candidates) == 1:
                qid = candidates[0]
        if not qid:
            continue
        row = dict(raw)
        row.setdefault("_id", str(qid))
        out[str(qid)] = row
    return out


def _load_aligned_sources(source_paths: Dict[str, Path], default_source: str) -> Dict[str, Dict[str, Any]]:
    raw_sources = {
        name: _load_raw_rows(path / "all_qa_results.json")
        for name, path in source_paths.items()
    }
    default_rows = _index_rows(raw_sources[default_source])
    default_ids = list(default_rows)
    sig_ids = _signature_to_ids(default_rows)
    out: Dict[str, Dict[str, Any]] = {}
    for name, rows in raw_sources.items():
        if name == default_source:
            out[name] = default_rows
        else:
            out[name] = _index_rows(
                rows,
                default_ids=default_ids,
                signature_ids=sig_ids,
            )
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
        category_source = parse_category_source(args.category_source)
        default_source = args.default_source
        if not default_source:
            raise SystemExit("--default-source is required with --source")
        if default_source not in source_paths:
            raise SystemExit(f"--default-source {default_source} is not a named source")
        for source_name in category_source.values():
            if source_name not in source_paths:
                raise SystemExit(f"unknown category source: {source_name}")
        source_rows = _load_aligned_sources(source_paths, default_source)
        ids = sorted(source_rows[default_source])
        primary_categories = set(category_source)
    else:
        if not args.primary_run or not args.fallback_run:
            raise SystemExit("set --primary-run/--fallback-run or use --source mixer mode")
        source_paths = {
            "primary": _resolve_path(args.primary_run),
            "fallback": _resolve_path(args.fallback_run),
        }
        source_rows = _load_aligned_sources(source_paths, "fallback")
        primary_categories = parse_categories(args.primary_categories)
        category_source = {cat: "primary" for cat in primary_categories}
        default_source = "fallback"
        ids = sorted(set(source_rows["primary"]) & set(source_rows["fallback"]))

    for prefix in args.exclude_prefix or []:
        ids = [qid for qid in ids if not qid.startswith(prefix)]

    rows: List[Dict[str, Any]] = []
    changes = {name: 0 for name in source_rows}
    missing = {name: 0 for name in source_rows}
    for qid in ids:
        ref = source_rows[default_source][qid]
        category = ref.get("category")
        try:
            category_int = int(category)
        except Exception:
            category_int = -1
        selected_source = category_source.get(category_int, default_source)
        if qid not in source_rows[selected_source]:
            missing[selected_source] += 1
            selected_source = default_source
        selected = source_rows[selected_source][qid]
        row = dict(selected)
        row["summary_answer"] = _answer(selected)
        row["pred"] = _answer(selected)
        row["category_guard"] = {
            "qid": qid,
            "category": category,
            "selected": selected_source,
            "answers": {
                name: _answer(rows_by_id[qid])
                for name, rows_by_id in source_rows.items()
                if qid in rows_by_id
            },
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
        "missing_selected_source_counts": missing,
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
