#!/usr/bin/env python
"""Collect Qwen run outputs into one portable folder.

The evaluation scripts write native outputs under r2m-api/results and
memory_directions/results. This collector mirrors the completed artifacts into
qwen_runs/<prefix>/ so the remote side has one directory to archive or copy
back.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


ROOT = Path(__file__).resolve().parents[2]
LOCOMO_CATEGORIES = [
    ("multi_hop", "Multi-hop"),
    ("temporal", "Temporal"),
    ("open_domain", "Open"),
    ("single_hop", "Single-hop"),
]
LOCOMO_METHOD_ORDER = [
    ("static_rag", "Static RAG"),
    ("mem0", "Mem0$^\\dagger$"),
    ("amem", "A-Mem$^\\dagger$"),
    ("memoryos", "MemoryOS$^\\dagger$"),
    ("lightmem", "LightMem$^\\dagger$"),
    ("memoryr1", "Memory-R1$^\\dagger$"),
    ("gam", "GAM"),
    ("r2mem", "R2Mem"),
    ("cave_mem", "\\method{}"),
    ("cave_mem_guarded", "\\method{}-Guard"),
    ("category_oracle", "Category Oracle"),
]


def result_defs(prefix: str) -> List[Dict[str, str]]:
    rows = [
        ("locomo", "static_rag", f"r2m-api/results/rag-locomo-{prefix}-full-no26"),
        ("locomo", "gam", f"r2m-api/results/gam-locomo-{prefix}-full-no26"),
        ("locomo", "r2mem", f"r2m-api/results/r2mem-locomo-{prefix}-full"),
        ("locomo", "cave_mem", f"memory_directions/results/locomo-v11-{prefix}"),
        ("hotpotqa", "static_rag", f"r2m-api/results/rag-hotpot400-{prefix}"),
        ("hotpotqa", "gam", f"r2m-api/results/gam-hotpot400-{prefix}"),
        ("hotpotqa", "r2mem", f"r2m-api/results/r2mem-hotpot400-{prefix}-full"),
        ("hotpotqa", "cave_mem", f"memory_directions/results/hotpotqa-eval400-v11-{prefix}"),
        ("narrativeqa", "static_rag", f"r2m-api/results/rag-nqa300-{prefix}"),
        ("narrativeqa", "gam", f"r2m-api/results/gam-nqa300-{prefix}"),
        ("narrativeqa", "r2mem", f"r2m-api/results/r2mem-nqa300-{prefix}-full"),
        ("narrativeqa", "cave_mem", f"memory_directions/results/narrativeqa300-v11-{prefix}"),
    ]
    rows.extend(
        ("locomo", method, f"r2m-api/results/{method}-locomo-{prefix}-full-no26")
        for method in ("mem0", "amem", "memoryos", "lightmem", "memoryr1")
    )
    return [{"dataset": d, "method": m, "source": s} for d, m, s in rows]


def optional_result_defs(prefix: str) -> List[Dict[str, str]]:
    rows = [
        (
            "locomo",
            "cave_mem_guarded",
            f"memory_directions/results/locomo-v11-category-guard-{prefix}",
        ),
        (
            "locomo",
            "category_oracle",
            f"memory_directions/results/locomo-category-oracle-{prefix}",
        )
    ]
    return [{"dataset": d, "method": m, "source": s} for d, m, s in rows]


def safe_name(path: str) -> str:
    return path.replace("/", "__").replace(":", "_")


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def metric(summary: Dict[str, Any], key: str) -> Any:
    overall = summary.get("metrics", {}).get("overall", {})
    if key == "bleu":
        return overall.get("bleu") if "bleu" in overall else overall.get("bleu1")
    return overall.get(key)


def mirror_dir(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def write_index(rows: Iterable[Dict[str, Any]], outdir: Path) -> None:
    rows = list(rows)
    columns = ["dataset", "method", "status", "n", "f1", "bleu1", "source", "artifact"]
    lines = ["\t".join(columns)]
    for row in rows:
        lines.append("\t".join(str(row.get(col, "")) for col in columns))
    (outdir / "summary_index.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    md = ["| dataset | method | n | F1 | BLEU | status |", "|---|---:|---:|---:|---:|---|"]
    for row in rows:
        f1 = row.get("f1")
        bleu1 = row.get("bleu1")
        md.append(
            "| {dataset} | {method} | {n} | {f1} | {bleu1} | {status} |".format(
                dataset=row.get("dataset", ""),
                method=row.get("method", ""),
                n=row.get("n", ""),
                f1="" if f1 in (None, "") else f"{float(f1):.4f}",
                bleu1="" if bleu1 in (None, "") else f"{float(bleu1):.4f}",
                status=row.get("status", ""),
            )
        )
    (outdir / "leaderboard.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def _fmt(value: Any) -> str:
    return "--" if value in (None, "") else f"{float(value):.2f}"


def _metric_pair(bucket: Dict[str, Any]) -> tuple[Any, Any]:
    if not bucket:
        return None, None
    bleu = bucket.get("bleu") if "bleu" in bucket else bucket.get("bleu1")
    return bucket.get("f1"), bleu


def write_locomo_category_tables(rows: Iterable[Dict[str, Any]], outdir: Path) -> None:
    by_method = {
        row["method"]: row
        for row in rows
        if row.get("dataset") == "locomo" and row.get("status") == "done"
    }
    if not by_method:
        return

    md = [
        "| method | Overall F1 | Overall BLEU | Multi-hop F1 | Multi-hop BLEU | Temporal F1 | Temporal BLEU | Open F1 | Open BLEU | Single-hop F1 | Single-hop BLEU |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    tex_rows: List[str] = []
    for method, label in LOCOMO_METHOD_ORDER:
        row = by_method.get(method)
        if row is None:
            continue
        summary = load_json(ROOT / row["source"] / "summary.json") or {}
        metrics = summary.get("metrics", {})
        values: List[str] = []
        overall_f1, overall_bleu = _metric_pair(metrics.get("overall", {}))
        values.extend([_fmt(overall_f1), _fmt(overall_bleu)])
        for key, _title in LOCOMO_CATEGORIES:
            f1, bleu = _metric_pair(metrics.get("by_category", {}).get(key, {}))
            values.extend([_fmt(f1), _fmt(bleu)])

        md_label = label.replace("$^\\dagger$", "")
        md.append(f"| {md_label} | " + " | ".join(values) + " |")
        tex_rows.append(f"& {label} & " + " & ".join(values) + r" \\")

    (outdir / "locomo_category_leaderboard.md").write_text(
        "\n".join(md) + "\n", encoding="utf-8"
    )
    (outdir / "locomo_category_rows.tex").write_text(
        "\n".join(tex_rows) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True, help="Run prefix, e.g. qwen25-7b")
    parser.add_argument("--out-root", default="qwen_runs")
    parser.add_argument("--copy-artifacts", action="store_true")
    args = parser.parse_args()

    outdir = ROOT / args.out_root / args.prefix
    summaries_dir = outdir / "summaries"
    artifacts_dir = outdir / "artifacts"
    summaries_dir.mkdir(parents=True, exist_ok=True)
    if args.copy_artifacts:
        artifacts_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    required = result_defs(args.prefix)
    optional = [
        item
        for item in optional_result_defs(args.prefix)
        if (ROOT / item["source"] / "summary.json").exists()
    ]
    for item in required + optional:
        src = ROOT / item["source"]
        summary = load_json(src / "summary.json")
        artifact_name = safe_name(item["source"])
        row: Dict[str, Any] = {
            "dataset": item["dataset"],
            "method": item["method"],
            "source": item["source"],
            "artifact": f"artifacts/{artifact_name}" if args.copy_artifacts else "",
        }
        if summary is None:
            row.update({"status": "missing", "n": "", "f1": "", "bleu1": ""})
            rows.append(row)
            continue

        summary_target = summaries_dir / f"{artifact_name}.summary.json"
        summary_target.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if args.copy_artifacts:
            mirror_dir(src, artifacts_dir / artifact_name)
        row.update(
            {
                "status": "done",
                "n": metric(summary, "n") or summary.get("n_questions", ""),
                "f1": metric(summary, "f1"),
                "bleu1": metric(summary, "bleu"),
            }
        )
        rows.append(row)

    manifest = {
        "prefix": args.prefix,
        "outdir": str(outdir.relative_to(ROOT)),
        "copy_artifacts": args.copy_artifacts,
        "rows": rows,
    }
    (outdir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_index(rows, outdir)
    write_locomo_category_tables(rows, outdir)
    done = sum(1 for row in rows if row["status"] == "done")
    print(f"collected {done}/{len(rows)} result groups -> {outdir}")
    print(f"index -> {outdir / 'summary_index.tsv'}")
    print(f"locomo categories -> {outdir / 'locomo_category_leaderboard.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
