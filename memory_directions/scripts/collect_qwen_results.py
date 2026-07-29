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
    for item in result_defs(args.prefix):
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
    done = sum(1 for row in rows if row["status"] == "done")
    print(f"collected {done}/{len(rows)} result groups -> {outdir}")
    print(f"index -> {outdir / 'summary_index.tsv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
