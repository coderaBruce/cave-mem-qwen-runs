#!/usr/bin/env python
"""Compare runs on a common set of samples.

Scores every run over the *intersection* of their completed samples, so a
partial run is still a fair comparison rather than a misleading one. Reports
F1 / BLEU-1 per category plus mean iterations and tokens, which is the shape
R²-Mem's Table 3 uses.

    python compare.py --runs gam-locomo-4omini r2mem-locomo-4omini --dataset locomo
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import apiharness  # noqa: F401
from apiharness.scoring import score

RESULTS = Path(__file__).resolve().parent / "results"


def load_run(tag: str):
    run_dir = RESULTS / tag
    per_sample = {}
    for qa_file in sorted(run_dir.glob("*/qa_results.json")):
        sid = qa_file.parent.name
        try:
            per_sample[sid] = json.loads(qa_file.read_text(encoding="utf-8"))
        except Exception:
            pass
    return per_sample


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--dataset", default="locomo")
    p.add_argument("--samples", nargs="*", default=None,
                   help="Restrict to these sample ids (default: intersection).")
    p.add_argument("--out", default=None, help="Write the comparison table as JSON.")
    args = p.parse_args()

    loaded = {tag: load_run(tag) for tag in args.runs}
    for tag, per in loaded.items():
        if not per:
            print(f"! run '{tag}' has no completed samples")
            return 1

    common = set.intersection(*(set(per) for per in loaded.values()))
    if args.samples:
        common &= set(args.samples)
    if not common:
        print("no samples in common")
        return 1
    common = sorted(common)
    print(f"comparing on {len(common)} shared samples: {', '.join(common)}\n")

    table = {}
    for tag, per in loaded.items():
        rows = [r for sid in common for r in per[sid]]
        ok = [r for r in rows if "error" not in r]
        metrics = score(ok, args.dataset)
        iters = [r.get("iterations", 0) for r in ok if r.get("iterations")]
        table[tag] = {
            "n_questions": len(rows),
            "n_errors": len(rows) - len(ok),
            "mean_iterations": round(sum(iters) / len(iters), 3) if iters else None,
            "metrics": metrics,
        }

    if args.dataset == "locomo":
        cats = ["multi_hop", "temporal", "open_domain", "single_hop"]
        header = f"{'run':<34}" + "".join(f"{c[:9]:>11}" for c in cats) + \
                 f"{'OVERALL':>11}{'iters':>8}"
        print(header)
        print("-" * len(header))
        for tag, t in table.items():
            bc = t["metrics"].get("by_category", {})
            cells = "".join(
                f"{bc[c]['f1']:>11.2f}" if c in bc else f"{'-':>11}" for c in cats
            )
            ov = t["metrics"].get("overall", {}).get("f1")
            it = t["mean_iterations"]
            print(f"{tag[:33]:<34}{cells}"
                  f"{(f'{ov:.2f}' if ov is not None else '-'):>11}"
                  f"{(f'{it:.2f}' if it is not None else '-'):>8}")
        print("\n(F1 shown; BLEU-1 in the JSON output)")
    else:
        print(f"{'run':<34}{'F1':>10}{'n':>8}{'iters':>8}")
        print("-" * 60)
        for tag, t in table.items():
            ov = t["metrics"].get("overall", {})
            print(f"{tag[:33]:<34}{ov.get('f1', 0):>10.2f}{ov.get('n', 0):>8}"
                  f"{(t['mean_iterations'] or 0):>8.2f}")

    if len(args.runs) == 2:
        a, b = args.runs
        oa = table[a]["metrics"].get("overall", {}).get("f1")
        ob = table[b]["metrics"].get("overall", {}).get("f1")
        if oa is not None and ob is not None:
            print(f"\ndelta ({b} - {a}): {ob - oa:+.2f} F1 "
                  f"({100 * (ob - oa) / oa:+.1f}% relative)")

    if args.out:
        Path(args.out).write_text(
            json.dumps({"samples": common, "table": table}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
