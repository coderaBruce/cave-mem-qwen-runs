#!/usr/bin/env python
"""OUR METHOD: outcome-verified experience selection.

R²-Mem admits an experience into the bank because a GPT-4o rubric liked the step
it was distilled from, and retrieves it because a cosine similarity is high.
Neither fact is evidence that injecting it improves anything. We measured the
consequences of that (notes/02-experiment-log.md E3/E4): on gpt-4o-mini the full
method is 0.92 F1 *below* the GAM baseline it is meant to beat.

This script replaces the rubric criterion with a measured one. For the
accumulation split we have gold answers, a baseline (GAM) answer, and an
experience-augmented answer, plus a per-step log of exactly which experiences
were retrieved. So for question q:

    delta(q) = F1(r2mem answer, gold) - F1(gam answer, gold)

and every experience retrieved while answering q is credited with delta(q):

    u(e) = mean{ delta(q) : e was retrieved for q }

Keep only u(e) > tau. Three consequences worth stating in the paper:

  * **The Evaluator disappears.** No GPT-4o, no rubric, no K_low/K_high. The
    filter is downstream task performance. This removes R²-Mem's external-judge
    dependency, which is the weakest part of their "RL-free and low-cost" pitch.
  * **The discarded middle is recoverable.** Once selection is by measured
    utility, there is no reason to throw away 41.5% of steps for scoring 6-9.
  * **Premature stopping is punished automatically.** Our E4 diagnosis was that
    injected experience makes the agent over-decisive and it stops too early.
    An experience that does that loses F1 and is removed by construction — the
    failure mode is priced in, rather than being invisible to a step-level rubric.

Credit assignment is uniform over the retrieved set. That is coarse — a good and
a bad experience retrieved together get the same credit — but it is unbiased in
expectation over enough questions and costs nothing extra, whereas per-experience
ablation would need one replay per experience.

    python outcome_utility.py --gam-run gam-locomo-4omini --exp-run r2mem-conv26-probe \
        --bank banks/locomo-faithful.json --samples conv-26 \
        --out banks/locomo-outcome.json --tau 0.0
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import apiharness  # noqa: F401
from apiharness.r2mem import ExperienceBank
from apiharness.retrievers import EmbeddingClient

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
CACHE = ROOT / ".cache"


def f1_of(pred: str, gold) -> float:
    from eval.locomo_test import f1_score

    if isinstance(gold, list):
        return max((f1_score(pred, str(g)) for g in gold if g is not None), default=0.0)
    return f1_score(pred, str(gold if gold is not None else ""))


def index_by_question(run_dir: Path, samples):
    """question -> qa record, for one run."""
    out = {}
    for sid in samples:
        f = run_dir / sid / "qa_results.json"
        if not f.exists():
            continue
        for r in json.loads(f.read_text(encoding="utf-8")):
            if "error" not in r and r.get("question"):
                out[(sid, r["question"])] = r
    return out


def experiences_used(qa_record) -> list[str]:
    """Pull the experience strings that were injected while answering."""
    path = qa_record.get("research_trace_file")
    if not path or not Path(path).exists():
        return []
    try:
        trace = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return []
    used = []
    for entry in trace.get("raw_memory", {}).get("experience_log", []):
        for hit in entry.get("retrieved", []):
            if hit.get("experience"):
                used.append(hit["experience"])
    return used


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gam-run", required=True, help="Baseline run tag.")
    p.add_argument("--exp-run", required=True,
                   help="Experience-augmented run over the SAME samples.")
    p.add_argument("--bank", required=True)
    p.add_argument("--samples", nargs="+", required=True,
                   help="Accumulation samples (must be held out from evaluation).")
    p.add_argument("--out", required=True, help="Annotated bank.")
    p.add_argument("--pruned-out", default=None, help="Bank with u(e) > tau only.")
    p.add_argument("--tau", type=float, default=0.0, help="Utility threshold to keep.")
    p.add_argument("--min-trials", type=int, default=1,
                   help="Drop entries measured fewer times than this.")
    p.add_argument("--embed-model", default="text-embedding-3-small")
    args = p.parse_args()

    gam = index_by_question(RESULTS / args.gam_run, args.samples)
    exp = index_by_question(RESULTS / args.exp_run, args.samples)
    common = sorted(set(gam) & set(exp))
    print(f"matched {len(common)} questions across both runs")
    if not common:
        print("no overlap — did both runs cover the same samples?")
        return 1

    embedder = EmbeddingClient(args.embed_model, str(CACHE / "embed"))
    bank = ExperienceBank.load(Path(args.bank), embedder)
    by_text = defaultdict(list)
    for i, e in enumerate(bank.entries):
        by_text[e.experience].append(i)

    deltas = defaultdict(list)          # bank index -> [delta, ...]
    q_deltas = []
    n_no_log = 0

    for key in common:
        gold = gam[key].get("gold_answer")
        f1_gam = f1_of(gam[key].get("summary_answer") or "", gold)
        f1_exp = f1_of(exp[key].get("summary_answer") or "", gold)
        delta = f1_exp - f1_gam
        q_deltas.append(delta)

        used = experiences_used(exp[key])
        if not used:
            n_no_log += 1
            continue
        for text in set(used):
            for idx in by_text.get(text, []):
                deltas[idx].append(delta)

    for idx, ds in deltas.items():
        bank.entries[idx].extra.update(
            {
                "outcome_utility": round(statistics.mean(ds), 5),
                "n_trials": len(ds),
                "n_positive": sum(1 for d in ds if d > 0),
                "n_negative": sum(1 for d in ds if d < 0),
            }
        )
    for e in bank.entries:
        e.extra.setdefault("outcome_utility", None)
        e.extra.setdefault("n_trials", 0)

    bank.save(Path(args.out))

    measured = [e for e in bank.entries if e.extra.get("n_trials", 0) >= args.min_trials]
    keep = [e for e in measured if (e.extra.get("outcome_utility") or 0) > args.tau]
    utils = [e.extra["outcome_utility"] for e in measured]

    print(f"\nquestion-level delta (r2mem - gam): "
          f"mean {statistics.mean(q_deltas):+.4f}  "
          f"median {statistics.median(q_deltas):+.4f}  "
          f"help {sum(1 for d in q_deltas if d>0)} / hurt {sum(1 for d in q_deltas if d<0)} "
          f"/ tie {sum(1 for d in q_deltas if d==0)}")
    print(f"questions with no experience log: {n_no_log}")
    print(f"\nbank: {len(bank)} entries, {len(measured)} measured "
          f"(>= {args.min_trials} trials), {len(keep)} with utility > {args.tau}")
    if utils:
        us = sorted(utils)
        print(f"utility: min {us[0]:+.4f}  q1 {us[len(us)//4]:+.4f}  "
              f"median {us[len(us)//2]:+.4f}  q3 {us[3*len(us)//4]:+.4f}  max {us[-1]:+.4f}")

    if args.pruned_out:
        pruned = ExperienceBank(embedder)
        for e in keep:
            pruned.add(e)
        pruned.save(Path(args.pruned_out))
        print(f"\npruned bank -> {args.pruned_out} ({len(pruned)} entries) "
              f"{json.dumps(pruned.stats())}")

    Path(args.out).with_suffix(".meta.json").write_text(
        json.dumps(
            {
                "gam_run": args.gam_run,
                "exp_run": args.exp_run,
                "samples": args.samples,
                "tau": args.tau,
                "n_questions": len(common),
                "mean_question_delta": statistics.mean(q_deltas),
                "n_measured": len(measured),
                "n_kept": len(keep),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
