#!/usr/bin/env python
"""Label the accumulation split, train the per-query router, and report honestly.

The gate this script exists to answer: **does per-query routing beat the best
fixed strategy?** If a learned router cannot beat "always do the single best
thing for this corpus", it is not a contribution and we should say so.

Three reference points are reported for every evaluation:
  * `best_fixed`  — the best single strategy for that corpus (what our earlier
                    corpus-level router effectively achieved)
  * `router`      — the learned per-query policy, cross-validated
  * `oracle`      — per-query argmax, the ceiling routing could ever reach

If oracle >> best_fixed there is headroom worth chasing; if router ≈ best_fixed
the features are not informative and we report that.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import apiharness  # noqa: F401
from apiharness.qrouter import FEATURE_NAMES, QueryRouter, featurise

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

# corpus -> (base_run, {strategy: run_tag}, accumulation samples, scorer)
CORPORA = {
    "locomo": ("gam-locomo-4omini",
               {"widen": "acc-loco-wide", "evidence": "acc-loco-ev",
                "answer_style": "acc-loco-sh"},
               ["conv-26"], "locomo"),
    "hotpotqa": ("gam-hotpot400",
                 {"widen": "acc-hp-wide", "evidence": "acc-hp-ev",
                  "answer_style": "acc-hp-sh"},
                 None, "hotpotqa"),
    "narrativeqa": ("gam-nqa300",
                    {"widen": "acc-nqa-wide", "evidence": "acc-nqa-ev",
                     "answer_style": "acc-nqa-sh"},
                    None, "narrativeqa"),
}


def scorer_for(kind: str):
    if kind == "locomo":
        from eval.locomo_test import f1_score

        def f(rec):
            g = rec.get("gold_answer")
            return f1_score(rec.get("summary_answer") or "", str(g if g is not None else ""))
    elif kind == "hotpotqa":
        from eval.hotpotqa_test import _calculate_f1

        def f(rec):
            return _calculate_f1(rec.get("pred") or "", rec.get("gold_answers") or [])
    else:
        from eval.narrativeqa_test import _calculate_f1

        def f(rec):
            return _calculate_f1(rec.get("pred") or "", rec.get("gold_answers") or [])
    return f


def load_run(tag: str, samples: Optional[List[str]]) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    run_dir = RESULTS / tag
    for qa in run_dir.glob("*/qa_results.json"):
        if samples and qa.parent.name not in samples:
            continue
        try:
            recs = json.loads(qa.read_text(encoding="utf-8"))
        except Exception:
            continue
        for i, r in enumerate(recs):
            if "error" in r:
                continue
            key = r.get("_id") or f"{qa.parent.name}|{i}"
            r["_sample_dir"] = str(qa.parent)
            out[key] = r
    return out


def build_dataset(name: str) -> Tuple[np.ndarray, List[str], np.ndarray, List[str]]:
    """Return (X, labels, per-strategy F1 matrix, strategy order)."""
    from gam.schemas import InMemoryPageStore

    from apiharness.retrievers import LiteBM25Retriever

    base_tag, strat_tags, samples, kind = CORPORA[name]
    score = scorer_for(kind)

    runs = {"base": load_run(base_tag, samples)}
    for s, tag in strat_tags.items():
        if (RESULTS / tag).exists():
            runs[s] = load_run(tag, None)
    order = [s for s in ["base", "widen", "evidence", "answer_style"] if s in runs]
    common = set(runs["base"])
    for s in order:
        common &= set(runs[s])
    common = sorted(common)
    if not common:
        return np.zeros((0, len(FEATURE_NAMES))), [], np.zeros((0, len(order))), order

    # The answer_style arm on the accumulation split necessarily draws its
    # demonstrations from that same split, so the demonstration questions are
    # leaked. Drop them from the labelled set.
    from apiharness.answer_style import build_style_shots

    shot_qs = set()
    try:
        base_run_dir = RESULTS / base_tag
        shot_samples = samples or sorted(
            {Path(r["_sample_dir"]).name for r in runs["base"].values()})
        for _cat, pairs in build_style_shots(base_run_dir, shot_samples,
                                             per_category=8, seed=42).items():
            shot_qs.update(q for q, _a in pairs)
    except Exception:
        pass

    bm_cache: Dict[str, Any] = {}
    X, labels, F = [], [], []
    n_excluded = 0
    for key in common:
        base = runs["base"][key]
        d = Path(base["_sample_dir"])
        if d not in bm_cache:
            store = InMemoryPageStore(dir_path=str(d))
            pages = store.load()
            if not pages:
                continue
            bm = LiteBM25Retriever({})
            bm.build(store)
            bm_cache[d] = (pages, bm)
        pages, bm = bm_cache[d]

        q = base.get("question") or ""
        if q in shot_qs:
            n_excluded += 1
            continue
        hits = bm.search([q], top_k=20)[0]
        scores = [h.meta.get("score", 0.0) for h in hits]
        retrieved = " ".join(h.snippet for h in hits[:5])
        ans = base.get("summary_answer") or base.get("pred") or ""

        X.append(featurise(q, pages, scores, ans, retrieved).vector())
        f1s = [score(runs[s][key]) for s in order]
        F.append(f1s)
        labels.append(order[int(np.argmax(f1s))])

    if n_excluded:
        print(f"  ({name}: excluded {n_excluded} demonstration questions)")
    return np.vstack(X), labels, np.array(F), order


def evaluate(name: str, X, labels, F, order, folds: int = 5, seed: int = 0):
    from sklearn.model_selection import StratifiedKFold

    n = len(labels)
    idx_of = {s: i for i, s in enumerate(order)}
    mean_by_strategy = F.mean(axis=0)
    best_fixed_i = int(np.argmax(mean_by_strategy))
    best_fixed = mean_by_strategy[best_fixed_i]
    oracle = F.max(axis=1).mean()

    # Cross-validated router
    preds = np.empty(n, dtype=object)
    y = np.array(labels)
    counts = Counter(labels)
    usable = all(c >= folds for c in counts.values()) and len(counts) > 1
    if usable:
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        for tr, te in skf.split(X, y):
            r = QueryRouter().fit(X[tr], list(y[tr]))
            preds[te] = r.predict(X[te])
    else:
        preds[:] = counts.most_common(1)[0][0]

    router_f1 = np.mean([F[i, idx_of[preds[i]]] for i in range(n)])
    acc = float(np.mean([preds[i] == labels[i] for i in range(n)]))

    return {
        "corpus": name,
        "n": n,
        "label_dist": dict(counts),
        "strategy_means": {s: round(100 * mean_by_strategy[i], 2)
                           for i, s in enumerate(order)},
        "best_fixed": round(100 * best_fixed, 2),
        "best_fixed_strategy": order[best_fixed_i],
        "router": round(100 * router_f1, 2),
        "oracle": round(100 * oracle, 2),
        "router_minus_fixed": round(100 * (router_f1 - best_fixed), 2),
        "headroom": round(100 * (oracle - best_fixed), 2),
        "label_accuracy": round(acc, 3),
        "cv_usable": usable,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpora", nargs="*", default=list(CORPORA))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--out", default="../notes/qrouter-report.json")
    args = ap.parse_args()

    allX, allY, reports = [], [], []
    for name in args.corpora:
        try:
            X, labels, F, order = build_dataset(name)
        except Exception as exc:
            print(f"{name}: SKIP ({type(exc).__name__}: {exc})")
            continue
        if not len(labels):
            print(f"{name}: SKIP (no overlapping queries across strategy runs)")
            continue
        rep = evaluate(name, X, labels, F, order, folds=args.folds)
        reports.append(rep)
        allX.append(X)
        allY.extend(labels)
        print(f"\n=== {name} ===")
        print(json.dumps(rep, indent=2))

    if allX:
        X = np.vstack(allX)
        r = QueryRouter().fit(X, allY)
        print("\n=== pooled router rules ===")
        print(r.rules())
        Path(args.out).write_text(
            json.dumps({"per_corpus": reports,
                        "pooled_label_dist": dict(Counter(allY))},
                       ensure_ascii=False, indent=2),
            encoding="utf-8")
        print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
