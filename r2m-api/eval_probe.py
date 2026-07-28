#!/usr/bin/env python
"""Gate for path 2: does the LLM bottleneck probe predict the winning strategy?

Compares four policies on the accumulation splits, where we know each query's
per-strategy F1:

  best_fixed   always the single best strategy for that corpus
  probe        route by the LLM's own bottleneck diagnosis
  probe+feat   learned tree over surface features PLUS the probe label
  oracle       per-query argmax (ceiling)

The bar is `best_fixed`. The blind-feature router lost to it on all three
corpora (E29); if the probe also loses, per-query routing is not viable with
anything we have and we should say so.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np

import apiharness  # noqa: F401
from apiharness.cached_generator import CachedOpenAIGenerator
from apiharness.probe import BOTTLENECKS, probe_bottleneck, strategy_for
from apiharness.qrouter import FEATURE_NAMES, QueryRouter, featurise
from train_qrouter import CORPORA, RESULTS, load_run, scorer_for

CACHE = Path(__file__).resolve().parent / ".cache"


def evidence_for(rec, pages) -> str:
    """Reconstruct the passages the base run actually integrated."""
    cited = []
    tf = rec.get("research_trace_file")
    if tf and Path(tf).exists():
        try:
            tr = json.loads(Path(tf).read_text(encoding="utf-8"))
            for it in tr.get("iterations", []):
                for s in (it.get("temp_memory", {}).get("sources") or []):
                    try:
                        cited.append(int(s))
                    except (TypeError, ValueError):
                        pass
        except Exception:
            pass
    seen, out = set(), []
    for pid in cited:
        if pid in seen or not (0 <= pid < len(pages)):
            continue
        seen.add(pid)
        out.append(pages[pid].content[:900])
        if len(out) >= 5:
            break
    return "\n\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpora", nargs="*", default=list(CORPORA))
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--out", default="../notes/probe-report.json")
    args = ap.parse_args()

    from gam.schemas import InMemoryPageStore
    from sklearn.model_selection import StratifiedKFold

    from apiharness.retrievers import LiteBM25Retriever
    import train_qrouter as T

    gen = CachedOpenAIGenerator({
        "model_name": args.model, "cache_dir": str(CACHE / "llm"),
        "temperature": 0.0, "max_tokens": 400, "role": "probe"})

    reports = []
    for name in args.corpora:
        base_tag, strat_tags, samples, kind = CORPORA[name]
        score = scorer_for(kind)
        runs = {"base": load_run(base_tag, samples)}
        for s, tag in strat_tags.items():
            if (RESULTS / tag).exists():
                runs[s] = load_run(tag, None)
        order = [s for s in ["base", "widen", "evidence", "answer_style"] if s in runs]
        common = sorted(set.intersection(*(set(runs[s]) for s in order)))
        if not common:
            print(f"{name}: SKIP")
            continue

        # Same demonstration-leak exclusion as train_qrouter.
        from apiharness.answer_style import build_style_shots
        shot_qs = set()
        try:
            ss = samples or sorted({Path(r["_sample_dir"]).name
                                    for r in runs["base"].values()})
            for _c, pairs in build_style_shots(RESULTS / base_tag, ss,
                                               per_category=8, seed=42).items():
                shot_qs.update(q for q, _a in pairs)
        except Exception:
            pass

        cache: Dict[Path, tuple] = {}
        X, F, labels, probes = [], [], [], []
        for key in common:
            rec = runs["base"][key]
            d = Path(rec["_sample_dir"])
            if d not in cache:
                st = InMemoryPageStore(dir_path=str(d))
                pgs = st.load()
                if not pgs:
                    continue
                bm = LiteBM25Retriever({})
                bm.build(st)
                cache[d] = (pgs, bm)
            pages, bm = cache[d]
            q = rec.get("question") or ""
            if not q or q in shot_qs:
                continue

            hits = bm.search([q], top_k=20)[0]
            scores = [h.meta.get("score", 0.0) for h in hits]
            draft = rec.get("summary_answer") or rec.get("pred") or ""
            ev = evidence_for(rec, pages) or " ".join(h.snippet[:600] for h in hits[:3])

            b, _conf, _tok = probe_bottleneck(gen, q, ev, draft)
            probes.append(b or "NONE")
            X.append(featurise(q, pages, scores, draft,
                               " ".join(h.snippet for h in hits[:5])).vector())
            f1s = [score(runs[s][key]) for s in order]
            F.append(f1s)
            labels.append(order[int(np.argmax(f1s))])

        if not labels:
            print(f"{name}: SKIP (no labelled queries)")
            continue

        X = np.vstack(X)
        F = np.array(F)
        idx = {s: i for i, s in enumerate(order)}
        means = F.mean(axis=0)
        bf_i = int(np.argmax(means))
        best_fixed, oracle = means[bf_i], F.max(axis=1).mean()

        # probe-only policy (fall back to base when the probe's pick isn't available)
        pf = np.mean([F[i, idx.get(strategy_for(probes[i]), idx["base"])]
                      for i in range(len(labels))])

        # probe as an extra feature in the learned tree
        onehot = np.array([[float(probes[i] == b) for b in BOTTLENECKS]
                           for i in range(len(labels))], dtype="float32")
        X2 = np.hstack([X, onehot])
        y = np.array(labels)
        cnt = Counter(labels)
        usable = len(cnt) > 1 and all(c >= args.folds for c in cnt.values())
        if usable:
            preds = np.empty(len(y), dtype=object)
            for tr, te in StratifiedKFold(args.folds, shuffle=True,
                                          random_state=0).split(X2, y):
                preds[te] = QueryRouter().fit(X2[tr], list(y[tr])).predict(X2[te])
            pff = np.mean([F[i, idx[preds[i]]] for i in range(len(y))])
        else:
            pff = float(means[bf_i])

        rep = {
            "corpus": name, "n": len(labels),
            "probe_dist": dict(Counter(probes)),
            "best_fixed": round(100 * best_fixed, 2),
            "best_fixed_strategy": order[bf_i],
            "probe_only": round(100 * pf, 2),
            "probe_plus_features": round(100 * pff, 2),
            "oracle": round(100 * oracle, 2),
            "probe_minus_fixed": round(100 * (pf - best_fixed), 2),
            "probe_label_acc": round(float(np.mean(
                [strategy_for(probes[i]) == labels[i] for i in range(len(labels))])), 3),
            "majority_rate": round(max(cnt.values()) / len(labels), 3),
        }
        reports.append(rep)
        print(f"\n=== {name} ===")
        print(json.dumps(rep, indent=2))

    Path(args.out).write_text(json.dumps(reports, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    print(f"\nprobe tokens billed: {gen.stats()['live_tokens']:,}")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
