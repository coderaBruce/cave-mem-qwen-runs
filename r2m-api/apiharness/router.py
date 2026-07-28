"""OUR METHOD: bottleneck-routed deep memory search.

GAM and R²-Mem apply one fixed policy to every query on every corpus — fixed
retrieval width (5 per retriever), fixed reflection depth (3), fixed answer
prompt. Our measurements say the three benchmarks have *different binding
constraints*, and that relieving a non-binding one costs more than it earns:

| benchmark | pages/sample | golds are spans? | binding constraint | helps | hurts |
|---|---|---|---|---|---|
| LoCoMo | ~19 (corpus fits) | yes | answer granularity | evidence +0.46 | wide −1.54 |
| HotpotQA | 26+ w/ distractors | yes | recall under distractors | wide +3.60 | evidence +0.22 |
| NarrativeQA | 23–140 | **no, 69% abstractive** | abstractive synthesis | shots +0.91 | wide −0.78, evidence −1.52 |

And stacking two strategies always lost to the better single one (E22). So the
contribution is *choosing correctly per query*, not doing everything.

**Scope, stated honestly.** The profile is *corpus-level*: both signals are
aggregates over the accumulation split, so the router emits one decision per
corpus, not per query. With three benchmarks that is three decisions from two
thresholds, which on its own is weak evidence however causal the signals are.
The claim only becomes testable by routing corpora the thresholds were never set
on — the HotpotQA 224K/448K splits, and a second backbone. Per-query routing on
the same signals is possible (recall_gain can be estimated per question from
score margins) but is not what is implemented here.

**Signals.** Both are estimated once on the *accumulation split* (the same
10-20% budget R²-Mem spends building its experience bank) and never touch
evaluation gold. An earlier design using `extractive_rate` and `coverage`
failed — it routed 1 of 3 correctly, because LoCoMo's golds turn out to be
almost as non-verbatim as NarrativeQA's (0.227 vs 0.237) and their page counts
are nearly identical (29 vs 26). These two work because each measures, directly
and causally, whether the corresponding intervention *can* help:

1. `recall_gain = R@20 − R@5` — how much widening retrieval adds gold-bearing
   pages. If a wider net catches no more gold, widening can only add noise.
2. `gold_page_fraction` — median fraction of a corpus's pages that contain the
   gold string. Low means the answer is localised, so handing the cited pages
   to the answer step isolates it; high means the gold tokens are diffuse
   boilerplate and passing pages cannot localise anything.

Measured on the accumulation splits:

| | recall_gain | gold_page_fraction | route | best measured |
|---|---|---|---|---|
| LoCoMo | +0.061 | 0.053 | EVIDENCE | evidence +0.46 |
| HotpotQA | **+0.375** | 0.038 | WIDEN | wide **+3.60** |
| NarrativeQA | +0.000 | **0.333** | ANSWER_STYLE | shots +0.91 |

**Rule:**

    if recall_gain > tau_recall:            WIDEN         # recall-bound
    elif gold_page_fraction < tau_local:    EVIDENCE      # localised evidence
    else:                                   ANSWER_STYLE  # diffuse / abstractive
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

WIDEN = "widen"
EVIDENCE = "evidence"
ANSWER_STYLE = "answer_style"

TAU_RECALL_GAIN = 0.15          # above this, widening still catches gold
TAU_GOLD_PAGE_FRACTION = 0.15   # below this, gold is localised enough to hand over
BASE_WIDTH = 5           # GAM's shipped per-retriever top_k
WIDE_WIDTH = 20


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", (s or "").lower())


@dataclass
class CorpusProfile:
    """Signals measured on the accumulation split."""
    recall_gain: float
    gold_page_fraction: float
    n_probes: int
    recall_at_base: float
    recall_at_wide: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "recall_gain": round(self.recall_gain, 4),
            "gold_page_fraction": round(self.gold_page_fraction, 4),
            "n_probes": self.n_probes,
            "recall_at_base": round(self.recall_at_base, 4),
            "recall_at_wide": round(self.recall_at_wide, 4),
        }


def profile_corpus(
    run_dir: Path,
    samples: List[str],
    base_width: int = BASE_WIDTH,
    wide_width: int = WIDE_WIDTH,
    min_gold_chars: int = 4,
) -> CorpusProfile:
    """Measure both routing signals from a finished baseline run's accumulation split."""
    import statistics

    from gam.schemas import InMemoryPageStore

    from .retrievers import LiteBM25Retriever

    r_base = r_wide = total = 0
    fracs: List[float] = []

    for sid in samples:
        d = Path(run_dir) / sid
        if not (d / "pages.json").exists() or not (d / "qa_results.json").exists():
            continue
        store = InMemoryPageStore(dir_path=str(d))
        pages = store.load()
        if not pages:
            continue
        bm25 = LiteBM25Retriever({})
        bm25.build(store)

        for rec in json.loads((d / "qa_results.json").read_text(encoding="utf-8")):
            if "error" in rec or not rec.get("question"):
                continue
            gold = rec.get("gold_answer")
            golds = (
                [gold] if (gold is not None and not isinstance(gold, list))
                else (rec.get("gold_answers") or [])
            )
            for cand in golds:
                g = _norm(str(cand)).strip()
                if len(g) < min_gold_chars:
                    continue
                bearing = [i for i, p in enumerate(pages) if g in _norm(p.content)]
                if not bearing:
                    continue
                total += 1
                fracs.append(len(bearing) / len(pages))
                ids = [int(h.page_id) for h in
                       bm25.search([rec["question"]], top_k=wide_width)[0]]
                if any(i in ids[:base_width] for i in bearing):
                    r_base += 1
                if any(i in ids[:wide_width] for i in bearing):
                    r_wide += 1
                break

    if not total:
        # No span-answerable probes at all -> maximally abstractive.
        return CorpusProfile(0.0, 1.0, 0, 0.0, 0.0)
    return CorpusProfile(
        recall_gain=(r_wide - r_base) / total,
        gold_page_fraction=statistics.median(fracs),
        n_probes=total,
        recall_at_base=r_base / total,
        recall_at_wide=r_wide / total,
    )


@dataclass
class RouteDecision:
    strategy: str
    width: int
    reason: str

    def as_dict(self) -> Dict[str, Any]:
        return {"strategy": self.strategy, "width": self.width, "reason": self.reason}


class BottleneckRouter:
    """Pick one strategy from the corpus profile."""

    def __init__(
        self,
        profile: CorpusProfile,
        tau_recall: float = TAU_RECALL_GAIN,
        tau_local: float = TAU_GOLD_PAGE_FRACTION,
        base_width: int = BASE_WIDTH,
        wide_width: int = WIDE_WIDTH,
    ):
        self.profile = profile
        self.tau_recall = tau_recall
        self.tau_local = tau_local
        self.base_width = base_width
        self.wide_width = wide_width

    def route(self) -> RouteDecision:
        p = self.profile
        if p.recall_gain > self.tau_recall:
            return RouteDecision(
                WIDEN, self.wide_width,
                f"recall_gain {p.recall_gain:.3f} > {self.tau_recall}: widening adds gold",
            )
        if p.gold_page_fraction < self.tau_local:
            return RouteDecision(
                EVIDENCE, self.base_width,
                f"recall saturated (gain {p.recall_gain:.3f}) and evidence is localised "
                f"(gold_page_fraction {p.gold_page_fraction:.3f} < {self.tau_local})",
            )
        return RouteDecision(
            ANSWER_STYLE, self.base_width,
            f"recall saturated and gold diffuse "
            f"(gold_page_fraction {p.gold_page_fraction:.3f} >= {self.tau_local}): "
            f"answer-side bottleneck",
        )
