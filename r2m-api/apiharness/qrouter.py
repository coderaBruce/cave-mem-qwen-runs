"""Per-query strategy routing, learned on the accumulation split.

Supersedes the corpus-level router in `router.py`. That one emitted one decision
per corpus from gold-dependent aggregates, which is three decisions on three
benchmarks — not defensible, and not per-query.

**Why hand-written per-query rules failed.** We measured the obvious query-time
signals on the accumulation splits:

| | tail_mass | margin@1-5 | lexically abstractive |
|---|---|---|---|
| LoCoMo | 0.439 | 0.595 | 13.2% |
| HotpotQA | 0.544 | 0.410 | 0.0% |
| NarrativeQA | 0.552 | 0.418 | 26.7% |

Retrieval shape cannot separate HotpotQA from NarrativeQA (0.544 vs 0.552), and a
lexical why/how detector fires on only 27% of NarrativeQA. What actually separates
them — whether the answer exists as a span in the corpus at all (96% vs 31%) — is a
property of the annotation style, not visible in the query or the score curve.

**So the router is learned, not hand-designed.** On the accumulation split we run all
three strategies, score each query under each, and label it with the argmax. A small
classifier then maps cheap query-time features to that label. Training data is the same
10-20% budget R²-Mem spends building its experience bank, and evaluation gold is never
touched.

**Features** (all observable at inference, none requiring gold):

- `n_pages`                 corpus size for this sample
- `tail_mass`               BM25 score mass below the base cutoff — is there comparable
                            material just past rank 5 that widening would catch?
- `margin_15`, `margin_110` peakedness of the score curve — retrieval confidence
- `top_score`, `score_std`  absolute scale and spread
- `q_len`, wh-type one-hot  question form
- `grounded`                does the *base* system's own answer appear verbatim in its
                            retrieved pages? This is the query-level proxy for the
                            extractive/abstractive regime, and it needs only the base
                            run, not gold.

`grounded` implies a two-pass design at inference: run base GAM, featurise, then re-run
under the chosen strategy if it differs. We report that cost honestly rather than hiding
it — the first pass is the baseline's own cost, so routed inference is at most ~2x
baseline, and only for queries the router moves off the default.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

STRATEGIES = ["base", "widen", "evidence", "answer_style"]
WH = ["who", "what", "when", "where", "why", "how", "which", "other"]


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", (s or "").lower())


def _wh_type(q: str) -> str:
    ql = (q or "").strip().lower()
    for w in WH[:-1]:
        if ql.startswith(w):
            return w
    for w in WH[:-1]:
        if re.search(rf"\b{w}\b", ql[:60]):
            return w
    return "other"


FEATURE_NAMES = (
    ["n_pages", "log_pages", "tail_mass", "margin_15", "margin_110",
     "top_score", "score_std", "q_len", "grounded", "answer_len"]
    + [f"wh_{w}" for w in WH]
)


@dataclass
class QueryFeatures:
    values: Dict[str, float] = field(default_factory=dict)

    def vector(self) -> np.ndarray:
        return np.array([self.values.get(n, 0.0) for n in FEATURE_NAMES], dtype="float32")


def featurise(
    question: str,
    pages: Sequence[Any],
    bm25_scores: Sequence[float],
    base_answer: str,
    retrieved_text: str,
) -> QueryFeatures:
    s = list(bm25_scores) or [0.0]
    total = sum(s) or 1e-9
    n = len(pages) or 1
    ans = _norm(base_answer).strip()
    v = {
        "n_pages": float(n),
        "log_pages": float(np.log1p(n)),
        "tail_mass": float(sum(s[5:]) / total),
        "margin_15": float((s[0] - s[min(4, len(s) - 1)]) / (s[0] + 1e-9)),
        "margin_110": float((s[0] - s[min(9, len(s) - 1)]) / (s[0] + 1e-9)),
        "top_score": float(s[0]),
        "score_std": float(np.std(s)),
        "q_len": float(len((question or "").split())),
        # Is the base system's own answer literally present in what it retrieved?
        "grounded": float(bool(ans) and len(ans) > 2 and ans in _norm(retrieved_text)),
        "answer_len": float(len((base_answer or "").split())),
    }
    for w in WH:
        v[f"wh_{w}"] = float(_wh_type(question) == w)
    return QueryFeatures(v)


class QueryRouter:
    """Small classifier over query features; falls back to the training majority."""

    def __init__(self, max_depth: int = 3, min_leaf: int = 15, seed: int = 0):
        from sklearn.tree import DecisionTreeClassifier

        self.clf = DecisionTreeClassifier(
            max_depth=max_depth, min_samples_leaf=min_leaf,
            class_weight="balanced", random_state=seed,
        )
        self.majority = "base"
        self.fitted = False
        self.classes_: List[str] = []

    def fit(self, X: np.ndarray, y: List[str]) -> "QueryRouter":
        if len(set(y)) < 2:
            self.majority = y[0] if y else "base"
            return self
        self.clf.fit(X, y)
        self.fitted = True
        self.classes_ = list(self.clf.classes_)
        vals, counts = np.unique(y, return_counts=True)
        self.majority = str(vals[int(np.argmax(counts))])
        return self

    def predict(self, X: np.ndarray) -> List[str]:
        if not self.fitted:
            return [self.majority] * len(X)
        return [str(p) for p in self.clf.predict(X)]

    def rules(self) -> str:
        if not self.fitted:
            return f"(constant: {self.majority})"
        from sklearn.tree import export_text

        return export_text(self.clf, feature_names=FEATURE_NAMES)
