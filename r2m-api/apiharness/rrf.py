"""OUR METHOD v6: rank-based hybrid fusion for GAM's retriever.

**The defect.** `ResearchAgent._search` collects hits from the three retrievers,
dedupes by `page_id`, then sorts them all by a single raw `meta["score"]`. Those
scores are not comparable:

    BM25      3.45 – 4.69   (unbounded Okapi score)
    cosine    0.20 – 0.23   (bounded inner product)
    page_index  0           (IndexRetriever writes meta={})

Measured on a real LoCoMo page store, the fused top-5 is **5/5 BM25 hits**.
Dense hits can essentially never outrank a keyword hit, and pages the planner
explicitly asked to re-read by ID sort dead last with score 0. GAM's "hybrid"
retrieval is BM25-only ranking with the others appended as a tail.

Their own ablation corroborates it: Table 3 gives BM25-alone 48.64 avg against
53.18 for all three tools, i.e. the other two retrievers add ~4.5 points, almost
all of which must come from pages BM25 missed entirely rather than from better
ordering.

**The fix.** Reciprocal Rank Fusion — combine by rank, not by score:

    RRF(p) = Σ_r  w_r / (k + rank_r(p))

Scale-free by construction, one hyperparameter (k=60, the standard value from
Cormack et al. 2009), and it gives `page_index` real standing instead of zero.

Chosen deliberately over the answer-side interventions that failed earlier: this
operates on *retrieval*, so it carries no assumption about answer style — which
is what made evidence-preserving answering lose on NarrativeQA, whose golds are
abstractive.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional

from gam.agents.research_agent import ResearchAgent
from gam.schemas import Hit, Result, SearchPlan

RRF_K = 60

# Source weights reflect standalone retriever quality from GAM's own Table 3
# (avg F1: BM25 48.64, embedding 32.31, page-id 28.96). Equal weighting lets the
# weak retrievers outvote the strong one.
DEFAULT_WEIGHTS = {"keyword": 1.0, "vector": 0.5, "page_index": 0.4}


def rrf_fuse(
    ranked_lists: Dict[str, List[Hit]],
    k: int = RRF_K,
    weights: Optional[Dict[str, float]] = None,
) -> List[Hit]:
    """Fuse per-retriever ranked lists into one ordering by reciprocal rank."""
    weights = weights or {}
    scores: Dict[str, float] = defaultdict(float)
    best: Dict[str, Hit] = {}
    contributors: Dict[str, set] = defaultdict(set)

    for source, hits in ranked_lists.items():
        w = weights.get(source, 1.0)
        seen_in_list = set()
        rank = 0
        for hit in hits:
            pid = hit.page_id
            if pid is None or pid in seen_in_list:
                continue
            seen_in_list.add(pid)
            rank += 1
            scores[pid] += w / (k + rank)
            contributors[pid].add(source)
            if pid not in best:
                best[pid] = hit

    fused: List[Hit] = []
    for pid in sorted(scores, key=lambda p: scores[p], reverse=True):
        hit = best[pid]
        fused.append(
            Hit(
                page_id=pid,
                snippet=hit.snippet,
                source="+".join(sorted(contributors[pid])),
                meta={"score": scores[pid], "rrf": True,
                      "n_retrievers": len(contributors[pid])},
            )
        )
    return fused


class RRFResearchAgent(ResearchAgent):
    """GAM's researcher with rank-based fusion instead of raw-score sorting.

    Only `_search` changes; planning, integration and reflection are inherited
    unchanged, so any measured difference is attributable to retrieval ordering.
    """

    def __init__(self, *args, rrf_k: int = RRF_K, top_pages: int = 5,
                 per_retriever_k: int = 5, weights: Optional[Dict[str, float]] = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.rrf_k = rrf_k
        self.top_pages = top_pages          # pages handed to _integrate
        self.per_retriever_k = per_retriever_k
        self.weights = weights if weights is not None else dict(DEFAULT_WEIGHTS)

    def _search(self, plan: SearchPlan, result: Result, question: str,
                searching_prompt: Optional[str] = None) -> Result:
        ranked: Dict[str, List[Hit]] = {}

        if "keyword" in plan.tools and plan.keyword_collection:
            combined = " ".join(plan.keyword_collection)
            out = self._search_by_keyword([combined], top_k=self.per_retriever_k)
            ranked["keyword"] = [h for sub in out for h in sub] if out and isinstance(
                out[0], list) else list(out or [])

        if "vector" in plan.tools and plan.vector_queries:
            out = self._search_by_vector(plan.vector_queries, top_k=self.per_retriever_k)
            # Merge the per-query lists into ONE dense source, keeping each page's
            # best rank. Registering each query separately (our first attempt) gave
            # dense N votes against keyword's 1 and cost 3.6 F1 -- GAM's Table 3
            # says BM25 alone (48.64) far outperforms embedding alone (32.31), so
            # the fusion must not let the weaker retriever outvote the stronger.
            subs = out if (out and isinstance(out[0], list)) else [list(out or [])]
            best_rank: Dict[str, int] = {}
            hit_of: Dict[str, Hit] = {}
            for sub in subs:
                for r, h in enumerate(sub, 1):
                    if h.page_id is None:
                        continue
                    if h.page_id not in best_rank or r < best_rank[h.page_id]:
                        best_rank[h.page_id] = r
                        hit_of[h.page_id] = h
            ranked["vector"] = [hit_of[p] for p in
                                sorted(best_rank, key=lambda p: best_rank[p])]

        if "page_index" in plan.tools and plan.page_index:
            out = self._search_by_page_index(plan.page_index)
            ranked["page_index"] = [h for sub in out for h in sub] if out and isinstance(
                out[0], list) else list(out or [])

        if not any(ranked.values()):
            return result

        fused = rrf_fuse(ranked, k=self.rrf_k, weights=self.weights)
        # GAM passes *every* deduped hit to _integrate (it only truncates per
        # retriever, at top_k=5 each). Truncating the fused list to 5 starved the
        # integrator and cost 3.6 F1 -- that was our bug, not a property of RRF.
        # top_pages=0 means "no truncation", matching GAM exactly, so the only
        # difference from baseline is the ordering.
        head = fused if not self.top_pages else fused[: self.top_pages]
        return self._integrate(head, result, question)
