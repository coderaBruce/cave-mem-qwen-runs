"""OUR METHOD v7: wider retrieval, GAM's ordering untouched.

GAM's Fig. 2 (bottom row) shows F1 rising monotonically with the number of
retrieved pages from 3 to 20 on every dataset -- up to ~+12 F1 at HotpotQA-224K --
and the shipped default is 5. We tested reflection *depth* earlier (saturated on
LoCoMo, max_iters=5 was slightly worse); breadth is the untested axis.

This subclass changes exactly one thing: the per-retriever `top_k` inside
`_search`. Fusion, dedup and ordering are byte-for-byte GAM's, so any difference
is attributable to breadth alone.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from gam.agents.research_agent import ResearchAgent
from gam.schemas import Hit, Result, SearchPlan


class WideResearchAgent(ResearchAgent):
    def __init__(self, *args, per_retriever_k: int = 5, **kwargs):
        super().__init__(*args, **kwargs)
        self.per_retriever_k = per_retriever_k

    def _search(self, plan: SearchPlan, result: Result, question: str,
                searching_prompt: Optional[str] = None) -> Result:
        k = self.per_retriever_k
        all_hits: List[Hit] = []

        for tool in plan.tools:
            hits: List[Hit] = []
            if tool == "keyword" and plan.keyword_collection:
                res = self._search_by_keyword([" ".join(plan.keyword_collection)], top_k=k)
            elif tool == "vector" and plan.vector_queries:
                res = self._search_by_vector(plan.vector_queries, top_k=k)
            elif tool == "page_index" and plan.page_index:
                res = self._search_by_page_index(plan.page_index)
            else:
                continue
            if res and isinstance(res[0], list):
                for sub in res:
                    hits.extend(sub)
            else:
                hits.extend(res or [])
            all_hits.extend(hits)

        if not all_hits:
            return result

        # Identical to GAM from here on.
        unique: Dict[str, Hit] = {}
        no_id: List[Hit] = []
        for h in all_hits:
            if h.page_id:
                if h.page_id not in unique:
                    unique[h.page_id] = h
                else:
                    e = unique[h.page_id]
                    if (h.meta.get("score", 0) if h.meta else 0) > (
                            e.meta.get("score", 0) if e.meta else 0):
                        unique[h.page_id] = h
            else:
                no_id.append(h)
        merged = list(unique.values()) + no_id
        merged.sort(key=lambda h: h.meta.get("score", 0) if h.meta else 0, reverse=True)
        return self._integrate(merged, result, question)
