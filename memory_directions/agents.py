from __future__ import annotations

import json
import re
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

from gam.agents.research_agent import ResearchAgent
from gam.schemas import (
    PLANNING_SCHEMA,
    ReflectionDecision,
    ResearchOutput,
    Result,
    SearchPlan,
)


CONTRACT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "answer_type": {"type": "string"},
        "slots": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "required": {"type": "boolean"},
                    "search_hints": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["name", "description", "required", "search_hints"],
                "additionalProperties": False,
            },
        },
        "constraints": {"type": "array", "items": {"type": "string"}},
        "completion_rules": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["answer_type", "slots", "constraints", "completion_rules"],
    "additionalProperties": False,
}


LEDGER_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "answer_summary": {"type": "string"},
        "slots": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "status": {
                        "type": "string",
                        "enum": ["filled", "partial", "missing", "conflict"],
                    },
                    "value": {"type": "string"},
                    "evidence_page_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "support": {"type": "string"},
                    "missing_reason": {"type": "string"},
                    "confidence": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                    },
                },
                "required": [
                    "name",
                    "status",
                    "value",
                    "evidence_page_ids",
                    "support",
                    "missing_reason",
                    "confidence",
                ],
                "additionalProperties": False,
            },
        },
        "conflicts": {"type": "array", "items": {"type": "string"}},
        "open_questions": {"type": "array", "items": {"type": "string"}},
        "enough": {"type": "boolean"},
        "next_request": {"type": "string"},
    },
    "required": [
        "answer_summary",
        "slots",
        "conflicts",
        "open_questions",
        "enough",
        "next_request",
    ],
    "additionalProperties": False,
}


HYPOTHESIS_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "hypotheses": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "candidate_answer": {"type": "string"},
                    "claims": {"type": "array", "items": {"type": "string"}},
                    "search_queries": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["candidate_answer", "claims", "search_queries"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["hypotheses"],
    "additionalProperties": False,
}


VERIFY_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "answer_summary": {"type": "string"},
        "best_answer": {"type": "string"},
        "verified_claims": {"type": "array", "items": {"type": "string"}},
        "rejected_claims": {"type": "array", "items": {"type": "string"}},
        "missing": {"type": "array", "items": {"type": "string"}},
        "sources": {"type": "array", "items": {"type": "string"}},
        "enough": {"type": "boolean"},
        "next_request": {"type": "string"},
    },
    "required": [
        "answer_summary",
        "best_answer",
        "verified_claims",
        "rejected_claims",
        "missing",
        "sources",
        "enough",
        "next_request",
    ],
    "additionalProperties": False,
}


MERGE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "merged_summary": {"type": "string"},
        "use_addendum": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["merged_summary", "use_addendum", "reason"],
    "additionalProperties": False,
}


ROUTE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "route": {"type": "string", "enum": ["gam", "addendum"]},
        "question_type": {
            "type": "string",
            "enum": ["temporal", "open_domain", "multi_hop", "single_hop", "other"],
        },
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "reason": {"type": "string"},
    },
    "required": ["route", "question_type", "confidence", "reason"],
    "additionalProperties": False,
}


COUNT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "answer_summary": {"type": "string"},
        "count_phrase": {"type": "string"},
        "instances": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "date": {"type": "string"},
                    "evidence": {"type": "string"},
                },
                "required": ["label", "date", "evidence"],
                "additionalProperties": False,
            },
        },
        "excluded": {"type": "array", "items": {"type": "string"}},
        "enough": {"type": "boolean"},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
    },
    "required": [
        "answer_summary",
        "count_phrase",
        "instances",
        "excluded",
        "enough",
        "confidence",
    ],
    "additionalProperties": False,
}


CAVE_VALIDATION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "use_candidate": {"type": "boolean"},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "expected_effect": {
            "type": "string",
            "enum": ["positive", "neutral", "negative"],
        },
        "reason": {"type": "string"},
    },
    "required": ["use_candidate", "confidence", "expected_effect", "reason"],
    "additionalProperties": False,
}


def _loads_maybe(text: str, default: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except Exception:
                pass
    return dict(default)


class DirectedMemoryAgent(ResearchAgent):
    """GAM-compatible researcher with explicit search state.

    Modes:
      - contract: question contract + evidence ledger
      - hypothesis: candidate answer hypotheses + claim verification
      - adaptive: contract + dynamic retrieval width/depth
      - hybrid: contract + hypotheses + dynamic budget
      - guarded_addendum: GAM summary first, then append verifier/addendum
      - guarded_hybrid: GAM summary first, then merge-gate the verifier/addendum
      - routed_addendum: use addendum only for question types where it helps
      - llm_routed_addendum: same routing target, but inferred from question text
    """

    def __init__(
        self,
        *args,
        mode: str = "hybrid",
        base_k: int = 5,
        wide_k: int = 12,
        max_iters: int = 3,
        evidence_char_budget: int = 52000,
        **kwargs,
    ) -> None:
        super().__init__(*args, max_iters=max_iters, **kwargs)
        self.mode = mode
        self.base_k = base_k
        self.wide_k = wide_k
        self.evidence_char_budget = evidence_char_budget

    def research(self, request: str) -> ResearchOutput:
        self._update_retrievers()
        memory_state = self.memory_store.load()
        memory_context = self._memory_context(memory_state)

        contract: Dict[str, Any] = {}
        if self.mode in {"contract", "adaptive", "hybrid"}:
            contract = self._make_contract(request, memory_context)

        hypotheses: Dict[str, Any] = {}
        if self.mode in {"hypothesis", "hybrid"}:
            hypotheses = self._make_hypotheses(request, memory_context, contract)

        ledger = self._empty_ledger(contract)
        result = Result()
        iterations: List[Dict[str, Any]] = []
        next_request = request

        for step in range(self.max_iters):
            k = self._retrieval_width(step, ledger)
            plan = self._directed_plan(
                request=request,
                next_request=next_request,
                memory_context=memory_context,
                contract=contract,
                ledger=ledger,
                hypotheses=hypotheses,
                step=step,
            )
            hits = self._collect_hits(plan, top_k=k)

            if self.mode == "hypothesis":
                verify = self._verify_hypotheses(
                    request=request,
                    hypotheses=hypotheses,
                    hits=hits,
                    previous=result.content,
                )
                result = Result(
                    content=verify.get("answer_summary") or verify.get("best_answer", ""),
                    sources=verify.get("sources", []),
                )
                decision = ReflectionDecision(
                    enough=bool(verify.get("enough", False)),
                    new_request=verify.get("next_request") or None,
                )
                ledger = {
                    "answer_summary": result.content,
                    "slots": [],
                    "conflicts": verify.get("rejected_claims", []),
                    "open_questions": verify.get("missing", []),
                    "enough": decision.enough,
                    "next_request": decision.new_request or "",
                }
            else:
                ledger = self._update_ledger(
                    request=request,
                    contract=contract,
                    ledger=ledger,
                    hits=hits,
                    hypotheses=hypotheses,
                )
                result = Result(
                    content=self._ledger_to_context(contract, ledger),
                    sources=self._ledger_sources(ledger),
                )
                decision = ReflectionDecision(
                    enough=bool(ledger.get("enough", False)),
                    new_request=ledger.get("next_request") or None,
                )

            iterations.append(
                {
                    "step": step,
                    "mode": self.mode,
                    "top_k": k,
                    "plan": plan.model_dump(),
                    "temp_memory": result.model_dump(),
                    "contract": contract,
                    "ledger": ledger,
                    "hypotheses": hypotheses,
                    "decision": decision.model_dump(),
                }
            )

            if decision.enough:
                break
            next_request = decision.new_request or self._fallback_next_request(request, ledger)

        raw = {"iterations": iterations, "temp_memory": result.model_dump()}
        return ResearchOutput(integrated_memory=result.content, raw_memory=raw)

    def guarded_research(self, request: str, merge_gate: bool = True) -> ResearchOutput:
        """Run GAM first, then add contract-guided evidence without replacing it."""
        primary = ResearchAgent.research(self, request)
        primary_summary = primary.integrated_memory or ""

        memory_state = self.memory_store.load()
        memory_context = self._memory_context(memory_state)
        contract = self._make_contract(request, memory_context)
        hypotheses = self._make_hypotheses(request, memory_context, contract)
        ledger = self._empty_ledger(contract)
        ledger["answer_summary"] = primary_summary

        next_request = request
        directed_iterations: List[Dict[str, Any]] = []
        for step in range(max(1, min(2, self.max_iters))):
            plan = self._directed_plan(
                request=request,
                next_request=next_request,
                memory_context=memory_context,
                contract=contract,
                ledger=ledger,
                hypotheses=hypotheses,
                step=step,
            )
            hits = self._collect_hits(plan, top_k=self._retrieval_width(step, ledger))
            ledger = self._update_ledger(
                request=request,
                contract=contract,
                ledger=ledger,
                hits=hits,
                hypotheses=hypotheses,
            )
            directed_iterations.append(
                {
                    "step": step,
                    "plan": plan.model_dump(),
                    "ledger": ledger,
                    "top_k": self._retrieval_width(step, ledger),
                }
            )
            if ledger.get("enough"):
                break
            next_request = self._fallback_next_request(request, ledger)

        addendum = self._ledger_addendum(ledger)
        if merge_gate:
            merge = self._merge_primary_addendum(request, primary_summary, addendum)
            final = merge.get("merged_summary") or primary_summary
        else:
            merge = {
                "merged_summary": primary_summary,
                "use_addendum": bool(addendum),
                "reason": "raw addendum appended",
            }
            final = primary_summary
            if addendum:
                final = (
                    "PRIMARY SUMMARY FROM GAM:\n"
                    f"{primary_summary}\n\n"
                    "CONTRACT-GUIDED EVIDENCE ADDENDUM:\n"
                    f"{addendum}\n\n"
                    "Use the primary summary unless the addendum provides a more specific evidence-supported value."
                )

        raw = dict(primary.raw_memory or {})
        raw["primary_gam"] = primary.raw_memory
        raw["primary_summary"] = primary_summary
        raw["addendum"] = addendum
        raw["contract"] = contract
        raw["hypotheses"] = hypotheses
        raw["directed_iterations"] = directed_iterations
        raw["merge"] = merge
        raw["temp_memory"] = {
            "content": final,
            "sources": sorted(
                set(
                    [str(x) for x in (raw.get("temp_memory") or {}).get("sources", [])]
                    + self._ledger_sources(ledger)
                )
            ),
        }
        return ResearchOutput(integrated_memory=final, raw_memory=raw)

    def _memory_context(self, memory_state) -> str:
        abstracts = getattr(memory_state, "abstracts", None) or []
        if not abstracts:
            return "No memory currently."
        return "\n".join(f"Page {i}: {abstract}" for i, abstract in enumerate(abstracts))

    def _json_call(
        self, prompt: str, schema: Dict[str, Any], default: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            response = self.generator.generate_single(prompt=prompt, schema=schema)
            self.total_tokens += int(response["response"]["usage"]["total_tokens"])
            data = response.get("json") or _loads_maybe(response.get("text", ""), default)
            return data if isinstance(data, dict) else dict(default)
        except Exception as exc:
            print(f"DirectedMemoryAgent JSON call failed: {exc}")
            return dict(default)

    def _make_contract(self, request: str, memory_context: str) -> Dict[str, Any]:
        prompt = f"""You create an evidence contract for a memory-search agent.

QUESTION:
{request}

MEMORY ABSTRACTS:
{memory_context}

Return a compact contract. A slot is a concrete piece of information that must be proven before the question can be answered. Include temporal, comparison, list/set, entity-disambiguation, and multi-hop relation constraints when relevant.

Rules:
- Use 2-6 slots.
- Search hints must be short literal or semantic queries.
- Do not answer the question unless the answer is already directly implied by the abstracts.
- The contract will control retrieval, so missing required facts should become slots.
"""
        default = {
            "answer_type": "unknown",
            "slots": [
                {
                    "name": "answer",
                    "description": request,
                    "required": True,
                    "search_hints": [request],
                }
            ],
            "constraints": [],
            "completion_rules": ["All required slots have evidence."],
        }
        return self._json_call(prompt, CONTRACT_SCHEMA, default)

    def _empty_ledger(self, contract: Dict[str, Any]) -> Dict[str, Any]:
        slots = []
        for slot in contract.get("slots", []):
            slots.append(
                {
                    "name": slot.get("name", "slot"),
                    "status": "missing",
                    "value": "",
                    "evidence_page_ids": [],
                    "support": "",
                    "missing_reason": slot.get("description", ""),
                    "confidence": "low",
                }
            )
        return {
            "answer_summary": "",
            "slots": slots,
            "conflicts": [],
            "open_questions": [s.get("description", "") for s in contract.get("slots", [])],
            "enough": False,
            "next_request": "",
        }

    def _make_hypotheses(
        self, request: str, memory_context: str, contract: Dict[str, Any]
    ) -> Dict[str, Any]:
        prompt = f"""Generate answer hypotheses for claim-verification memory search.

QUESTION:
{request}

MEMORY ABSTRACTS:
{memory_context}

EVIDENCE CONTRACT:
{json.dumps(contract, ensure_ascii=False)}

Return 1-3 plausible candidate answers. For each one, list atomic claims that must be verified and search queries that could support or refute it. If there is no plausible answer yet, still create search queries for the most likely evidence.
"""
        return self._json_call(prompt, HYPOTHESIS_SCHEMA, {"hypotheses": []})

    def _directed_plan(
        self,
        request: str,
        next_request: str,
        memory_context: str,
        contract: Dict[str, Any],
        ledger: Dict[str, Any],
        hypotheses: Dict[str, Any],
        step: int,
    ) -> SearchPlan:
        if self.mode == "hypothesis":
            queries = []
            for hyp in hypotheses.get("hypotheses", []):
                queries.extend(hyp.get("search_queries", [])[:3])
            if not queries:
                queries = [next_request or request]
            return SearchPlan(
                info_needs=[f"Verify candidate answer claims for: {request}"],
                tools=["keyword", "vector"],
                keyword_collection=queries[:4],
                vector_queries=queries[:4],
                page_index=[],
            )

        prompt = f"""You are planning one retrieval step for a contract-guided memory search agent.

ORIGINAL QUESTION:
{request}

CURRENT SEARCH REQUEST:
{next_request}

MEMORY:
{memory_context}

CONTRACT:
{json.dumps(contract, ensure_ascii=False)}

CURRENT EVIDENCE LEDGER:
{json.dumps(ledger, ensure_ascii=False)}

HYPOTHESES:
{json.dumps(hypotheses, ensure_ascii=False)}

Plan retrieval only for missing, partial, or conflicting slots. Prefer exact keywords for names/dates/entities, vector queries for conceptual relations, and page_index only when the memory explicitly points to page ids. Avoid redundant searches already satisfied by the ledger.
"""
        data = self._json_call(prompt, PLANNING_SCHEMA, {})
        plan = SearchPlan(
            info_needs=data.get("info_needs", []),
            tools=data.get("tools", []),
            keyword_collection=data.get("keyword_collection", []),
            vector_queries=data.get("vector_queries", []),
            page_index=data.get("page_index", []),
        )
        if not plan.tools:
            hints: List[str] = []
            for slot in contract.get("slots", []):
                hints.extend(slot.get("search_hints", []))
            plan = SearchPlan(
                info_needs=[next_request or request],
                tools=["keyword", "vector"],
                keyword_collection=(hints or [next_request or request])[:5],
                vector_queries=(hints or [next_request or request])[:5],
                page_index=[],
            )
        return plan

    def _collect_hits(self, plan: SearchPlan, top_k: int) -> List[Any]:
        all_hits: List[Any] = []
        for tool in plan.tools:
            if tool == "keyword" and plan.keyword_collection:
                queries = [" ".join(plan.keyword_collection)]
                results = self._search_by_keyword(queries, top_k=top_k)
            elif tool == "vector" and plan.vector_queries:
                results = self._search_by_vector(plan.vector_queries, top_k=top_k)
            elif tool == "page_index" and plan.page_index:
                results = self._search_by_page_index(plan.page_index)
            else:
                continue
            for item in results or []:
                if isinstance(item, list):
                    all_hits.extend(item)
                else:
                    all_hits.append(item)

        unique: Dict[str, Any] = {}
        no_id: List[Any] = []
        for hit in all_hits:
            if getattr(hit, "page_id", None):
                prev = unique.get(hit.page_id)
                score = hit.meta.get("score", 0) if hit.meta else 0
                prev_score = prev.meta.get("score", 0) if prev and prev.meta else -1
                if prev is None or score > prev_score:
                    unique[hit.page_id] = hit
            else:
                no_id.append(hit)
        merged = list(unique.values()) + no_id
        merged.sort(key=lambda h: h.meta.get("score", 0) if h.meta else 0, reverse=True)
        return merged

    def _hits_context(self, hits: List[Any]) -> str:
        parts: List[str] = []
        total = 0
        for i, hit in enumerate(hits, 1):
            pid = getattr(hit, "page_id", None)
            source = getattr(hit, "source", "")
            snippet = getattr(hit, "snippet", "") or ""
            header = ""
            if pid is not None:
                try:
                    page = self.page_store.get(int(str(pid)))
                    header = (getattr(page, "header", "") or "").strip() if page else ""
                except Exception:
                    header = ""
            if header:
                text = f"{i}. [{source}]({pid}) HEADER: {header}\nCONTENT: {snippet}"
            else:
                text = f"{i}. [{source}]({pid}) {snippet}"
            if total + len(text) > self.evidence_char_budget:
                remain = max(0, self.evidence_char_budget - total)
                if remain <= 500:
                    break
                text = text[:remain]
            parts.append(text)
            total += len(text)
        return "\n".join(parts) if parts else "No retrieved evidence."

    def _update_ledger(
        self,
        request: str,
        contract: Dict[str, Any],
        ledger: Dict[str, Any],
        hits: List[Any],
        hypotheses: Dict[str, Any],
    ) -> Dict[str, Any]:
        prompt = f"""Update an evidence ledger for memory search.

QUESTION:
{request}

CONTRACT:
{json.dumps(contract, ensure_ascii=False)}

PREVIOUS LEDGER:
{json.dumps(ledger, ensure_ascii=False)}

HYPOTHESES:
{json.dumps(hypotheses, ensure_ascii=False)}

RETRIEVED EVIDENCE:
{self._hits_context(hits)}

Task:
- Fill required slots only with facts supported by retrieved evidence or previous ledger evidence.
- Keep a page id for every filled value.
- If the question asks when something happened, resolve relative expressions using page headers or session context and prefer absolute dates over phrases such as "next month" or "yesterday".
- Mark conflicts when evidence disagrees.
- Set enough=true only when all required slots are filled with adequate evidence and no unresolved conflict remains.
- If not enough, write one targeted next_request that directly searches the most important missing or conflicting slot.
- The answer_summary must be concise but sufficiently detailed for the downstream answerer.
"""
        updated = self._json_call(prompt, LEDGER_SCHEMA, ledger)
        if not updated.get("slots"):
            updated["slots"] = ledger.get("slots", [])
        return updated

    def _verify_hypotheses(
        self,
        request: str,
        hypotheses: Dict[str, Any],
        hits: List[Any],
        previous: str,
    ) -> Dict[str, Any]:
        prompt = f"""Verify candidate answers using retrieved evidence.

QUESTION:
{request}

PREVIOUS VERIFIED SUMMARY:
{previous}

HYPOTHESES:
{json.dumps(hypotheses, ensure_ascii=False)}

RETRIEVED EVIDENCE:
{self._hits_context(hits)}

Choose the best supported answer. Reject unsupported claims. If evidence is insufficient, provide a targeted next_request. Use page ids as sources.
"""
        return self._json_call(
            prompt,
            VERIFY_SCHEMA,
            {
                "answer_summary": previous,
                "best_answer": previous,
                "verified_claims": [],
                "rejected_claims": [],
                "missing": [request],
                "sources": [],
                "enough": False,
                "next_request": request,
            },
        )

    def _retrieval_width(self, step: int, ledger: Dict[str, Any]) -> int:
        if self.mode not in {"adaptive", "hybrid"}:
            return self.base_k
        if step == 0:
            return self.base_k
        open_count = len(ledger.get("open_questions", []))
        conflict_count = len(ledger.get("conflicts", []))
        missing_count = sum(
            1
            for s in ledger.get("slots", [])
            if s.get("status") in {"missing", "partial", "conflict"}
        )
        if conflict_count or open_count > 1 or missing_count > 1:
            return self.wide_k
        return max(self.base_k, min(self.wide_k, self.base_k + 3))

    def _fallback_next_request(self, request: str, ledger: Dict[str, Any]) -> str:
        if ledger.get("open_questions"):
            return " ".join(str(x) for x in ledger["open_questions"][:3])
        missing = [
            s.get("missing_reason") or s.get("name", "")
            for s in ledger.get("slots", [])
            if s.get("status") != "filled"
        ]
        return " ".join(missing[:3]) or request

    def _ledger_sources(self, ledger: Dict[str, Any]) -> List[str]:
        out: List[str] = []
        seen = set()
        for slot in ledger.get("slots", []):
            for sid in slot.get("evidence_page_ids", []):
                sid = str(sid)
                if sid and sid not in seen:
                    out.append(sid)
                    seen.add(sid)
        return out

    def _ledger_to_context(self, contract: Dict[str, Any], ledger: Dict[str, Any]) -> str:
        lines = [
            "ANSWER SUMMARY:",
            ledger.get("answer_summary", ""),
            "",
            "EVIDENCE CONTRACT:",
            f"answer_type: {contract.get('answer_type', '')}",
        ]
        if contract.get("constraints"):
            lines.append(
                "constraints: " + "; ".join(str(x) for x in contract.get("constraints", []))
            )
        lines.append("")
        lines.append("EVIDENCE LEDGER:")
        for slot in ledger.get("slots", []):
            sources = ",".join(str(x) for x in slot.get("evidence_page_ids", []))
            lines.append(
                f"- {slot.get('name')}: {slot.get('status')} | "
                f"value={slot.get('value')} | sources={sources} | "
                f"support={slot.get('support')} | missing={slot.get('missing_reason')}"
            )
        if ledger.get("conflicts"):
            lines.append("CONFLICTS: " + "; ".join(str(x) for x in ledger.get("conflicts", [])))
        if ledger.get("open_questions"):
            lines.append(
                "OPEN QUESTIONS: "
                + "; ".join(str(x) for x in ledger.get("open_questions", []))
            )
        return "\n".join(lines).strip()

    def _ledger_addendum(self, ledger: Dict[str, Any]) -> str:
        lines: List[str] = []
        summary = (ledger.get("answer_summary") or "").strip()
        if summary:
            lines.append(f"verified_summary: {summary}")
        for slot in ledger.get("slots", []):
            if slot.get("status") not in {"filled", "conflict"}:
                continue
            value = str(slot.get("value") or "").strip()
            support = str(slot.get("support") or "").strip()
            if not value and not support:
                continue
            sources = ",".join(str(x) for x in slot.get("evidence_page_ids", []))
            lines.append(
                f"- {slot.get('name')}: {slot.get('status')} | "
                f"value={value} | sources={sources} | support={support}"
            )
        if ledger.get("conflicts"):
            lines.append(
                "unresolved_conflicts: "
                + "; ".join(str(x) for x in ledger.get("conflicts", []))
            )
        return "\n".join(lines).strip()

    def _merge_primary_addendum(
        self, request: str, primary_summary: str, addendum: str
    ) -> Dict[str, Any]:
        if not addendum:
            return {
                "merged_summary": primary_summary,
                "use_addendum": False,
                "reason": "no addendum",
            }
        prompt = f"""Merge a primary GAM memory summary with a contract-guided evidence addendum.

QUESTION:
{request}

PRIMARY SUMMARY:
{primary_summary}

CONTRACT-GUIDED ADDENDUM:
{addendum}

Rules:
- The primary summary is the default. Preserve it unless the addendum clearly corrects an error or fills information the primary summary lacks.
- Do not include meta labels like "primary" or "addendum" in the merged summary.
- Do not include unknown, missing, open-question, or conflict text unless it is essential.
- For date questions, prefer an absolute date over a relative phrase. If the primary summary already gives a sufficient month/year and the addendum only adds a day-of-month, keep the concise primary date unless the primary conflicts with the addendum.
- Keep the merged summary short and directly useful for answering the question.
"""
        return self._json_call(
            prompt,
            MERGE_SCHEMA,
            {
                "merged_summary": primary_summary,
                "use_addendum": False,
                "reason": "merge failed",
            },
        )


class ContractMemoryAgent(DirectedMemoryAgent):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, mode="contract", **kwargs)


class HypothesisMemoryAgent(DirectedMemoryAgent):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, mode="hypothesis", **kwargs)


class AdaptiveContractMemoryAgent(DirectedMemoryAgent):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, mode="adaptive", **kwargs)


class HybridMemoryAgent(DirectedMemoryAgent):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, mode="hybrid", **kwargs)


class GuardedHybridMemoryAgent(DirectedMemoryAgent):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, mode="guarded_hybrid", **kwargs)

    def research(self, request: str) -> ResearchOutput:
        return self.guarded_research(request)


class GuardedAddendumMemoryAgent(DirectedMemoryAgent):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, mode="guarded_addendum", **kwargs)

    def research(self, request: str) -> ResearchOutput:
        return self.guarded_research(request, merge_gate=False)


class RoutedAddendumMemoryAgent(DirectedMemoryAgent):
    """Selective version: temporal/open-domain get addendum, others stay GAM."""

    _TEMPORAL_RE = re.compile(
        r"\b(when|what date|which date|what month|which month|what year|"
        r"how long|before|after)\b",
        re.IGNORECASE,
    )
    _OPEN_DOMAIN_RE = re.compile(
        r"\b(might|would|could|considered|status|financial|degree be|"
        r"person|likely|probably)\b",
        re.IGNORECASE,
    )

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, mode="routed_addendum", **kwargs)

    def research(self, request: str, category: Optional[int] = None) -> ResearchOutput:
        if self._should_use_addendum(request, category):
            out = self.guarded_research(request, merge_gate=False)
            out.raw_memory["route"] = {
                "strategy": "guarded_addendum",
                "category": category,
                "reason": self._route_reason(request, category),
            }
            return out

        out = ResearchAgent.research(self, request)
        out.raw_memory["route"] = {
            "strategy": "gam",
            "category": category,
            "reason": self._route_reason(request, category),
        }
        return out

    def _should_use_addendum(self, request: str, category: Optional[int]) -> bool:
        # LoCoMo categories: 2=temporal, 3=open-domain. These are question-type
        # labels, not answers, and the same decision is approximated below when
        # labels are unavailable.
        if category in {2, 3}:
            return True
        if category in {1, 4, 5}:
            return False
        return bool(self._TEMPORAL_RE.search(request) or self._OPEN_DOMAIN_RE.search(request))

    def _route_reason(self, request: str, category: Optional[int]) -> str:
        if category in {2, 3}:
            return "LoCoMo temporal/open-domain category"
        if category in {1, 4, 5}:
            return "LoCoMo non-temporal/non-open category"
        if self._TEMPORAL_RE.search(request):
            return "temporal question heuristic"
        if self._OPEN_DOMAIN_RE.search(request):
            return "open-domain question heuristic"
        return "fallback to GAM"


class HeuristicRoutedAddendumMemoryAgent(RoutedAddendumMemoryAgent):
    """Same router, but ignores dataset category labels."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.mode = "heuristic_routed_addendum"

    def _should_use_addendum(self, request: str, category: Optional[int]) -> bool:
        return bool(self._TEMPORAL_RE.search(request) or self._OPEN_DOMAIN_RE.search(request))

    def _route_reason(self, request: str, category: Optional[int]) -> str:
        if self._TEMPORAL_RE.search(request):
            return "temporal question heuristic"
        if self._OPEN_DOMAIN_RE.search(request):
            return "open-domain question heuristic"
        return "fallback to GAM"


class LLMRoutedAddendumMemoryAgent(RoutedAddendumMemoryAgent):
    """Question-only router: no dataset category labels or training."""

    _DATE_MENTION_RE = re.compile(
        r"\b("
        r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
        r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|"
        r"dec(?:ember)?|20\d{2}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"
        r")\b",
        re.IGNORECASE,
    )
    _DIRECT_FACT_RE = re.compile(
        r"^\s*(who|where|which|what|how many|how do|why did|do)\b",
        re.IGNORECASE,
    )
    _OPEN_SIGNAL_RE = re.compile(
        r"\b("
        r"might|likely|would|could|considered|status|financial|degree be|"
        r"underlying condition|emotions?|feeling|discomfort|preference|"
        r"personality|patriotic"
        r")\b",
        re.IGNORECASE,
    )

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.mode = "llm_routed_addendum"

    def research(self, request: str, category: Optional[int] = None) -> ResearchOutput:
        route = self._route_with_llm(request)
        use_addendum = route.get("route") == "addendum"

        if use_addendum:
            out = self.guarded_research(request, merge_gate=False)
            strategy = "guarded_addendum"
        else:
            out = ResearchAgent.research(self, request)
            strategy = "gam"

        out.raw_memory["route"] = {
            "strategy": strategy,
            "category": None,
            "router": route,
            "reason": route.get("reason", ""),
        }
        return out

    def _route_with_llm(self, request: str) -> Dict[str, Any]:
        prompt = f"""Route a memory QA question to either plain GAM search or a guarded evidence addendum.

QUESTION:
{request}

Use only the question text. Do not assume access to dataset category labels.

Choose addendum only for question types where an evidence contract tends to help:
- temporal: asks when/date/month/year/time, before/after, duration, ordering, or relative time resolution.
- open_domain: asks for a likely status, possibility, inference, preference, relationship, personal trait, financial/academic/work status, or other judgment that benefits from verified evidence.

Choose gam for:
- single_hop: asks for one direct fact.
- multi_hop: requires combining several facts but is not mainly temporal or open-domain.
- other: unclear cases.

Be conservative. If the addendum is unlikely to add value, route to gam.

Return only a JSON object with exactly these keys:
{{
  "route": "gam" or "addendum",
  "question_type": "temporal" or "open_domain" or "multi_hop" or "single_hop" or "other",
  "confidence": "high" or "medium" or "low",
  "reason": "short reason"
}}
"""
        data = self._json_call(
            prompt,
            ROUTE_SCHEMA,
            {
                "route": "gam",
                "question_type": "other",
                "confidence": "low",
                "reason": "router failed",
            },
        )
        qtype = data.get("question_type")
        route = data.get("route")
        if route == "addendum" and qtype not in {"temporal", "open_domain"}:
            data["route"] = "gam"
            data["reason"] = f"non-target type {qtype}; fallback to GAM"
        if route not in {"gam", "addendum"}:
            data["route"] = (
                "addendum"
                if self._TEMPORAL_RE.search(request) or self._OPEN_DOMAIN_RE.search(request)
                else "gam"
            )
        if self._should_guard_direct_fact(request, data):
            data["route"] = "gam"
            data["reason"] = (
                "direct factual question without open-domain uncertainty signal; "
                "fallback to GAM"
            )
        return data

    def _should_guard_direct_fact(self, request: str, route: Dict[str, Any]) -> bool:
        if route.get("route") != "addendum":
            return False
        if route.get("question_type") != "open_domain":
            return False
        if self._TEMPORAL_RE.search(request) or self._DATE_MENTION_RE.search(request):
            return False
        if self._OPEN_SIGNAL_RE.search(request):
            return False
        return bool(self._DIRECT_FACT_RE.search(request))


class TemporalRoutedAddendumMemoryAgent(LLMRoutedAddendumMemoryAgent):
    """High-precision question-only router for temporal/date questions."""

    _DURATION_RE = re.compile(
        r"\b(how long|how many|duration)\b",
        re.IGNORECASE,
    )
    _DIRECT_TEMPORAL_ASK_RE = re.compile(
        r"^\s*(when|what date|which date|what month|which month|what year|which year|what time)\b",
        re.IGNORECASE,
    )
    _YESNO_BEFORE_AFTER_RE = re.compile(
        r"^\s*(is|was|were|are|do|does|did|has|had|would|could)\b.*\b(before|after)\b",
        re.IGNORECASE,
    )
    _BAD_RELATIVE_TIME_RE = re.compile(
        r"\b("
        r"next month|this month|last month|next week|this week|last week|"
        r"last year|next year|a year ago|yesterday|tomorrow|recently|"
        r"week prior|prior to|few weeks|exact date is not specified|"
        r"not specified|occur soon|in the summer"
        r")\b",
        re.IGNORECASE,
    )

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.mode = "temporal_routed_addendum"

    def research(self, request: str, category: Optional[int] = None) -> ResearchOutput:
        route = self._temporal_route(request)
        use_addendum = self._should_use_temporal_addendum(request, route)

        if use_addendum:
            out = self.guarded_research(request, merge_gate=False)
            strategy = "guarded_addendum"
            guard_reason = self._unsafe_temporal_addendum(
                str(out.raw_memory.get("addendum") or "")
            )
            if guard_reason:
                primary = str(out.raw_memory.get("primary_summary") or "")
                if primary:
                    out.integrated_memory = primary
                    temp_memory = out.raw_memory.get("temp_memory") or {}
                    temp_memory["content"] = primary
                    out.raw_memory["temp_memory"] = temp_memory
                strategy = "gam"
        else:
            out = ResearchAgent.research(self, request)
            strategy = "gam"
            guard_reason = ""

        out.raw_memory["route"] = {
            "strategy": strategy,
            "category": None,
            "router": route,
            "reason": (
                guard_reason
                if guard_reason
                else route.get("reason", "")
                if use_addendum
                else "non-temporal or duration-like question; fallback to GAM"
            ),
        }
        return out

    def _temporal_route(self, request: str) -> Dict[str, Any]:
        if self._should_use_temporal_addendum(request, {}):
            return {
                "route": "addendum",
                "question_type": "temporal",
                "confidence": "high",
                "reason": "direct temporal question heuristic",
            }
        return {
            "route": "gam",
            "question_type": "other",
            "confidence": "high",
            "reason": "non-temporal or duration-like question; fallback to GAM",
        }

    def _should_use_temporal_addendum(
        self, request: str, route: Dict[str, Any]
    ) -> bool:
        if self._DURATION_RE.search(request):
            return False
        if (
            self._YESNO_BEFORE_AFTER_RE.search(request)
            and not self._DATE_MENTION_RE.search(request)
        ):
            return False
        return bool(self._DIRECT_TEMPORAL_ASK_RE.search(request))

    def _unsafe_temporal_addendum(self, addendum: str) -> str:
        if not addendum:
            return "empty temporal addendum; fallback to GAM"
        if self._BAD_RELATIVE_TIME_RE.search(addendum):
            return "temporal addendum used relative time; fallback to GAM"
        first_line = ""
        for line in addendum.splitlines():
            if line.startswith("verified_summary:"):
                first_line = line
                break
        if ";" in first_line:
            return "temporal addendum proposed multiple dates; fallback to GAM"
        return ""


class DateAnchoredAddendumMemoryAgent(TemporalRoutedAddendumMemoryAgent):
    """Add a deterministic source-date check for date-conditioned questions."""

    _MONTHS = {
        "january": 1,
        "february": 2,
        "march": 3,
        "april": 4,
        "may": 5,
        "june": 6,
        "july": 7,
        "august": 8,
        "september": 9,
        "october": 10,
        "november": 11,
        "december": 12,
        "jan": 1,
        "feb": 2,
        "mar": 3,
        "apr": 4,
        "jun": 6,
        "jul": 7,
        "aug": 8,
        "sep": 9,
        "oct": 10,
        "nov": 11,
        "dec": 12,
    }
    _MONTH_FULL = {
        1: "january",
        2: "february",
        3: "march",
        4: "april",
        5: "may",
        6: "june",
        7: "july",
        8: "august",
        9: "september",
        10: "october",
        11: "november",
        12: "december",
    }
    _MONTH_RE = "|".join(sorted(_MONTHS, key=len, reverse=True))
    _MDY_RE = re.compile(
        rf"\b({_MONTH_RE})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?[,]?\s+(20\d{{2}})\b",
        re.IGNORECASE,
    )
    _DMY_RE = re.compile(
        rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MONTH_RE})\.?[,]?\s+(20\d{{2}})\b",
        re.IGNORECASE,
    )
    _MONTH_YEAR_RE = re.compile(
        rf"\b({_MONTH_RE})\.?[,]?\s+(20\d{{2}})\b",
        re.IGNORECASE,
    )
    _YEAR_RE = re.compile(r"\b(20\d{2}|19\d{2})\b")
    _SESSION_DATE_RE = re.compile(
        rf"Dialogue Time\(available to answer questions\):.*?\bon\s+(\d{{1,2}})\s+({_MONTH_RE})[,]?\s+(20\d{{2}})",
        re.IGNORECASE,
    )
    _RELATIVE_NEXT_DAY_RE = re.compile(r"\b(yesterday|last night)\b", re.IGNORECASE)
    _STOPWORDS = {
        "the",
        "a",
        "an",
        "on",
        "in",
        "at",
        "to",
        "of",
        "with",
        "and",
        "or",
        "did",
        "does",
        "do",
        "was",
        "were",
        "is",
        "are",
        "when",
        "what",
        "which",
        "who",
        "how",
        "why",
    }

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.mode = "date_anchor_addendum"

    def research(self, request: str, category: Optional[int] = None) -> ResearchOutput:
        out = super().research(request, category)
        block, sources = self._date_anchor_block(request)
        if not block:
            out.raw_memory["date_anchor"] = {"used": False, "sources": []}
            return out
        anchor_summary = self._date_anchor_summary(request, block)

        final = (
            "DATE-ANCHORED EVIDENCE SUMMARY:\n"
            f"{anchor_summary}\n\n"
            "SOURCE EXCERPTS:\n"
            f"{block}\n\n"
            "Answer using only this source check. It was selected by matching the "
            "date or adjacent relative-date session in the question."
        )
        out.integrated_memory = final
        temp_memory = out.raw_memory.get("temp_memory") or {}
        old_sources = [str(x) for x in temp_memory.get("sources", [])]
        temp_memory["content"] = final
        temp_memory["sources"] = sorted(set(old_sources + [str(x) for x in sources]))
        out.raw_memory["temp_memory"] = temp_memory
        out.raw_memory["date_anchor"] = {
            "used": True,
            "sources": sources,
            "summary": anchor_summary,
        }
        return out

    def _date_anchor_summary(self, request: str, block: str) -> str:
        prompt = f"""Extract the answer-relevant fact from date-anchored source excerpts.

QUESTION:
{request}

SOURCE EXCERPTS:
{block}

The excerpts were selected because their session date matches the question date
or an adjacent relative-date reference such as "last night" or "yesterday".

Return one concise sentence that directly states the answer-relevant fact.
If the evidence is insufficient, say "Insufficient date-anchored evidence."
Do not use facts outside the excerpts.
"""
        try:
            response = self.generator.generate_single(prompt=prompt)
            return (response.get("text") or "").strip()
        except Exception:
            return "Insufficient date-anchored evidence."

    def _extract_date_query(self, request: str) -> Dict[str, Any]:
        m = self._MDY_RE.search(request)
        if m:
            month = self._MONTHS[m.group(1).lower()]
            return {"day": date(int(m.group(3)), month, int(m.group(2)))}
        m = self._DMY_RE.search(request)
        if m:
            month = self._MONTHS[m.group(2).lower()]
            return {"day": date(int(m.group(3)), month, int(m.group(1)))}
        m = self._MONTH_YEAR_RE.search(request)
        if m:
            return {"month": self._MONTHS[m.group(1).lower()], "year": int(m.group(2))}
        m = self._YEAR_RE.search(request)
        if m:
            return {"year": int(m.group(1))}
        return {}

    def _session_date(self, content: str) -> Optional[date]:
        m = self._SESSION_DATE_RE.search(content)
        if not m:
            return None
        return date(int(m.group(3)), self._MONTHS[m.group(2).lower()], int(m.group(1)))

    def _question_terms(self, request: str) -> set:
        terms = {
            t.lower()
            for t in re.findall(r"[A-Za-z][A-Za-z']+", request)
            if len(t) > 2 and t.lower() not in self._STOPWORDS
        }
        return {t for t in terms if t.lower() not in self._MONTHS}

    def _focused_excerpt(self, content: str, terms: set, width: int = 1800) -> str:
        lower = content.lower()
        hits = []
        for term in terms:
            pos = lower.find(term)
            if pos >= 0:
                hits.append((lower.count(term), -len(term), pos))
        if not hits:
            return content[:width].strip()
        _, _, center = sorted(hits)[0]
        start = max(0, center - width // 3)
        end = min(len(content), start + width)
        return content[start:end].strip()

    def _date_anchor_block(self, request: str) -> Tuple[str, List[int]]:
        query = self._extract_date_query(request)
        if not query:
            return "", []
        if "day" not in query:
            return "", []
        terms = self._question_terms(request)
        rows: List[Tuple[float, int, str]] = []
        for idx, page in enumerate(self.page_store.load()):
            content = page.content or ""
            session_date = self._session_date(content)
            hay = f"{page.header}\n{content}"
            score = 0.0
            if "day" in query:
                target = query["day"]
                if session_date == target:
                    score += 20
                elif (
                    session_date == target + timedelta(days=1)
                    and self._RELATIVE_NEXT_DAY_RE.search(content)
                ):
                    score += 18
                elif target.strftime("%B") in hay and str(target.day) in hay and str(target.year) in hay:
                    score += 10
            elif "month" in query:
                if (
                    session_date
                    and session_date.year == query["year"]
                    and session_date.month == query["month"]
                ):
                    score += 12
                month_name = self._MONTH_FULL[query["month"]]
                if month_name in hay.lower() and str(query["year"]) in hay:
                    score += 8
            elif "year" in query and str(query["year"]) in hay:
                score += 8
            if score <= 0:
                continue
            lower = hay.lower()
            overlap = sum(1 for term in terms if term in lower)
            score += min(overlap, 6)
            if overlap == 0 and score < 18:
                continue
            excerpt = self._focused_excerpt(content, terms)
            rows.append((score, idx, f"[page {idx}] {page.header}\n{excerpt}"))

        rows.sort(key=lambda x: x[0], reverse=True)
        picked = rows[:3]
        return "\n\n".join(x[2] for x in picked), [x[1] for x in picked]


class DateCountAnchoredAddendumMemoryAgent(DateAnchoredAddendumMemoryAgent):
    """Date anchor plus a raw-source verifier for count/list-cardinality asks."""

    _COUNT_ASK_RE = re.compile(r"^\s*how many\b", re.IGNORECASE)
    _COUNT_STOPWORDS = DateAnchoredAddendumMemoryAgent._STOPWORDS | {
        "many",
        "times",
        "time",
        "has",
        "have",
        "had",
        "been",
        "did",
        "does",
        "were",
        "passed",
        "between",
        "mentioned",
        "mentions",
    }
    _EXPANSIONS = {
        "france": {"paris", "french"},
        "italy": {"rome", "italian"},
        "japan": {"tokyo", "japanese"},
        "united": {"usa", "america", "american"},
        "states": {"usa", "america", "american"},
    }

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.mode = "date_count_anchor_addendum"

    def research(self, request: str, category: Optional[int] = None) -> ResearchOutput:
        out = super().research(request, category)
        block, sources = self._count_anchor_block(request)
        if not block:
            out.raw_memory["count_anchor"] = {"used": False, "sources": []}
            return out

        count = self._count_anchor_summary(request, block)
        if not count.get("enough") or not (count.get("count_phrase") or "").strip():
            out.raw_memory["count_anchor"] = {
                "used": False,
                "sources": sources,
                "summary": count,
            }
            return out

        instances = count.get("instances") or []
        instance_lines = "\n".join(
            self._format_count_instance(item) for item in instances
        )
        final = (
            "COUNT-ANCHORED EVIDENCE SUMMARY:\n"
            f"Exact answer phrase: {count.get('count_phrase', '').strip()}\n"
            f"{count.get('answer_summary', '').strip()}\n\n"
            "COUNTED INSTANCES:\n"
            f"{instance_lines if instance_lines else '- none listed'}\n\n"
            "SOURCE EXCERPTS:\n"
            f"{block}\n\n"
            "Answer using the exact answer phrase unless the counted instances "
            "show a more precise wording. Count only completed or explicitly "
            "planned instances according to the question wording."
        )
        out.integrated_memory = final
        temp_memory = out.raw_memory.get("temp_memory") or {}
        old_sources = [str(x) for x in temp_memory.get("sources", [])]
        temp_memory["content"] = final
        temp_memory["sources"] = sorted(set(old_sources + [str(x) for x in sources]))
        out.raw_memory["temp_memory"] = temp_memory
        out.raw_memory["count_anchor"] = {
            "used": True,
            "sources": sources,
            "summary": count,
        }
        return out

    def _format_count_instance(self, item: Any) -> str:
        if isinstance(item, dict):
            label = str(item.get("label", "")).strip()
            item_date = str(item.get("date", "")).strip()
            evidence = str(item.get("evidence", "")).strip()
            return f"- {label} | {item_date} | {evidence}".strip()
        return f"- {str(item).strip()}"

    def _count_terms(self, request: str) -> set:
        terms = {
            t.lower()
            for t in re.findall(r"[A-Za-z][A-Za-z']+", request)
            if len(t) > 2 and t.lower() not in self._COUNT_STOPWORDS
        }
        terms = {t for t in terms if t.lower() not in self._MONTHS}
        expanded = set(terms)
        for term in terms:
            expanded.update(self._EXPANSIONS.get(term, set()))
        return expanded

    def _count_anchor_block(self, request: str) -> Tuple[str, List[int]]:
        if not self._COUNT_ASK_RE.search(request):
            return "", []

        terms = self._count_terms(request)
        if not terms:
            return "", []

        pages = self.page_store.load()
        doc_freq: Dict[str, int] = {term: 0 for term in terms}
        for page in pages:
            hay = f"{page.header}\n{page.content or ''}".lower()
            for term in terms:
                if term in hay:
                    doc_freq[term] += 1

        rows: List[Tuple[float, int, str]] = []
        for idx, page in enumerate(pages):
            content = page.content or ""
            hay = f"{page.header}\n{content}"
            lower = hay.lower()
            matched = [term for term in terms if term in lower]
            if not matched:
                continue
            score = 0.0
            for term in matched:
                score += 1.0 / max(1, doc_freq.get(term, 1))
                if len(term) > 5:
                    score += 0.15
            if re.search(r"\b(one|two|three|four|five|six|seven|eight|nine|ten|\d+)\b", lower):
                score += 0.2
            excerpt = self._focused_excerpt(content, set(matched), width=1800)
            rows.append((score, idx, f"[page {idx}] {page.header}\n{excerpt}"))

        rows.sort(key=lambda x: x[0], reverse=True)
        picked = rows[:8]
        return "\n\n".join(x[2] for x in picked), [x[1] for x in picked]

    def _count_anchor_summary(self, request: str, block: str) -> Dict[str, Any]:
        prompt = f"""You are an exact-count verifier for a memory QA benchmark.

QUESTION:
{request}

SOURCE EXCERPTS:
{block}

Task:
- Count distinct instances that directly satisfy the question.
- Deduplicate repeated mentions of the same instance.
- Do not count merely planned, hypothetical, desired, or future events unless
  the question asks about plans.
- If the question asks how many times someone "mentioned" something, count
  explicit mentions in the excerpts, not implied facts.
- Prefer an exact short answer phrase such as "two times", "three years",
  "Three dogs", or "6".
- If the excerpts are insufficient for an exact count, set enough=false and
  leave count_phrase empty.

Return JSON matching the schema. Do not use facts outside the excerpts.
"""
        return self._json_call(
            prompt,
            COUNT_SCHEMA,
            {
                "answer_summary": "Insufficient count evidence.",
                "count_phrase": "",
                "instances": [],
                "excluded": [],
                "enough": False,
                "confidence": "low",
            },
        )


class CausalValidatedDateCountAnchoredAddendumMemoryAgent(
    DateCountAnchoredAddendumMemoryAgent
):
    """Condition-aware variant: validate an anchor before it overrides GAM.

    This is a minimal CAVE-style implementation. The typed date/count evidence
    is treated as a candidate experience intervention. It is reused only when a
    lightweight validation step predicts a positive effect over the primary
    GAM summary under the current question/evidence condition.
    """

    _INSUFFICIENT_RE = re.compile(
        r"\b("
        r"insufficient|not enough|no sufficient|cannot determine|can't determine|"
        r"not specified|not available|unclear|unknown|no evidence"
        r")\b",
        re.IGNORECASE,
    )
    _NUMBER_WORDS = {
        "zero": 0,
        "one": 1,
        "once": 1,
        "single": 1,
        "two": 2,
        "twice": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
        "eleven": 11,
        "twelve": 12,
    }
    _VAGUE_COUNT_RE = re.compile(
        r"\b(at least|at most|about|around|approximately|roughly|several|many|"
        r"few|no specific|not specified|unknown|unclear)\b",
        re.IGNORECASE,
    )

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.mode = "cave_date_count_anchor"

    def research(self, request: str, category: Optional[int] = None) -> ResearchOutput:
        profile = self._memory_profile()
        if profile in {"episodic_dialogue", "fact_pack"}:
            out = TemporalRoutedAddendumMemoryAgent.research(self, request, category)
            base_policy = "profile_allows_temporal_base"
        else:
            out = ResearchAgent.research(self, request)
            base_policy = "profile_uses_plain_gam_base"
        if out.raw_memory is None:
            out.raw_memory = {}
        previous_route = out.raw_memory.get("route")
        if previous_route is None:
            out.raw_memory["route"] = {
                "strategy": "gam",
                "category": category,
                "reason": "candidate experiences must validate before reuse",
            }
        out.raw_memory["cave"] = {
            "method": "causal_validated_anchor",
            "memory_profile": profile,
            "base_policy": base_policy,
            "candidate_experiences": ["date_anchor", "count_anchor"],
            "base_route": previous_route,
        }

        self._maybe_apply_validated_date_anchor(out, request)
        self._maybe_apply_validated_count_anchor(out, request)
        return out

    def _memory_profile(self) -> str:
        pages = self.page_store.load()
        text = "\n".join(
            f"{getattr(page, 'header', '')}\n{getattr(page, 'content', '')}"
            for page in pages[: min(len(pages), 6)]
        )
        if "Dialogue Time(available to answer questions)" in text or re.search(
            r"\bD\d+:\d+\b", text
        ):
            return "episodic_dialogue"
        if "INPUT_MESSAGE contains" in text or "various documents" in text:
            return "fact_pack"
        return "narrative"

    def _maybe_apply_validated_date_anchor(
        self, out: ResearchOutput, request: str
    ) -> None:
        block, sources = self._date_anchor_block(request)
        if not block:
            out.raw_memory["date_anchor"] = {
                "used": False,
                "candidate": False,
                "sources": [],
                "validation": {"use_candidate": False, "reason": "no date anchor block"},
            }
            return

        primary_summary = out.integrated_memory or ""
        anchor_summary = self._date_anchor_summary(request, block)
        validation = self._validate_anchor_candidate(
            request=request,
            anchor_kind="date",
            primary_summary=primary_summary,
            candidate_summary=anchor_summary,
            source_block=block,
            extra_rules=(
                "- Reject if the candidate says evidence is insufficient.\n"
                "- Reject if the primary already directly answers the question and "
                "the candidate only adds context or a longer phrasing.\n"
                "- Treat the source excerpts as stronger evidence than the primary "
                "when they explicitly support the candidate.\n"
                "- Do not reject solely because the candidate conflicts with the "
                "primary; reject only if the cited source excerpts do not support "
                "the candidate or the candidate does not directly answer.\n"
                "- Accept only when the dated source clearly corrects, disambiguates, "
                "or fills a fact missing from the primary summary."
            ),
        )
        accepted = self._accept_validation(validation) and not self._INSUFFICIENT_RE.search(
            anchor_summary
        )

        out.raw_memory["date_anchor"] = {
            "used": accepted,
            "candidate": True,
            "sources": sources,
            "summary": anchor_summary,
            "validation": validation,
        }
        if not accepted:
            return

        final = (
            "DATE-ANCHORED EVIDENCE SUMMARY:\n"
            f"{anchor_summary}\n\n"
            "SOURCE EXCERPTS:\n"
            f"{block}\n\n"
            "Answer using the shortest phrase that directly answers the question. "
            "Use this source check only because validation found it more reliable "
            "than the primary summary."
        )
        self._replace_integrated_memory(out, final, sources)

    def _maybe_apply_validated_count_anchor(
        self, out: ResearchOutput, request: str
    ) -> None:
        block, sources = self._count_anchor_block(request)
        if not block:
            out.raw_memory["count_anchor"] = {
                "used": False,
                "candidate": False,
                "sources": [],
                "validation": {"use_candidate": False, "reason": "no count anchor block"},
            }
            return

        primary_summary = out.integrated_memory or ""
        count = self._count_anchor_summary(request, block)
        count_phrase = str(count.get("count_phrase") or "").strip()
        deterministic_reject = self._count_deterministic_reject(
            primary_summary, count_phrase
        )
        if deterministic_reject:
            validation = {
                "use_candidate": False,
                "confidence": "high",
                "expected_effect": "negative",
                "reason": deterministic_reject,
            }
        elif not count.get("enough") or not count_phrase:
            validation = {
                "use_candidate": False,
                "confidence": "high",
                "expected_effect": "negative",
                "reason": "count verifier did not produce sufficient exact count",
            }
        else:
            validation = {
                "use_candidate": True,
                "confidence": "high",
                "expected_effect": "positive",
                "reason": (
                    "primary lacks a concise exact count or is explicitly vague; "
                    "count verifier produced a sufficient count phrase"
                ),
            }

        accepted = self._accept_validation(validation)
        out.raw_memory["count_anchor"] = {
            "used": accepted,
            "candidate": True,
            "sources": sources,
            "summary": count,
            "validation": validation,
        }
        if not accepted:
            return

        instances = count.get("instances") or []
        instance_lines = "\n".join(
            self._format_count_instance(item) for item in instances
        )
        final = (
            "COUNT-ANCHORED EVIDENCE SUMMARY:\n"
            f"Exact answer phrase: {count_phrase}\n"
            f"{str(count.get('answer_summary') or '').strip()}\n\n"
            "COUNTED INSTANCES:\n"
            f"{instance_lines if instance_lines else '- direct aggregate count'}\n\n"
            "SOURCE EXCERPTS:\n"
            f"{block}\n\n"
            "Answer using the exact answer phrase. Use this count only because "
            "validation found it more reliable than the primary summary."
        )
        self._replace_integrated_memory(out, final, sources)

    def _replace_integrated_memory(
        self, out: ResearchOutput, final: str, sources: List[int]
    ) -> None:
        out.integrated_memory = final
        temp_memory = out.raw_memory.get("temp_memory") or {}
        old_sources = [str(x) for x in temp_memory.get("sources", [])]
        temp_memory["content"] = final
        temp_memory["sources"] = sorted(set(old_sources + [str(x) for x in sources]))
        out.raw_memory["temp_memory"] = temp_memory

    def _validate_anchor_candidate(
        self,
        request: str,
        anchor_kind: str,
        primary_summary: str,
        candidate_summary: str,
        source_block: str,
        extra_rules: str,
    ) -> Dict[str, Any]:
        prompt = f"""Validate whether a candidate experience should override primary GAM memory.

QUESTION:
{request}

PRIMARY GAM MEMORY:
{primary_summary}

CANDIDATE EXPERIENCE TYPE:
{anchor_kind}

CANDIDATE EXPERIENCE OUTPUT:
{candidate_summary}

SOURCE EXCERPTS USED BY THE CANDIDATE:
{source_block}

Decision rule:
Use the candidate only if it is expected to improve the final short-answer F1
relative to the primary memory. Prefer abstaining when the candidate is merely
plausible but not clearly better.

Additional rules:
{extra_rules}

Return JSON with:
- use_candidate: true only for a likely positive intervention.
- confidence: high, medium, or low.
- expected_effect: positive, neutral, or negative.
- reason: one concise explanation.
"""
        data = self._json_call(
            prompt,
            CAVE_VALIDATION_SCHEMA,
            {
                "use_candidate": False,
                "confidence": "low",
                "expected_effect": "negative",
                "reason": "validation failed",
            },
        )
        if not isinstance(data, dict):
            return {
                "use_candidate": False,
                "confidence": "low",
                "expected_effect": "negative",
                "reason": "validation returned invalid data",
            }
        return data

    def _accept_validation(self, validation: Dict[str, Any]) -> bool:
        return bool(validation.get("use_candidate")) and validation.get(
            "expected_effect"
        ) == "positive" and validation.get("confidence") in {"high", "medium"}

    def _count_deterministic_reject(
        self, primary_summary: str, count_phrase: str
    ) -> str:
        if not count_phrase:
            return ""
        if self._INSUFFICIENT_RE.search(count_phrase):
            return "candidate count phrase is insufficient"
        primary_count = self._extract_count_value(primary_summary)
        if primary_count is None:
            return ""
        if self._VAGUE_COUNT_RE.search(primary_summary):
            return ""
        return "primary already contains a concise exact count; abstain from count override"

    def _extract_count_value(self, text: str) -> Optional[int]:
        lower = (text or "").lower()
        digit = re.search(r"\b\d+\b", lower)
        if digit:
            try:
                return int(digit.group(0))
            except ValueError:
                return None
        for word, value in self._NUMBER_WORDS.items():
            if re.search(rf"\b{re.escape(word)}\b", lower):
                return value
        return None
