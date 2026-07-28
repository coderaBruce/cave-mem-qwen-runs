"""R²-Mem on the API harness.

Faithful port of arXiv:2605.13486 onto our API-backed GAM. Prompt fidelity is
the priority, so everything that can be reused from `R2M-official/exp/` is:

  * Evaluator  -> `exp.judge_model.gpt_judger` is called *as-is*, with the module's
    `OpenAI` symbol patched so its hardcoded third-party gateway
    ("chatapi.littlewheat.com") is replaced by the real endpoint. The rubric
    text, the 4x0-3 dimensions, model (gpt-4o), temperature (0.2) and
    `response_format=json_object` are all upstream's.
  * Learner    -> `exp.prompts.self_reflection_prompts` verbatim, including the
    role_desc / thinking / task strings built in `exp/self_reflection.py`.
  * Online     -> `exp.prompts.research_prompts_exp.*_PROMPT_exp` verbatim.

Replaced: FAISS `IndexFlatIP` -> numpy exact inner product (a few hundred
entries; FAISS buys nothing), BGE-M3 -> API embeddings.

Thresholds default to the paper's K_low=5 / K_high=10 and Top-K=3.
"""
from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from gam.agents.research_agent import ResearchAgent
from gam.schemas import (
    GENERATE_REQUESTS_SCHEMA,
    INFO_CHECK_SCHEMA,
    PLANNING_SCHEMA,
    MemoryState,
    ReflectionDecision,
    SearchPlan,
)

from .retrievers import EmbeddingClient

K_LOW_DEFAULT = 5
K_HIGH_DEFAULT = 10
TOPK_DEFAULT = 3

_SITUATION_SCHEMA = {
    "type": "object",
    "properties": {"situation": {"type": "string"}},
    "required": ["situation"],
    "additionalProperties": False,
}

_EXPERIENCE_SCHEMA = {
    "type": "object",
    "properties": {
        "thinking": {"type": "string"},
        "summary": {"type": "string"},
        "situation": {"type": "string"},
        "experience": {"type": "string"},
    },
    "required": ["thinking", "summary", "situation", "experience"],
    "additionalProperties": False,
}


# ==========================================================================
# Experience bank
# ==========================================================================
@dataclass
class Experience:
    module: str          # "Planning" | "Reflection"
    quality: str         # "good" | "bad"
    condition: str
    situation: str
    summary: str
    experience: str
    score: Optional[int] = None
    source: Optional[str] = None
    # Filled in by our extensions, unused by the faithful R²-Mem port.
    extra: Dict[str, Any] = field(default_factory=dict)

    def index_text(self) -> str:
        """Upstream keys the index on "(situation)" + condition."""
        return f"({self.situation})" + self.condition


class ExperienceBank:
    """Numpy-backed replacement for R²-Mem's two FAISS banks."""

    def __init__(self, embedder: EmbeddingClient):
        self.embedder = embedder
        self.entries: List[Experience] = []
        self._matrix: Optional[np.ndarray] = None
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self.entries)

    def add(self, exp: Experience) -> None:
        with self._lock:
            self.entries.append(exp)
            self._matrix = None  # invalidate

    def _ensure_matrix(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._matrix is None and self.entries:
                self._matrix = self.embedder.embed(
                    [e.index_text() for e in self.entries]
                )
            return self._matrix

    def search(
        self, query_text: str, module: str, top_k: int = TOPK_DEFAULT
    ) -> List[Tuple[Experience, float]]:
        matrix = self._ensure_matrix()
        if matrix is None or not self.entries:
            return []

        mask = np.array([e.module == module for e in self.entries])
        if not mask.any():
            return []

        q = self.embedder.embed([query_text])[0]
        scores = matrix @ q
        scores = np.where(mask, scores, -np.inf)

        k = min(top_k, int(mask.sum()))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(self.entries[int(i)], float(scores[int(i)])) for i in top]

    # ---- persistence ----
    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                [
                    {
                        "module": e.module,
                        "quality": e.quality,
                        "condition": e.condition,
                        "situation": e.situation,
                        "summary": e.summary,
                        "experience": e.experience,
                        "score": e.score,
                        "source": e.source,
                        "extra": e.extra,
                    }
                    for e in self.entries
                ],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path, embedder: EmbeddingClient) -> "ExperienceBank":
        bank = cls(embedder)
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        bank.entries = [Experience(**d) for d in data]
        return bank

    def stats(self) -> Dict[str, int]:
        out: Dict[str, int] = {"total": len(self.entries)}
        for e in self.entries:
            out[f"{e.module}/{e.quality}"] = out.get(f"{e.module}/{e.quality}", 0) + 1
        return out


# ==========================================================================
# Offline stage: Evaluator + Learner
# ==========================================================================
def _patched_judge_client(api_key: str, base_url: Optional[str]):
    """Redirect `judge_model`'s hardcoded gateway to the real endpoint."""
    import exp.judge_model as jm
    from openai import OpenAI as _RealOpenAI

    def _factory(*_args, **kwargs):
        return _RealOpenAI(api_key=api_key, base_url=base_url)

    jm.OpenAI = _factory
    return jm


def trace_to_string(trace: Dict[str, Any]) -> str:
    """Upstream `Trace_str`, but taking a dict rather than a file path."""
    out = ""
    for step in trace.get("iterations", []):
        out += f"(step {step['step']})\n"
        out += (
            "[Planning]\n"
            f"plan: {step['plan']}\n"
            "[Reflection]\n"
            f"'temp_memory': {step['temp_memory']}\n"
            f"'decision': {step['decision']}\n\n"
        )
    out += "[Answering]\n" + f"integrated_memory:{trace.get('integrated_memory','')}"
    return out


EVAL_CACHE_DIR = Path(__file__).resolve().parents[1] / ".cache" / "evaluator"


def evaluate_trace(
    question: str,
    reference_answer: str,
    model_answer: str,
    trace: Dict[str, Any],
    api_key: str,
    base_url: Optional[str] = None,
    use_cache: bool = True,
) -> Tuple[Dict[str, Any], int]:
    """Rubric-guided Evaluator (upstream `gpt_judger`, gpt-4o, temperature 0.2).

    Cached on disk: we build several bank variants off the same traces, and
    GPT-4o is the expensive part of the whole pipeline. A cache hit reports the
    original token count so cost accounting stays honest.
    """
    import hashlib

    trace_str = trace_to_string(trace)
    key = hashlib.sha256(
        json.dumps(
            [question, reference_answer, model_answer, trace_str], ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()
    path = EVAL_CACHE_DIR / key[:2] / f"{key}.json"

    if use_cache and path.exists():
        try:
            blob = json.loads(path.read_text(encoding="utf-8"))
            return blob["content"], blob["tokens"]
        except Exception:
            pass

    jm = _patched_judge_client(api_key, base_url)
    content, tokens = jm.gpt_judger(
        question, reference_answer, model_answer, trace_str, api_key
    )
    if use_cache and content.get("results"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"content": content, "tokens": tokens}, ensure_ascii=False),
            encoding="utf-8",
        )
    return content, tokens


def step_context(trace: Dict[str, Any], step: int, module: str) -> Tuple[str, str, str]:
    """Upstream `info_prompt`: recover (current_query, info, condition) for a step."""
    question = trace.get("question", "")
    current_question, info, condition = question, "", ""
    for item in trace.get("iterations", []):
        if str(item["step"]) != str(step):
            continue
        if module == "Planning":
            if str(step) == "0":
                current_question = question
            else:
                prev = trace["iterations"][int(step) - 1]["decision"]
                current_question = prev.get("new_request") or question
            condition = current_question
            info = f'"question": {current_question}\n"plan": {item["plan"]}'
        else:
            content = item["temp_memory"].get("content", "")
            info = f'"temp_memory": {content}\n"decision": {item["decision"]}'
            current_question = question
            condition = f"question:{question} temp_memory:{content}"
        break
    return current_question, info, condition


def distil_experience(
    module: str,
    quality: str,
    question: str,
    info: str,
    reason_and_advice: str,
    generator,
) -> Tuple[Optional[Dict[str, Any]], int]:
    """self-Reflection Learner. Role strings copied from `exp/self_reflection.py`."""
    import exp.prompts.self_reflection_prompts as sr

    if module == "Planning":
        if quality == "bad":
            role_desc = ("You are an AI [TRACE] Auditor: Extract abstract situation "
                         "and corrective planning experience from low-quality planning traces.")
            task = ("summarize a corrective and discriminative planning experience to "
                    "prevent similar low-quality planning behaviors.")
            thinking = ("Why is this [TRACE] considered low quality under the "
                        "[DIAGNOSED REASON]. Where did it perform worst?")
        else:
            role_desc = ("You are an AI [TRACE] Strategist: Distill best-practice planning "
                         "patterns and reusable experience from high-quality planning traces.")
            task = ("summarize an effective and discriminative planning experience that "
                    "leads to high-quality planning behaviors.")
            thinking = ("Why is this [TRACE] considered high quality under the "
                        "[DIAGNOSED REASON]. Where did it perform best?")
        system_prompt = sr.Planning_system_prompt.format(role_desc=role_desc)
        user_prompt = sr.Planning_prompt.format(
            QUESTION=question,
            content_type="low_quality" if quality == "bad" else "high_quality",
            info=info,
            content_reason=reason_and_advice,
            thinking=thinking,
            planning_task=task,
        )
    else:
        if quality == "bad":
            role_desc = ("You are an AI Memory [TRACE] Auditor: Extract abstract abstract "
                         "situation and corrective from experience low-quality reasoning traces.")
            thinking = ("Why is this [TRACE] considered low quality under the evaluation? "
                        "Where did it perform worst?")
            experience = ("summarize an abstract and reusable experience to prevent similar "
                          "low-quality reflection behaviors in future scenarios.")
        else:
            role_desc = ("You are an AI Memory [TRACE] Strategist: Distill best-practice "
                         "patterns and reusable experience from high-quality reasoning traces.")
            thinking = ("Why is this [TRACE] considered high quality under the evaluation? "
                        "Where did it perform best?")
            experience = ("summarize an abstract and reusable experience to lead high-quality "
                          "reflection behaviors in future scenarios.")
        system_prompt = sr.Reflection_system_prompt.format(role_desc=role_desc)
        user_prompt = sr.Reflection_prompt.format(
            QUESTION=question,
            content_type="low_quality" if quality == "bad" else "high_quality",
            info=info,
            content_reason=reason_and_advice,
            thinking=thinking,
            experience=experience,
        )

    resp = generator.generate_single(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        schema=_EXPERIENCE_SCHEMA,
    )
    tokens = 0
    try:
        tokens = int(resp["response"]["usage"]["total_tokens"])
    except Exception:
        pass
    data = resp.get("json") or _loose_json(resp.get("text", ""))
    return data, tokens


def _loose_json(text: str) -> Optional[Dict[str, Any]]:
    m = re.search(r"```json\n(.*?)\n```", text, re.DOTALL)
    blob = m.group(1) if m else text
    try:
        return json.loads(blob)
    except Exception:
        start, end = blob.find("{"), blob.rfind("}")
        if start != -1 and end != -1:
            try:
                return json.loads(blob[start : end + 1])
            except Exception:
                return None
    return None


def build_bank_from_traces(
    records: List[Dict[str, Any]],
    bank: ExperienceBank,
    learner_generator,
    evaluator_api_key: str,
    evaluator_base_url: Optional[str] = None,
    k_low: int = K_LOW_DEFAULT,
    k_high: int = K_HIGH_DEFAULT,
    keep_middle: bool = False,
    on_progress=None,
) -> Dict[str, Any]:
    """Algorithm 1: evaluate each step, distil the extremes, store.

    `records` is a list of {question, gold, pred, trace}.
    `keep_middle=True` is *our* extension (see notes/01-baseline-critique.md
    §3.2 — upstream `continue`s over mid-scoring steps and throws the mass of
    the distribution away). It is off by default so the port stays faithful.
    """
    stats = {
        "n_traces": len(records),
        "n_steps_scored": 0,
        "n_good": 0,
        "n_bad": 0,
        "n_middle_skipped": 0,
        "evaluator_tokens": 0,
        "learner_tokens": 0,
        "score_hist": {},
        "errors": [],
    }

    for ridx, rec in enumerate(records):
        try:
            content, ev_tokens = evaluate_trace(
                rec["question"],
                str(rec.get("gold", "")),
                str(rec.get("pred", "")),
                rec["trace"],
                evaluator_api_key,
                evaluator_base_url,
            )
            stats["evaluator_tokens"] += ev_tokens
            results = content.get("results", [])
            if not results:
                stats["errors"].append(
                    {"idx": ridx, "why": content.get("error", "no results")}
                )
                continue

            for item in results:
                rubrics = item.get("rubrics", {}) or {}
                if not rubrics:
                    continue
                total = sum(int(v) for v in rubrics.values())
                module = item.get("module")
                if module not in ("Planning", "Reflection"):
                    continue
                stats["n_steps_scored"] += 1
                stats["score_hist"][total] = stats["score_hist"].get(total, 0) + 1

                if total >= k_high:
                    quality = "good"
                elif total <= k_low:
                    quality = "bad"
                elif keep_middle:
                    quality = "middle"
                else:
                    stats["n_middle_skipped"] += 1
                    continue

                question, info, condition = step_context(
                    rec["trace"], item.get("step", 0), module
                )
                data, lt = distil_experience(
                    module,
                    "bad" if quality == "bad" else "good",
                    question,
                    info,
                    item.get("reason and advice", ""),
                    learner_generator,
                )
                stats["learner_tokens"] += lt
                if not data or not data.get("experience"):
                    continue

                bank.add(
                    Experience(
                        module=module,
                        quality=quality,
                        condition=condition,
                        situation=data.get("situation", ""),
                        summary=data.get("summary", ""),
                        experience=data["experience"],
                        score=total,
                        source=rec.get("source"),
                        extra={"rubrics": rubrics},
                    )
                )
                stats["n_good" if quality == "good" else "n_bad"] += 1
        except Exception as exc:
            stats["errors"].append({"idx": ridx, "why": f"{type(exc).__name__}: {exc}"})

        if on_progress:
            on_progress(ridx + 1, len(records), stats)

    return stats


# ==========================================================================
# Online stage: experience-augmented researcher
# ==========================================================================
def _abstract_situation_planning(generator, query: str) -> Tuple[str, int]:
    """Verbatim from `exp/eval/research_agent_exp_main.generate_situation_planning`."""
    system_prompt = "\n    You are an expert in abstracting situation.\n    "
    user_prompt = f"""
    [question]: {query}

    ***YOUR TASK***
    abstract situation of the [question](Do NOT include concrete entities):
    Summarize the abstract planning situation, What is this question trying to ask?
    - Multi-entity query
    - Time-related query
    - list/set
    - single_fact
    - multi_hop
    ...

    Generate the standardized situation(one concise sentence).

    Output ONLY valid JSON:
    {{
        "situation": "<abstract situation>"
    }}

    """
    prompt = f"System Prompt: {system_prompt}\n\nUser Instructions: {user_prompt}"
    resp = generator.generate_single(prompt=prompt, schema=_SITUATION_SCHEMA)
    tokens = int(resp["response"]["usage"]["total_tokens"])
    data = resp.get("json") or _loose_json(resp.get("text", "")) or {}
    return data.get("situation", query), tokens


def _abstract_situation_reflection(generator, query: str, temp_memory) -> Tuple[str, int]:
    """Verbatim from `..._main.generate_situation_reflection`."""
    system_prompt = "\n    You are an expert in abstracting situation.\n    "
    user_prompt = f"""
    [question]: {query}
    [temp_memory]: {temp_memory}

    ***YOUR TASK***
    abstract situation (Do NOT include concrete entities):
        Summarize the abstract situation(question and temp memory), considering:
        What is this question trying to ask?
        What does the temp_memory already contain?
        Whether the current information in temp_memory is already sufficient to solve the question?

    Generate the standardized situation.

    Output ONLY valid JSON:
    {{
        "situation": "<abstract situation>"
    }}

    """
    prompt = f"System Prompt: {system_prompt}\n\nUser Instructions: {user_prompt}"
    resp = generator.generate_single(prompt=prompt, schema=_SITUATION_SCHEMA)
    tokens = int(resp["response"]["usage"]["total_tokens"])
    data = resp.get("json") or _loose_json(resp.get("text", "")) or {}
    return data.get("situation", query), tokens


def format_experiences(hits: List[Tuple[Experience, float]]) -> str:
    if not hits:
        return "No relevant experience."
    return "\n".join(
        f"{i}. [{e.quality}] {e.experience}" for i, (e, _) in enumerate(hits, 1)
    )


class ResearchAgentExp(ResearchAgent):
    """GAM's researcher with experience injected into planning and reflection.

    Only `_planning` and `_reflection` are overridden; searching and integration
    are inherited unchanged, matching upstream's `ResearchAgent_exp`.
    """

    def __init__(self, *args, bank: ExperienceBank, top_k: int = TOPK_DEFAULT,
                 select=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.bank = bank
        self.top_k = top_k
        # Hook for our variants: (hits, module, condition) -> filtered hits.
        self.select = select
        self.experience_log: List[Dict[str, Any]] = []

    def research(self, request: str, category: int = 0):
        """Reset the per-question experience log, then attach it to the trace."""
        self.experience_log = []
        out = super().research(request)
        try:
            out.raw_memory["experience_log"] = list(self.experience_log)
        except Exception:
            pass
        return out

    def _retrieve(self, module: str, condition: str, situation: str):
        hits = self.bank.search(f"({situation}){condition}", module, self.top_k)
        if self.select is not None:
            hits = self.select(hits, module, condition)
        self.experience_log.append(
            {
                "module": module,
                "situation": situation,
                "retrieved": [
                    {"quality": e.quality, "score": s, "experience": e.experience}
                    for e, s in hits
                ],
            }
        )
        return hits

    def _planning(self, request, memory_state, planning_prompt=None) -> SearchPlan:
        import exp.prompts.research_prompts_exp as rp

        if not memory_state.abstracts:
            memory_context = "No memory currently."
        else:
            memory_context = "\n".join(
                f"Page {i}: {a}" for i, a in enumerate(memory_state.abstracts)
            )

        situation, tok = _abstract_situation_planning(self.generator, request)
        self.total_tokens += tok
        hits = self._retrieve("Planning", request, situation)

        prompt = rp.Planning_PROMPT_exp.format(
            request=request,
            memory=memory_context,
            experience=format_experiences(hits),
        )
        try:
            resp = self.generator.generate_single(prompt=prompt, schema=PLANNING_SCHEMA)
            self.total_tokens += int(resp["response"]["usage"]["total_tokens"])
            data = resp.get("json") or json.loads(resp["text"])
            return SearchPlan(
                info_needs=data.get("info_needs", []),
                tools=data.get("tools", []),
                keyword_collection=data.get("keyword_collection", []),
                vector_queries=data.get("vector_queries", []),
                page_index=data.get("page_index", []),
            )
        except Exception as exc:
            print(f"Error in planning(exp): {exc}")
            return SearchPlan()

    def _reflection(self, request, result, reflection_prompt=None) -> ReflectionDecision:
        import exp.prompts.research_prompts_exp as rp

        situation, tok = _abstract_situation_reflection(
            self.generator, request, result.content
        )
        self.total_tokens += tok
        condition = f"question:{request} temp_memory:{result.content}"
        hits = self._retrieve("Reflection", condition, situation)
        experience_text = format_experiences(hits)

        try:
            check_prompt = rp.InfoCheck_PROMPT_exp.format(
                request=request, result=result.content, experience=experience_text
            )
            resp = self.generator.generate_single(
                prompt=check_prompt, schema=INFO_CHECK_SCHEMA
            )
            self.total_tokens += int(resp["response"]["usage"]["total_tokens"])
            data = resp.get("json") or json.loads(resp["text"])
            if data.get("enough", False):
                return ReflectionDecision(enough=True, new_request=None)

            gen_prompt = rp.GenerateRequests_PROMPT_exp.format(
                request=request, result=result.content, experience=experience_text
            )
            resp2 = self.generator.generate_single(
                prompt=gen_prompt, schema=GENERATE_REQUESTS_SCHEMA
            )
            self.total_tokens += int(resp2["response"]["usage"]["total_tokens"])
            data2 = resp2.get("json") or json.loads(resp2["text"])
            reqs = data2.get("new_requests", [])
            return ReflectionDecision(
                enough=False,
                new_request=" ".join(reqs) if isinstance(reqs, list) and reqs else None,
            )
        except Exception as exc:
            print(f"Error in reflection(exp): {exc}")
            return ReflectionDecision(enough=False, new_request=None)
