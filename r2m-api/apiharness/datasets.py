"""Unified dataset layer over the three benchmarks GAM and R²-Mem use.

Wherever possible this delegates to the *upstream* loaders, chunkers, prompt
builders and metrics in `R2M-official/eval/`, so our numbers stay comparable to
the papers. This module only normalises their three different shapes into one.

Structural difference worth knowing:
  * LoCoMo    -> one sample is a conversation carrying many questions; memory is
                 built once per conversation and shared across its questions.
  * HotpotQA  -> one sample IS one question, with its own concatenated context.
  * NarrativeQA  Memory must be rebuilt per question, which is why these two are
                 far more expensive per question than LoCoMo.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"


@dataclass
class QAItem:
    question: str
    gold: Any                      # str (LoCoMo) or List[str] (Hotpot/Narrative)
    category: Optional[int] = None
    qid: Optional[str] = None


@dataclass
class Sample:
    sample_id: str
    chunks: List[str]              # units fed to MemoryAgent.memorize
    qas: List[QAItem] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetSpec:
    name: str
    samples: List[Sample]
    # (category, summary, question, generator) -> answer string
    answerer: Callable[[Optional[int], str, str, Any], str]
    multi_qa_per_sample: bool


# --------------------------------------------------------------------------
def load_locomo(path: Optional[str] = None, max_tokens: int = 2048) -> DatasetSpec:
    from eval.locomo_test import (
        answer_with_summary,
        build_session_chunks_for_sample,
        collect_qa_items_for_sample,
        load_locomo as _load,
    )

    path = path or str(DATA_DIR / "locomo" / "locomo10.json")
    raw = _load(path)

    samples: List[Sample] = []
    for idx, s in enumerate(raw):
        qas = [
            QAItem(
                question=q.get("question") or "",
                gold=q.get("answer"),
                category=q.get("category"),
            )
            for q in collect_qa_items_for_sample(s)
            # Category 5 has no usable ground truth; upstream skips it too.
            if q.get("category") != 5
        ]
        samples.append(
            Sample(
                sample_id=s.get("sample_id", f"conv-{idx}"),
                chunks=build_session_chunks_for_sample(s),
                qas=qas,
            )
        )

    return DatasetSpec(
        name="locomo",
        samples=samples,
        answerer=lambda cat, summary, q, gen: answer_with_summary(cat, summary, q, gen),
        multi_qa_per_sample=True,
    )


# --------------------------------------------------------------------------
def _single_qa_spec(
    name: str, raw, chunker, make_prompt, max_tokens: int, question_key: str
):
    # Upstream is inconsistent: hotpotqa_test normalises the question to "input",
    # narrativeqa_test to "question". Chunking is deferred until the sample is
    # actually run, because tokenising every context up front is slow and these
    # files are hundreds of MB.
    samples: List[Sample] = []
    for idx, item in enumerate(raw):
        qid = item.get("_id", f"{name}-{idx}")
        samples.append(
            Sample(
                sample_id=str(qid),
                chunks=[],
                qas=[
                    QAItem(
                        question=item.get(question_key) or "",
                        gold=item.get("answers", []),
                        qid=str(qid),
                    )
                ],
                meta={"_id": qid, "_raw": item, "_chunker": chunker,
                      "_max_tokens": max_tokens},
            )
        )

    def answerer(cat, summary, question, gen, evidence=None):
        prompt = make_prompt(summary, question)
        if evidence:
            # Same intervention as on LoCoMo: the answer step otherwise sees only
            # the integrated paraphrase, never the cited source text. Spliced in
            # before the answer cue so the upstream instructions stay intact.
            block = ("\nSOURCE EXCERPTS (verbatim; prefer exact wording from here):\n"
                     f"{evidence}\n")
            for cue in ("Answer:", "Short answer:"):
                idx = prompt.rfind(cue)
                if idx != -1:
                    prompt = prompt[:idx] + block + "\n" + prompt[idx:]
                    break
            else:
                prompt = prompt + block
        out = gen.generate_single(prompt=prompt)
        return (out.get("text") or "").strip()

    return DatasetSpec(
        name=name, samples=samples, answerer=answerer, multi_qa_per_sample=False
    )


def load_hotpotqa(split: str = "eval_400", max_tokens: int = 2048) -> DatasetSpec:
    from eval.hotpotqa_test import (
        build_context_chunks_for_sample,
        load_hotpotqa as _load,
        make_prompt,
    )

    path = DATA_DIR / "hotpotqa" / f"{split}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing. Run the download step (see r2m-api/README.md)."
        )
    return _single_qa_spec(
        f"hotpotqa-{split}",
        _load(str(path)),
        build_context_chunks_for_sample,
        make_prompt,
        max_tokens,
        question_key="input",
    )


def _load_narrativeqa_parquet(data_dir: Path, split: str) -> List[Dict[str, Any]]:
    """Replicate `eval/narrativeqa_test.load_narrativeqa` without HF `datasets`.

    Upstream calls `load_dataset("parquet", data_dir=...)`, which drags in the
    whole `datasets` stack. We read the shards with pyarrow instead and emit
    field-for-field identical records, so the upstream chunker and metric work
    unchanged.
    """
    import pyarrow.parquet as pq

    shards = sorted(data_dir.glob(f"{split}-*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No {split}-*.parquet in {data_dir}.")

    rows: List[Dict[str, Any]] = []
    for shard in shards:
        rows.extend(pq.read_table(shard).to_pylist())

    out: List[Dict[str, Any]] = []
    for idx, item in enumerate(rows):
        document = item.get("document") or {}
        question = item.get("question") or {}
        document_text = document.get("text", "") if isinstance(document, dict) else ""
        document_id = (
            document.get("id", f"doc-{idx}") if isinstance(document, dict) else f"doc-{idx}"
        )
        question_text = question.get("text", "") if isinstance(question, dict) else ""

        answers: List[str] = []
        for ans in item.get("answers") or []:
            if isinstance(ans, dict):
                if ans.get("text"):
                    answers.append(ans["text"])
            elif isinstance(ans, str):
                answers.append(ans)

        out.append(
            {
                "index": idx,
                "document_text": document_text,
                "document_id": document_id,
                "question": question_text,
                "answers": answers,
                "_id": f"narrativeqa-{document_id}-{idx}",
            }
        )
    return out


def _narrativeqa_chunker(sample: Dict[str, Any], max_tokens: int, _unused=None) -> List[str]:
    """Fixed NarrativeQA chunker.

    Upstream's `narrativeqa_test._smart_split_by_tokens` calls
    `tokenizer.decode(chunk_tokens, skip_special_tokens=True)`, which is a
    HuggingFace kwarg that tiktoken's `Encoding.decode` rejects. The identical
    function in `hotpotqa_test.py` wraps that call in a try/except (their own
    comment marks it "关键修复点") but the fix was never ported here, so the
    tiktoken path crashes and NarrativeQA silently requires a local BGE-M3.

    This is upstream's algorithm with their fix applied: fixed-width splits of
    `max_tokens`, each prefixed `[Session {i}]`.
    """
    import tiktoken

    text = sample.get("document_text") or ""
    if not text:
        return []

    tokenizer = tiktoken.encoding_for_model("gpt-4o-2024-08-06")
    tokens = tokenizer.encode(text, disallowed_special=())
    if len(tokens) <= max_tokens:
        return [f"[Session 1]\n{text}"]

    chunks: List[str] = []
    session_id, start = 0, 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_text = tokenizer.decode(tokens[start:end])
        if chunk_text.strip():
            chunks.append(f"[Session {session_id}]\n{chunk_text.strip()}")
            session_id += 1
        start = end
    return chunks


def load_narrativeqa(split: str = "test", max_tokens: int = 2048) -> DatasetSpec:
    from eval.narrativeqa_test import make_prompt

    data_dir = DATA_DIR / "narrativeqa"
    if not any(data_dir.glob("*.parquet")):
        raise FileNotFoundError(
            f"No parquet files in {data_dir}. Run the download step."
        )
    return _single_qa_spec(
        "narrativeqa",
        _load_narrativeqa_parquet(data_dir, split),
        _narrativeqa_chunker,
        make_prompt,
        max_tokens,
        question_key="question",
    )


def materialise_chunks(sample: Sample) -> List[str]:
    """Chunk a single-QA sample on demand (see note in `_single_qa_spec`)."""
    if sample.chunks:
        return sample.chunks
    chunker = sample.meta.get("_chunker")
    if chunker is None:
        return []
    sample.chunks = chunker(
        sample.meta["_raw"], sample.meta["_max_tokens"], None
    )
    return sample.chunks


def use_boundary_chunkers(spec: DatasetSpec) -> DatasetSpec:
    """Swap in boundary-respecting chunkers (see apiharness/chunking.py)."""
    from .chunking import hotpotqa_chunker, narrativeqa_chunker

    if spec.name.startswith("hotpotqa"):
        ch = hotpotqa_chunker
    elif spec.name == "narrativeqa":
        ch = narrativeqa_chunker
    else:
        return spec  # LoCoMo already chunks by session, a natural unit
    for s in spec.samples:
        s.chunks = []
        s.meta["_chunker"] = ch
    return spec


LOADERS = {
    "locomo": load_locomo,
    "hotpotqa": load_hotpotqa,
    "narrativeqa": load_narrativeqa,
}
