"""Memory representation: boundary-respecting pages for the Memorizer.

GAM slices context into fixed 2048-token pages with no regard for structure.
Measured on HotpotQA eval_400 (400 Wikipedia documents concatenated into one
context):

    400 documents  ->  26 pages
    26/26 pages straddle more than one document boundary
    23/25 pages begin mid-sentence

Every page is therefore a blend of ~15 unrelated articles with both ends
severed. For multi-hop QA whose gold evidence is two specific documents, a
retrieved page delivers one relevant article plus fourteen distractors, and
possibly only half of the relevant one. The same slicing cuts NarrativeQA books
mid-scene.

This packs *whole units* into pages instead: split the context on its natural
boundaries, then greedily fill pages up to the token budget without ever
splitting a unit across pages. A unit larger than the budget is split, but only
as a fallback.

The unit is corpus-dependent and detected from the text, not hard-coded per
dataset:
  * an explicit ``Document N:`` marker (HotpotQA's MemAgent format)
  * otherwise paragraph breaks (NarrativeQA prose)

LoCoMo is untouched: it already chunks by conversation session, which is a
natural unit.
"""
from __future__ import annotations

import re
from typing import List, Optional

DOC_MARKER = re.compile(r"^(?=Document\s+\d+\s*:)", re.M)


def _tokenizer():
    import tiktoken

    return tiktoken.encoding_for_model("gpt-4o-2024-08-06")


def split_units(text: str) -> List[str]:
    """Split into natural units, preferring explicit document markers."""
    parts = [p for p in DOC_MARKER.split(text) if p.strip()]
    if len(parts) > 1:
        return parts
    parts = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    return parts if parts else ([text] if text.strip() else [])


def pack_units(
    text: str,
    max_tokens: int = 2048,
    label: str = "Session",
) -> List[str]:
    """Greedily pack whole units into pages of at most `max_tokens`."""
    if not text or not text.strip():
        return []

    enc = _tokenizer()
    units = split_units(text)

    pages: List[str] = []
    buf: List[str] = []
    buf_len = 0

    def flush():
        nonlocal buf, buf_len
        if buf:
            pages.append("\n\n".join(buf).strip())
            buf, buf_len = [], 0

    for unit in units:
        toks = enc.encode(unit, disallowed_special=())
        n = len(toks)

        if n > max_tokens:
            # A single oversized unit: emit what we have, then split this one.
            flush()
            for s in range(0, n, max_tokens):
                pages.append(enc.decode(toks[s : s + max_tokens]).strip())
            continue

        if buf_len + n > max_tokens:
            flush()
        buf.append(unit)
        buf_len += n

    flush()
    return [f"[{label} {i}]\n{p}" for i, p in enumerate(pages) if p]


def hotpotqa_chunker(sample, max_tokens: int = 2048, _unused=None) -> List[str]:
    return pack_units(sample.get("context") or "", max_tokens)


def narrativeqa_chunker(sample, max_tokens: int = 2048, _unused=None) -> List[str]:
    return pack_units(sample.get("document_text") or "", max_tokens)


def boundary_stats(text: str, pages: List[str]) -> dict:
    """Diagnostics for the paper: how well do pages respect document units?"""
    n_docs = len(DOC_MARKER.split(text)) - 1 if DOC_MARKER.search(text) else 0
    straddling = sum(1 for p in pages
                     if len(re.findall(r"Document\s+\d+\s*:", p)) > 1)
    mid_start = sum(
        1 for p in pages[1:]
        if not re.match(r"\[\w+ \d+\]\s*\nDocument", p)
    )
    return {
        "n_units": n_docs,
        "n_pages": len(pages),
        "pages_straddling_units": straddling,
        "pages_starting_mid_unit": mid_start,
    }
