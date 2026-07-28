"""API-backed replacements for GAM's two heavyweight retrievers.

GAM ships:
  * `DenseRetriever`  -> BGE-M3 via FlagEmbedding/sentence-transformers (needs torch + GPU)
  * `BM25Retriever`   -> pyserini (needs a JVM + Lucene)

Neither is usable in an API-only setting, so this module provides drop-in
replacements implementing the same `build/update/search/load` interface and
returning the same `List[List[Hit]]`.

Deviations from upstream, recorded deliberately so they can be reported:

  * Dense: OpenAI `text-embedding-3-small` (1536-d, cosine via normalised inner
    product) instead of BGE-M3 (1024-d). Different embedding space, so absolute
    dense-retrieval quality will differ from the paper.
  * BM25: pure-Python Okapi BM25 with Lucene's default parameters
    (k1=0.9, b=0.4) rather than pyserini's Lucene implementation. Tokenisation
    is lowercase alphanumeric, where Lucene applies its own analyzer chain.
    Rankings will be close but not identical.
  * By default both index `page.content` only, matching upstream. Note that BOTH
    of GAM's retrievers contain the same dead-code pattern -- they build
    `header + " " + content` and then overwrite it with `content` -- so the
    Memorizer's abstract is computed, stored and never indexed. Set
    `index_header=True` to restore it (see `apiharness/contextual.py`).
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import threading
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from openai import OpenAI

from gam.schemas import Hit

from .env import openai_embed_base_url, openai_embed_key

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


# --------------------------------------------------------------------------
# Embeddings
# --------------------------------------------------------------------------
class EmbeddingClient:
    """OpenAI embeddings with a content-addressed disk cache.

    Pages are re-embedded on every run otherwise, which dominates cost on the
    long-context benchmarks (a 448K-token HotpotQA sample is ~220 pages).
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        cache_dir: Optional[str] = None,
        batch_size: int = 128,
    ):
        self.model = model
        self.batch_size = batch_size
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._client = OpenAI(api_key=openai_embed_key(), base_url=openai_embed_base_url())
        self._mem: Dict[str, np.ndarray] = {}
        self._lock = threading.Lock()
        self.n_embedded = 0  # texts actually sent to the API

    def _key(self, text: str) -> str:
        return hashlib.sha256(f"{self.model}\x00{text}".encode("utf-8")).hexdigest()

    def _disk_path(self, key: str) -> Optional[Path]:
        if not self.cache_dir:
            return None
        sub = self.cache_dir / key[:2]
        sub.mkdir(parents=True, exist_ok=True)
        return sub / f"{key}.npy"

    def embed(self, texts: List[str]) -> np.ndarray:
        """Return L2-normalised embeddings, shape (len(texts), dim)."""
        if not texts:
            return np.zeros((0, 1), dtype="float32")

        out: List[Optional[np.ndarray]] = [None] * len(texts)
        missing: List[int] = []

        for i, t in enumerate(texts):
            key = self._key(t)
            with self._lock:
                vec = self._mem.get(key)
            if vec is None:
                path = self._disk_path(key)
                if path and path.exists():
                    try:
                        vec = np.load(path)
                    except Exception:
                        vec = None
            if vec is None:
                missing.append(i)
            else:
                out[i] = vec
                with self._lock:
                    self._mem[key] = vec

        for start in range(0, len(missing), self.batch_size):
            chunk = missing[start : start + self.batch_size]
            # The API rejects empty strings; substitute a single space.
            payload = [texts[i] if texts[i].strip() else " " for i in chunk]
            resp = self._client.embeddings.create(model=self.model, input=payload)
            self.n_embedded += len(chunk)
            for idx, item in zip(chunk, resp.data):
                vec = np.asarray(item.embedding, dtype="float32")
                norm = float(np.linalg.norm(vec))
                if norm > 0:
                    vec = vec / norm
                out[idx] = vec
                key = self._key(texts[idx])
                with self._lock:
                    self._mem[key] = vec
                path = self._disk_path(key)
                if path:
                    try:
                        np.save(path, vec)
                    except Exception:
                        pass

        return np.vstack([v for v in out if v is not None])


# --------------------------------------------------------------------------
# Dense retriever
# --------------------------------------------------------------------------
class APIDenseRetriever:
    """Drop-in for GAM's DenseRetriever, backed by API embeddings + numpy.

    Exact inner-product search. FAISS buys nothing at these corpus sizes
    (hundreds of pages); a dense matmul is both faster and dependency-free.
    """

    name = "vector"

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.embedder: EmbeddingClient = config["embedder"]
        self.index_header: bool = bool(config.get("index_header", False))
        self._matrix: Optional[np.ndarray] = None
        self._pages: List[Any] = []

    def _index_text(self, p) -> str:
        if self.index_header and getattr(p, "header", None):
            return f"{p.header}\n{p.content}"
        return p.content

    def build(self, page_store) -> None:
        self._pages = list(page_store.load())
        texts = [self._index_text(p) for p in self._pages]
        self._matrix = self.embedder.embed(texts) if texts else None

    def update(self, page_store) -> None:
        self.build(page_store)

    def load(self):
        return self._matrix

    def search(self, query_list: List[str], top_k: int = 10) -> List[List[Hit]]:
        if self._matrix is None or not len(self._pages):
            return [[] for _ in query_list]

        q = self.embedder.embed(list(query_list))
        scores = q @ self._matrix.T  # (n_queries, n_pages), cosine
        k = min(top_k, len(self._pages))

        results: List[List[Hit]] = []
        for row in scores:
            top = np.argpartition(-row, k - 1)[:k]
            top = top[np.argsort(-row[top])]
            results.append(
                [
                    Hit(
                        page_id=str(int(i)),
                        snippet=self._pages[int(i)].content,
                        source="vector",
                        meta={"score": float(row[int(i)])},
                    )
                    for i in top
                ]
            )
        return results


# --------------------------------------------------------------------------
# BM25 retriever
# --------------------------------------------------------------------------
class LiteBM25Retriever:
    """Pure-Python Okapi BM25 with Lucene defaults (k1=0.9, b=0.4)."""

    name = "keyword"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.k1 = float(self.config.get("k1", 0.9))
        self.b = float(self.config.get("b", 0.4))
        self.index_header: bool = bool(self.config.get("index_header", False))
        self._docs: List[List[str]] = []
        self._pages: List[Any] = []
        self._df: Counter = Counter()
        self._avgdl: float = 0.0
        self._tf: List[Counter] = []

    def _index_text(self, p) -> str:
        if self.index_header and getattr(p, "header", None):
            return f"{p.header}\n{p.content}"
        return p.content

    def build(self, page_store) -> None:
        self._pages = list(page_store.load())
        self._docs = [_tokenize(self._index_text(p)) for p in self._pages]
        self._tf = [Counter(d) for d in self._docs]
        self._df = Counter()
        for tf in self._tf:
            for term in tf:
                self._df[term] += 1
        self._avgdl = (
            sum(len(d) for d in self._docs) / len(self._docs) if self._docs else 0.0
        )

    def update(self, page_store) -> None:
        self.build(page_store)

    def load(self):
        return self._docs

    def _idf(self, term: str) -> float:
        n = len(self._docs)
        df = self._df.get(term, 0)
        # Lucene's BM25 idf, always positive.
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def _score_query(self, query: str) -> np.ndarray:
        terms = _tokenize(query)
        scores = np.zeros(len(self._docs), dtype="float32")
        if not terms or not self._docs:
            return scores
        for term in set(terms):
            idf = self._idf(term)
            if idf <= 0:
                continue
            for i, tf in enumerate(self._tf):
                f = tf.get(term, 0)
                if not f:
                    continue
                dl = len(self._docs[i]) or 1
                denom = f + self.k1 * (1 - self.b + self.b * dl / (self._avgdl or 1))
                scores[i] += idf * (f * (self.k1 + 1)) / denom
        return scores

    def search(self, query_list: List[str], top_k: int = 10) -> List[List[Hit]]:
        if not self._docs:
            return [[] for _ in query_list]
        results: List[List[Hit]] = []
        for query in query_list:
            scores = self._score_query(query)
            k = min(top_k, len(self._docs))
            top = np.argsort(-scores)[:k]
            results.append(
                [
                    Hit(
                        page_id=str(int(i)),
                        snippet=self._pages[int(i)].content,
                        source="keyword",
                        meta={"score": float(scores[int(i)])},
                    )
                    for i in top
                    if scores[int(i)] > 0
                ]
            )
        return results
